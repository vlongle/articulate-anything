import argparse
import trimesh
import xml.etree.ElementTree as ET
import copy
import base64
import cv2
import numpy as np
import sapien.core as sapien
import re
import logging
import os
import glob
from PIL import Image
from articulate_anything.agent.agent import Agent
import json
from articulate_anything.utils.utils import (
    save_json,
    join_path,
    file_to_string,
)

PARTNET_MESH_ANNOTATOR_SYSTEM_INSTRUCTION = """
You are a helpful part annotator whose job is to give each link specified in the semantic label more detailed descriptions. You are given a short clip of the assembled object and image of each part.
Details include: (material properties, shape, function, etc.)

The semantic label file look like this:

```
link_0 <joint_type> <semantic_label>
link_1 <joint_type> <semantic_label>
...
```

return a detailed description of each part mentioned in the semantic label file in JSON format:
```json
{
    "reasoning": "I see... ",
    "annotation": {
        "object": "<overall object description>", # eg. a wooden cabinet with a drawer and a door.
        "link_0": "<description>", # eg. "A wooden drawer that slides in and out."
        "link_1": "<description>",
    }
    ...
}
```
Tips:

- Do not label parts as things they are not. If the semantic says it is a drawer, do not label it as a door.
- If you cannot see the part in the frames, simply say: "a <semantic_label> for <the object that you see>"
- Give EXACTLY the number of parts mentioned in the semantic label file. No more, no less.
- Do not speculate on the motion of the parts if you do not see any motion in the frames.
- Use strong, confident language.
- You might comment on the material, shap, texture but do NOT comment on the color as we will not render the color in the final output.
- Later, we will run a CLIP model to get embeddings for each part mesh, and retrieve the part meshes as part of our text-to-3D pipeline.
    - Thus, you need to be specific in your `link` descriptions to ensure the CLIP model can accurately retrieve the correct part mesh.
    For example, there are `toilet_body` that are intended to be a public_toilet or be wall-mounted. These toilets would visually look very different
    from say a in-home toilet. Your descriptions should be specific enough to differentiate between these two types of toilets.
"""


class PartNetMeshAnnotator(Agent):
    OUT_RESULT_PATH = "object_annotation_gemini.json"

    def _make_system_instruction(self):
        return PARTNET_MESH_ANNOTATOR_SYSTEM_INSTRUCTION

    def _make_prompt_parts(self, partnet_dir):
        object_image_paths = glob.glob(f'{partnet_dir}/object_view_*.jpg')
        object_image_prompt = ['Images of the Assembled Object:\n']
        for object_image_path in object_image_paths:
            object_image_prompt.append(Image.open(object_image_path))

        part_image_paths = glob.glob(f'{partnet_dir}/part_link*.jpg')

        part_image_prompt = ['Images of the Parts:\n']
        for part_image_path in part_image_paths:
            link_name = 'link' + \
                os.path.basename(part_image_path).split('_')[2].split('.')[0]
            part_image_prompt.append(f"Part: {link_name}\n")
            part_image_prompt.append(Image.open(part_image_path))

        semantics = file_to_string(join_path(partnet_dir, 'semantics.txt'))
        message = """
            Please help me with this object:

            ```
            <semantics>
            ```
            """
        message = message.replace("<semantics>", "".join(semantics))
        prompt = [message] + object_image_prompt + part_image_prompt
        return prompt

    def parse_response(self, response, **kwargs):
        json_str = response.text.strip().strip('```json').strip()

        parsed_response = json.loads(json_str, strict=False)
        logging.info(f"PartNet mesh annotator response: {parsed_response}")

        save_json(parsed_response, join_path(
            self.cfg.out_dir, self.OUT_RESULT_PATH))


def combine_meshes(obj_dir):
    urdf_path = join_path(obj_dir, "mobility.urdf")
    logging.info(f"Processing {urdf_path}")
    try:
        tree = ET.parse(urdf_path)
        root = tree.getroot()

        for link in root.findall('link'):
            link_name = link.attrib['name']
            mesh_export_path = join_path(
                obj_dir, f"{link_name}_combined_mesh.obj")
            if os.path.exists(mesh_export_path):
                continue
            if link_name == "base_link":
                continue
            link_meshes = []

            for visual in link.findall('visual'):
                origin = visual.find('origin')
                xyz, rpy = np.zeros(3), np.zeros(3)
                if 'xyz' in origin.attrib:
                    xyz = np.array([float(x)
                                   for x in origin.attrib['xyz'].split()])
                if 'rpy' in origin.attrib:
                    rpy = np.array([float(x)
                                   for x in origin.attrib['rpy'].split()])
                for geometry in visual.findall('geometry'):
                    mesh = geometry.find('mesh')
                    mesh_path = mesh.attrib['filename']
                    mesh_path = join_path(obj_dir, mesh_path)
                    mesh = trimesh.load(mesh_path)
                    link_meshes.append(mesh)
            combined_mesh = trimesh.util.concatenate(link_meshes)
            combined_mesh.export(mesh_export_path)
    except Exception as e:
        logging.error(f"Failed to process {urdf_path}")
        logging.error(e)

    return None


# =========================================================
# Legacy code to render object in different views
# =========================================================


def regex_path_exists(directory, regex_pattern):
    # Compile the regular expression pattern
    pattern = re.compile(regex_pattern)

    # List all files and directories in the specified directory
    entries = os.listdir(directory)

    # Check each entry against the regex pattern
    for entry in entries:
        if pattern.match(entry):
            return True  # Return True if a match is found

    return False


def load_scene():
    # Initialize the SAPIEN engine and renderer
    engine = sapien.Engine()
    renderer = sapien.SapienRenderer(device="cuda")
    engine.set_renderer(renderer)
    scene = engine.create_scene()
    scene.set_timestep(1 / 100.0)
    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5], shadow=True)
    scene.add_point_light([1, 2, 2], [1, 1, 1])
    scene.add_point_light([1, -2, 2], [1, 1, 1])
    scene.add_point_light([-1, 0, 1], [1, 1, 1])
    return scene


def get_view_matrix(cam_pos):
    forward = -cam_pos / np.linalg.norm(cam_pos)
    left = np.cross([0, 0, 1], forward)
    left = left / np.linalg.norm(left)
    up = np.cross(forward, left)
    mat44 = np.eye(4)
    mat44[:3, :3] = np.stack([forward, left, up], axis=1)
    mat44[:3, 3] = cam_pos
    return mat44


def step(scene, camera):
    scene.step()  # run a physical step
    scene.update_render()  # sync pose from SAPIEN to renderer
    camera.take_picture()

    rgba = camera.get_float_texture("Color")
    rgba_img = (rgba * 255).clip(0, 255).astype("uint8")
    bgr_img = cv2.cvtColor(rgba_img, cv2.COLOR_RGB2BGR)
    return bgr_img


def save_isolated_link(link, root, file_name):
    # Create a new tree structure for the robot
    robot = ET.Element('robot', name=root.attrib['name'])
    new_tree = ET.ElementTree(element=robot)

    # Add the isolated link to the new robot tree as child of the root
    robot.append(copy.deepcopy(link))
    # Add base_link
    link_base = ET.Element('link', name='base')
    robot.append(link_base)
    # Add joint between base and isolated link
    joint = ET.Element('joint', name='base_joint', type='fixed')
    origin = ET.Element('origin', xyz='0 0 0', rpy='0 0 0')
    joint.append(origin)
    parent = ET.Element('parent', link='base')
    joint.append(parent)
    child = ET.Element('child', link=link.attrib['name'])
    joint.append(child)
    robot.append(joint)

    # Write to file
    new_tree.write(file_name, encoding='utf-8', xml_declaration=True)


def render_partnet_views(partnet_dir, overwrite=False):
    scene = load_scene()
    loader = scene.create_urdf_loader()
    logging.info(f"Processing {partnet_dir}..")
    urdf_file = join_path(partnet_dir, 'mobility.urdf')

    # rendering
    if not regex_path_exists(partnet_dir, r'object_view_.*\.jpg') \
            or not regex_path_exists(partnet_dir, r'part_.*\.jpg') or overwrite:

        render_xml_tree = ET.parse(urdf_file)
        root = render_xml_tree.getroot()
        logging.info(f"  loaded urdf tree from {urdf_file}")

        loader.fix_root_link = True
        logging.info(f"  🏗️ Loaded scene")

        # asset = loader.load(urdf_file)
        asset = loader.load_kinematic(urdf_file)
        # for link in asset.get_links():
        #     link.disable_gravity = True

        limits = []
        for joint in asset.get_joints():
            if joint.type in ["revolute", "prismatic"]:
                lower, upper = joint.get_limits()[0]
                limits.append([lower, upper])

        limits = np.array(limits)
        limits = np.clip(limits, -np.pi, np.pi)

        near, far = 0.1, 100
        width, height = 480, 480

        # Compute the camera pose by specifying forward(x), left(y) and up(z)
        cam_pos_front = np.array([2, 2, 2])
        cam_pos_back = np.array([-2, -2, 2])
        mat_front = get_view_matrix(cam_pos_front)
        mat_back = get_view_matrix(cam_pos_back)
        view_mats = [mat_front, mat_back]

        camera = scene.add_camera(
            name="camera",
            width=width,
            height=height,
            fovy=np.deg2rad(35),
            near=near,
            far=far,
        )
        images_base64 = []

        for i, mat in enumerate(view_mats):
            camera.set_pose(sapien.Pose(mat))

            steps = 3

            for j in range(steps):
                progress = j / steps
                qpos = limits[:, 0] + progress * \
                    (limits[:, 1] - limits[:, 0])
                asset.set_qpos(qpos)

                img = step(scene, camera)

                # save image
                cv2.imwrite(
                    f'{partnet_dir}/object_view_{i}_step{j}.jpg', img)

                _, buffer = cv2.imencode('.jpg', img)
                images_base64.append(
                    base64.b64encode(buffer).decode('utf-8'))

                logging.info(f"  📸 Rendered object_view_{i}_step{j}.jpg")
        # also render each part separatelt by creating temporary urdf files that isolates each link
        scene.remove_camera(camera)
        # scene.remove_articulation(asset)
        scene.remove_kinematic_articulation(asset)

        for i, link in enumerate(root.findall('link')):
            link_name = link.attrib['name']
            if link_name == 'base':
                continue

            temp_urdf_path = join_path(
                partnet_dir, f'temp_link_{i}.urdf')
            save_isolated_link(link, root, temp_urdf_path)

            camera = scene.add_camera(
                name="camera",
                width=width,
                height=height,
                fovy=np.deg2rad(65),
                near=near,
                far=far,
            )
            camera.set_pose(sapien.Pose(mat_front))

            # asset = loader.load(temp_urdf_path)
            asset = loader.load_kinematic(temp_urdf_path)
            img = step(scene, camera)
            cv2.imwrite(f'{partnet_dir}/part_{link_name}.jpg', img)
            os.remove(temp_urdf_path)
            scene.remove_camera(camera)
            # scene.remove_articulation(asset)
            scene.remove_kinematic_articulation(asset)

        logging.info(f"  ✨✅ Rendered {partnet_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a textured 3D model with spotted texture."
    )
    parser.add_argument("partnet_dir", help="PartNet object directory")
    parser.add_argument("--render_part_views", action="store_true",)
    args = parser.parse_args()
    if args.render_part_views:
        render_partnet_views(args.partnet_dir)
    combine_meshes(args.partnet_dir)
