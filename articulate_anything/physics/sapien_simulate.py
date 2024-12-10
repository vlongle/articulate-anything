import hydra
from omegaconf import OmegaConf, DictConfig
from scipy.spatial.transform import Rotation as R
import sapien.core as sapien
import numpy as np
import cv2
from sapien.utils import Viewer
from articulate_anything.utils.utils import (
    load_json,
    join_path,
    create_dir,
)
from functools import partial
import os
from omegaconf import DictConfig
import logging
from articulate_anything.physics.sapien_render import (
    setup_cameras,
    take_camera_pic,
    VideoWriterManager,
    save_image,
    flip_video,
)
from typing import Any, Dict
from sapien.asset import create_checkerboard
from articulate_anything.physics.pybullet_utils import setup_pybullet
import pybullet as p


def add_checkerboard(scene, renderer):
    # Set up a default camera to initialize the renderer
    scene.add_camera("init_camera", width=1, height=1,
                     fovy=1.0, near=0.1, far=100)
    # Create a checkered ground
    checker_size = 0.5  # Size of each checker square
    checker_cols = 50  # Number of columns
    checker_rows = 50  # Number of rows
    color1 = [0.8, 0.8, 0.8, 1]  # Light gray
    color2 = [0.3, 0.3, 0.3, 1]  # Dark gray

    checker_mesh, checker_material = create_checkerboard(
        renderer,
        shape=(checker_rows, checker_cols),
        length=checker_size,
        color1=color1,
        color2=color2
    )

    ground_actor = scene.create_actor_builder().add_visual_from_mesh(
        checker_mesh,
        material=checker_material,
    ).build_static()

    # Position the ground to be centered under the obj
    # Slightly below z=0 to avoid z-fighting
    ground_actor.set_pose(sapien.Pose([-8, -8, 0]))


def make_floor(floor_texture, scene, renderer):
    if floor_texture == "plain":
        scene.add_ground(altitude=0)
    elif floor_texture == "checkerboard":
        add_checkerboard(scene, renderer)
    else:
        raise ValueError(
            f"Texture {floor_texture} not recognized. Supported textures are 'plain' and 'checkerboard'")


def setup_sapien(cfg):
    if cfg.ray_tracing:
        sapien.render_config.camera_shader_dir = "rt"
        sapien.render_config.viewer_shader_dir = "rt"
        sapien.render_config.rt_samples_per_pixel = 64
        sapien.render_config.rt_use_denoiser = True

    engine = sapien.Engine()
    renderer = sapien.SapienRenderer(device="cuda")
    engine.set_renderer(renderer)

    scene = engine.create_scene()
    scene.set_timestep(cfg.engine.timestep)

    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    robot = loader.load_kinematic(cfg.urdf.file)
    assert robot, "URDF not loaded."

    if cfg.urdf.raise_distance_file is None:
        cfg.urdf.raise_distance_file = join_path(
            os.path.dirname(cfg.urdf.file), "raise_distances.json")

    if not os.path.exists(cfg.urdf.raise_distance_file):
        # NOTE: Couldn't figure out how to correctly compute the raise distance
        # in sapien so we'll use pybullet to compute it
        client, robot_id = setup_pybullet(
            cfg.urdf.file)
        cfg.urdf.raise_distance_file = join_path(
            os.path.dirname(cfg.urdf.file), "raise_distances.json")
        p.disconnect(client)

    raise_distance = load_json(cfg.urdf.raise_distance_file)
    # Apply rotation and translation
    rotation = R.from_euler('xyz', [
                            cfg.urdf.rotation_pose.rx, cfg.urdf.rotation_pose.ry, cfg.urdf.rotation_pose.rz])
    quat = rotation.as_quat()  # [x, y, z, w]
    # Sapien convention: wxyz
    quat = [quat[3], quat[0], quat[1], quat[2]]

    pose = sapien.Pose(
        p=[0, 0, max(raise_distance) + cfg.urdf.raise_distance_offset],
        q=quat,
    )
    robot.set_pose(pose)

    make_floor(cfg.floor_texture, scene, renderer)
    scene.set_ambient_light(cfg.lighting.ambient)
    scene.add_directional_light(
        cfg.lighting.directional.direction, cfg.lighting.directional.color)

    return engine, scene, robot


def get_manipulatable_joints(robot):
    joints = robot.get_joints()
    manipulatable_joints = []
    for i, joint in enumerate(joints):
        if joint.type in ["revolute", "prismatic"]:
            limits = joint.get_limits()[0]
            if (
                joint.type == "revolute"
                and limits[0] == -float("inf")
                and limits[1] == float("inf")
            ):
                # This is likely a continuous joint
                new_limits = (0, 4.89)  # Approximately 0 to 280 degrees
            else:
                new_limits = limits
            manipulatable_joints.append((joint.name, joint.type, new_limits))
    return manipulatable_joints


def get_joint_idx_by_name(manipulatable_joints, joint_name):
    for i, joint in enumerate(manipulatable_joints):
        if joint[0] == joint_name:
            return i
    return None


def move_joint(cfg: DictConfig, scene, cameras, robot, joint_name, joints_dict, video_writers):
    joint_idx = get_joint_idx_by_name(joints_dict, joint_name)
    if joint_idx is None:
        raise ValueError(f"Joint name '{joint_name}' not found")

    _, _, (lower_limit, upper_limit) = joints_dict[joint_idx]

    # Define movement functions for the specific joint
    def move_up(num_steps, step):
        # from lower_limit to upper_limit
        return lower_limit + step * (upper_limit - lower_limit) / num_steps

    def move_down(num_steps, step):
        # from upper_limit to lower_limit
        return upper_limit - step * (upper_limit - lower_limit) / num_steps

    move_func_dict = {
        "auto": move_down if lower_limit < 0 else move_up,
        "move_up": move_up,
        "move_down": move_down,
    }
    move_func = move_func_dict[cfg.simulation_params.joint_move_dir]

    compute_target_pos = partial(move_func, cfg.simulation_params.num_steps)

    trajectory = []
    for step in range(cfg.simulation_params.num_steps):
        qpos = robot.get_qpos()
        qpos[joint_idx] = compute_target_pos(step)
        robot.set_qpos(qpos)
        scene.step()
        scene.update_render()

        for camera_name, camera in cameras.items():
            bgr_img = take_camera_pic(
                robot, camera,
                use_segmentation=cfg.use_segmentation,
                output_json=cfg.output.seg_json,
                object_white=cfg.object_white,
            )
            video_writers[camera_name].write(bgr_img)

        trajectory.append(qpos.tolist())

    if cfg.flip_video:
        for camera_name, video_writer in video_writers.items():
            video_writer.release()
            video_path = os.path.join(
                cfg.output.dir, f"video_{joint_name}_{camera_name}.mp4")
            flip_video(video_path)

    return trajectory


def simulate_sapien(cfg: DictConfig):
    logging.info(
        f">>>> Generating a {cfg.simulation_params.stationary_or_move} of joint: {cfg.simulation_params.joint_name} | file: {cfg.urdf.file}")

    if cfg.output.dir is None:
        cfg.output.dir = join_path(os.path.dirname(cfg.urdf.file))

    if cfg.output.seg_json is None:
        cfg.output.seg_json = join_path(cfg.output.dir, "seg.json")

    create_dir(cfg.output.dir)
    engine, scene, robot = setup_sapien(cfg)
    manipulatable_joints = get_manipulatable_joints(robot)
    cameras = setup_cameras(cfg, scene)

    if cfg.simulation_params.stationary_or_move == "move":
        capture_video(cfg, scene, robot, cameras, manipulatable_joints)
    elif cfg.simulation_params.stationary_or_move == "stationary":
        capture_photo(cfg, scene, robot, cameras)
    else:
        raise ValueError(
            f"Stationary or move option '{cfg.simulation_params.stationary_or_move}' not recognized. Must be 'move' or 'stationary'.")

    if not cfg.headless:
        viewer = setup_viewer(engine, scene)
        while not viewer.closed:
            scene.step()
            scene.update_render()
            viewer.render()


def capture_video(cfg: DictConfig, scene, robot, cameras: dict, manipulatable_joints):
    if cfg.simulation_params.joint_name != "all" and cfg.simulation_params.joint_name not in [joint[0] for joint in manipulatable_joints]:
        raise ValueError(
            f"Joint name '{cfg.simulation_params.joint_name}' not found in the list of manipulatable joints of the URDF file. {manipulatable_joints}")

    initial_qpos = robot.get_qpos()
    for joint_name, joint_type, (lower_limit, upper_limit) in manipulatable_joints:
        # reset all joints to the initial position
        robot.set_qpos(initial_qpos)
        scene.step()
        scene.update_render()

        if cfg.simulation_params.joint_name != "all" and cfg.simulation_params.joint_name != joint_name:
            continue

        with VideoWriterManager(cfg, joint_name) as video_writers:
            move_joint(cfg, scene, cameras, robot, joint_name,
                       manipulatable_joints, video_writers)


def get_photo_name(cfg, camera_name):
    if not cfg.use_segmentation:
        return f"robot_{camera_name}.png"
    if cfg.object_white:
        return f"robot_{camera_name}_seg.png"
    return f"robot_{camera_name}_seg_color.png"


def capture_photo(cfg: DictConfig, scene, robot, cameras: Dict[str, Any]):
    set_joint_to_target_limit(robot, cfg.simulation_params.joint_move_dir)

    scene.step()
    scene.update_render()

    for camera_name, camera in cameras.items():
        bgr_img = take_camera_pic(
            robot, camera,
            use_segmentation=cfg.use_segmentation,
            output_json=cfg.output.seg_json,
            object_white=cfg.object_white,
        )

        photo_filename = get_photo_name(cfg, camera_name)
        # Generate a unique filename for each camera
        photo_path = join_path(cfg.output.dir, photo_filename)
        save_image(bgr_img, photo_path)


def setup_viewer(engine, scene):
    viewer = Viewer(engine.get_renderer())
    viewer.set_scene(scene)
    viewer.set_camera_xyz(x=-4, y=0, z=2)
    viewer.set_camera_rpy(r=0, p=-np.arctan2(2, 4), y=0)
    viewer.window.set_camera_parameters(near=0.05, far=100, fovy=1)
    return viewer


def set_joint_to_target_limit(robot, joint_move_dir):
    manipulatable_joints = get_manipulatable_joints(robot)
    qpos = robot.get_qpos()

    for joint_idx, (joint_name, joint_type, (lower_limit, upper_limit)) in enumerate(manipulatable_joints):
        if joint_move_dir == "auto":
            target_limit = upper_limit if lower_limit < 0 else lower_limit
        elif joint_move_dir == "move_up":
            target_limit = lower_limit
        elif joint_move_dir == "move_down":
            target_limit = upper_limit
        else:
            raise ValueError(
                f"Invalid joint_move_dir: {joint_move_dir}. Must be 'auto', 'move_up', or 'move_down'.")

        qpos[joint_idx] = target_limit

    robot.set_qpos(qpos)


@ hydra.main(version_base=None, config_path="../../conf/simulator", config_name="default")
def main(cfg: DictConfig):
    simulate_sapien(cfg)


if __name__ == "__main__":
    main()
