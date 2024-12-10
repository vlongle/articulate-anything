import xml.etree.ElementTree as ET
import shutil
import os
from articulate_anything.api.odio_urdf import (
    load_urdf,
    Robot,
    Visual,
    Collision,
    Geometry,
    Link,
    Mesh,
    Origin,
    load_semantic,
)


def mask_urdf(input_file, format_rviz=False, skip_helper=True,
              full_mesh_path=True):
    """
    Masks a URDF file by removing joint, collision, and visual origin information.

    Args:
        input_file (str): Path to the original URDF file.
        output_file (str): Path to save the masked URDF file.
    """
    input_dir = os.path.dirname(input_file)
    semantic_file = os.path.join(input_dir, "semantics.txt")
    link_semantics = load_semantic(semantic_file)

    root = load_urdf(input_file)
    masked_links = []

    for link in root.findall("link"):
        link_name = link.attrib["name"]
        visuals = []
        Collisions = []
        if link_name not in link_semantics and link_name != "base" and skip_helper:
            # deliberately skipping links like helper
            continue

        for visual in link.findall("visual"):
            mesh_filename = visual.find("geometry").find(
                "mesh").attrib["filename"]
            my_pkg_name = "my_robot_pkg"
            prefix = os.path.join(
                "package://",
                my_pkg_name,
                "urdf",
                input_dir,
            )
            if full_mesh_path:
                # Use the full path for the mesh file
                mesh_filename = os.path.abspath(
                    os.path.join(input_dir, mesh_filename))

            elif format_rviz and not mesh_filename.startswith(prefix):
                mesh_filename = os.path.join(prefix, mesh_filename)
            # visuals.append(Visual(Geometry(Mesh(filename=mesh_filename))))
            visuals.append(
                Visual(
                    Geometry(Mesh(filename=mesh_filename)), Origin(
                        [0, 0, 0], [0, 0, 0])
                )
            )
            Collisions.append(
                Collision(
                    Geometry(Mesh(filename=mesh_filename)), Origin(
                        [0, 0, 0], [0, 0, 0])
                )
            )

        semantic_link_name = f"{link_semantics.get(link_name, link_name)}"
        masked_links.append(
            Link(name=semantic_link_name, *visuals, *Collisions)
        )  # Changed here

    masked_robot = Robot(input_dir, *masked_links)
    return masked_robot
