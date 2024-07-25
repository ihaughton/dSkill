from pathlib import Path

PACKAGE_PATH = Path(__file__).parent
ASSETS_PATH = PACKAGE_PATH.joinpath("assets")
WORLD_XML = ASSETS_PATH / "scene.xml"
UR5_XML = ASSETS_PATH / "mujoco_menagerie" / "universal_robots_ur5e" / "ur5e.xml"
ROBOTIQ_MODEL = "2f140"
ROBOTIQ_XML = (
    ASSETS_PATH
    / "mujoco_menagerie"
    / f"robotiq_{ROBOTIQ_MODEL}"
    / f"{ROBOTIQ_MODEL}.xml"
)
SEG_GROUP_COLORS = {
    0: [0, 0, 0],
    1: [255, 0, 0],
    2: [0, 255, 0],
    3: [0, 0, 255],
}
TARGET_SEG_GROUP = 2
GRIPPER_SEG_GROUP = 1
