from pathlib import Path

PACKAGE_PATH = Path(__file__).parent
XMLS_PATH = PACKAGE_PATH.joinpath("envs", "xmls")
ASSETS_PATH = PACKAGE_PATH.joinpath("envs", "assets")
WORLD_XML = XMLS_PATH / "world.xml"
