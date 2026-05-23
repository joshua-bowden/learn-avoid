"""Local registration of the red_obstacle object for LIBERO.

Import this module before creating the environment to register the object.
"""
import os
import pathlib

from robosuite.models.objects import MujocoXMLObject
from libero.libero.envs.base_object import register_object

_ASSET_DIR = pathlib.Path(__file__).parent / "assets"


@register_object
class RedObstacle(MujocoXMLObject):
    def __init__(
        self,
        name="red_obstacle",
        joints=[dict(type="free", damping="0.0005")],
    ):
        super().__init__(
            os.path.join(str(_ASSET_DIR), "red_obstacle.xml"),
            name=name,
            joints=joints,
            obj_type="all",
            duplicate_collision_geoms=False,
        )
        self.category_name = "red_obstacle"
        self.rotation = (0, 0)
        self.rotation_axis = "x"
        self.object_properties = {"vis_site_names": {}}
