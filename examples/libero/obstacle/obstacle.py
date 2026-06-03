"""Static red box obstacle registered with LIBERO object system."""

import os
import re

from libero.libero.envs.base_object import register_object
from robosuite.models.objects import MujocoXMLObject

_OBSTACLE_DIR = os.path.dirname(os.path.abspath(__file__))


@register_object
class RedObstacle(MujocoXMLObject):
    """Unmovable table obstacle (no free joint)."""

    def __init__(self, name="red_obstacle_1", joints=None):
        super().__init__(
            os.path.join(_OBSTACLE_DIR, "assets/red_obstacle.xml"),
            name=name,
            joints=joints,
            obj_type="all",
            duplicate_collision_geoms=False,
        )
        self.category_name = "_".join(
            re.sub(r"([A-Z])", r" \1", self.__class__.__name__).split()
        ).lower()
        self.rotation = (0.0, 0.0)
        self.rotation_axis = "z"
        self.object_properties = {"vis_site_names": {}}
