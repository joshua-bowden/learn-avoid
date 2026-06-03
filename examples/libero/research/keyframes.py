"""Extract pick/place keyframes from demonstration actions via gripper channel."""

from __future__ import annotations

import dataclasses

import numpy as np

from common import SKIP_STEPS, is_gripper_closed_action


@dataclasses.dataclass
class Keyframes:
    initial: int
    pick: int
    place: int

    def as_dict(self) -> dict[str, int]:
        return {"initial": self.initial, "pick": self.pick, "place": self.place}


def extract_keyframes(actions: np.ndarray, skip_steps: int = SKIP_STEPS) -> Keyframes:
    """
    Key poses from gripper transitions:
      - initial: first valid index after stabilization
      - pick: last index before first gripper close
      - place: last index before first gripper open after close
    """
    n = len(actions)
    start = min(skip_steps, n - 1)

    first_close = None
    for i in range(start, n):
        if is_gripper_closed_action(actions[i]):
            first_close = i
            break
    if first_close is None:
        raise ValueError("No gripper close found in demonstration")

    pick = max(start, first_close - 1)

    first_open_after_close = None
    for i in range(first_close + 1, n):
        if not is_gripper_closed_action(actions[i]):
            first_open_after_close = i
            break
    if first_open_after_close is None:
        raise ValueError("No gripper open after close found in demonstration")

    place = max(first_close + 1, first_open_after_close - 1)

    return Keyframes(initial=start, pick=pick, place=place)
