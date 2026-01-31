import dataclasses
from typing import ClassVar

import numpy as np

from openpi import transforms


def _to_hwc_uint8(img) -> np.ndarray:
    img = np.asarray(img)

    # Convert CHW -> HWC if needed.
    if img.ndim == 3 and img.shape[0] in (1, 3, 4) and img.shape[-1] not in (1, 3, 4):
        img = np.transpose(img, (1, 2, 0))

    # Convert float images (commonly [0, 1]) to uint8.
    if np.issubdtype(img.dtype, np.floating):
        img = np.clip(img, 0.0, 1.0)
        img = (255.0 * img).astype(np.uint8)

    if img.dtype != np.uint8:
        img = img.astype(np.uint8)

    return img


@dataclasses.dataclass(frozen=True)
class So101FollowerInputs(transforms.DataTransformFn):
    """Inputs for a simple single-arm follower robot (e.g. so101_follower LeRobot datasets).

    Expected inputs:
    - images: dict[name, img] where img is CHW float32 in [0, 1] (torch) or HWC uint8.
    - state: [d] (typically 6)
    - actions: [action_horizon, d] (typically 6)
    """

    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("cam_head", "cam_right_wrist")

    def __call__(self, data: dict) -> dict:
        in_images = data["images"]
        if set(in_images) - set(self.EXPECTED_CAMERAS):
            raise ValueError(f"Expected images to contain a subset of {self.EXPECTED_CAMERAS}, got {tuple(in_images)}")

        # Head image is required.
        head = _to_hwc_uint8(in_images["cam_head"])

        images = {
            "base_0_rgb": head,
        }
        image_masks = {
            "base_0_rgb": np.True_,
        }

        # Optional wrist image(s).
        right_name = "cam_right_wrist"
        if right_name in in_images:
            images["right_wrist_0_rgb"] = _to_hwc_uint8(in_images[right_name])
            image_masks["right_wrist_0_rgb"] = np.True_
        else:
            images["right_wrist_0_rgb"] = np.zeros_like(head)
            image_masks["right_wrist_0_rgb"] = np.False_

        # No left wrist camera on many follower setups; keep the slot but mask it out.
        images["left_wrist_0_rgb"] = np.zeros_like(head)
        image_masks["left_wrist_0_rgb"] = np.False_

        inputs = {
            "image": images,
            "image_mask": image_masks,
            "state": np.asarray(data["state"]),
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class So101FollowerOutputs(transforms.DataTransformFn):
    """Outputs for so101_follower style policies."""

    action_dim: int = 6

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"])
        return {"actions": actions[:, : self.action_dim]}

