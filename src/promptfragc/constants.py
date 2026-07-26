"""Frozen study constants.

Changing this file after inspecting results changes the preregistered study.
"""

from __future__ import annotations

CIFAR10_CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

# Prompts deliberately keep the class label fixed and only change the framing.
# IDs are stable and are written to every result row.
PROMPTS = (
    ("p00", "a photo of a {label}."),
    ("p01", "a photograph depicting {label}."),
    ("p02", "an image of the object {label}."),
    ("p03", "a picture showing {label}."),
    ("p04", "this image shows {label}."),
    ("p05", "the main subject in this image is {label}."),
    ("p06", "the object depicted here is {label}."),
    ("p07", "a visual representation of {label}."),
    ("p08", "an example image of the class {label}."),
    ("p09", "this picture contains {label}."),
    ("p10", "a natural image containing {label}."),
    ("p11", "{label}, shown in a photograph."),
)

DEFAULT_PROMPT_ID = "p00"

STANDARD_CORRUPTIONS = (
    "gaussian_noise",
    "shot_noise",
    "impulse_noise",
    "defocus_blur",
    "glass_blur",
    "motion_blur",
    "zoom_blur",
    "snow",
    "frost",
    "fog",
    "brightness",
    "contrast",
    "elastic_transform",
    "pixelate",
    "jpeg_compression",
)

METRIC_COLUMNS = (
    "accuracy",
    "ece",
    "nll",
    "brier",
    "aurc",
    "risk_at_80",
)

