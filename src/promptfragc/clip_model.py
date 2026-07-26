"""Thin OpenCLIP inference adapter."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass


@dataclass
class ClipBundle:
    model: object
    preprocess: object
    tokenizer: object
    device: object
    logit_scale: float


def load_clip_bundle(model_name: str, pretrained: str, device_name: str) -> ClipBundle:
    import open_clip
    import torch

    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "configuration requests CUDA, but torch.cuda.is_available() is false"
        )
    device = torch.device(device_name)
    pretrained_config = open_clip.get_pretrained_cfg(model_name, pretrained) or {}
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
        force_quick_gelu=bool(pretrained_config.get("quick_gelu", False)),
    )
    model = model.to(device)
    model.eval()
    tokenizer = open_clip.get_tokenizer(model_name)
    raw_scale = getattr(model, "logit_scale", None)
    logit_scale = float(raw_scale.exp().detach().cpu()) if raw_scale is not None else 100.0
    return ClipBundle(
        model=model,
        preprocess=preprocess,
        tokenizer=tokenizer,
        device=device,
        logit_scale=logit_scale,
    )


def autocast_context(device):
    import torch

    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def encode_prompt_bank(bundle: ClipBundle, classes, prompt_specs):
    """Return normalized text features with shape [prompts, classes, dim]."""
    import torch

    prompt_features = []
    with torch.inference_mode(), autocast_context(bundle.device):
        for _, template in prompt_specs:
            text = [template.format(label=class_name) for class_name in classes]
            tokens = bundle.tokenizer(text).to(bundle.device)
            features = bundle.model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)
            prompt_features.append(features.float())
    return torch.stack(prompt_features, dim=0)


def infer_prompt_probabilities(bundle: ClipBundle, text_features, images):
    """Return probabilities shaped [prompts, batch, classes]."""
    import torch

    images = images.to(bundle.device, non_blocking=True)
    with torch.inference_mode(), autocast_context(bundle.device):
        image_features = bundle.model.encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logits = bundle.logit_scale * torch.einsum(
            "bd,pcd->pbc",
            image_features.float(),
            text_features,
        )
        probabilities = logits.softmax(dim=-1)
    return probabilities
