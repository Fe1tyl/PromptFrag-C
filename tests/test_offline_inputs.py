from __future__ import annotations

def test_clean_cifar10_is_never_downloaded_during_inference(
    monkeypatch, tmp_path
) -> None:
    from promptfragc.data import make_dataset
    from torchvision import datasets

    observed: dict[str, object] = {}

    class FakeCIFAR10:
        def __init__(self, **kwargs) -> None:
            observed.update(kwargs)

        def __len__(self) -> int:
            return 10_000

    monkeypatch.setattr(datasets, "CIFAR10", FakeCIFAR10)
    dataset = make_dataset(
        data_root=tmp_path,
        corruption="clean",
        severity=0,
        transform=lambda image: image,
        sample_limit=3,
        seed=7,
    )

    assert len(dataset) == 3
    assert observed["download"] is False


def test_openai_pretrained_config_enables_quick_gelu(monkeypatch) -> None:
    import open_clip
    from promptfragc.clip_model import load_clip_bundle

    observed: dict[str, object] = {}

    class FakeModel:
        logit_scale = None

        def to(self, device):
            return self

        def eval(self) -> None:
            return None

    monkeypatch.setattr(
        open_clip,
        "get_pretrained_cfg",
        lambda model_name, pretrained: {"quick_gelu": True},
    )

    def fake_create_model_and_transforms(model_name, pretrained, **kwargs):
        observed.update(kwargs)
        return FakeModel(), None, object()

    monkeypatch.setattr(
        open_clip,
        "create_model_and_transforms",
        fake_create_model_and_transforms,
    )
    monkeypatch.setattr(open_clip, "get_tokenizer", lambda model_name: object())

    bundle = load_clip_bundle("ViT-B-32", "openai", "cpu")

    assert bundle.model is not None
    assert observed["force_quick_gelu"] is True


def test_runner_sets_cublas_determinism_environment() -> None:
    import os

    import promptfragc.runner  # noqa: F401

    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
