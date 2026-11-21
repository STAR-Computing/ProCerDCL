"""Task loading utilities for the ProCerDCL evaluation subjects.

``task_id`` enumerates one (model, dataset) pair each:

    task_id  Model              Dataset
    -------  -----------------  -----------
    1        LeNet-5            MNIST
    2        ResNet-18          CIFAR-10
    3        ResNet-50          ImageNet-1k
    4        MobileNetV2        CIFAR-10
    5        ViT-Base/16        CIFAR-10
    6        DeiT-Small         ImageNet-1k
    7        ResNet-18          CIFAR-100
    8        ResNet-18          SVHN
    9        ResNet-18          ImageNet-1k
    10       ResNet-50          CIFAR-100
    11       ResNet-50          SVHN
    12       ResNet-34          CIFAR-10
    13       ResNet-34          CIFAR-100
    14       ResNet-34          SVHN
    15       ResNet-50          CIFAR-10
    16       VGG-16-BN          CIFAR-10
    17       VGG-16-BN          CIFAR-100
    18       VGG-16-BN          SVHN
    19       DenseNet-121       CIFAR-10
    20       DenseNet-121       CIFAR-100
    21       DenseNet-121       SVHN
    22       ViT-B/16 in21k-ft  CIFAR-10
    23       ViT-B/16 in21k-ft  CIFAR-100
    24       ViT-B/16 in21k-ft  SVHN
    25       BERT-base          SST-2
    26       BERT-base          AG News
    27       DistilBERT         SST-2

Use :func:`load_task` to obtain ``(model, tokenizer, dataset)`` for a given id
(``tokenizer`` is ``None`` for the vision tasks). Every dataset is loaded
through the HuggingFace ``datasets`` library. Vision splits have eval transforms
attached on-the-fly via ``Dataset.with_transform`` so that indexing yields
``{"pixel_values", "label"}`` tensors. Vision checkpoints are downloaded from the
Hugging Face Hub (``hf_repo``); the text classifiers are built from their
pretrained Hugging Face models.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

from src.models import build_model


RANDOM_SEED = 123

# Vision models first try to load trained parameters from the HuggingFace Hub
# (per-task ``hf_repo`` in :data:`TASKS`). Only if no Hub repo is configured or
# the download fails do they fall back to the local cache below: the first build
# caches its (random-init) parameters here and later builds reload them, so the
# "original model" stays identical across runs (verifier / attack / post).
# (Gitignored; see .gitignore.)
MODEL_CACHE_DIR = Path("checkpoints")

# Where attack and verification results are written / read. (Gitignored.)
RESULT_DIR = Path("result")

# ImageNet normalisation, also used when low-res RGB sets are upscaled to 224 for
# the transformers, and for Tiny-ImageNet (an ImageNet subset) at native res.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2762)
SVHN_MEAN = (0.4377, 0.4438, 0.4728)
SVHN_STD = (0.1980, 0.2010, 0.1970)
MNIST_MEAN, MNIST_STD = (0.1307,), (0.3081,)
FASHION_MNIST_MEAN, FASHION_MNIST_STD = (0.2860,), (0.3530,)
KMNIST_MEAN, KMNIST_STD = (0.1918,), (0.3483,)
EMNIST_MEAN, EMNIST_STD = (0.1751,), (0.3332,)


# Canonical (namespaced) HuggingFace Hub repo ids -- bare aliases like "mnist"
# are rejected by recent huggingface_hub versions.
HF_MNIST = "ylecun/mnist"
HF_FASHION_MNIST = "zalando-datasets/fashion_mnist"
HF_KMNIST = "tanganke/kmnist"
HF_EMNIST_LETTERS = "tanganke/emnist_letters"
HF_CIFAR10 = "uoft-cs/cifar10"
HF_CIFAR100 = "uoft-cs/cifar100"
HF_SVHN = "ufldl-stanford/svhn"
HF_TINY_IMAGENET = "zh-plus/tiny-imagenet"
HF_IMAGENET = "ILSVRC/imagenet-1k"
# Text-classification subjects (E1). SST-2 is the GLUE sentiment task; AG News
# is a 4-class topic task with longer inputs.
HF_SST2 = "stanfordnlp/sst2"
HF_AGNEWS = "fancyzhx/ag_news"


# --------------------------------------------------------------------------- #
# Dataset loaders (all via the HuggingFace `datasets` library)
# --------------------------------------------------------------------------- #
def _image_transform(mean, std, resize=None, center_crop=None, grayscale=False):
    """Build the eval-time torchvision transform applied to each PIL image."""
    from torchvision import transforms as T

    ops = []
    if resize is not None:
        ops.append(T.Resize(resize))
    if center_crop is not None:
        ops.append(T.CenterCrop(center_crop))
    ops += [T.ToTensor(), T.Normalize(mean, std)]
    return T.Compose(ops)


def _attach_image_transform(ds, transform, grayscale: bool):
    """Attach an on-the-fly transform: indexing yields pixel_values + label."""
    cols = ds.column_names
    img_col = "img" if "img" in cols else "image"
    # Datasets disagree on the label column name (CIFAR-100 uses "fine_label").
    label_col = next(
        (c for c in ("label", "fine_label", "labels") if c in cols), "label"
    )
    mode = "L" if grayscale else "RGB"

    def _apply(batch):
        images = [transform(im.convert(mode)) for im in batch[img_col]]
        return {"pixel_values": images, "label": batch[label_col]}

    return ds.with_transform(_apply)


def _hf_vision(path, split, default_split, transform, grayscale, name=None):
    from datasets import load_dataset

    ds = load_dataset(path, name, split=split or default_split)
    return _attach_image_transform(ds, transform, grayscale)


def _mnist_like(path, mean, std, split):
    transform = _image_transform(mean, std, grayscale=True)
    return _hf_vision(path, split, "test", transform, grayscale=True)


def _rgb(path, native_mean, native_std, split, default_split,
         image_size, name=None, native_resize=None):
    """Generic RGB image loader.

    ``image_size is None`` -> keep native resolution + the dataset's own stats
    (the CNN setup). Otherwise upscale to ``image_size`` square + ImageNet stats
    (the ViT / DeiT setup). ``native_resize`` optionally forces the native size.
    """
    if image_size is None:
        transform = _image_transform(native_mean, native_std, resize=native_resize)
    else:
        transform = _image_transform(
            IMAGENET_MEAN, IMAGENET_STD, resize=(image_size, image_size)
        )
    return _hf_vision(path, split, default_split, transform, grayscale=False, name=name)


def _text_classification(path, split, default_split, text_col, max_length,
                         name=None):
    """Text-classification loader for the E1 subjects.

    Rows keep their raw text (``verifier.collate`` tokenizes on the fly for the
    dataset-level sampler) but also carry ``text``/``label`` under canonical
    names so callers do not need per-dataset column knowledge.

    Tokenization is left to the caller because it needs the subject's own
    tokenizer. ``max_length`` is recorded on the returned dataset so the
    embedding-space drivers can pad every input to one fixed length -- a
    constant sequence length keeps ``torch.compile`` from going dynamic on the
    shape, the same reason the vision samplers pad partial batches.
    """
    from datasets import load_dataset

    ds = load_dataset(path, name, split=split or default_split)
    if text_col != "text":
        ds = ds.rename_column(text_col, "text")
    keep = {"text", "label"}
    drop = [c for c in ds.column_names if c not in keep]
    if drop:
        ds = ds.remove_columns(drop)
    ds.max_length = max_length  # advisory; consumed by the text drivers
    return ds


def _imagenet(split, image_size=224):
    # ImageNet-1k is a gated dataset: accept its terms on the Hub and log in
    # (`huggingface-cli login`) before first use.
    if image_size >= 224:
        transform = _image_transform(
            IMAGENET_MEAN, IMAGENET_STD, resize=256, center_crop=224
        )
    else:  # smaller resolution (e.g. 32x32) to cut attack cost
        transform = _image_transform(
            IMAGENET_MEAN, IMAGENET_STD, resize=(image_size, image_size)
        )
    norm = "validation" if split in ("test", "val", "validation", None) else split
    return _hf_vision(HF_IMAGENET, norm, "validation", transform, grayscale=False)


# --------------------------------------------------------------------------- #
# Task table
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Task:
    model_name: str  # human-readable
    dataset_name: str  # human-readable
    model_key: str  # key into src.models.MODEL_REGISTRY
    model_kwargs: Dict[str, Any]
    dataset_kind: str  # dispatch key for the loaders above
    dataset_kwargs: Dict[str, Any]
    # HuggingFace Hub repo id holding this model's trained checkpoint, and the
    # checkpoint filename within it. A task tries this before the local
    # random-init cache; ``None`` skips the Hub and uses the cache directly.
    hf_repo: Optional[str] = None
    hf_filename: Optional[str] = None


# Tasks 1-6 keep their original ids/config (public checkpoints verified to load
# strictly into these from-scratch architectures). Tasks 7-29 have no compatible
# public checkpoint for the (model, dataset) combination and so leave
# ``hf_repo=None``, falling back to the local random-init cache. If a configured
# repo is unreachable the loader warns and falls back too, so a run never
# crashes on a missing checkpoint.
#
# Resolution / stem convention for the added pairs:
#   * LeNet-5            : native 28x28 grayscale (MNIST family).
#   * ResNet / MobileNet : 32x32 sets (CIFAR/SVHN) at native res with the CIFAR
#                          stem; Tiny-ImageNet at native 64x64; ImageNet at 224
#                          with the ImageNet stem.
#   * ViT / DeiT         : every dataset upscaled to 224x224 (patchify needs a
#                          fixed input), ImageNet normalisation.
TASKS: Dict[int, Task] = {
    1: Task("LeNet-5", "MNIST", "lenet5",
            dict(num_classes=10, in_channels=1),
            "mnist", dict(),
            hf_repo="ChufanSuki/LeNet5",
            hf_filename="lenet_mnist_model.pth"),
    2: Task("ResNet-18", "CIFAR-10", "resnet18",
            dict(num_classes=10, stem="cifar"),
            "cifar10", dict(image_size=None),
            hf_repo="edadaltocg/resnet18_cifar10",
            hf_filename="pytorch_model.bin"),
    3: Task("ResNet-50", "ImageNet-1k", "resnet50",
            dict(num_classes=1000, stem="imagenet"),
            "imagenet", dict(),
            hf_repo="timm/resnet50.tv_in1k",
            hf_filename="pytorch_model.bin"),
    4: Task("MobileNetV2", "CIFAR-10", "mobilenet_v2",
            dict(num_classes=10, stem_stride=1),
            "cifar10", dict(image_size=None),
            hf_repo="Makeh3ne/cifar10-mobilenetv2",
            hf_filename="best_mobilenet_v2_model.pth"),
    5: Task("ViT-Base/16", "CIFAR-10", "vit_base_patch16",
            dict(num_classes=10, img_size=224),
            "cifar10", dict(image_size=224),
            hf_repo="anonauthors/cifar10-timm-vit_base_patch16_224.orig_in21k_ft_in1k",
            hf_filename="pytorch_model.bin"),
    # Switched from CIFAR-10 to ImageNet-1k (both README-listed for DeiT) because
    # no CIFAR-10 distilled-DeiT checkpoint exists on the Hub; the timm ImageNet
    # weights load strictly into this distilled DeiT-Small.
    6: Task("DeiT-Small", "ImageNet-1k", "deit_small_patch16",
            dict(num_classes=1000, img_size=224),
            "imagenet", dict(),
            hf_repo="timm/deit_small_distilled_patch16_224.fb_in1k",
            hf_filename="pytorch_model.bin"),
    # ---- ResNet-18 ----
    7: Task("ResNet-18", "CIFAR-100", "resnet18",
             dict(num_classes=100, stem="cifar"),
             "cifar100", dict(image_size=None),
             hf_repo="edadaltocg/resnet18_cifar100",
             hf_filename="pytorch_model.bin"),
    8: Task("ResNet-18", "SVHN", "resnet18",
             dict(num_classes=10, stem="cifar"),
             "svhn", dict(image_size=None),
             hf_repo="edadaltocg/resnet18_svhn",
             hf_filename="pytorch_model.bin"),
    9: Task("ResNet-18", "ImageNet-1k", "resnet18",
             dict(num_classes=1000, stem="imagenet"),
             "imagenet", dict(),
             hf_repo="timm/resnet18.tv_in1k",
             hf_filename="pytorch_model.bin"),

    # ---- ResNet-50 ----
    10: Task("ResNet-50", "CIFAR-100", "resnet50",
             dict(num_classes=100, stem="cifar"),
             "cifar100", dict(image_size=None),
             hf_repo="edadaltocg/resnet50_cifar100",
             hf_filename="pytorch_model.bin"),
    11: Task("ResNet-50", "SVHN", "resnet50",
             dict(num_classes=10, stem="cifar"),
             "svhn", dict(image_size=None),
             hf_repo="edadaltocg/resnet50_svhn",
             hf_filename="pytorch_model.bin"),
    # ===== edadaltocg classifier zoo: architecture via `detectors`+timm,
    #       trained weights from the matching HuggingFace repo (strict load
    #       verified). CIFAR/SVHN at native 32x32; ViT upscaled to 224. =====
    12: Task("ResNet-34", "CIFAR-10", "detectors", dict(timm_name="resnet34_cifar10"),
             "cifar10", dict(image_size=None),
             hf_repo="edadaltocg/resnet34_cifar10", hf_filename="pytorch_model.bin"),
    13: Task("ResNet-34", "CIFAR-100", "detectors", dict(timm_name="resnet34_cifar100"),
             "cifar100", dict(image_size=None),
             hf_repo="edadaltocg/resnet34_cifar100", hf_filename="pytorch_model.bin"),
    14: Task("ResNet-34", "SVHN", "detectors", dict(timm_name="resnet34_svhn"),
             "svhn", dict(image_size=None),
             hf_repo="edadaltocg/resnet34_svhn", hf_filename="pytorch_model.bin"),
    15: Task("ResNet-50", "CIFAR-10", "detectors", dict(timm_name="resnet50_cifar10"),
             "cifar10", dict(image_size=None),
             hf_repo="edadaltocg/resnet50_cifar10", hf_filename="pytorch_model.bin"),
    16: Task("VGG-16-BN", "CIFAR-10", "detectors", dict(timm_name="vgg16_bn_cifar10"),
             "cifar10", dict(image_size=None),
             hf_repo="edadaltocg/vgg16_bn_cifar10", hf_filename="pytorch_model.bin"),
    17: Task("VGG-16-BN", "CIFAR-100", "detectors", dict(timm_name="vgg16_bn_cifar100"),
             "cifar100", dict(image_size=None),
             hf_repo="edadaltocg/vgg16_bn_cifar100", hf_filename="pytorch_model.bin"),
    18: Task("VGG-16-BN", "SVHN", "detectors", dict(timm_name="vgg16_bn_svhn"),
             "svhn", dict(image_size=None),
             hf_repo="edadaltocg/vgg16_bn_svhn", hf_filename="pytorch_model.bin"),
    19: Task("DenseNet-121", "CIFAR-10", "detectors", dict(timm_name="densenet121_cifar10"),
             "cifar10", dict(image_size=None),
             hf_repo="edadaltocg/densenet121_cifar10", hf_filename="pytorch_model.bin"),
    20: Task("DenseNet-121", "CIFAR-100", "detectors", dict(timm_name="densenet121_cifar100"),
             "cifar100", dict(image_size=None),
             hf_repo="edadaltocg/densenet121_cifar100", hf_filename="pytorch_model.bin"),
    21: Task("DenseNet-121", "SVHN", "detectors", dict(timm_name="densenet121_svhn"),
             "svhn", dict(image_size=None),
             hf_repo="edadaltocg/densenet121_svhn", hf_filename="pytorch_model.bin"),
    22: Task("ViT-B/16 in21k-ft", "CIFAR-10", "detectors",
             dict(timm_name="vit_base_patch16_224_in21k_ft_cifar10"),
             "cifar10", dict(image_size=224),
             hf_repo="edadaltocg/vit_base_patch16_224_in21k_ft_cifar10", hf_filename="pytorch_model.bin"),
    23: Task("ViT-B/16 in21k-ft", "CIFAR-100", "detectors",
             dict(timm_name="vit_base_patch16_224_in21k_ft_cifar100"),
             "cifar100", dict(image_size=224),
             hf_repo="edadaltocg/vit_base_patch16_224_in21k_ft_cifar100", hf_filename="pytorch_model.bin"),
    24: Task("ViT-B/16 in21k-ft", "SVHN", "detectors",
             dict(timm_name="vit_base_patch16_224_in21k_ft_svhn"),
             "svhn", dict(image_size=224),
             hf_repo="edadaltocg/vit_base_patch16_224_in21k_ft_svhn", hf_filename="pytorch_model.bin"),

    # --- E1: text-classification subjects (Transformer / NLP coverage) ------
    # Fine-tuned public checkpoints so the "original model" is a real trained
    # classifier, not a random head. The epsilon-ball for these lives in the
    # frozen input-embedding space (see src/models/text_embed.py).
    25: Task("BERT-base", "SST-2", "bert",
             dict(pretrained_name="textattack/bert-base-uncased-SST-2",
                  num_labels=2),
             "sst2", dict(max_length=64)),
    26: Task("BERT-base", "AG News", "bert",
             dict(pretrained_name="textattack/bert-base-uncased-ag-news",
                  num_labels=4),
             "agnews", dict(max_length=128)),
    27: Task("DistilBERT", "SST-2", "distilbert",
             dict(pretrained_name="distilbert-base-uncased-finetuned-sst-2-english",
                  num_labels=2),
             "sst2", dict(max_length=64)),
}

COMPILER : Dict[int, str] = {
    0: "inductor",
    1: "tensorrt",
    2: "onnxrt",
}

# Model keys whose builders also produce a tokenizer (NLP / LLM tasks).
NLP_MODEL_KEYS = {"bert", "distilbert", "gpt2", "qwen"}

# Native (CNN-setup) normalisation stats per RGB dataset kind. When a task
# upscales to 224 for a transformer (image_size set) ImageNet stats are used.
_RGB_NATIVE_STATS = {
    "cifar10": (CIFAR10_MEAN, CIFAR10_STD),
    "cifar100": (CIFAR100_MEAN, CIFAR100_STD),
    "svhn": (SVHN_MEAN, SVHN_STD),
    "tiny_imagenet": (IMAGENET_MEAN, IMAGENET_STD),
}
_GRAY_STATS = {
    "mnist": (MNIST_MEAN, MNIST_STD),
    "fashion_mnist": (FASHION_MNIST_MEAN, FASHION_MNIST_STD),
    "kmnist": (KMNIST_MEAN, KMNIST_STD),
    "emnist_letters": (EMNIST_MEAN, EMNIST_STD),
}


def list_tasks() -> Dict[int, str]:
    """Return ``{task_id: 'Model on Dataset'}`` for every task."""
    return {tid: f"{t.model_name} on {t.dataset_name}" for tid, t in TASKS.items()}


def task_slug(task: "Task") -> str:
    """Filesystem-safe name derived from a task's model and dataset.

    e.g. model "LeNet-5", dataset "MNIST" -> ``"lenet-5_on_mnist"``. Used to
    organize saved results under the result directory.
    """
    import re

    def slug(s: str) -> str:
        return re.sub(r"[^0-9a-z]+", "-", s.lower()).strip("-")

    return f"{slug(task.model_name)}_on_{slug(task.dataset_name)}"


def result_dir(task: "Task", backend: str, method: str) -> Path:
    """Create and return ``<RESULT_DIR>/<task_slug>/<backend>/<method>/``.

    Shared by the verifier, attack and bound-search scripts so they all write to
    the same per-(task, backend, method) layout.
    """
    out = RESULT_DIR / task_slug(task) / backend / method
    out.mkdir(parents=True, exist_ok=True)
    return out


def model_cache_path(task_id: int) -> Path:
    """Path of the cached parameter file for ``task_id``'s model."""
    task = TASKS[task_id]
    return MODEL_CACHE_DIR / f"task{task_id}_{task.model_key}.pt"


def normalization_stats(task: "Task") -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    """Return the ``(mean, std)`` used to normalize ``task``'s input images.

    Mirrors the per-task choice made by the dataset loaders, so saved tensors
    can be inverted back to original ``[0, 1]`` pixel space. Raises for non-image
    tasks.
    """
    kind = task.dataset_kind
    if kind in _GRAY_STATS:
        return _GRAY_STATS[kind]
    if kind in _RGB_NATIVE_STATS:
        # Transformers upscale to 224 (image_size set) and use ImageNet stats;
        # CNNs keep native resolution and the dataset's own stats.
        if task.dataset_kwargs.get("image_size") is None:
            return _RGB_NATIVE_STATS[kind]
        return IMAGENET_MEAN, IMAGENET_STD
    if kind == "imagenet":
        return IMAGENET_MEAN, IMAGENET_STD
    raise ValueError(f"no image normalization for dataset kind {kind!r}")


def _hf_checkpoint_filename(task: "Task") -> str:
    """Filename of ``task``'s checkpoint within its HuggingFace Hub repo."""
    return task.hf_filename or f"{task.model_key}.pt"


def _load_hf_params(model, repo_id: str, filename: str) -> bool:
    """Download a trained ``state_dict`` from the HuggingFace Hub and load it.

    Accepts either a bare ``state_dict`` or a checkpoint dict wrapping one under
    a ``"state_dict"`` / ``"model"`` key. Returns ``True`` on success; on any
    failure (repo missing, key mismatch, offline) it logs a warning and returns
    ``False`` so the caller can fall back to the local cache.
    """
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import HfHubHTTPError

    try:
        path = hf_hub_download(repo_id=repo_id, filename=filename)
        state = torch.load(path, map_location="cpu")
        if isinstance(state, dict):
            for key in ("state_dict", "model"):
                if key in state and isinstance(state[key], dict):
                    state = state[key]
                    break
        model.load_state_dict(state)
    except (HfHubHTTPError, OSError, RuntimeError, KeyError) as exc:
        import warnings

        warnings.warn(
            f"Could not load HuggingFace params from {repo_id!r}/{filename!r} "
            f"({type(exc).__name__}: {exc}); falling back to local cache."
        )
        return False
    return True


def _load_or_cache_params(model, cache_path: Path):
    """Reload cached parameters if present, otherwise cache the current ones.

    Returns ``model`` so calls can be chained. The first time a task's model is
    built its (random-init) weights are written to ``cache_path``; subsequent
    builds load that file so every run sees the same model.
    """
    if cache_path.exists():
        model.load_state_dict(torch.load(cache_path, map_location="cpu"))
    else:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), cache_path)
    return model


def _build_dataset(task: Task, split: Optional[str], number: int = 1000,
                   image_size: Optional[int] = None):
    kind = task.dataset_kind
    kw = task.dataset_kwargs
    img = kw.get("image_size", image_size)  # task config wins over the arg

    if kind == "mnist":
        dataset = _mnist_like(HF_MNIST, MNIST_MEAN, MNIST_STD, split)
    elif kind == "fashion_mnist":
        dataset = _mnist_like(
            HF_FASHION_MNIST, FASHION_MNIST_MEAN, FASHION_MNIST_STD, split
        )
    elif kind == "kmnist":
        dataset = _mnist_like(HF_KMNIST, KMNIST_MEAN, KMNIST_STD, split)
    elif kind == "emnist_letters":
        dataset = _mnist_like(HF_EMNIST_LETTERS, EMNIST_MEAN, EMNIST_STD, split)
    elif kind == "cifar10":
        dataset = _rgb(HF_CIFAR10, CIFAR10_MEAN, CIFAR10_STD, split, "test", img)
    elif kind == "cifar100":
        dataset = _rgb(HF_CIFAR100, CIFAR100_MEAN, CIFAR100_STD, split, "test", img)
    elif kind == "svhn":
        dataset = _rgb(HF_SVHN, SVHN_MEAN, SVHN_STD, split, "test", img,
                       name="cropped_digits")
    elif kind == "tiny_imagenet":
        dataset = _rgb(HF_TINY_IMAGENET, IMAGENET_MEAN, IMAGENET_STD, split,
                       "valid", img, native_resize=(64, 64))
    elif kind == "imagenet":
        dataset = _imagenet(split, img if img else 224)
    elif kind == "sst2":
        # SST-2's test split is unlabelled; validation is the evaluation split.
        dataset = _text_classification(
            HF_SST2, split, "validation", "sentence",
            kw.get("max_length", 64),
        )
    elif kind == "agnews":
        dataset = _text_classification(
            HF_AGNEWS, split, "test", "text", kw.get("max_length", 128),
        )
    else:
        raise ValueError(f"Unknown dataset kind: {kind}")

    return dataset.select(range(min(number, len(dataset))))


def load_task(
    task_id: int,
    split: Optional[str] = None,
    build_model_: bool = True,
    build_dataset: bool = True,
    number: int = 1000,
    image_size: Optional[int] = None,
) -> Tuple[Any, Any, Any]:
    """Load the model, tokenizer and dataset for ``task_id``.

    All datasets are pulled through the HuggingFace ``datasets`` library and
    cached under ``$HF_HOME`` / ``$HF_DATASETS_CACHE`` (defaults to
    ``~/.cache/huggingface``).

    Args:
        task_id: integer in ``1..29`` (see the table at the top of this module
            and :func:`list_tasks`).
        split: dataset split. Defaults to the evaluation split per task
            (``test`` for the MNIST family / CIFAR / SVHN, ``valid`` for
            Tiny-ImageNet, ``validation`` for ImageNet).
        build_model_: if False, the model and tokenizer slots are ``None``
            (useful when only the dataset is needed, avoiding weight downloads).
        build_dataset: if False, the dataset slot is ``None``.
        number: take only the first ``number`` samples of the split.
        image_size: optional override for datasets whose task config does not
            already pin a resolution.

    Returns:
        ``(model, tokenizer, dataset)``. All tasks here are vision tasks, so
        ``tokenizer`` is always ``None``; the model's parameters come from the
        task's HuggingFace Hub ``hf_repo`` when configured, else from the local
        random-init cache (cached on first build, reused thereafter). The
        dataset is a HuggingFace ``Dataset`` whose items are
        ``{"pixel_values": Tensor, "label": int}`` (eval transforms applied
        on-the-fly). ImageNet-1k is gated -- accept its terms on the Hub and
        ``huggingface-cli login`` first.
    """
    if task_id not in TASKS:
        raise KeyError(
            f"Unknown task_id {task_id}. Valid ids: {sorted(TASKS)}. "
            f"See utils.list_tasks()."
        )
    task = TASKS[task_id]

    model = None
    tokenizer = None
    if build_model_:
        if task.model_key in NLP_MODEL_KEYS:
            # NLP/LLM models load deterministic pretrained weights already.
            model, tokenizer = build_model(
                task.model_key, return_tokenizer=True, **task.model_kwargs
            )
            model = _load_or_cache_params(model, model_cache_path(task_id))
        else:
            # Vision models: load trained params from the HuggingFace Hub if a
            # repo is configured; otherwise fall back to the local random-init
            # cache (cache on first build, reload after).
            model = build_model(task.model_key, **task.model_kwargs)

            loaded = False
            if task.hf_repo:
                loaded = _load_hf_params(
                    model, task.hf_repo, _hf_checkpoint_filename(task)
                )
            if not loaded:
                model = _load_or_cache_params(model, model_cache_path(task_id))

    dataset = (
        _build_dataset(task, split, number, image_size)
        if build_dataset else None
    )
    return model, tokenizer, dataset


if __name__ == "__main__":
    print("Available tasks:")
    for tid, desc in list_tasks().items():
        print(f"  {tid:2d}: {desc}")
