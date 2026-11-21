"""PyTorch models for the DLCLProbVerification evaluation subjects.

Each evaluation subject from the project README is implemented here:

    #  Model                 Dataset(s)                     Builder
    -  --------------------  -----------------------------  -----------------------
    1  LeNet-5 / small CNN   MNIST, Fashion-MNIST           lenet5
    2  ResNet-18             CIFAR-10, ImageNet-1k          resnet18
    3  ResNet-50             ImageNet-1k                    resnet50
    4  MobileNetV2           CIFAR-10, ImageNet-1k          mobilenet_v2
    5  ViT-Base/16           CIFAR-10, ImageNet-1k          vit_base_patch16
    6  DeiT-Small            CIFAR-10, ImageNet-1k          deit_small_patch16
    7  BERT-base-uncased     SST-2, MRPC (GLUE)             build_bert
    8  DistilBERT            SST-2, MRPC (GLUE)             build_distilbert
    9  GPT-2 (small, 124M)   WikiText-2                     build_gpt2
    10 Qwen3.5-4B-Base       WikiText-2                     build_qwen

Vision models (#1-6) are implemented from scratch in pure PyTorch. NLP / LLM
models (#7-10) wrap HuggingFace ``transformers``.

The ``build_model`` helper provides a single string-keyed entry point, and
``MODEL_REGISTRY`` maps each key to its builder.
"""

from __future__ import annotations

from .lenet import LeNet5, lenet5
from .resnet import ResNet, BasicBlock, Bottleneck, resnet18, resnet50
from .mobilenetv2 import MobileNetV2, mobilenet_v2
from .vit import VisionTransformer, vit_base_patch16
from .deit import DistilledVisionTransformer, deit_small_patch16
from .bert import build_bert
from .distilbert import build_distilbert
from .gpt2 import build_gpt2
from .qwen import build_qwen

__all__ = [
    # vision (from scratch)
    "LeNet5",
    "lenet5",
    "ResNet",
    "BasicBlock",
    "Bottleneck",
    "resnet18",
    "resnet50",
    "MobileNetV2",
    "mobilenet_v2",
    "VisionTransformer",
    "vit_base_patch16",
    "DistilledVisionTransformer",
    "deit_small_patch16",
    # nlp / llm (transformers wrappers)
    "build_bert",
    "build_distilbert",
    "build_gpt2",
    "build_qwen",
    # registry
    "MODEL_REGISTRY",
    "build_model",
]

def detectors_model(timm_name: str, **kwargs):
    """Build an edadaltocg CIFAR/SVHN classifier architecture via timm.

    The `detectors` package registers these CIFAR/SVHN/ImageNet variants into
    timm (e.g. ``resnet34_cifar10``, ``vgg16_bn_cifar100``, ``densenet121_svhn``,
    ``vit_base_patch16_224_in21k_ft_cifar10``). We build the architecture only
    (``pretrained=False``); the trained weights are loaded separately from the
    HuggingFace repo via the task's ``hf_repo`` (network certs work there, unlike
    detectors' own downloader). Import is lazy so the registry stays usable when
    `detectors` is not installed.
    """
    import detectors  # noqa: F401  registers the models into timm
    import timm
    return timm.create_model(timm_name, pretrained=False)


# Maps a stable string key (as used by the README subject names) to a builder.
MODEL_REGISTRY = {
    "lenet5": lenet5,
    "resnet18": resnet18,
    "resnet50": resnet50,
    "mobilenet_v2": mobilenet_v2,
    "vit_base_patch16": vit_base_patch16,
    "deit_small_patch16": deit_small_patch16,
    "bert": build_bert,
    "distilbert": build_distilbert,
    "gpt2": build_gpt2,
    "qwen": build_qwen,
    "detectors": detectors_model,
}


def build_model(name: str, **kwargs):
    """Build a model by registry key.

    Example:
        >>> from src.models import build_model
        >>> net = build_model("resnet18", num_classes=10, stem="cifar")
    """
    if name not in MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model '{name}'. Available: {sorted(MODEL_REGISTRY)}"
        )
    return MODEL_REGISTRY[name](**kwargs)
