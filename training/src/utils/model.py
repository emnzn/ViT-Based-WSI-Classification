from typing import Dict, Any, Union

import torch.nn as nn
from torchvision.models.resnet import ResNet
from torchvision.models.swin_transformer import SwinTransformer
from torchvision.models import (
    resnet18, resnet34, resnet50, resnet101, resnet152,
    swin_t, swin_s, swin_b, swin_v2_t, swin_v2_s, swin_v2_b
)

from .abmil import AttentionBasedMIL

def swin_transformer(
    version: str,
    variant: str,
    dropout: float,
    num_classes: int
    ) -> SwinTransformer:

    """
    Initializes a Swin Transformer for classification.

    Parameters
    ----------
    version: str
        The version of the Swin Transformer to be initialized.
        Must be one of [v1, v2].

    variant: str
        The variant of the Swin Transformer to be initialized.
        Must be one of [tiny, small, base].

    dropout: float
        The dropout probability.

    num_classes: int
        The number of classes to be predicted.

    Returns
    -------
    model: SwinTransformer
        The initialized Swin Transformer.
    """

    valid_versions = ["v1", "v2"]
    valid_variants = ["tiny", "small", "base"]
    assertion_message = f"Swin transformer version must be one of {valid_versions} and variant must be one of {valid_variants}"

    assert version in valid_versions and variant in valid_variants, assertion_message

    if version == "v1":
        if variant == "tiny":
            model = swin_t(dropout=dropout, attention_dropout=dropout)
            model.features[0][0] = nn.Conv2d(1024, 96, kernel_size=(4, 4), stride=(4, 4))
            model.head = nn.Linear(in_features=768, out_features=num_classes)

        if variant == "small":
            model = swin_s(dropout=dropout, attention_dropout=dropout)
            model.features[0][0] = nn.Conv2d(1024, 96, kernel_size=(4, 4), stride=(4, 4))
            model.head = nn.Linear(in_features=768, out_features=num_classes)

        if variant == "base":
            model = swin_b(dropout=dropout, attention_dropout=dropout)
            model.features[0][0] = nn.Conv2d(1024, 128, kernel_size=(4, 4), stride=(4, 4))
            model.head = nn.Linear(in_features=1024, out_features=num_classes)

    elif version == "v2":
        if variant == "tiny":
            model = swin_v2_t(dropout=dropout, attention_dropout=dropout)
            model.features[0][0] = nn.Conv2d(1024, 96, kernel_size=(4, 4), stride=(4, 4))
            model.head = nn.Linear(in_features=768, out_features=num_classes)

        if variant == "small":
            model = swin_v2_s(dropout=dropout, attention_dropout=dropout)
            model.features[0][0] = nn.Conv2d(1024, 96, kernel_size=(4, 4), stride=(4, 4))
            model.head = nn.Linear(in_features=768, out_features=num_classes)

        if variant == "base":
            model = swin_v2_b(dropout=dropout, attention_dropout=dropout)
            model.features[0][0] = nn.Conv2d(1024, 128, kernel_size=(4, 4), stride=(4, 4))
            model.head = nn.Linear(in_features=1024, out_features=num_classes)

    return model


def resnet(
    variant: str,
    num_classes: int,
    normalization_method: str
    ) -> ResNet:

    """
    Initializes a ResNet model for classification.

    Parameters
    ----------
    variant: str
        Must be one of the following:
            - resnet18
            - resnet34
            - resnet50
            - resnet101
            - resnet152

    Returns
    -------
    model: ResNet
        The initialized ResNet model.
    """

    valid_variants = ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]
    assertion_message = f"ResNet variant must be one of {valid_variants}."

    assert variant in valid_variants, assertion_message

    valid_normalization_methods = ["batch", "group"]
    assertion_message = f"normalization_method must be one of {valid_normalization_methods}"
    
    assert normalization_method in valid_normalization_methods, assertion_message

    if variant == "resnet18":
        model = resnet18()
        model.conv1 = nn.Conv2d(1024, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.fc = nn.Linear(in_features=512, out_features=num_classes)

    if variant == "resnet34":
        model = resnet34()
        model.conv1 = nn.Conv2d(1024, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.fc = nn.Linear(in_features=512, out_features=num_classes)

    if variant == "resnet50":
        model = resnet50()
        model.conv1 = nn.Conv2d(1024, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.fc = nn.Linear(in_features=2048, out_features=num_classes)

    if variant == "resnet101":
        model = resnet101()
        model.conv1 = nn.Conv2d(1024, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.fc = nn.Linear(in_features=2048, out_features=num_classes)

    if variant == "resnet152":
        model = resnet152()
        model.conv1 = nn.Conv2d(1024, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.fc = nn.Linear(in_features=2048, out_features=num_classes)

    if normalization_method == "group":
        model = convert_bn(model, num_groups=32)

    return model


def attention_mil(num_classes: int) -> AttentionBasedMIL:

    """
    Initializes a two-layer Gated Attention MIL model.
    """

    input_dim = 1024
    embed_dim = 512
    hidden_dim = 384    
    
    model = AttentionBasedMIL(
        input_dim,
        embed_dim,
        hidden_dim,
        num_classes
        )
    
    return model


def convert_bn(
    model: ResNet, 
    num_groups: int
    ) -> ResNet:
    
    """
    Recursively replace all BatchNorm layers with GroupNorm in a given model.
    
    Parameters
    ----------
    model: ResNet
        The model containing BatchNorm layers.
    
    num_groups: int 
        The number of groups to be used in GroupNorm.
        
    Returns
    -------
    model: ResNet
        The model with batchnorm layers replaced to groupnorm.
    """
    for name, module in model.named_children():
        if isinstance(module, nn.BatchNorm2d):
            num_channels = module.num_features
            setattr(model, name, nn.GroupNorm(num_groups=num_groups, num_channels=num_channels))

        else:
            convert_bn(module, num_groups)

    return model


def get_model(
    args: Dict[str, Any]
    ) -> Union[ResNet, SwinTransformer, AttentionBasedMIL]:

    """
    Returns model and save path given a set of arguments.
    """

    if args["model"] == "attention-mil":
        model = attention_mil(num_classes=args["num_classes"])
        save_base_name = args["model"]

    if args["model"] == "resnet":
        model = resnet(
            variant=args["variant"], num_classes=args["num_classes"], 
            normalization_method=args["normalization_method"]
            )
        
        save_base_name = f"{args['variant']}"

    if args["model"] == "swin":
        model = swin_transformer(
            version=args["version"], variant=args["variant"], dropout=args["dropout_probability"], num_classes=args["num_classes"]
            )
        
        save_base_name = f"{args['model']}-{args['version']}-{args['variant']}"

    return model, save_base_name