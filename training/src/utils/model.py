import torch
from torchvision.models.resnet import ResNet
from torchvision.models.swin_transformer import SwinTransformer
from torchvision.models import (
    resnet18, resnet34, resnet50, resnet101, resnet152,
    swin_t, swin_s, swin_b, swin_v2_t, swin_v2_s, swin_v2_b
)

def swin_transformer(
    version: str,
    variant: str,
    num_classes: int
    ) -> SwinTransformer:

    """
    Initializes a Swin Transformer for classification.

    Parameters
    ----------
    variant: str
        The variant of the Swin Transformer to be initialized.
        Must be one of [tiny, small, base].
    
    version: str
        The version of the Swin Transformer to be initialized.
        Must be one of [v1, v2].

    num_classes: int
        The number of classes to be predicted.

    Returns
    -------
    model: SwinTransformer
        The initialized Swin Transformer.
    """

    if version == "v1":
        if variant == "tiny":
            model = swin_t()
            model.features[0][0] = torch.nn.Conv2d(1024, 96, kernel_size=(4, 4), stride=(4, 4))
            model.head = torch.nn.Linear(in_features=768, out_features=num_classes)

        if variant == "small":
            model = swin_s()
            model.features[0][0] = torch.nn.Conv2d(1024, 96, kernel_size=(4, 4), stride=(4, 4))
            model.head = torch.nn.Linear(in_features=768, out_features=num_classes)

        if variant == "base":
            model = swin_b()
            model.features[0][0] = torch.nn.Conv2d(1024, 128, kernel_size=(4, 4), stride=(4, 4))
            model.head = torch.nn.Linear(in_features=1024, out_features=num_classes)

    elif version == "v2":
        if variant == "tiny":
            model = swin_v2_t()
            model.features[0][0] = torch.nn.Conv2d(1024, 96, kernel_size=(4, 4), stride=(4, 4))
            model.head = torch.nn.Linear(in_features=768, out_features=num_classes)

        if variant == "small":
            model = swin_v2_s()
            model.features[0][0] = torch.nn.Conv2d(1024, 96, kernel_size=(4, 4), stride=(4, 4))
            model.head = torch.nn.Linear(in_features=768, out_features=num_classes)

        if variant == "base":
            model = swin_v2_b()
            model.features[0][0] = torch.nn.Conv2d(1024, 96, kernel_size=(4, 4), stride=(4, 4))
            model.head = torch.nn.Linear(in_features=1024, out_features=num_classes)

    return model


def resnet(
    variant: str,
    num_classes: int
    ) -> ResNet:

    if variant == "resnet18":
        model = resnet18()
        model.conv1 = torch.nn.Conv2d(1024, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.fc = torch.nn.Linear(in_features=512, out_features=num_classes)

    if variant == "resnet34":
        model = resnet34()
        model.conv1 = torch.nn.Conv2d(1024, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.fc = torch.nn.Linear(in_features=512, out_features=num_classes)

    if variant == "resnet50":
        model = resnet50()
        model.conv1 = torch.nn.Conv2d(1024, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.fc = torch.nn.Linear(in_features=2048, out_features=num_classes)

    if variant == "resnet101":
        model = resnet101()
        model.conv1 = torch.nn.Conv2d(1024, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.fc = torch.nn.Linear(in_features=2048, out_features=num_classes)

    if variant == "resnet152":
        model.conv1 = torch.nn.Conv2d(1024, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.fc = torch.nn.Linear(in_features=2048, out_features=num_classes)

    return model



    