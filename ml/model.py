import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50

NUM_CLASSES = 104


def create_model():
    """
    DeepLabV3 with an ImageNet-pretrained ResNet50 backbone
    and a 104-class FoodSeg103 segmentation head.
    """

    model = deeplabv3_resnet50(weights="DEFAULT")

    # Replace the original segmentation classifier.
    model.classifier[4] = nn.Conv2d(
        in_channels=256,
        out_channels=NUM_CLASSES,
        kernel_size=1
    )

    return model


if __name__ == "__main__":
    print("Creating DeepLabV3 model...")

    model = create_model()
    model.eval()

    test_input = torch.randn(1, 3, 256, 256)

    with torch.no_grad():
        output = model(test_input)

    print("Output shape:", output["out"].shape)
    print("Expected: [1, 104, 256, 256]")