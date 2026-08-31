import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50
#103 food categories and 1 background category 
NUM_CLASSES = 104
def create_model():


   # Load DeepLabV3 with a ResNet-50 backbone
   model = deeplabv3_resnet50(
      weights="DEFAULT"
   )
   # Load DeepLabV3 with a ResNet-50 backbone
   model = deeplabv3_resnet50(
      weights="DEFAULT"
   ) 

   # Replace the final classifier so that it predicts
   # 104 classes instead of the original number of classes.
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
   print("Model created successfully!")

   #create a test image
   test_input = torch.randn(1, 3, 256, 256)
   print("\nInput shape:") 
   print(test_input.shape)

   #run the image through the model
   with torch.no_grad():
      output = model(test_input)

   print("\nOutput shape:") 
   print(output["out"].shape)

   print("\nExpected:") 
   print("(1, 104, 256, 256)")
