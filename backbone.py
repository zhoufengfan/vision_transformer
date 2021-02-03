import torch.nn as nn
import torchvision
import cv2
import torch
import torchvision.transforms as transforms
from vision_transformer_pytorch import VisionTransformer


class Network2(nn.Module):
    def __init__(self):
        super().__init__()
        # self.resnet50 = torchvision.models.resnet50(pretrained=False)
        self.vit = VisionTransformer.from_name('ViT-B_16', image_size=32)
        # for param in self.parameters():
        #     param.requires_grad = False
        self.fc = nn.Linear(1000, 10)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x = self.vit(x)
        x = self.relu(x)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    dog_img_ndarray = cv2.imread(r"C:\Users\zff\Pictures\dog.jpg")
    # dog_img_tensor = torch.tensor(dog_img_ndarray)
    transform2 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    dog_img_tensor = transform2(dog_img_ndarray[:32, :32])
    dog_img_tensor_batch = torch.unsqueeze(dog_img_tensor, dim=0)
    net = Network2()
    feature_vector = net(dog_img_tensor_batch)
    print("feature_vector.shape is", feature_vector.shape)
