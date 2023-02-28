import torch
import torch.nn as nn
import torchvision.models as models

class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()

        # Load ResNet152, ResNet18, and DenseNet
        resnet152 = models.resnet152(pretrained=True)
        resnet18 = models.resnet18(pretrained=True)
        densenet = models.densenet161(pretrained=True)

        # Remove last fully connected layer from each model
        self.resnet152 = nn.Sequential(*list(resnet152.children())[:-1])
        self.resnet18 = nn.Sequential(*list(resnet18.children())[:-1])
        self.densenet = nn.Sequential(*list(densenet.children())[:-1])

        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        # Compute ResNet152 features
        resnet152_features = self.resnet152(x)

        # Compute ResNet18 features
        resnet18_features = self.resnet18(x)

        # Compute DenseNet features
        densenet_features = self.densenet(x)

        # Concatenate features from all three models
        features = torch.cat([resnet152_features, resnet18_features, densenet_features], dim=1)

        return features
