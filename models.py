# Shadow model architechture
import torch
import torch.nn as nn
from torchvision.models import resnet18

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def setup_model():
    local_path = "weights_resnet18_cifar10.pth"
    weights_pretrained = torch.load(local_path, map_location=DEVICE)

    model = resnet18(weights=None, num_classes=10)
    model.load_state_dict(weights_pretrained)
    model.to(DEVICE)
    model.eval()
    return model

class ShadowNet(nn.Module):
    def __init__(self, num_classes, use_batchnorm=True, use_dropout=True):
        super(ShadowNet, self).__init__()

        # Load the pre-trained ResNet18 model
        pretrained_model = setup_model()

        # Extract layers from the pre-trained ResNet18
        self.conv1 = pretrained_model.conv1
        self.bn1 = pretrained_model.bn1
        self.relu = pretrained_model.relu
        self.maxpool = pretrained_model.maxpool
        self.layer1 = pretrained_model.layer1
        self.layer2 = pretrained_model.layer2
        self.layer3 = pretrained_model.layer3
        self.layer4 = pretrained_model.layer4

        # Custom fully connected layers for CIFAR-10
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

        # Additional layers for improving model performance
        self.use_batchnorm = use_batchnorm
        self.use_dropout = use_dropout

        if self.use_batchnorm:
            self.bn2 = nn.BatchNorm1d(512)  # Batch Normalization before the FC layer

        if self.use_dropout:
            self.dropout = nn.Dropout(0.5)  # Dropout layer with a dropout rate of 0.5

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        if self.use_batchnorm:
            x = self.bn2(x)

        if self.use_dropout:
            x = self.dropout(x)

        x = self.fc(x)

        return x

#
#shadow_model = ShadowNet(num_classes=10, use_batchnorm=False, use_dropout=False)
#print(shadow_model)




#Attack model architechture

class BinaryClassifier(nn.Module):
    def __init__(self, input_size=3):
        super(BinaryClassifier, self).__init__()
        self.fc1 = nn.Linear(3, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        nn.Dropout(0.5)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x
