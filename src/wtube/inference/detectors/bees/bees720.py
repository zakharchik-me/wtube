import torch.nn as nn
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(__file__, "../../../../utils")))
import time

from utils.registry import register

@register("detectors")
class SegmentationModel720(nn.Module):
    def __init__(self, input_channels=3, out_channels=1):
        super(SegmentationModel720, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.batchnorm3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.batchnorm4 = nn.BatchNorm2d(512)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc_conv1 = nn.Conv2d(512, 1024, kernel_size=1, padding=0)
        self.batchnorm_fc1 = nn.BatchNorm2d(1024)
        self.relu_fc1 = nn.ReLU()

        self.fc_conv2 = nn.Conv2d(1024, 512, kernel_size=1, padding=0)
        self.batchnorm_fc2 = nn.BatchNorm2d(512)
        self.relu_fc2 = nn.ReLU()

        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv4u = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.batchnorm4u = nn.BatchNorm2d(256)
        self.relu4u = nn.ReLU()

        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv3u = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.batchnorm3u = nn.BatchNorm2d(128)
        self.relu3u = nn.ReLU()

        self.upsample3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv2u = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.batchnorm2u = nn.BatchNorm2d(64)
        self.relu2u = nn.ReLU()

        self.upsample4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.detector = nn.Conv2d(64, out_channels, kernel_size=1, padding=0)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        start = time.time()
        x = self.relu1(self.batchnorm1(self.conv1(x)))
        x = self.pool1(x)

        x = self.relu2(self.batchnorm2(self.conv2(x)))
        x = self.pool2(x)

        x = self.relu3(self.batchnorm3(self.conv3(x)))
        x = self.pool3(x)

        x = self.relu4(self.batchnorm4(self.conv4(x)))
        x = self.pool4(x)

        x = self.relu_fc1(self.batchnorm_fc1(self.fc_conv1(x)))
        x = self.relu_fc2(self.batchnorm_fc2(self.fc_conv2(x)))

        x = self.upsample1(x)
        x = self.relu4u(self.batchnorm4u(self.conv4u(x)))

        x = self.upsample2(x)
        x = self.relu3u(self.batchnorm3u(self.conv3u(x)))

        x = self.upsample3(x)
        x = self.relu2u(self.batchnorm2u(self.conv2u(x)))

        x = self.upsample4(x)
        x = self.sigmoid(self.detector(x))
        print(time.time() - start)
        return x
