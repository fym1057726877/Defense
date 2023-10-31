import torch.nn as nn
import torch
import torchvision


def getDefinedClsModel(model_name, dataset_name, num_classes, device) -> nn.Module:
    # define the model
    if model_name == "FV-CNN":  # FV-CNN
        if dataset_name == "Fingervein2":
            model = Tifs2019CnnWithoutMaxPool(num_classes, fingervein1=True)
        else:
            model = Tifs2019CnnWithoutMaxPool(num_classes)
    elif model_name == "ModelB":
        model = TargetedModelB(num_classes)
    elif model_name == "PV-CNN":
        if dataset_name == "Fingervein1" or "Fingervein2":
            model = MSMDGANetCnn_wo_MaxPool(num_classes, fingervein1=True)
        else:
            model = MSMDGANetCnn_wo_MaxPool(num_classes)
    elif model_name == "FVRASNet_wo_Maxpooling":  # FVRAS-Net
        if dataset_name == "Fingervein1" or "Fingervein2":
            model = FVRASNet_wo_Maxpooling(num_classes, fingervein1=True)
        else:
            model = FVRASNet_wo_Maxpooling(num_classes)
    elif model_name == "LightweightDeepConvNN":  # Lightweight_CNN
        if dataset_name == "Fingervein1" or "Fingervein2":
            model = LightweightDeepConvNN(num_classes, fingervein1=True)
        else:
            model = LightweightDeepConvNN(num_classes)
    elif model_name == "FineTune":
        model = FineTuneClassifier(num_classes)
    elif model_name == "Resnet50":
        model = Resnet50(num_classes)
    else:
        raise NotImplementedError(f"classifier {model_name} is not implemented")
    model.to(device)
    return model


class Resnet50(nn.Module):
    def __init__(self, num_classes):
        super(Resnet50, self).__init__()
        self.conv_in = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=1)
        self.model = torchvision.models.resnet50(num_classes=num_classes)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.model(x)
        return x


class FineTuneClassifier(nn.Module):
    def __init__(self, out_channel):
        super(FineTuneClassifier, self).__init__()
        self.conv_in = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=1)
        self.model = torchvision.models.resnet18(pretrained=True)
        # 修改最后一层
        self.model.fc = nn.Linear(self.model.fc.in_features, out_channel)
        self.softmax = nn.Softmax(-1)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.model(x)
        x = self.softmax(x)
        return x


# FV-CNN
class Tifs2019CnnWithoutMaxPool(nn.Module):
    def __init__(self, out_channel, fingervein1=False):
        super(Tifs2019CnnWithoutMaxPool, self).__init__()
        self.fingervein1 = fingervein1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(128)

        self.conv2 = nn.Conv2d(in_channels=128, out_channels=512, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(512)

        self.conv3 = nn.Conv2d(in_channels=512, out_channels=768, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(768)

        if fingervein1:
            self.conv4 = nn.Conv2d(in_channels=768, out_channels=1024, kernel_size=(4, 12))
        else:
            self.conv4 = nn.Conv2d(in_channels=768, out_channels=1024, kernel_size=4)
        self.bn4 = nn.BatchNorm2d(1024)
        self.relu1 = nn.ReLU()

        self.conv5 = nn.Conv2d(in_channels=1024, out_channels=out_channel, kernel_size=2)

        self.softmax = nn.Softmax(-1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu1(x)
        x = self.conv5(x)
        x = x.squeeze(3).squeeze(2)
        x = self.softmax(x)
        return x
    

class TargetedModelB(nn.Module):
    def __init__(self, num_classes):
        super(TargetedModelB, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU()

        self.dropout = nn.Dropout(0.2)

        self.fc_out = nn.Linear(in_features=256 * 4 * 4, out_features=num_classes)
        self.softmax = nn.Softmax(-1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)

        x = self.dropout(x)
        x = x.reshape(x.size(0), -1)

        x = self.fc_out(x)
        x = self.softmax(x)
        return x


# PV-CNN
class MSMDGANetCnn_wo_MaxPool(nn.Module):
    def __init__(self, out_channel, fingervein1=False):

        super(MSMDGANetCnn_wo_MaxPool, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, padding=2, stride=2)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=2, stride=2)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, padding=2, stride=2)
        self.bn3 = nn.BatchNorm2d(256)

        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4)
        self.bn4 = nn.BatchNorm2d(512)

        if fingervein1:
            self.fc = nn.Linear(5 * 13 * 512, 800)
        else:
            self.fc = nn.Linear(5*5*512, 800)
        self.dropout = nn.Dropout()
        self.output = nn.Linear(800, out_channel)
        self.softmax = nn.Softmax(-1)

        self.relu = nn.ReLU()
        self.leaklyrelu = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.output(x)
        x = self.softmax(x)
        return x


class ConvBlock_wo_Maxpooling(nn.Module):
    def __init__(self, in_c, out_c):
        super(ConvBlock_wo_Maxpooling, self).__init__()
        self.conv3x3 = nn.Conv2d(in_channels=in_c, out_channels=out_c,kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv1x1 = nn.Conv2d(in_channels=out_c, out_channels=out_c, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv3x3(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv1x1(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x
    

class FVRASNet_wo_Maxpooling(nn.Module):
    def __init__(self, out_channel, fingervein1=False):
        super(FVRASNet_wo_Maxpooling, self).__init__()
        channels = [64, 128, 256]
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        self.block1 = ConvBlock_wo_Maxpooling(in_c=32, out_c=channels[0])
        self.block2 = ConvBlock_wo_Maxpooling(in_c=channels[0], out_c=channels[1])
        self.block3 = ConvBlock_wo_Maxpooling(in_c=channels[1], out_c=channels[2])
        if fingervein1:
            self.fc1 = nn.Linear(in_features=256 * 4 * 8, out_features=256)
        else:
            self.fc1 = nn.Linear(in_features=256 * 4 * 4, out_features=256)
        self.dropout1 = nn.Dropout()
        self.fc_out = nn.Linear(in_features=256, out_features=out_channel)
        self.softmax = nn.Softmax(-1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.fc_out(x)
        x = self.softmax(x)
        return x


# Lightweight FV CNN
class LightweightDeepConvNN(nn.Module):
    def __init__(self, out_channel, fingervein1=False):
        super(LightweightDeepConvNN, self).__init__()
        self.stemBlock = StemBlock()
        self.stageblock1 = StageBlock(in_c=32)
        self.stageblock2 = StageBlock(in_c=64)
        self.stageblock3 = StageBlock(in_c=96, output_layer=True)
        if fingervein1:
            self.fc_out = nn.Linear(in_features=128 * 4 * 8, out_features=out_channel)
        else:
            self.fc_out = nn.Linear(in_features=128 * 4 * 4, out_features=out_channel)
        self.softmax = nn.Softmax(-1)

    def forward(self, x):
        x = self.stemBlock(x)
        x = self.stageblock1(x)
        x = self.stageblock2(x)
        x = self.stageblock3(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc_out(x)
        x = self.softmax(x)
        return x

class StemBlock(nn.Module):
    def __init__(self):
        super(StemBlock, self).__init__()
        self.stem1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=2, padding=1)
        self.bn_stem1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=1, stride=1)
        self.bn_conv1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=2, padding=1)
        self.bn_conv2 = nn.BatchNorm2d(16)
        self.stem3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1)
        self.bn_stem3 = nn.BatchNorm2d(32)

    def forward(self, x):
        x = self.stem1(x)
        x = self.bn_stem1(x)
        x = self.relu(x)
        x1 = self.maxpool(x)
        x2 = self.conv1(x)
        x2 = self.bn_conv1(x2)
        x2 = self.relu(x2)
        x2 = self.conv2(x2)
        x2 = self.bn_conv2(x2)
        x2 = self.relu(x2)
        x = torch.cat((x1, x2), dim=1)
        x = self.stem3(x)
        x = self.bn_stem3(x)
        x = self.relu(x)
        return x


class StageBlock(nn.Module):
    def __init__(self, in_c, output_layer=False):
        super(StageBlock, self).__init__()
        self.smallStage1 = SmallStageBlock(in_c=in_c)
        self.smallStage2 = SmallStageBlock(in_c=in_c + 8)
        self.smallStage3 = SmallStageBlock(in_c=in_c + 16)
        self.smallStage4 = SmallStageBlock(in_c=in_c + 24)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1x1 = nn.Conv2d(in_channels=in_c+32, out_channels=in_c+32, kernel_size=1)
        self.bn = nn.BatchNorm2d(in_c+32)
        self.isOutput_layer = output_layer

    def forward(self, x):
        x = self.smallStage1(x)
        x = self.smallStage2(x)
        x = self.smallStage3(x)
        x = self.smallStage4(x)
        if self.isOutput_layer:
            x = self.conv1x1(x)
            x = self.bn(x)
        else:
            x = self.pool(x)
        return x


class SmallStageBlock(nn.Module):
    def __init__(self, in_c):
        super(SmallStageBlock, self).__init__()
        self.branch1_conv1 = nn.Conv2d(in_channels=in_c, out_channels=4, kernel_size=1)
        self.branch1_bn_conv1 = nn.BatchNorm2d(4)
        self.relu = nn.ReLU()
        self.branch1_conv2 = nn.Conv2d(in_channels=4, out_channels=4, kernel_size=1)
        self.branch1_bn_conv2 = nn.BatchNorm2d(4)
        self.branch3_conv1 = nn.Conv2d(in_channels=in_c, out_channels=4, kernel_size=1)
        self.branch3_bn_conv1 = nn.BatchNorm2d(4)
        self.branch3_conv2 = nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, stride=1, padding=1)
        self.branch3_bn_conv2 = nn.BatchNorm2d(4)
        self.branch3_conv3 = nn.Conv2d(in_channels=4, out_channels=4, kernel_size=1)
        self.branch3_bn_conv3 = nn.BatchNorm2d(4)

    def forward(self, x):
        x1 = self.branch1_conv1(x)
        x1 = self.branch1_bn_conv1(x1)
        x1 = self.relu(x1)
        x1 = self.branch1_conv2(x1)
        x1 = self.branch1_bn_conv2(x1)
        x1 = self.relu(x1)
        x3 = self.branch3_conv1(x)
        x3 = self.branch3_bn_conv1(x3)
        x3 = self.relu(x3)
        x3 = self.branch3_conv2(x3)
        x3 = self.branch3_bn_conv2(x3)
        x3 = self.relu(x3)
        x3 = self.branch3_conv3(x3)
        x3 = self.branch3_bn_conv3(x3)
        x3 = self.relu(x3)
        x = torch.cat((x1, x, x3), dim=1)
        return x