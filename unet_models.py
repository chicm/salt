from torch import nn
from torch.nn import functional as F
import torch
from torchvision import models
from torchvision.models import resnet34, resnet101, resnet50, resnet152
import torchvision
import pdb

"""
This script has been taken (and modified) from :
https://github.com/ternaus/TernausNet

@ARTICLE{arXiv:1801.05746,
         author = {V. Iglovikov and A. Shvets},
          title = {TernausNet: U-Net with VGG11 Encoder Pre-Trained on ImageNet for Image Segmentation},
        journal = {ArXiv e-prints},
         eprint = {1801.05746}, 
           year = 2018
        }
"""


def conv3x3(in_, out):
    return nn.Conv2d(in_, out, 3, padding=1)


class ConvRelu(nn.Module):
    def __init__(self, in_, out):
        super().__init__()
        self.conv = conv3x3(in_, out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x


class NoOperation(nn.Module):
    def forward(self, x):
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            ConvRelu(in_channels, middle_channels),
            nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class UNet11(nn.Module):
    def __init__(self, num_classes=1, num_filters=32, pretrained=False):
        """
        :param num_classes:
        :param num_filters:
        :param pretrained:
            False - no pre-trained network is used
            True  - encoder is pre-trained with VGG11
        """
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)

        self.encoder = models.vgg11(pretrained=pretrained).features

        self.relu = self.encoder[1]
        self.conv1 = self.encoder[0]
        self.conv2 = self.encoder[3]
        self.conv3s = self.encoder[6]
        self.conv3 = self.encoder[8]
        self.conv4s = self.encoder[11]
        self.conv4 = self.encoder[13]
        self.conv5s = self.encoder[16]
        self.conv5 = self.encoder[18]

        self.center = DecoderBlock(num_filters * 8 * 2, num_filters * 8 * 2, num_filters * 8)
        self.dec5 = DecoderBlock(num_filters * (16 + 8), num_filters * 8 * 2, num_filters * 8)
        self.dec4 = DecoderBlock(num_filters * (16 + 8), num_filters * 8 * 2, num_filters * 4)
        self.dec3 = DecoderBlock(num_filters * (8 + 4), num_filters * 4 * 2, num_filters * 2)
        self.dec2 = DecoderBlock(num_filters * (4 + 2), num_filters * 2 * 2, num_filters)
        self.dec1 = ConvRelu(num_filters * (2 + 1), num_filters)

        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, x):
        conv1 = self.relu(self.conv1(x))
        conv2 = self.relu(self.conv2(self.pool(conv1)))
        conv3s = self.relu(self.conv3s(self.pool(conv2)))
        conv3 = self.relu(self.conv3(conv3s))
        conv4s = self.relu(self.conv4s(self.pool(conv3)))
        conv4 = self.relu(self.conv4(conv4s))
        conv5s = self.relu(self.conv5s(self.pool(conv4)))
        conv5 = self.relu(self.conv5(conv5s))

        center = self.center(self.pool(conv5))

        dec5 = self.dec5(torch.cat([center, conv5], 1))
        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(torch.cat([dec2, conv1], 1))
        return self.final(dec1)


def unet11(pretrained=False, **kwargs):
    """
    pretrained:
            False - no pre-trained network is used
            True  - encoder is pre-trained with VGG11
            carvana - all weights are pre-trained on
                Kaggle: Carvana dataset https://www.kaggle.com/c/carvana-image-masking-challenge
    """
    model = UNet11(pretrained=pretrained, **kwargs)

    if pretrained == 'carvana':
        state = torch.load('TernausNet.pt')
        model.load_state_dict(state['model'])
    return model


class DecoderBlockV2(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, is_deconv=True):
        super(DecoderBlockV2, self).__init__()
        self.in_channels = in_channels

        if is_deconv:
            """
                Paramaters for Deconvolution were chosen to avoid artifacts, following
                link https://distill.pub/2016/deconv-checkerboard/
            """

            self.block = nn.Sequential(
                ConvRelu(in_channels, middle_channels),
                nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=4, stride=2,
                                   padding=1),
                nn.ReLU(inplace=True)
            )
        else:
            self.block = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear'),
                ConvRelu(in_channels, middle_channels),
                ConvRelu(middle_channels, out_channels),
            )

    def forward(self, x):
        return self.block(x)


class AlbuNet(nn.Module):
    """
        UNet (https://arxiv.org/abs/1505.04597) with Resnet34(https://arxiv.org/abs/1512.03385) encoder

        Proposed by Alexander Buslaev: https://www.linkedin.com/in/al-buslaev/

        """

    def __init__(self, num_classes=1, num_filters=32, pretrained=False, is_deconv=False):
        """
        :param num_classes:
        :param num_filters:
        :param pretrained:
            False - no pre-trained network is used
            True  - encoder is pre-trained with resnet34
        :is_deconv:
            False: bilinear interpolation is used in decoder
            True: deconvolution is used in decoder
        """
        super().__init__()
        self.num_classes = num_classes

        self.pool = nn.MaxPool2d(2, 2)

        self.encoder = torchvision.models.resnet34(pretrained=pretrained)

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(self.encoder.conv1,
                                   self.encoder.bn1,
                                   self.encoder.relu,
                                   self.pool)

        self.conv2 = self.encoder.layer1

        self.conv3 = self.encoder.layer2

        self.conv4 = self.encoder.layer3

        self.conv5 = self.encoder.layer4

        self.center = DecoderBlockV2(512, num_filters * 8 * 2, num_filters * 8, is_deconv)

        self.dec5 = DecoderBlockV2(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec4 = DecoderBlockV2(256 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec3 = DecoderBlockV2(128 + num_filters * 8, num_filters * 4 * 2, num_filters * 2, is_deconv)
        self.dec2 = DecoderBlockV2(64 + num_filters * 2, num_filters * 2 * 2, num_filters * 2 * 2, is_deconv)
        self.dec1 = DecoderBlockV2(num_filters * 2 * 2, num_filters * 2 * 2, num_filters, is_deconv)
        self.dec0 = ConvRelu(num_filters, num_filters)
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        center = self.center(self.pool(conv5))

        dec5 = self.dec5(torch.cat([center, conv5], 1))

        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(dec2)
        dec0 = self.dec0(dec1)

        return self.final(dec0)


class UNetVGG16(nn.Module):
    """PyTorch U-Net model using VGG16 encoder.

    UNet: https://arxiv.org/abs/1505.04597
    VGG: https://arxiv.org/abs/1409.1556
    Proposed by Vladimir Iglovikov and Alexey Shvets: https://github.com/ternaus/TernausNet

    Args:
            num_classes (int): Number of output classes.
            num_filters (int, optional): Number of filters in the last layer of decoder. Defaults to 32.
            dropout_2d (float, optional): Probability factor of dropout layer before output layer. Defaults to 0.2.
            pretrained (bool, optional):
                False - no pre-trained weights are being used.
                True  - VGG encoder is pre-trained on ImageNet.
                Defaults to False.
            is_deconv (bool, optional):
                False: bilinear interpolation is used in decoder.
                True: deconvolution is used in decoder.
                Defaults to False.

    """

    def __init__(self, num_classes=1, num_filters=32, dropout_2d=0.2, pretrained=False, is_deconv=False):
        super().__init__()
        self.num_classes = num_classes
        self.dropout_2d = dropout_2d

        self.pool = nn.MaxPool2d(2, 2)

        self.encoder = torchvision.models.vgg16(pretrained=pretrained).features

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(self.encoder[0],
                                   self.relu,
                                   self.encoder[2],
                                   self.relu)

        self.conv2 = nn.Sequential(self.encoder[5],
                                   self.relu,
                                   self.encoder[7],
                                   self.relu)

        self.conv3 = nn.Sequential(self.encoder[10],
                                   self.relu,
                                   self.encoder[12],
                                   self.relu,
                                   self.encoder[14],
                                   self.relu)

        self.conv4 = nn.Sequential(self.encoder[17],
                                   self.relu,
                                   self.encoder[19],
                                   self.relu,
                                   self.encoder[21],
                                   self.relu)

        self.conv5 = nn.Sequential(self.encoder[24],
                                   self.relu,
                                   self.encoder[26],
                                   self.relu,
                                   self.encoder[28],
                                   self.relu)

        self.center = DecoderBlockV2(512, num_filters * 8 * 2, num_filters * 8, is_deconv)

        self.dec5 = DecoderBlockV2(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec4 = DecoderBlockV2(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec3 = DecoderBlockV2(256 + num_filters * 8, num_filters * 4 * 2, num_filters * 2, is_deconv)
        self.dec2 = DecoderBlockV2(128 + num_filters * 2, num_filters * 2 * 2, num_filters, is_deconv)
        self.dec1 = ConvRelu(64 + num_filters, num_filters)
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(self.pool(conv1))
        conv3 = self.conv3(self.pool(conv2))
        conv4 = self.conv4(self.pool(conv3))
        conv5 = self.conv5(self.pool(conv4))

        center = self.center(self.pool(conv5))

        dec5 = self.dec5(torch.cat([center, conv5], 1))

        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(torch.cat([dec2, conv1], 1))

        return self.final(F.dropout2d(dec1, p=self.dropout_2d))


class UNetResNet(nn.Module):
    """PyTorch U-Net model using ResNet(34, 101 or 152) encoder.

    UNet: https://arxiv.org/abs/1505.04597
    ResNet: https://arxiv.org/abs/1512.03385
    Proposed by Alexander Buslaev: https://www.linkedin.com/in/al-buslaev/

    Args:
            encoder_depth (int): Depth of a ResNet encoder (34, 101 or 152).
            num_classes (int): Number of output classes.
            num_filters (int, optional): Number of filters in the last layer of decoder. Defaults to 32.
            dropout_2d (float, optional): Probability factor of dropout layer before output layer. Defaults to 0.2.
            pretrained (bool, optional):
                False - no pre-trained weights are being used.
                True  - ResNet encoder is pre-trained on ImageNet.
                Defaults to False.
            is_deconv (bool, optional):
                False: bilinear interpolation is used in decoder.
                True: deconvolution is used in decoder.
                Defaults to False.

    """

    def __init__(self, encoder_depth, num_classes=1, num_filters=32, dropout_2d=0.2,
                 pretrained=True, is_deconv=True):
        super().__init__()
        #pdb.set_trace()
        self.name = 'UNetResNet_'+str(encoder_depth)
        self.num_classes = num_classes
        self.dropout_2d = dropout_2d

        if encoder_depth == 34:
            self.encoder = torchvision.models.resnet34(pretrained=pretrained)
            bottom_channel_nr = 512
        elif encoder_depth == 50:
            self.encoder = torchvision.models.resnet50(pretrained=pretrained)
            bottom_channel_nr = 2048
        elif encoder_depth == 101:
            self.encoder = torchvision.models.resnet101(pretrained=pretrained)
            bottom_channel_nr = 2048
        elif encoder_depth == 152:
            self.encoder = torchvision.models.resnet152(pretrained=pretrained)
            bottom_channel_nr = 2048
        else:
            raise NotImplementedError('only 34, 101, 152 version of Resnet are implemented')

        self.pool = nn.MaxPool2d(2, 2)

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(self.encoder.conv1,
                                   self.encoder.bn1,
                                   self.encoder.relu)
                                   #self.pool)

        self.conv2 = self.encoder.layer1

        self.conv3 = self.encoder.layer2

        self.conv4 = self.encoder.layer3

        self.conv5 = self.encoder.layer4

        self.center = DecoderBlockV2(bottom_channel_nr, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec5 = DecoderBlockV2(bottom_channel_nr + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec4 = DecoderBlockV2(bottom_channel_nr // 2 + num_filters * 8, num_filters * 8 * 2, num_filters * 8,
                                   is_deconv)
        self.dec3 = DecoderBlockV2(bottom_channel_nr // 4 + num_filters * 8, num_filters * 4 * 2, num_filters * 2,
                                   is_deconv)
        self.dec2 = DecoderBlockV2(bottom_channel_nr // 8 + num_filters * 2, num_filters * 2 * 2, num_filters * 2 * 2,
                                   is_deconv)
        self.dec1 = DecoderBlockV2(num_filters * 2 * 2, num_filters * 2 * 2, num_filters, is_deconv)
        self.dec0 = ConvRelu(num_filters, num_filters)
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)
        self.classifier = nn.Linear(num_filters * 256 * 256, 1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        pool = self.pool(conv5)
        center = self.center(pool)

        dec5 = self.dec5(torch.cat([center, conv5], 1))

        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(dec2)
        dec0 = self.dec0(dec1)
        out = self.pool(dec0)

        cls_out = self.classifier(F.dropout(dec0.view(dec0.size(0), -1), p=0.25))

        return self.final(F.dropout2d(out, p=self.dropout_2d)), cls_out
    
    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def get_params(self, base_lr):
        group1 = [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5]
        group2 = [self.dec0, self.dec1, self.dec2, self.dec3, self.dec4, self.dec5, self.center]
        group3 = [self.classifier, self.final]

        params1 = []
        for x in group1:
            for p in x.parameters():
                params1.append(p)
        
        param_group1 = {'params': params1, 'lr': base_lr / 100}

        params2 = []
        for x in group2:
            for p in x.parameters():
                params2.append(p)
        param_group2 = {'params': params2, 'lr': base_lr / 10}

        params3 = []
        for x in group3:
            for p in x.parameters():
                params3.append(p)
        param_group3 = {'params': params3, 'lr': base_lr}

        return [param_group1, param_group2, param_group3]

class ConvBn2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3,3), stride=(1,1), padding=(1,1)):
        super(ConvBn2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class ChannelAttentionGate(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ChannelAttentionGate, self).__init__()
        self.fc1 = nn.Conv2d(channel, reduction, kernel_size=1, padding=0)
        self.fc2 = nn.Conv2d(reduction, channel, kernel_size=1, padding=0)

    def forward(self, x):
        x = F.adaptive_avg_pool2d(x,1)
        x = self.fc1(x)
        x = F.relu(x, inplace=True)
        x = self.fc2(x)
        x = F.sigmoid(x)
        return x

class SpatialAttentionGate(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SpatialAttentionGate, self).__init__()
        self.fc1 = nn.Conv2d(channel, reduction, kernel_size=1, padding=0)
        self.fc2 = nn.Conv2d(reduction, 1, kernel_size=1, padding=0)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x, inplace=True)
        x = self.fc2(x)
        x = F.sigmoid(x)
        #print(x.size())
        return x

class DecoderV3(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, is_deconv=True):
        super(DecoderV3, self).__init__()
        self.conv1 = ConvBn2d(in_channels, middle_channels)
        self.conv2 = ConvBn2d(middle_channels, out_channels)
        self.spatial_gate = SpatialAttentionGate(out_channels)
        self.channel_gate = ChannelAttentionGate(out_channels)

    def forward(self, x, e=None):
        x = F.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)
        if e is not None:
            x = torch.cat([x,e], 1)

        x = F.relu(self.conv1(x), inplace=True)
        x = F.relu(self.conv2(x), inplace=True)

        g1 = self.spatial_gate(x)
        g2 = self.channel_gate(x)
        x = x*g1 + x*g2

        return x

class UNetResNet2(nn.Module):
    """PyTorch U-Net model using ResNet(34, 101 or 152) encoder.

    UNet: https://arxiv.org/abs/1505.04597
    ResNet: https://arxiv.org/abs/1512.03385
    Proposed by Alexander Buslaev: https://www.linkedin.com/in/al-buslaev/

    Args:
            encoder_depth (int): Depth of a ResNet encoder (34, 101 or 152).
            num_classes (int): Number of output classes.
            num_filters (int, optional): Number of filters in the last layer of decoder. Defaults to 32.
            dropout_2d (float, optional): Probability factor of dropout layer before output layer. Defaults to 0.2.
            pretrained (bool, optional):
                False - no pre-trained weights are being used.
                True  - ResNet encoder is pre-trained on ImageNet.
                Defaults to False.
            is_deconv (bool, optional):
                False: bilinear interpolation is used in decoder.
                True: deconvolution is used in decoder.
                Defaults to False.

    """

    def __init__(self, encoder_depth, num_classes=1, num_filters=32, dropout_2d=0.3,
                 pretrained=True, is_deconv=True):
        super().__init__()
        #pdb.set_trace()
        self.name = 'UNetResNetV2_'+str(encoder_depth)
        self.num_classes = num_classes
        self.dropout_2d = dropout_2d

        if encoder_depth == 34:
            self.encoder = resnet34(pretrained=pretrained)
            bottom_channel_nr = 512
        elif encoder_depth == 50:
            self.encoder = resnet50(pretrained=pretrained)
            bottom_channel_nr = 2048
        elif encoder_depth == 101:
            self.encoder = resnet101(pretrained=pretrained)
            bottom_channel_nr = 2048
        elif encoder_depth == 152:
            self.encoder = resnet152(pretrained=pretrained)
            bottom_channel_nr = 2048
        else:
            raise NotImplementedError('only 34, 101, 152 version of Resnet are implemented')

        self.pool = nn.MaxPool2d(2, 2)
        self.enc1 = nn.Sequential(self.encoder.conv1, self.encoder.bn1, self.encoder.relu)#self.pool)
        self.enc2 = self.encoder.layer1
        self.enc3 = self.encoder.layer2
        self.enc4 = self.encoder.layer3
        self.enc5 = self.encoder.layer4

        #self.center = DecoderV3(bottom_channel_nr, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.center = nn.Sequential(
            ConvBn2d(bottom_channel_nr, bottom_channel_nr, kernel_size=(3,3), padding=1),
            nn.ReLU(inplace=True),
            ConvBn2d(bottom_channel_nr, num_filters*8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.dec5 = DecoderV3(bottom_channel_nr + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec4 = DecoderV3(bottom_channel_nr // 2 + num_filters * 8, num_filters * 8 * 2, num_filters * 8,
                                   is_deconv)
        self.dec3 = DecoderV3(bottom_channel_nr // 4 + num_filters * 8, num_filters * 4 * 2, num_filters * 2,
                                   is_deconv)
        self.dec2 = DecoderV3(bottom_channel_nr // 8 + num_filters * 2, num_filters * 2 * 2, num_filters * 2 * 2,
                                   is_deconv)
        self.dec1 = DecoderV3(num_filters * 2 * 2, num_filters * 2 * 2, num_filters, is_deconv)
        #self.dec0 = ConvBn2d(num_filters, num_filters)

        #self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

        self.logit = nn.Sequential(
            nn.Conv2d(736, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1, padding=0)
        )

        self.classifier = nn.Linear(128 * 128, 1)

    def forward(self, x):
        x = self.enc1(x)
        e2 = self.enc2(x)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4) #; print(e5.size())

        #pool = self.pool(e5); remove
        center = self.center(e5) #; print(center.size())

        d5 = self.dec5(center, e5)

        d4 = self.dec4(d5, e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2)
        #d0 = self.dec0(d1)
        #d0 = F.relu(d0, inplace=True)

        # Hyper column
        f = torch.cat([
            d1,
            F.upsample(d2, scale_factor=2, mode='bilinear', align_corners=False),
            F.upsample(d3, scale_factor=4, mode='bilinear', align_corners=False),
            F.upsample(d4, scale_factor=8, mode='bilinear', align_corners=False),
            F.upsample(d5, scale_factor=16, mode='bilinear', align_corners=False),
        ], 1) 

        f = F.dropout2d(f, p=self.dropout_2d)
        #print(f.size())
        #print('f:', f.size())
        logit = self.logit(f)

        #out = self.pool(d0)
        #out = F.dropout2d(d0, p=self.dropout_2d)
        #print(out.size())
        cls_out = self.classifier(F.dropout(logit.view(logit.size(0), -1), p=0.25))

        return logit, cls_out
    
    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()
    def get_params(self, base_lr):
        group1 = [self.enc1, self.enc2, self.enc3, self.enc4, self.enc5]
        group2 = [self.dec1, self.dec2, self.dec3, self.dec4, self.dec5, self.center]
        group3 = [self.classifier, self.logit]

        params1 = []
        for x in group1:
            for p in x.parameters():
                params1.append(p)
            
        param_group1 = {'params': params1, 'lr': base_lr / 100}

        params2 = []
        for x in group2:
            for p in x.parameters():
                params2.append(p)
        param_group2 = {'params': params2, 'lr': base_lr / 10}

        params3 = []
        for x in group3:
            for p in x.parameters():
                params3.append(p)
        param_group3 = {'params': params3, 'lr': base_lr}

        return [param_group1, param_group2, param_group3]



class EncoderAttention(nn.Module):
    def __init__(self, channels):
        super(EncoderAttention, self).__init__()
        self.spatial_gate = SpatialAttentionGate(channels)
        self.channel_gate = ChannelAttentionGate(channels)

    def forward(self, x):
        g1 = self.spatial_gate(x)
        g2 = self.channel_gate(x)
        x = x*g1 + x*g2

        return x

class UNetResNetV3(nn.Module):
    def __init__(self, encoder_depth, num_classes=1, num_filters=32, dropout_2d=0.2,
                 pretrained=True, is_deconv=True):
        super(UNetResNetV3, self).__init__()
        #pdb.set_trace()
        self.name = 'UNetResNetV3_'+str(encoder_depth)
        self.num_classes = num_classes
        self.dropout_2d = dropout_2d

        if encoder_depth == 34:
            self.encoder = torchvision.models.resnet34(pretrained=pretrained)
            bottom_channel_nr = 512
        elif encoder_depth == 50:
            self.encoder = torchvision.models.resnet50(pretrained=pretrained)
            bottom_channel_nr = 2048
        elif encoder_depth == 101:
            self.encoder = torchvision.models.resnet101(pretrained=pretrained)
            bottom_channel_nr = 2048
        elif encoder_depth == 152:
            self.encoder = torchvision.models.resnet152(pretrained=pretrained)
            bottom_channel_nr = 2048
        else:
            raise NotImplementedError('only 34, 101, 152 version of Resnet are implemented')

        self.pool = nn.MaxPool2d(2, 2)

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(self.encoder.conv1,
                                   self.encoder.bn1,
                                   self.encoder.relu,
                                   self.pool)
        self.conv2 = self.encoder.layer1

        self.conv3 = self.encoder.layer2

        self.conv4 = self.encoder.layer3

        self.conv5 = self.encoder.layer4

        self.center = DecoderV3(bottom_channel_nr, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec5 = DecoderV3(bottom_channel_nr + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec4 = DecoderV3(bottom_channel_nr // 2 + num_filters * 8, num_filters * 8 * 2, num_filters * 8,
                                   is_deconv)
        self.dec3 = DecoderV3(bottom_channel_nr // 4 + num_filters * 8, num_filters * 4 * 2, num_filters * 2,
                                   is_deconv)
        self.dec2 = DecoderV3(bottom_channel_nr // 8 + num_filters * 2, num_filters * 2 * 2, num_filters * 2 * 2,
                                   is_deconv)
        self.dec1 = DecoderV3(num_filters * 2 * 2, num_filters * 2 * 2, num_filters, is_deconv)
        #self.dec0 = ConvRelu(num_filters, num_filters)
        #self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

        #self.logit = nn.Sequential(
        #    EncoderAttention(736),
        #    nn.Conv2d(736, 64, kernel_size=3, padding=1),
        #    EncoderAttention(64),
        #    nn.ReLU(inplace=True),
        #    nn.Conv2d(64, 1, kernel_size=1, padding=0)
        #)
        self.logit = nn.Sequential(
            nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
            EncoderAttention(num_filters),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_filters, 1, kernel_size=1, padding=0)
        )

    def forward(self, x):
        conv1 = self.conv1(x) #;print('conv1:', conv1.size())
        conv2 = self.conv2(conv1) #;print('conv2:', conv2.size())
        conv3 = self.conv3(conv2) #;print('conv3:', conv3.size())
        conv4 = self.conv4(conv3) #;print('conv4:', conv4.size())
        conv5 = self.conv5(conv4) #;print('conv5:', conv5.size())

        pool = self.pool(conv5)
        center = self.center(pool)

        dec5 = self.dec5(torch.cat([center, conv5], 1))

        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1)) #print('dec2:', dec2.size())
        dec1 = self.dec1(dec2) #; print('dec1:', dec1.size())
        #dec0 = self.dec0(dec1); print('dec0:', dec0.size())

        #f = torch.cat([
        #    dec1,
        #    F.upsample(dec2, scale_factor=2, mode='bilinear', align_corners=False),
        #    F.upsample(dec3, scale_factor=4, mode='bilinear', align_corners=False),
        #    F.upsample(dec4, scale_factor=8, mode='bilinear', align_corners=False),
        #    F.upsample(dec5, scale_factor=16, mode='bilinear', align_corners=False),
        #], 1) 

        f = F.dropout2d(dec1, p=self.dropout_2d)
        #out = self.pool(dec0)

        return self.logit(f), None
    
    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def get_params(self, base_lr):
        group1 = [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5]
        group2 = [self.dec1, self.dec2, self.dec3, self.dec4, self.dec5, self.center]
        group3 = [self.logit]

        params1 = []
        for x in group1:
            for p in x.parameters():
                params1.append(p)
        
        param_group1 = {'params': params1, 'lr': base_lr / 10}

        params2 = []
        for x in group2:
            for p in x.parameters():
                params2.append(p)
        param_group2 = {'params': params2, 'lr': base_lr / 2}

        params3 = []
        for x in group3:
            for p in x.parameters():
                params3.append(p)
        param_group3 = {'params': params3, 'lr': base_lr}

        return [param_group1, param_group2, param_group3]



class UNetResNetV4(nn.Module):
    def __init__(self, encoder_depth, num_classes=1, num_filters=32, dropout_2d=0.2,
                 pretrained=True, is_deconv=True):
        super(UNetResNetV4, self).__init__()
        #pdb.set_trace()
        self.name = 'UNetResNetV4_'+str(encoder_depth)
        self.num_classes = num_classes
        self.dropout_2d = dropout_2d

        if encoder_depth == 34:
            self.encoder = torchvision.models.resnet34(pretrained=pretrained)
            bottom_channel_nr = 512
        elif encoder_depth == 50:
            self.encoder = torchvision.models.resnet50(pretrained=pretrained)
            bottom_channel_nr = 2048
        elif encoder_depth == 101:
            self.encoder = torchvision.models.resnet101(pretrained=pretrained)
            bottom_channel_nr = 2048
        elif encoder_depth == 152:
            self.encoder = torchvision.models.resnet152(pretrained=pretrained)
            bottom_channel_nr = 2048
        else:
            raise NotImplementedError('only 34, 101, 152 version of Resnet are implemented')

        self.pool = nn.MaxPool2d(2, 2)

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(self.encoder.conv1,
                                   self.encoder.bn1,
                                   self.encoder.relu,
                                   self.pool)
        self.att1 = EncoderAttention(num_filters*2)
        self.conv2 = self.encoder.layer1
        self.att2 = EncoderAttention(num_filters*8)

        self.conv3 = self.encoder.layer2
        self.att3 = EncoderAttention(num_filters*16)

        self.conv4 = self.encoder.layer3
        self.att4 = EncoderAttention(num_filters*32)

        self.conv5 = self.encoder.layer4
        self.att5 = EncoderAttention(num_filters*64)

        self.center = DecoderV3(bottom_channel_nr, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec5 = DecoderV3(bottom_channel_nr + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec4 = DecoderV3(bottom_channel_nr // 2 + num_filters * 8, num_filters * 8 * 2, num_filters * 8,
                                   is_deconv)
        self.dec3 = DecoderV3(bottom_channel_nr // 4 + num_filters * 8, num_filters * 4 * 2, num_filters * 2,
                                   is_deconv)
        self.dec2 = DecoderV3(bottom_channel_nr // 8 + num_filters * 2, num_filters * 2 * 2, num_filters * 2 * 2,
                                   is_deconv)
        self.dec1 = DecoderV3(num_filters * 2 * 2, num_filters * 2 * 2, num_filters, is_deconv)
        #self.dec0 = ConvRelu(num_filters, num_filters)
        #self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

        self.logit = nn.Sequential(
            EncoderAttention(736),
            nn.Conv2d(736, 64, kernel_size=3, padding=1),
            EncoderAttention(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1, padding=0)
        )

    def forward(self, x):
        conv1 = self.conv1(x) #;print('conv1:', conv1.size())
        att1 = self.att1(conv1) #; print('att1:', att1.size())
        conv2 = self.conv2(att1) #;print('conv2:', conv2.size())
        att2 = self.att2(conv2) #; print('att2:', att2.size())
        conv3 = self.conv3(att2) #;print('conv3:', conv3.size())
        att3 = self.att3(conv3) #; print('att3:', att3.size())
        conv4 = self.conv4(att3) #;print('conv4:', conv4.size())
        att4 = self.att4(conv4) #; print('att4:', att4.size())
        conv5 = self.conv5(att4) #;print('conv5:', conv5.size())
        att5 = self.att5(conv5) #; print('att5:', att5.size())

        pool = self.pool(att5)
        center = self.center(pool)

        dec5 = self.dec5(torch.cat([center, att5], 1))

        dec4 = self.dec4(torch.cat([dec5, att4], 1))
        dec3 = self.dec3(torch.cat([dec4, att3], 1))
        dec2 = self.dec2(torch.cat([dec3, att2], 1)); #print('dec2:', dec2.size())
        dec1 = self.dec1(dec2); #print('dec1:', dec1.size())
        #dec0 = self.dec0(dec1); print('dec0:', dec0.size())

        f = torch.cat([
            dec1,
            F.upsample(dec2, scale_factor=2, mode='bilinear', align_corners=False),
            F.upsample(dec3, scale_factor=4, mode='bilinear', align_corners=False),
            F.upsample(dec4, scale_factor=8, mode='bilinear', align_corners=False),
            F.upsample(dec5, scale_factor=16, mode='bilinear', align_corners=False),
        ], 1) 

        f = F.dropout2d(f, p=self.dropout_2d)
        #out = self.pool(dec0)

        return self.logit(f), None
    
    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def get_params(self, base_lr):
        group1 = [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5]
        group2 = [self.dec1, self.dec2, self.dec3, self.dec4, self.dec5, self.center]
        group3 = [self.att1, self.att2, self.att3, self.att4, self.att5,]
        group4 = [self.logit]

        params1 = []
        for x in group1:
            for p in x.parameters():
                params1.append(p)
        
        param_group1 = {'params': params1, 'lr': base_lr / 100}

        params2 = []
        for x in group2:
            for p in x.parameters():
                params2.append(p)
        param_group2 = {'params': params2, 'lr': base_lr / 10}

        params3 = []
        for x in group3:
            for p in x.parameters():
                params3.append(p)
        param_group3 = {'params': params3, 'lr': base_lr / 20}

        params4 = []
        for x in group4:
            for p in x.parameters():
                params4.append(p)
        param_group4 = {'params': params4, 'lr': base_lr}

        return [param_group1, param_group2, param_group3, param_group4]

def test():
    model = UNetResNetV3(152).cuda()
    model.freeze_bn()
    inputs = torch.randn(2,3,128,128).cuda()
    out, _ = model(inputs)
    #print(model)
    print(out.size()) #, cls_taret.size())
    #print(out)


if __name__ == '__main__':
    test()