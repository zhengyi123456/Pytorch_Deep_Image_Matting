import torch
import torch.nn as nn
import torchvision.models as models

cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']

def migrage_vgg(encoder):
    vgg_model = models.vgg16(pretrained=True)
    features = vgg_model.features
    pointer = -1
    vgg_parameters = []
    for i, k in enumerate(vgg_model.modules()):
        if isinstance(k, nn.Conv2d):
            vgg_parameters.append(k)

    for i, k in enumerate(encoder.modules()):
        if isinstance(k, nn.Conv2d):
            pointer += 1
            para = vgg_parameters[pointer]
            if pointer == 0:
                pass
                weight = para.weight
                zero_channel = torch.zeros(weight.size(0), 1, weight.size(2), weight.size(3))
                weight = torch.cat((weight, zero_channel), dim=1)
                k.weight = torch.nn.Parameter(weight)
            else:
                k.weight = torch.nn.Parameter(para.weight)

    return encoder

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # input [10, 4, 320, 320]

        self.conv_1 = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.maxpool_1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.maxpool_2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.maxpool_3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.maxpool_4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.maxpool_5 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv_1(x)
        orig_1 = x
        x = self.maxpool_1(x)
        x = self.conv_2(x)
        orig_2 = x
        x = self.maxpool_2(x)
        x = self.conv_3(x)
        orig_3 = x
        x = self.maxpool_3(x)
        x = self.conv_4(x)
        orig_4 = x
        x = self.maxpool_5(x)
        x = self.conv_5(x)
        orig_5 = x
        x = self.maxpool_5(x)
        orig = []
        orig.append(orig_5)
        orig.append(orig_4)
        orig.append(orig_3)
        orig.append(orig_2)
        orig.append(orig_1)
        return x, orig

class Unpool(nn.Module):
    def __init__(self):
        super(Unpool, self).__init__()

    # [10, 2, None, None, None] input [10, None, None, None] output
    def forward(self, input):
        orig = input[:, 0]
        x = input[:, 1]
        bool_mask = orig >= x
        bool_mask = bool_mask.float()
        x = bool_mask * x

        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.Unpooling = Unpool()

        self.upsample_1 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.deconv_1 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512)
        )

        self.upsample_2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.deconv_2 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256)
        )

        self.upsample_3 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.deconv_3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128)
        )

        self.upsample_4 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.deconv_4 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64)
        )

        self.upsample_5 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.deconv_5 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64)
        )

        self.deconv_6 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.deconv_6_bn = nn.Sigmoid()
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                #nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                #nn.init.constant_(m.weight, 1)
                #nn.init.xavier_normal_(m.weight)
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, orig):
        #[10, 512, 20, 20]
        x = self.upsample_1(x)
        orig_5 = orig[0] #[10, 512, 20, 20]
        together = torch.cat((orig_5.unsqueeze(1), x.unsqueeze(1)), dim=1)#[10, 2, 512, 20, 20]
        x = self.Unpooling(together) #[10,512,20,20]
        x = self.deconv_1(x)#[10, 512, 20, 20]

        x = self.upsample_2(x)#[10, 512, 40, 40]
        orig_4 = orig[1]
        together = torch.cat((orig_4.unsqueeze(1), x.unsqueeze(1)), dim=1)
        x = self.Unpooling(together)
        x= self.deconv_2(x) #[10, 256, 40, 40]

        x = self.upsample_3(x)
        orig_3 = orig[2]
        together = torch.cat((orig_3.unsqueeze(1), x.unsqueeze(1)), dim=1)
        x = self.Unpooling(together)
        x = self.deconv_3(x) #[10, 128, 80, 80]

        x = self.upsample_4(x)
        orig_4 = orig[3]
        together = torch.cat((orig_4.unsqueeze(1), x.unsqueeze(1)), dim=1)
        x = self.Unpooling(together)
        x = self.deconv_4(x) #[10, 64, 160, 160]

        x = self.upsample_5(x)
        orig_5 = orig[4]
        together = torch.cat((orig_5.unsqueeze(1), x.unsqueeze(1)), dim=1)
        x = self.Unpooling(together)
        x = self.deconv_5(x) #[10, 64, 320, 320]

        x = self.deconv_6(x) #[10, 1, 320, 320]
        x = self.deconv_6_bn(x)

        return x

class EncoderDecoder(nn.Module):
    def __init__(self):
        super(EncoderDecoder, self).__init__()
        self.encoder = Encoder()
        self.encoder = migrage_vgg(self.encoder)
        self.decoder = Decoder()
    def forward(self, x):
        x, orig = self.encoder(x)
        x = self.decoder(x, orig)

        output = x
        output = output.transpose(1, 3)
        output = output.transpose(1, 2)
        x = output

        return x

if __name__ == '__main__':
    encoder = Encoder()
    encoder = migrage_vgg(encoder)

    decoder = Decoder()

    for i, k in enumerate(encoder.conv_5.modules()):
        if i==1:
            print(k.weight)
