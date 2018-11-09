import torch
import torch.nn as nn
import torchvision.models as models

class Unpooling(nn.Module):
    def __init__(self):
        super(Unpooling, self).__init__()
    #input [10, 2, 512, 20, 20]
    def forward(self, input):
        original = input[:, 1, :, :, :] #original output
        x = input[:, 0, :, :, :]
        original = original.squeeze(1)
        x = x.squeeze(1)
        bool_mask = x >= original
        bool_mask = bool_mask.float()
        x = bool_mask * x
        return x

class VGG(nn.Module):
    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        return x

    def _initialize_weights(self):
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

# def make_layers(cfg, batch_norm=False):
#     layers = []
#     in_channels = 3
#     for v in cfg:
#         if v == 'M':
#             layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
#         else:
#             conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
#             if batch_norm:
#                 layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
#             else:
#                 layers += [conv2d, nn.ReLU(inplace=True)]
#             in_channels = v
#     return nn.Sequential(*layers)

def vgg16(**kwargs):
    model = VGG(make_layers(cfg, batch_norm=True), **kewargs)
    return model

cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']

# VGG 16 D
def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 4
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

def get_encoder():
    model = models.vgg16(pretrained=True)
    features = model.features
    encoder = VGG(make_layers(cfg))
    for i, v in enumerate(encoder.features):
        vgg16_module = features[i]
        encoder_module = v
        if isinstance(v, nn.Conv2d):
            if i==0:
                # add the channel
                weight = vgg16_module.weight
                zero_channel = torch.zeros(weight.size(0),1,weight.size(2),weight.size(3))
                weight = torch.cat((weight, zero_channel), dim=1)
                encoder_module.weight = torch.nn.Parameter(weight)
                #encoder_module.weight.data.assign(torch.nn.Parameter(weight))
            else:
                encoder_module.weight = torch.nn.Parameter(vgg16_module.weight)
                #encoder_module.weight.data.assign(torch.nn.Parameter(weight))
    return encoder

decode_cfg = ['u', 512, 'u', 256, 'u', 128, 'u', 64, 'u', 64]

def decode_make_layers(decode_cfg, batch_norm=True):
    layers = []
    in_channels = 512

    conv2d = nn.Conv2d(in_channels, 512, kernel_size=1, stride=1, padding=1)
    if batch_norm:
        layers += [conv2d, nn.BatchNorm2d(512), nn.ReLU(inplace=True)]
    else:
        layers += [conv2d, nn.ReLU(inplace=True)]
    in_channels = 512

    for v in decode_cfg:
        if v == 'u':
            # layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            layers += [nn.Upsample(scale_factor=2, mode='bilinear')]
            # layers += [nn.MaxUnpool2d(2, stride=2)]
            #layers += [nn.ConvTranspose2d(in_channels, in_channels, ())]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=5, stride=1, padding=1)
            conv2d_1 = nn.Conv2d(v, v, kernel_size=5, stride=1, padding=1)
            conv2d_2 = nn.Conv2d(v, v, kernel_size=5, stride=1, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                #layers += [conv2d_1, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                #layers += [conv2d_2, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
                #layers += [conv2d_1, nn.ReLU(inplace=True)]
                #layers += [conv2d_2, nn.ReLU(inplace=True)]
            in_channels = v

    conv2d = nn.Conv2d(in_channels, 1, kernel_size=5, stride=1, padding=1)
    if batch_norm:
        layers += [conv2d, nn.BatchNorm2d(1)]
    else:
        layers += [conv2d]

    return nn.Sequential(*layers)

class Decoder(nn.Module):
    def __init__(self, features):
        super(Decoder, self).__init__()
        self.features = features
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                #nn.init.constant_(m.weight, 1)
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.features(x)

class Decoder123(nn.Module):
    def __init__(self):
        super(Decoder123, self).__init__()
        self.unpool = Unpooling()
        self.maxpool_encode = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upsample_5 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.deconv_5 = nn.Sequential([nn.Conv2d(512, 512, kernel_size=5, stride=1, padding=1),
                                       nn.BatchNorm2d(512),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(512, 512, kernel_size=5, stride=1, padding=1),
                                       nn.BatchNorm2d(512),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(512, 512, kernel_size=5, stride=1, padding=1),
                                       nn.BatchNorm2d(512),
                                       nn.ReLU(inplace=True),
                                       ])
    def forward(self, x):
        # x [batch, 512, 20 ,20]
        origin_5 = x
        #[10, 512, 10, 10]
        x = self.maxpool_encode(x)
        #[10, 512, 20, 20]
        x = self.upsample_5(x)
        origin_5_reshape = origin_5.unsqueeze(1)
        x_reshape = x.unsqueeze(1)
        #[10, 2, 512, 20, 20]
        together = torch.cat((origin_5_reshape, x_reshape), dim=1)
        #[10, 512, 20, 20]
        x = Unpooling(together)



def get_decoder():
    decoder = Decoder(decode_make_layers(decode_cfg))
    return decoder

class EncoderDecoder(nn.Module):
    def __init__(self):
        super(EncoderDecoder, self).__init__()
        self.encoder = get_encoder()
        self.decoder = get_decoder()

    def forward(self, x):
        temp = self.encoder(x)
        # [16, 1, 320, 320]
        output = self.decoder(temp)
        output = output.transpose(1,3)
        output = output.transpose(1,2)
        return output



if __name__ == '__main__':
    encoder = get_encoder()
    #print(encoder)
    decoder = get_decoder()
    print(decoder)
    print('Success')
