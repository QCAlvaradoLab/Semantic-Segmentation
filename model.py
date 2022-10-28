import torch
from torch.nn.init import xavier_uniform_, uniform_
import torch.nn.functional as F
from torch import nn

class encoder_module(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=False):
        super(encoder_module, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        
        for layer in layers:
            if isinstance(layer, nn.Conv2d):
                xavier_uniform_(layer.weight.data)
                layer.bias.data.zero_()    
            elif not isinstance(layer, nn.ReLU):
                uniform_(layer.weight.data)
                layer.bias.data.zero_()    

        if dropout:
            layers.append(nn.Dropout())
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.encode = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.encode(x)

class decoder_module(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(decoder_module, self).__init__()
        layers = [
            nn.Conv2d(in_channels, middle_channels, kernel_size=3),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, middle_channels, kernel_size=3),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=2, stride=2),
        ]
        self.decode = nn.Sequential(*layers)

        for layer in layers:
            if isinstance(layer, nn.Conv2d):
                xavier_uniform_(layer.weight.data)
                layer.bias.data.zero_()    
            elif not isinstance(layer, nn.ReLU):
                uniform_(layer.weight.data)
                layer.bias.data.zero_()    
    
    def forward(self, x):
        return self.decode(x)

class UNet(nn.Module):
    def __init__(self, num_classes):
        super(UNet, self).__init__()
        self.enc1 = encoder_module(3, 64)
        self.enc2 = encoder_module(64, 128)
        self.enc3 = encoder_module(128, 256)
        self.enc4 = encoder_module(256, 512, dropout=True)
        self.feat = decoder_module(512, 1024, 512)
        self.dec4 = decoder_module(1024, 512, 256)
        self.dec3 = decoder_module(512, 256, 128)
        self.dec2 = decoder_module(256, 128, 64)
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        for layer in self.dec1:
            
            if isinstance(layer, nn.Conv2d):
                xavier_uniform_(layer.weight.data)
                layer.bias.data.zero_() 
            elif not isinstance(layer, nn.ReLU):
                uniform_(layer.weight)
                layer.bias.data.zero_() 
            
        if num_classes == 2:
            num_classes = 1
        self.num_classes = num_classes
        
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)

        xavier_uniform_(self.final.weight.data)
        self.final.bias.data.zero_()

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        feat = self.feat(enc4)
        dec4 = self.dec4(torch.cat([feat, F.interpolate(enc4, feat.size()[2:], mode='bilinear')], 1))
        dec3 = self.dec3(torch.cat([dec4, F.interpolate(enc3, dec4.size()[2:], mode='bilinear')], 1))
        dec2 = self.dec2(torch.cat([dec3, F.interpolate(enc2, dec3.size()[2:], mode='bilinear')], 1))
        dec1 = self.dec1(torch.cat([dec2, F.interpolate(enc1, dec2.size()[2:], mode='bilinear')], 1))
        final = self.final(dec1)

        seg_mask = F.interpolate(final, x.size()[2:], mode='bilinear')
        
        if self.num_classes == 1:
            return torch.sigmoid(seg_mask)
        else:
            return seg_mask

if __name__=='__main__':
    
    model = UNet(num_classes=4)
    model.forward(torch.rand((32, 3,512,512)))
