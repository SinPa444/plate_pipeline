import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.quantization import QuantStub, DeQuantStub

# --- Bottleneck تغییری نکرده چون ResNet18 از Block استفاده می‌کند ---
# اما برای کامل بودن کد، اگر خواستید برای ResNet50 استفاده کنید باید مثل Block تغییرش دهید
class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, padding=0)
        self.batch_norm3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()
        # اضافه شده برای کوانتایز (جایگزین جمع معمولی)
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        identity = x.clone()
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.relu(self.batch_norm2(self.conv2(x)))
        x = self.conv3(x)
        x = self.batch_norm3(x)
        if self.i_downsample is not None:
            identity = self.i_downsample(identity)
        
        # تغییر مهم: استفاده از skip_add برای جمع
        x = self.skip_add.add(x, identity)
        x = self.relu(x)
        return x

class Block(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()
        
        # تغییر ۱: ماژول مخصوص جمع در کوانتایز
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        identity = x.clone()
        
        # این بخش بعدا Fuse می‌شود
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.batch_norm2(self.conv2(x))

        if self.i_downsample is not None:
            identity = self.i_downsample(identity)
            
        # تغییر ۲: استفاده از skip_add به جای x += identity
        x = self.skip_add.add(x, identity)
        x = self.relu(x)
        return x

class ResNet(nn.Module):
    def __init__(self, ResBlock, layer_list, num_classes, num_channels=3):
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(ResBlock, layer_list[0], planes=64)
        self.layer2 = self._make_layer(ResBlock, layer_list[1], planes=128, stride=2)
        self.layer3 = self._make_layer(ResBlock, layer_list[2], planes=256, stride=2)
        self.layer4 = self._make_layer(ResBlock, layer_list[3], planes=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * ResBlock.expansion, num_classes)

        # تغییر ۳: دروازه‌های ورود و خروج کوانتایزیشن
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, x):
        # تبدیل ورودی به int8
        x = self.quant(x)

        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        # تبدیل خروجی به float32
        x = self.dequant(x)
        return x

    def _make_layer(self, ResBlock, blocks, planes, stride=1):
        ii_downsample = None
        layers = []

        if stride != 1 or self.in_channels != planes * ResBlock.expansion:
            ii_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, planes * ResBlock.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes * ResBlock.expansion)
            )

        layers.append(ResBlock(self.in_channels, planes, i_downsample=ii_downsample, stride=stride))
        self.in_channels = planes * ResBlock.expansion

        for i in range(blocks - 1):
            layers.append(ResBlock(self.in_channels, planes))

        return nn.Sequential(*layers)

    # تغییر ۴: تابعی برای ادغام لایه‌ها
    def fuse_model(self):
        # ادغام لایه اول (Conv + BN + ReLU)
        torch.quantization.fuse_modules(self, [['conv1', 'batch_norm1', 'relu']], inplace=True)
        
        # ادغام بلوک‌های ResNet
        for m in self.modules():
            if type(m) == Block:
                # Conv1 + BN1 + ReLU
                torch.quantization.fuse_modules(m, [['conv1', 'batch_norm1', 'relu']], inplace=True)
                # Conv2 + BN2 (دقت کنید ReLU آخر جداست چون بعد از جمع انجام میشه)
                torch.quantization.fuse_modules(m, [['conv2', 'batch_norm2']], inplace=True)
                # اگر downsample داشت، آن را هم فیوز کن
                if m.i_downsample:
                    torch.quantization.fuse_modules(m.i_downsample, ['0', '1'], inplace=True)

def ResNet18(num_classes, channels=3):
    return ResNet(Block, [2, 2, 2, 2], num_classes, channels)