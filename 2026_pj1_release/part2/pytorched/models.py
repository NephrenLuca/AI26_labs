import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, drop_p: float = 0.0) -> None:
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        ]
        if drop_p > 0:
            layers.append(nn.Dropout2d(p=drop_p))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class HanziCNN(nn.Module):
    def __init__(self, num_classes: int = 12) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1, 32, drop_p=0.05),
            ConvBlock(32, 64, drop_p=0.10),
            ConvBlock(64, 128, drop_p=0.15),
            ConvBlock(128, 256, drop_p=0.20),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x
