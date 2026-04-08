from mynn import AdaptiveAvgPool2d, Conv2d, Dropout, Flatten, Linear, MaxPool2d, Module, ReLU, Sequential


class HanziCNN(Module):
    def __init__(self, num_classes: int = 12) -> None:
        super().__init__()
        self.net = Sequential(
            Conv2d(1, 16, kernel_size=3, padding=1),
            ReLU(),
            Conv2d(16, 16, kernel_size=3, padding=1),
            ReLU(),
            MaxPool2d(2, 2),
            Conv2d(16, 32, kernel_size=3, padding=1),
            ReLU(),
            Conv2d(32, 32, kernel_size=3, padding=1),
            ReLU(),
            MaxPool2d(2, 2),
            Conv2d(32, 64, kernel_size=3, padding=1),
            ReLU(),
            MaxPool2d(2, 2),
            AdaptiveAvgPool2d((1, 1)),
            Flatten(),
            Linear(64, 64),
            ReLU(),
            Dropout(0.2),
            Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net.forward(x)

    def backward(self, grad):
        return self.net.backward(grad)
