from torch import nn

FUNCS = {
    "conv": nn.Conv2d,
    "bn": nn.BatchNorm2d,
    "relu": nn.ReLU,
    "max_pool": nn.MaxPool2d,
    "avg_pool": nn.AvgPool2d,
    "dropout": nn.Dropout2d,
    "fc": nn.Linear,
}
