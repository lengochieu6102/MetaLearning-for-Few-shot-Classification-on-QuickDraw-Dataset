import torch.nn as nn

def maml_init_(module):
    nn.init.xavier_uniform_(module.weight.data, gain=1.0)
    nn.init.constant_(module.bias.data, 0.0)
    return module
    
class ConvBlock(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 max_pool=True,
                 max_pool_factor=1.0):
        super(ConvBlock, self).__init__()
        stride = (int(2 * max_pool_factor), int(2 * max_pool_factor))
        if max_pool:
            self.max_pool = nn.MaxPool2d(
                kernel_size=stride,
                stride=stride,
                # ceil_mode=False,
            )
            stride = (1, 1)
        else:
            self.max_pool = lambda x: x
        self.normalize = nn.BatchNorm2d(
            out_channels,
            affine=True,
            # eps=1e-3,
            # momentum=0.999,
            # track_running_stats=False,
        )
        nn.init.uniform_(self.normalize.weight)
        self.relu = nn.ReLU()
        
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=1,
            bias=True,
        )
        maml_init_(self.conv)

    def forward(self, x):
        x = self.conv(x)
        x = self.normalize(x)
        x = self.relu(x)
        x = self.max_pool(x)

        return x

from collections import OrderedDict
class QuicDrawCNN(nn.Module):
    def __init__(self,
                channels,
                out_features,
                hidden=64,
                max_pool=True,
                layers=4,
                max_pool_factor=1.0):
        super(QuicDrawCNN, self).__init__()
        self.in_channels = channels
        self.out_features = out_features
        self.hidden_size = hidden

        core = [('layer1',ConvBlock(channels,
                          hidden,
                          (3, 3),
                          max_pool=max_pool,
                          max_pool_factor=max_pool_factor),
                )]
        for i in range(layers - 1):
            core.append((f'layer{i+2}',ConvBlock(hidden,
                                  hidden,
                                  kernel_size=(3, 3),
                                  max_pool=max_pool,
                                  max_pool_factor=max_pool_factor)))
        self.features = nn.Sequential(OrderedDict(core))

    def forward(self, inputs, params=None):
        features = self.features(inputs)
        return features