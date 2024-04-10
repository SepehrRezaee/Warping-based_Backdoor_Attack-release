"""VGG11/13/16/19 in PyTorch."""
import torch
import torch.nn as nn

# cfg = {
#     "VGG11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
#     "VGG13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
#     "VGG16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
#     "VGG19": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
# }


# class VGG(nn.Module):
#     def __init__(self, vgg_name, num_classes=10):
#         super(VGG, self).__init__()
#         self.features = self._make_layers(cfg[vgg_name])
#         self.classifier = nn.Linear(512, num_classes)

#     def forward(self, x):
#         out = self.features(x)
#         out = out.view(out.size(0), -1)
#         out = self.classifier(out)
#         return out

#     def _make_layers(self, cfg):
#         layers = []
#         in_channels = 3
#         for x in cfg:
#             if x == "M":
#                 layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
#             else:
#                 layers += [
#                     nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
#                     nn.BatchNorm2d(x),
#                     nn.ReLU(inplace=True),
#                 ]
#                 in_channels = x
#         layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
#         return nn.Sequential(*layers)


class VGG(nn.Module):
    """ MNIST Encoder from Original Paper's Keras based Implementation.
        Args:
            init_num_filters (int): initial number of filters from encoder image channels
            lrelu_slope (float): positive number indicating LeakyReLU negative slope
            inter_fc_dim (int): intermediate fully connected dimensionality prior to embedding layer
            embedding_dim (int): embedding dimensionality
    """
    def __init__(self, init_num_filters=32, lrelu_slope=0.2, inter_fc_dim=128, nofclasses=10,nofchannels=3):
        super(VGG, self).__init__()
        # self.use_stn=use_stn
        self.init_num_filters_ = init_num_filters
        self.lrelu_slope_ = lrelu_slope
        self.inter_fc_dim_ = inter_fc_dim
        self.nofclasses_ = nofclasses
        # if self.use_stn:
        #     self.stn = STN()

        self.features = nn.Sequential(
            nn.Conv2d(nofchannels, self.init_num_filters_ * 1, kernel_size=5, padding=2),
            nn.BatchNorm2d(self.init_num_filters_ * 1),
            nn.ReLU(True),

            nn.Conv2d(self.init_num_filters_ * 1, self.init_num_filters_ * 1, kernel_size=5,padding=2),
            nn.BatchNorm2d(self.init_num_filters_ * 1),
            nn.ReLU(True),

            nn.MaxPool2d(2,2),

            nn.Conv2d(self.init_num_filters_ * 1, self.init_num_filters_ * 1, kernel_size=5,padding=2),
            nn.BatchNorm2d(self.init_num_filters_ * 1),
            nn.ReLU(True),

            nn.Conv2d(self.init_num_filters_ * 1, self.init_num_filters_ * 1, kernel_size=5,padding=2),
            nn.BatchNorm2d(self.init_num_filters_ * 1),
            nn.ReLU(True),

            nn.MaxPool2d(2,2),

            nn.Conv2d(self.init_num_filters_ * 1, self.init_num_filters_ * 1, kernel_size=5,padding=2),
            nn.BatchNorm2d(self.init_num_filters_ * 1),
            nn.ReLU(True),

            nn.Conv2d(self.init_num_filters_ * 1, self.init_num_filters_ * 1, kernel_size=5,padding=2),
            nn.BatchNorm2d(self.init_num_filters_ * 1),
            nn.ReLU(True),

            nn.MaxPool2d(2,2),
        )

        self.fc = nn.Sequential(
            nn.Linear(self.init_num_filters_ *4*4, self.inter_fc_dim_),
            nn.BatchNorm1d(self.inter_fc_dim_),
            nn.ReLU(True),
            nn.Dropout(p=.2),

            nn.Linear(self.inter_fc_dim_, int(self.inter_fc_dim_/2)),
            nn.BatchNorm1d(int(self.inter_fc_dim_/2)),
            nn.ReLU(True),
            nn.Dropout(p=.2),

            nn.Linear(int(self.inter_fc_dim_/2), self.nofclasses_)
        )

    def forward(self, x):
        # if self.use_stn:
        #     x = self.stn(x)
        x = self.features(x)
#         print(x.shape)
        x = x.view(-1, self.init_num_filters_ *4*4)
        x = self.fc(x)
        return x


def create_vgg():
    return VGG()


def test():
    net = create_vgg()
    x = torch.randn(2, 3, 32, 32)
    y = net(x)
    print(y.size())


if __name__ == "__main__":
    test()