import math
import numpy as np
from tqdm import tqdm_notebook as tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
import torch
from torch import Tensor
from torch import nn
from torch.nn  import functional as F
from torch.autograd import Variable
from torchdiffeq import odeint, odeint_adjoint


MAX_NUM_STEPS = 1000

class Conv2dTime(nn.Conv2d):
    def __init__(self, in_channels, *args, **kwargs):
        """
        Code adapted from https://github.com/EmilienDupont/augmented-neural-odes
        Conv2d module where time gets concatenated as a feature map.
        Makes ODE func aware of the current time step.
        """
        super(Conv2dTime, self).__init__(in_channels + 1, *args, **kwargs)

    def forward(self, t, x):
        # Shape (batch_size, 1, height, width)
        t_img = torch.ones_like(x[:, :1, :, :]) * t
        # Shape (batch_size, channels + 1, height, width)
        t_and_x = torch.cat([t_img, x], 1)
        return super(Conv2dTime, self).forward(t_and_x)


def conv_bn(inp, oup, stride, conv_layer=nn.Conv2d, norm_layer=nn.BatchNorm2d, nlin_layer=nn.ReLU):
    return nn.Sequential(
        conv_layer(inp, oup, 3, stride, 1, bias=False),
        norm_layer(oup),
        nlin_layer(inplace=True)
    )


def conv_1x1_bn(inp, oup, conv_layer=nn.Conv2d, norm_layer=nn.BatchNorm2d, nlin_layer=nn.ReLU):
    return nn.Sequential(
        conv_layer(inp, oup, 1, 1, 0, bias=False),
        norm_layer(oup),
        nlin_layer(inplace=True)
    )


class Hswish(nn.Module):
    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3., inplace=self.inplace) / 6.


class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3., inplace=self.inplace) / 6.


class SEModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            Hsigmoid()
            # nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SEModule2(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SEModule2, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel+1, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            Hsigmoid()
            # nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        print(b)
        print(c)
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class Identity(nn.Module):
    def __init__(self, channel):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def make_divisible(x, divisible_by=8):
    import numpy as np
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)


class ODEfunc(nn.Module):
    def __init__(self, inp, oup, kernel, stride, exp, se=False, nl='RE', time_dependent =True):
        super(ODEfunc, self).__init__()
        assert stride in [1, 2]
        assert kernel in [3, 5]
        padding = (kernel - 1) // 2
        if nl == 'RE':
            nlin_layer = nn.ReLU # or ReLU6
        elif nl == 'HS':
            nlin_layer = Hswish
        else:
            raise NotImplementedError
        if se:
            SELayer = SEModule2
        else:
            SELayer = Identity
        
        self.nfe = 0
        self.time_dependent = time_dependent
        
        if self.time_dependent: 
            self.conv1 = Conv2dTime(inp, exp, 1, 1, 0, bias=False)
            self.conv2 = Conv2dTime(exp, exp+1, kernel, stride, padding, groups=exp + 1, bias=False)
            self.conv3 = Conv2dTime(exp+1, oup, 1, 1, 0)
            self.nl = nlin_layer(inplace=True)
            self.se = SELayer(exp)
            self.norm0 = nn.BatchNorm2d(exp)
            self.norm1 = nn.BatchNorm2d(exp+1)
            self.norm2 = nn.BatchNorm2d(oup)
        else:
            self.conv1 = nn.Conv2d(inp, exp, 1, 1, 0, bias=False)
            self.conv2 = nn.Conv2d(exp, exp, kernel, stride, padding, groups=exp, bias=False)
            self.conv3 = nn.Conv2d(exp, oup, 1, 1, 0, bias=False)
            self.nl = nlin_layer(inplace=True)
            self.se = SELayer(exp)
            self.norm1 = nn.BatchNorm2d(exp)
            self.norm2 = nn.BatchNorm2d(oup)


    def forward(self, t, x):
        self.nfe += 1
        
        if self.time_dependent: 
            out = self.conv1(t, x)
            out = self.norm0(out)
            out = self.nl(out)
            out = self.conv2(t, out)
            out = self.norm1(out)
            out = self.se(out)
            out = self.nl(out)
            out = self.conv3(t, out)
            out = self.norm2(out)
        else:
            out = self.conv1(x)
            out = self.norm1(out)
            out = self.nl(out)
            out = self.conv2(out)
            out = self.norm1(out)
            out = self.se(out)
            out = self.nl(out)
            out = self.conv3(out)
            out = self.norm2(out)

        return out

class ODEBottleNeck(nn.Module):
    def __init__(self, input_channel, output_channel, k, s, exp_channel, se, nl, tol=1e-3, adjoint=False):
        """
        Code adapted from https://github.com/EmilienDupont/augmented-neural-odes
        Utility class that wraps odeint and odeint_adjoint.
        Args:
            odefunc (nn.Module): the module to be evaluated
            tol (float): tolerance for the ODE solver
            adjoint (bool): whether to use the adjoint method for gradient calculation
        """
        super(ODEBottleNeck, self).__init__()
        self.adjoint = adjoint
        self.odefunc = ODEfunc(input_channel, output_channel, k, s, exp_channel, se, nl)
        self.tol = tol

    def forward(self, x):
        # Forward pass corresponds to solving ODE, so reset number of function
        # evaluations counter
        self.odefunc.nfe = 0

        integration_time = torch.tensor([0, 1]).float().type_as(x)

        if self.adjoint:
            out = odeint_adjoint(self.odefunc, x, integration_time,
                                 rtol=self.tol, atol=self.tol, method='dopri5',
                                 options={'max_num_steps': MAX_NUM_STEPS})
        else:
            out = odeint(self.odefunc, x, integration_time,
                         rtol=self.tol, atol=self.tol, method='dopri5',
                         options={'max_num_steps': MAX_NUM_STEPS})

        return out[1]  # Return only final time


class MobileBottleneck(nn.Module):
    def __init__(self, inp, oup, kernel, stride, exp, se=False, nl='RE'):
        super(MobileBottleneck, self).__init__()
        assert stride in [1, 2]
        assert kernel in [3, 5]
        padding = (kernel - 1) // 2
        if stride == 1 and inp == oup:
            self.use_res_connect = True
        else:
            self.use_res_connect = False

        conv_layer = nn.Conv2d
        norm_layer = nn.BatchNorm2d
        if nl == 'RE':
            nlin_layer = nn.ReLU # or ReLU6
        elif nl == 'HS':
            nlin_layer = Hswish
        else:
            raise NotImplementedError
        if se:
            SELayer = SEModule
        else:
            SELayer = Identity

        self.conv = nn.Sequential(
            # pw
            conv_layer(inp, exp, 1, 1, 0, bias=False),
            norm_layer(exp),
            nlin_layer(inplace=True),
            # dw
            conv_layer(exp, exp, kernel, stride, padding, groups=exp, bias=False),
            norm_layer(exp),
            SELayer(exp),
            nlin_layer(inplace=True),
            # pw-linear
            conv_layer(exp, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class ContinuousMobileNetV3(nn.Module):
    def __init__(self, n_class=200, input_size=64, dropout=0.3, mode='small', width_mult=1.0):
        super(ContinuousMobileNetV3, self).__init__()
        input_channel = 16
        last_channel = 1280

        # refer to Table 2 in paper
        mobile_setting1 = [
            # k, exp, c,  se,     nl,  s,
            [3, 16,  16,  True,  'RE', 1],
            [3, 72,  24,  False, 'RE', 1],
            [3, 88,  24,  False, 'RE', 1],
            [5, 96,  40,  True,  'HS', 2]
        ]

        # ODE 1
        ODE_setting1 = [
            # k, exp, c,  se,     nl,  s,
            [5, 120, 48,  False,  'HS', 1]
        ]
        
        ODE_setting2 = [
            # k, exp, c,  se,     nl,  s,
            [5, 288, 96,  False,  'HS', 2]
        ]

        mobile_setting2 = [
            # k, exp, c,  se,     nl,  s,
            [5, 576, 96,  True,  'HS', 1],
            [5, 576, 96,  True,  'HS', 1],
        ]

        # building Down Sampling layer
        assert input_size % 32 == 0
        last_channel = make_divisible(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(3, input_channel, 2, nlin_layer=Hswish)]
        self.classifier = []

        # building mobile BottleNeck blocks
        for k, exp, c, se, nl, s in mobile_setting1:
            output_channel = make_divisible(c * width_mult)
            exp_channel = make_divisible(exp * width_mult)
            self.features.append(MobileBottleneck(input_channel, output_channel, k, s, exp_channel, se, nl))
            input_channel = output_channel

        # building mobile ODE blocks
        for k, exp, c, se, nl, s in ODE_setting1:
            output_channel = make_divisible(c * width_mult)
            exp_channel = make_divisible(exp * width_mult)
            self.features.append(ODEBottleNeck(input_channel, output_channel, k, s, exp_channel, se, nl))
            input_channel = output_channel

        # building mobile ODE blocks
        for k, exp, c, se, nl, s in ODE_setting2:
            output_channel = make_divisible(c * width_mult)
            exp_channel = make_divisible(exp * width_mult)
            self.features.append(ODEBottleNeck(input_channel, output_channel, k, s, exp_channel, se, nl))
            input_channel = output_channel

        # building mobile BottleNeck blocks
        for k, exp, c, se, nl, s in mobile_setting2:
            output_channel = make_divisible(c * width_mult)
            exp_channel = make_divisible(exp * width_mult)
            self.features.append(MobileBottleneck(input_channel, output_channel, k, s, exp_channel, se, nl))
            input_channel = output_channel

        # building last several layers
        last_conv = make_divisible(576 * width_mult)
        self.features.append(conv_1x1_bn(input_channel, last_conv, nlin_layer=Hswish))
        # self.features.append(SEModule(last_conv))  # refer to paper Table2, but I think this is a mistake
        self.features.append(nn.AdaptiveAvgPool2d(1))
        self.features.append(nn.Conv2d(last_conv, last_channel, 1, 1, 0))
        self.features.append(Hswish(inplace=True))
        
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),    # refer to paper section 6
            nn.Linear(last_channel, n_class),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)