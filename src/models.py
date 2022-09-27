
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.elements import create_conv_if, bddi_if


class BITnet6(nn.Module):
    def __init__(self, num_classes=10, img_size=32, feature_scale=6, with_bn=False):
        super(BITnet6, self).__init__()
        self.num_classes = num_classes
        feature_scale = feature_scale
        out_feature = 4
        self.extra_max = False

        if img_size == 64:
            self.extra_max = True

        self.begin = create_conv_if(3, int(64*feature_scale), 3, 1, 1, size=32)
        self.layer1 = create_conv_if(int(64*feature_scale), int(64*feature_scale), 3, 1, 1, size=32)
        self.layer2 = create_conv_if(int(64*feature_scale), int(128*feature_scale), 3, 1, 1, size=16)
        self.layer3 = create_conv_if(int(128*feature_scale), int(256*feature_scale), 3, 1, 1, size=8)
        self.layer4 = create_conv_if(int(256*feature_scale), int(512*feature_scale), 3, 1, 1, size=4)
        self.mlp1 = bddi_if(nn.Linear(int(512*out_feature*out_feature*feature_scale), num_classes))

        self.flatten = nn.Flatten()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x, state=0, steps=3, select_first=True, normalize=True):
        if type(state)!=list:
            s0, s1, s2, s3, s4, s5 = [0,]*6
        else:
            s0, s1, s2, s3, s4, s5 = state

        output = 0
        steps_count = 0
        for i_iter in range(steps):
            o0, s0 = self.begin(x, s0)
            if self.extra_max:
                o0 = self.maxpool(o0)

            o1, s1 = self.layer1(o0, s1)
            o1 = self.maxpool(o1)
            o2, s2 = self.layer2(o1, s2)
            o2 = self.maxpool(o2)
            o3, s3 = self.layer3(o2, s3)
            o4, s4 = self.layer4(o3, s4)
            o4 = self.maxpool(o4)

            o4_flat = self.flatten(o4)
            o5, s5 = self.mlp1(o4_flat, s5)

            if i_iter==0 and not select_first:
                continue
            steps_count +=1
            output += o5
        if normalize:
            output = output / (steps_count)
        return output, [s0, s1, s2, s3, s4, s5]


class BITnet6_v101(nn.Module):
    def __init__(self, num_classes=10, img_size=32, feature_scale=6, with_bn=False):
        super(BITnet6_v101, self).__init__()
        self.num_classes = num_classes
        feature_scale = feature_scale
        out_feature = 4
        self.extra_max = False

        if img_size == 64:
            self.extra_max = True

        self.begin = create_conv_if(3, int(64*feature_scale), 3, 1, 1, size=32)
        self.layer1 = create_conv_if(int(64*feature_scale), int(64*feature_scale), 3, 1, 1, size=32)
        self.layer2 = create_conv_if(int(64*feature_scale), int(128*feature_scale), 3, 1, 1, size=16)
        self.layer3 = create_conv_if(int(128*feature_scale), int(256*feature_scale), 3, 1, 1, size=8)
        self.layer4 = create_conv_if(int(256*feature_scale), int(512*feature_scale), 3, 1, 1, size=4)
        self.mlp1 = bddi_if(nn.Linear(int(512*out_feature*out_feature*feature_scale), num_classes))

        self.flatten = nn.Flatten()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x, state=0, steps=3, select_first=True, normalize=True):
        if type(state)!=list:
            s0, s1, s2, s3, s4, s5 = [0,]*6
        else:
            s0, s1, s2, s3, s4, s5 = state

        output = 0
        steps_count = 0
        for i_iter in range(steps):
            o0, s0 = self.begin(x, s0)
            if self.extra_max:
                o0 = self.maxpool(o0)

            o1, s1 = self.layer1(o0, s1)
            o1 = self.maxpool(o1)
            o2, s2 = self.layer2(o1, s2)
            o3, s3 = self.layer3(o2, s3)
            o3 = self.maxpool(o3)
            o4, s4 = self.layer4(o3, s4)
            o4 = self.maxpool(o4)

            o4_flat = self.flatten(o4)
            o5, s5 = self.mlp1(o4_flat, s5)

            if i_iter==0 and not select_first:
                continue
            steps_count +=1
            output += o5
        if normalize:
            output = output / (steps_count)
        return output, [s0, s1, s2, s3, s4, s5]
