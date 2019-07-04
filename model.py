import numpy as np
import time
import torch
import torch.nn as nn
import sys
import torch.functional as F


class CPModule(nn.Module):

    def __init__(self, t, feature_num, k=3):
        super(CPModule, self).__init__()
        self.t = t
        self.k = k
        self.feature_num = feature_num
        self.MLP = torch.nn.Sequential(
            torch.nn.Linear(feature_num * 2 + 3, 16),
            torch.nn.Linear(16, 8),
            torch.nn.Linear(8, feature_num)
        ).cuda()

    def forward(self, input):
        size = input.size()
        bs = size[0]
        t = size[1]
        h = size[3]
        w = size[4]
        hw = h * w
        thw = t * hw
        feature_num = size[2]
        # (bs, segments, channels, height, width) -> (bs, segments, height, width, channels)
        input = input.permute([0, 1, 3, 4, 2])
        # -> (bs, segments*height*width, channels)
        input = input.contiguous().view((-1, thw, feature_num))
        # Compute semantic similarity (bs, thw, thw)
        # https://blog.csdn.net/frankzd/article/details/80251042
        # Square
        square1 = torch.sum(input ** 2, dim=2).view((-1, thw, 1)).expand(-1, -1, thw)
        square2 = torch.transpose(square1, 1, 2)
        # Negative l2 distance
        # There might appear `nan` in the similarity matrix during torch.sqrt(),
        # it will result in getting super enormous index numbers while performing `torch.topK()`,
        # which finally cause device-side assert triggered
        # `index >= -sizes[i] && index < sizes[i] && "index out of bounds"`
        # Hense, perform torch.clamp() to constrain value of each element of similarity in [MAX, MIN] before perform torch.sqrt()
        similarity_unsqrt = torch.add(torch.add(square1, square2),
                                      -2 * torch.matmul(input, torch.transpose(input, 1, 2)))
        # Do clamp()
        similarity = torch.sqrt(torch.clamp(similarity_unsqrt, 1e-5, 100)) * -1
        # print(similarity_sqrted)
        # Set the values of the elements in the similarity matrix diagonal block matrices of shape HW × HW to be −∞.
        x = torch.arange(thw).view(-1, 1).expand(-1, hw).contiguous().view(-1)
        y = torch.arange(thw).view(-1, 1, hw).expand(-1, hw, -1).contiguous().view(-1)
        similarity[:, x, y] = -1000
        # Select features and locations of K correspondences by similarity
        corres_idx = torch.topk(similarity, self.k, dim=2)[1]
        corres_idx_viewed = corres_idx.view(-1)
        # Get features of correspondences grouping by respective input features.
        bs_dim_idx = torch.arange(bs).view(-1, 1).expand(-1, self.k * thw).contiguous().view(-1)
        correspondences = input[bs_dim_idx, corres_idx_viewed, :].view(bs, thw, self.k, feature_num)
        # Embedding
        # Construct the inputs of MLP
        # The format of each input is [f_input, f_correspondence, normalized_t_diff, normalized_h_diff, normalized_w_diff]
        input_expanded = input.view([bs, thw, 1, feature_num]).expand(-1, -1, self.k, -1)
        MLP_features_inputs = torch.cat([input_expanded, correspondences], dim=3)
        corres_t_pos = (corres_idx_viewed // hw).float() / t
        corres_h_pos = (corres_idx_viewed % hw // w).float()
        corres_w_pos = (corres_idx_viewed % hw % w).float()
        corres_positions = torch.cat([corres_t_pos.view(-1, 1), corres_h_pos.view(-1, 1), corres_w_pos.view(-1, 1)],
                                     dim=1).view(bs, thw, self.k, 3).cuda()
        input_idx_viewed = torch.arange(thw).view(-1, h * w * t).expand(bs, -1).contiguous().view(-1, 1).expand(-1, self.k).contiguous().view(-1)
        input_t_pos = (input_idx_viewed // h*w).float() / t
        input_h_pos = (input_idx_viewed % h*w // w).float()
        input_w_pos = (input_idx_viewed % h*w % w).float()
        input_positions = torch.cat([input_t_pos.view(-1, 1), input_h_pos.view(-1, 1), input_w_pos.view(-1, 1)], dim=1).view(bs, t * h * w, self.k, 3) * -1
        input_positions = input_positions.cuda()
        displacements = corres_positions - input_positions
        MLP_inputs = torch.cat([MLP_features_inputs.cuda(), displacements.cuda()], dim=3).view(-1, feature_num * 2 + 3).cuda()
        # Get MLP outputs
        MLP_outputs = self.MLP(MLP_inputs)
        MLP_outputs = MLP_outputs.view(bs, thw, self.k, feature_num)
        # Perform max
        max_MLP = torch.max(MLP_outputs, dim=2)[0].view(bs, t, h, w, feature_num)
        # Transfer to original input shape
        embedded = max_MLP.permute(0, 1, 4, 2, 3).view(bs, t, feature_num, h, w)
        return embedded


class CPNet(nn.Module):

    def __init__(self, input_channel, t, class_num):
        super(CPNet, self).__init__()
        self.input_channel = input_channel
        self.conv1 = nn.Conv2d(self.input_channel, 16, 3)
        self.t = t
        self.cp = CPModule(self.t, 16)
        self.conv2 = nn.Conv2d(16, 16, 3).cuda()
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(t * 14 * 14 * 16, class_num).cuda()

    def forward(self, input):
        input = input.view((-1, self.input_channel) + input.size()[-2:])  # (bs*segments, channels, height, width)
        conv1 = self.conv1(input)
        conv1 = conv1.view((-1, self.t) + conv1.size()[1:])  # (bs, segments, channels, height, width)
        cp1 = self.cp(conv1)
        cp1 = cp1.view(-1, 16, 30, 30).cuda()
        conv2 = self.conv2(cp1)
        pool = self.pool(conv2)
        pool = pool.view(-1, 4 * 16 * 14 * 14)
        fc = self.fc(pool.view(-1, self.t * 14 * 14 * 16))
        return fc


if __name__ == '__main__':
    cpnet = CPNet(3, 4, 4).cuda()
    input = torch.rand(16, 4, 3, 32, 32).cuda()
    out = cpnet(input)
    print(out)
