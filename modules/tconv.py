import pdb
import copy
import torch
import collections
import torch.nn as nn
import torch.nn.functional as F

class TemporalConv(nn.Module):
    def __init__(self, input_size, hidden_size, conv_type=2, use_bn=False, num_classes=-1):
        super(TemporalConv, self).__init__()
        self.use_bn = use_bn
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.conv_type = conv_type

        if self.conv_type == 0:
            self.kernel_size = ['K3']
        elif self.conv_type == 1:
            self.kernel_size = ['K5', "P2"]
        elif self.conv_type == 2:
            self.kernel_size = ['K5', "P2", 'K5', "P2"]
        elif self.conv_type == 3:
            self.kernel_size = ['K5', 'K5', "P2"]
        elif self.conv_type == 4:
            self.kernel_size = ['K5', 'K5']
        elif self.conv_type == 5:
            self.kernel_size = ['K5', "P2", 'K5']
        elif self.conv_type == 6:
            self.kernel_size = ["P2", 'K5', 'K5']
        elif self.conv_type == 7:
            self.kernel_size = ["P2", 'K5', "P2", 'K5']
        elif self.conv_type == 8:
            self.kernel_size = ["P2", "P2", 'K5', 'K5']
        elif self.conv_type == 9:
            self.kernel_size = ["K5", "K5", "P2"]
        elif self.conv_type == 10:
            self.kernel_size = ["K5", "K5"]

        modules = []
        for layer_idx, ks in enumerate(self.kernel_size):
            input_sz = self.input_size if layer_idx == 0 or self.conv_type == 6 and layer_idx == 1 or self.conv_type == 7 and layer_idx == 1 or self.conv_type == 8 and layer_idx == 2 else self.hidden_size
            if ks[0] == 'P':
                modules.append(nn.MaxPool1d(kernel_size=int(ks[1]), ceil_mode=False))
            elif ks[0] == 'K':
                modules.append(
                    nn.Conv1d(input_sz, self.hidden_size, kernel_size=int(ks[1]), stride=1, padding=0)
                )
                modules.append(nn.BatchNorm1d(self.hidden_size))
                modules.append(nn.ReLU(inplace=True))
        self.temporal_conv = nn.Sequential(*modules)

        if self.num_classes != -1:
            self.fc = nn.Linear(self.hidden_size, self.num_classes)

    def update_lgt(self, lgt):
        feat_len = copy.deepcopy(lgt)
        for ks in self.kernel_size:
            if ks[0] == 'P':
                feat_len = torch.div(feat_len, 2)
            else:
                feat_len -= int(ks[1]) - 1
                #pass
        return feat_len

    def forward(self, frame_feat, lgt):
        visual_feat = self.temporal_conv(frame_feat)
        lgt = self.update_lgt(lgt)
        logits = None if self.num_classes == -1 \
            else self.fc(visual_feat.transpose(1, 2)).transpose(1, 2)
        return {
            "visual_feat": visual_feat.permute(2, 0, 1),
            "conv_logits": logits.permute(2, 0, 1),
            "feat_len": lgt.cpu(),
        }


class TemporalSlowFastFuse(nn.Module):
    def __init__(self, fast_input_size, slow_input_size, hidden_size, conv_type=2, use_bn=False, num_classes=-1):
        super(TemporalSlowFastFuse, self).__init__()
        self.use_bn = use_bn
        self.fast_input_size = fast_input_size
        self.slow_input_size = slow_input_size
        self.main_input_size = fast_input_size + slow_input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.conv_type = conv_type

        if self.conv_type == 0:
            self.kernel_size = ['K3']
        elif self.conv_type == 1:
            self.kernel_size = ['K5', "P2"]
        elif self.conv_type == 2:
            self.kernel_size = ['K5', "P2", 'K5', "P2"]
        elif self.conv_type == 3:
            self.kernel_size = ['K5', 'K5', "P2"]
        elif self.conv_type == 4:
            self.kernel_size = ['K5', 'K5']
        elif self.conv_type == 5:
            self.kernel_size = ['K5', "P2", 'K5']
        elif self.conv_type == 6:
            self.kernel_size = ["P2", 'K5', 'K5']
        elif self.conv_type == 7:
            self.kernel_size = ["P2", 'K5', "P2", 'K5']
        elif self.conv_type == 8:
            self.kernel_size = ["P2", "P2", 'K5', 'K5']
        elif self.conv_type == 9:
            self.kernel_size = ["K5", "K5", "P2"]
        elif self.conv_type == 10:
            self.kernel_size = ["K5", "K5"]

        fast_modules = []
        slow_modules = []
        main_modules = []
        for layer_idx, ks in enumerate(self.kernel_size):
            fast_input_sz = self.fast_input_size if layer_idx == 0 or self.conv_type == 6 and layer_idx == 1 or self.conv_type == 7 and layer_idx == 1 or self.conv_type == 8 and layer_idx == 2 else self.hidden_size
            slow_input_sz = self.slow_input_size if layer_idx == 0 or self.conv_type == 6 and layer_idx == 1 or self.conv_type == 7 and layer_idx == 1 or self.conv_type == 8 and layer_idx == 2 else self.hidden_size
            main_input_sz = self.main_input_size if layer_idx == 0 or self.conv_type == 6 and layer_idx == 1 or self.conv_type == 7 and layer_idx == 1 or self.conv_type == 8 and layer_idx == 2 else self.hidden_size
            if ks[0] == 'P':
                fast_modules.append(nn.MaxPool1d(kernel_size=int(ks[1]), ceil_mode=False))
                slow_modules.append(nn.MaxPool1d(kernel_size=int(ks[1]), ceil_mode=False))
                main_modules.append(nn.MaxPool1d(kernel_size=int(ks[1]), ceil_mode=False))
            elif ks[0] == 'K':
                fast_modules.append(
                    nn.Conv1d(fast_input_sz, self.hidden_size, kernel_size=int(ks[1]), stride=1, padding=0)
                )
                fast_modules.append(nn.BatchNorm1d(self.hidden_size))
                fast_modules.append(nn.ReLU(inplace=True))
                slow_modules.append(
                    nn.Conv1d(slow_input_sz, self.hidden_size, kernel_size=int(ks[1]), stride=1, padding=0)
                )
                slow_modules.append(nn.BatchNorm1d(self.hidden_size))
                slow_modules.append(nn.ReLU(inplace=True))
                main_modules.append(
                    nn.Conv1d(main_input_sz, self.hidden_size, kernel_size=int(ks[1]), stride=1, padding=0)
                )
                main_modules.append(nn.BatchNorm1d(self.hidden_size))
                main_modules.append(nn.ReLU(inplace=True))
        self.fast_temporal_conv = nn.Sequential(*fast_modules)
        self.slow_temporal_conv = nn.Sequential(*slow_modules)
        self.main_temporal_conv = nn.Sequential(*main_modules)

        if self.num_classes != -1:
            self.fc = nn.ModuleList([nn.Linear(self.hidden_size, self.num_classes) for i in range(3)])
    
    def update_lgt(self, lgt):
        feat_len = copy.deepcopy(lgt)
        for ks in self.kernel_size:
            if ks[0] == 'P':
                feat_len = torch.div(feat_len, 2)
            else:
                feat_len -= int(ks[1]) - 1
                #pass
        return feat_len

    def forward(self, frame_feat, lgt):
        visual_feat = [self.main_temporal_conv(frame_feat)]
        if self.training:
            slow_path = frame_feat[:,:self.slow_input_size]
            fast_path = frame_feat[:,self.slow_input_size:]
            slow_feat = self.slow_temporal_conv(slow_path)
            fast_feat = self.fast_temporal_conv(fast_path)
            visual_feat.extend([slow_feat, fast_feat])
        num_paths = len(visual_feat)
        lgt = self.update_lgt(lgt)
        logits = None if self.num_classes == -1 \
                else [self.fc[i](visual_feat[i].transpose(1, 2)).transpose(1, 2) for i in range(num_paths)]
        return {
            "visual_feat": [visual_feat[i].permute(2, 0, 1) for i in range(num_paths)],
            "conv_logits": [logits[i].permute(2, 0, 1) for i in range(num_paths)],
            "feat_len": lgt.cpu(),
        }


class SlowFastFuse(nn.Module):

    def __init__(self, fast_input_size, slow_input_size, hidden_size, num_classes):
        super(SlowFastFuse, self).__init__()
        self.fast_input_size = fast_input_size
        self.slow_input_size = slow_input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.kernel_size = ['P4']
        self.fast_conv = nn.Conv1d(fast_input_size, hidden_size // 2, kernel_size=1, stride=1, padding=0)
        self.slow_conv = nn.Conv1d(slow_input_size, hidden_size // 2, kernel_size=1, stride=1, padding=0)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.fc_intra = nn.ModuleList([nn.Linear(hidden_size // 2, num_classes) for i in range(2)])
    
    def forward(self, frame_feat, lgt):
        slow_path = frame_feat[0]
        fast_path = frame_feat[1]
        slow_feat = self.slow_conv(slow_path)
        fast_feat = self.fast_conv(fast_path)
        intra_feat = [slow_feat, fast_feat]
        inter_feat = torch.cat(intra_feat, dim=1)
        inter_logits = None if self.num_classes == -1 \
            else self.fc(inter_feat.transpose(1, 2)).transpose(1, 2)
        intra_logits = None if self.num_classes == -1 \
            else [self.fc_intra[i](intra_feat[i].transpose(1, 2)).transpose(1, 2) for i in range(len(intra_feat))]
        return {
            "inter_feat": inter_feat.permute(2, 0, 1),
            "intra_feat": [intra_feat[i].permute(2, 0, 1) for i in range(2)],
            "conv_inter_logits": inter_logits.permute(2, 0, 1),
            "conv_intra_logits": [intra_logits[i].permute(2, 0, 1) for i in range(2)],
            "feat_len": lgt.cpu(),
        }


class FastConv(nn.Module):
    def __init__(self, fast_input_size, slow_input_size, hidden_size, num_classes):
        super(FastConv, self).__init__()
        self.fast_input_size = fast_input_size
        self.slow_input_size = slow_input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.kernel_size = ['K5', 'P2', 'K5', 'P2']
        self.slow_conv = nn.Conv1d(slow_input_size, hidden_size // 2, kernel_size=1, stride=1, padding=0)
        fast_modules = []
        for layer_idx, ks in enumerate(self.kernel_size):
            fast_input_sz = self.fast_input_size if layer_idx == 0 else hidden_size // 2
            
            if ks[0] == 'P':
                fast_modules.append(nn.MaxPool1d(kernel_size=int(ks[1]), ceil_mode=False))
            elif ks[0] == 'K':
                fast_modules.append(
                    nn.Conv1d(fast_input_sz, hidden_size // 2, kernel_size=int(ks[1]), stride=1, padding='same')
                )
                fast_modules.append(nn.BatchNorm1d(hidden_size // 2))
                fast_modules.append(nn.ReLU(inplace=True))
        
        self.fast_conv = nn.Sequential(*fast_modules)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.fc_intra = nn.ModuleList([nn.Linear(hidden_size // 2, num_classes) for i in range(2)])
    
    def update_lgt(self, lgt):
        feat_len = copy.deepcopy(lgt)
        for ks in self.kernel_size:
            if ks[0] == 'P':
                feat_len = torch.div(feat_len, 2)
            else:
                feat_len -= int(ks[1]) - 1
                #pass
        return feat_len
    
    def forward(self, frame_feat, lgt):
        slow_path = frame_feat[0]
        fast_path = frame_feat[1]
        slow_feat = self.slow_conv(slow_path)
        fast_feat = self.fast_conv(fast_path)
        intra_feat = [slow_feat, fast_feat]
        inter_feat = torch.cat(intra_feat, dim=1)
        inter_logits = None if self.num_classes == -1 \
            else self.fc(inter_feat.transpose(1, 2)).transpose(1, 2)
        intra_logits = None if self.num_classes == -1 \
            else [self.fc_intra[i](intra_feat[i].transpose(1, 2)).transpose(1, 2) for i in range(len(intra_feat))]
        return {
            "inter_feat": inter_feat.permute(2, 0, 1),
            "intra_feat": [intra_feat[i].permute(2, 0, 1) for i in range(2)],
            "conv_inter_logits": inter_logits.permute(2, 0, 1),
            "conv_intra_logits": [intra_logits[i].permute(2, 0, 1) for i in range(2)],
            "feat_len": (lgt // 4).cpu(),
        }