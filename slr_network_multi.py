import pdb
import copy
import utils
import torch
import types
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from modules.criterions import SeqKD
from modules import BiLSTMLayer, TemporalSlowFastFuse
import slowfast_modules.slowfast as slowfast
import importlib

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class NormLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(NormLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_dim, out_dim))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, x):
        outputs = torch.matmul(x, F.normalize(self.weight, dim=0))
        return outputs


class SLRModel(nn.Module):
    def __init__(
            self, num_classes, c2d_type, conv_type, load_pkl, slowfast_config, slowfast_args=None,
            use_bn=False, hidden_size=1024, gloss_dict=None, loss_weights=None,
            weight_norm=True, share_classifier=1
    ):
        super(SLRModel, self).__init__()
        self.decoder = None
        self.loss = dict()
        self.criterion_init()
        self.num_classes = num_classes
        self.loss_weights = loss_weights
        self.conv2d = getattr(slowfast, c2d_type)(slowfast_config=slowfast_config,  slowfast_args=slowfast_args,
                                                  load_pkl=load_pkl, multi=True)

        self.conv1d = TemporalSlowFastFuse(fast_input_size=256, slow_input_size=2048, hidden_size=hidden_size, conv_type=conv_type, use_bn=use_bn, num_classes=num_classes)
        self.decoder = utils.Decode(gloss_dict, num_classes, 'beam')

        self.temporal_model = nn.ModuleList([BiLSTMLayer(rnn_type='LSTM', input_size=hidden_size, hidden_size=hidden_size,
                                        num_layers=2, bidirectional=True) for i in range(3)])
        if weight_norm:
            self.classifier = nn.ModuleList([NormLinear(hidden_size, self.num_classes) for i in range(3)])
            self.conv1d.fc = nn.ModuleList([NormLinear(hidden_size, self.num_classes) for i in range(3)])
        else:
            self.classifier = nn.ModuleList([nn.Linear(hidden_size, self.num_classes) for i in range(3)])
            self.conv1d.fc = nn.ModuleList([nn.Linear(hidden_size, self.num_classes) for i in range(3)])
        if share_classifier == 1:
            self.conv1d.fc = self.classifier
        elif share_classifier == 2:
            classifier = self.classifier[0]
            self.classifier = nn.ModuleList([classifier for i in range(3)])
            self.conv1d.fc = nn.ModuleList([classifier for i in range(3)])
        #self.register_backward_hook(self.backward_hook)

    def backward_hook(self, module, grad_input, grad_output):
        for g in grad_input:
            g[g != g] = 0

    def masked_bn(self, inputs, len_x):
        def pad(tensor, length):
            return torch.cat([tensor, tensor.new(length - tensor.size(0), *tensor.size()[1:]).zero_()])

        x = torch.cat([inputs[len_x[0] * idx:len_x[0] * idx + lgt] for idx, lgt in enumerate(len_x)])
        x = self.conv2d(x)
        x = torch.cat([pad(x[sum(len_x[:idx]):sum(len_x[:idx + 1])], len_x[0])
                       for idx, lgt in enumerate(len_x)])
        return x

    def forward(self, x, len_x, label=None, label_lgt=None):
        if len(x.shape) == 5:
            framewise = self.conv2d(x.permute(0,2,1,3,4))
        else:
            # frame-wise features
            framewise = x
        
        conv1d_outputs = self.conv1d(framewise, len_x)
        lgt = conv1d_outputs['feat_len']
        
        outputs = []
        for i in range(len(conv1d_outputs['visual_feat'])):
            tm_outputs = self.temporal_model[i](conv1d_outputs['visual_feat'][i], lgt)
            outputs.append(self.classifier[i](tm_outputs['predictions']))

        pred = None if self.training \
            else self.decoder.decode(outputs[0], lgt, batch_first=False, probs=False)
        conv_pred = None if self.training \
            else self.decoder.decode(conv1d_outputs['conv_logits'][0], lgt, batch_first=False, probs=False)

        return {
            #"framewise_features": framewise,
            #"visual_features": conv1d_outputs['visual_feat'],
            "feat_len": lgt,
            "conv_logits": conv1d_outputs["conv_logits"],
            "sequence_logits": outputs,
            "conv_sents": conv_pred,
            "recognized_sents": pred,
        }

    def criterion_calculation(self, ret_dict, label, label_lgt):
        loss = 0
        for k, weight in self.loss_weights.items():
            if k == 'SeqCTC':
                loss += weight * self.loss['CTCLoss'](ret_dict["sequence_logits"][0].log_softmax(-1),
                                                      label.cpu().int(), ret_dict["feat_len"].cpu().int(),
                                                      label_lgt.cpu().int()).mean()
            elif k == 'Slow' or k == 'Fast':
                i = 1 if k == 'Slow' else 2
                loss += weight * self.loss_weights['SeqCTC'] * self.loss['CTCLoss'](ret_dict["sequence_logits"][i].log_softmax(-1),
                                                                                label.cpu().int(), ret_dict["feat_len"].cpu().int(),
                                                                                label_lgt.cpu().int()).mean()
                if 'ConvCTC' in self.loss_weights:
                    loss += weight * self.loss_weights['ConvCTC'] * self.loss['CTCLoss'](ret_dict["conv_logits"][i].log_softmax(-1),
                                                                                        label.cpu().int(), ret_dict["feat_len"].cpu().int(),
                                                                                        label_lgt.cpu().int()).mean()
                if 'Dist' in self.loss_weights:
                    # loss += weight * self.loss_weights['Dist'] * self.loss['distillation'](ret_dict["conv_intra_logits"][i],
                    #                                                                     ret_dict["sequence_logits"].detach(),
                    #                                                                     use_blank=False)
                    loss += weight * self.loss_weights['Dist'] * self.loss['distillation'](ret_dict["conv_logits"][i],
                                                                                        ret_dict["sequence_logits"][i].detach(),
                                                                                        use_blank=False)
            elif k == 'ConvCTC':
                loss += weight * self.loss['CTCLoss'](ret_dict["conv_logits"][0].log_softmax(-1),
                                                      label.cpu().int(), ret_dict["feat_len"].cpu().int(),
                                                      label_lgt.cpu().int()).mean()
            elif k == 'Dist':
                loss += weight * self.loss['distillation'](ret_dict["conv_logits"][0],
                                                           ret_dict["sequence_logits"][0].detach(),
                                                           use_blank=False)
        return loss

    def criterion_init(self):
        self.loss['CTCLoss'] = torch.nn.CTCLoss(reduction='none', zero_infinity=False)
        self.loss['distillation'] = SeqKD(T=8)
        return self.loss
