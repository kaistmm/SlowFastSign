import copy
import math
import numpy as np
import os
import pickle
from collections import OrderedDict
import torch
import re

def get_name_convert_func():
    """
    Get the function to convert Caffe2 layer names to PyTorch layer names.
    Returns:
        (func): function to convert parameter name from Caffe2 format to PyTorch
        format.
    """
    pairs = [
        # ------------------------------------------------------------
        # 'nonlocal_conv3_1_theta_w' -> 's3.pathway0_nonlocal3.conv_g.weight'
        [
            r"^nonlocal_conv([0-9]+)_([0-9]+)_(.*)",
            r"s\1.pathway0_nonlocal\2_\3",
        ],
        # 'theta' -> 'conv_theta'
        [r"^(.*)_nonlocal([0-9]+)_(theta)(.*)", r"\1_nonlocal\2.conv_\3\4"],
        # 'g' -> 'conv_g'
        [r"^(.*)_nonlocal([0-9]+)_(g)(.*)", r"\1_nonlocal\2.conv_\3\4"],
        # 'phi' -> 'conv_phi'
        [r"^(.*)_nonlocal([0-9]+)_(phi)(.*)", r"\1_nonlocal\2.conv_\3\4"],
        # 'out' -> 'conv_out'
        [r"^(.*)_nonlocal([0-9]+)_(out)(.*)", r"\1_nonlocal\2.conv_\3\4"],
        # 'nonlocal_conv4_5_bn_s' -> 's4.pathway0_nonlocal3.bn.weight'
        [r"^(.*)_nonlocal([0-9]+)_(bn)_(.*)", r"\1_nonlocal\2.\3.\4"],
        # ------------------------------------------------------------
        # 't_pool1_subsample_bn' -> 's1_fuse.conv_f2s.bn.running_mean'
        [r"^t_pool1_subsample_bn_(.*)", r"s1_fuse.bn.\1"],
        # 't_pool1_subsample' -> 's1_fuse.conv_f2s'
        [r"^t_pool1_subsample_(.*)", r"s1_fuse.conv_f2s.\1"],
        # 't_res4_5_branch2c_bn_subsample_bn_rm' -> 's4_fuse.conv_f2s.bias'
        [
            r"^t_res([0-9]+)_([0-9]+)_branch2c_bn_subsample_bn_(.*)",
            r"s\1_fuse.bn.\3",
        ],
        # 't_pool1_subsample' -> 's1_fuse.conv_f2s'
        [
            r"^t_res([0-9]+)_([0-9]+)_branch2c_bn_subsample_(.*)",
            r"s\1_fuse.conv_f2s.\3",
        ],
        # ------------------------------------------------------------
        # 'res4_4_branch_2c_bn_b' -> 's4.pathway0_res4.branch2.c_bn_b'
        [
            r"^res([0-9]+)_([0-9]+)_branch([0-9]+)([a-z])_(.*)",
            r"s\1.pathway0_res\2.branch\3.\4_\5",
        ],
        # 'res_conv1_bn_' -> 's1.pathway0_stem.bn.'
        [r"^res_conv1_bn_(.*)", r"s1.pathway0_stem.bn.\1"],
        # 'conv1_xy_w_momentum' -> 's1.pathway0_stem.conv_xy.'
        [r"^conv1_xy(.*)", r"s1.pathway0_stem.conv_xy\1"],
        # 'conv1_w_momentum' -> 's1.pathway0_stem.conv.'
        [r"^conv1_(.*)", r"s1.pathway0_stem.conv.\1"],
        # 'res4_0_branch1_w' -> 'S4.pathway0_res0.branch1.weight'
        [
            r"^res([0-9]+)_([0-9]+)_branch([0-9]+)_(.*)",
            r"s\1.pathway0_res\2.branch\3_\4",
        ],
        # 'res_conv1_' -> 's1.pathway0_stem.conv.'
        [r"^res_conv1_(.*)", r"s1.pathway0_stem.conv.\1"],
        # ------------------------------------------------------------
        # 'res4_4_branch_2c_bn_b' -> 's4.pathway0_res4.branch2.c_bn_b'
        [
            r"^t_res([0-9]+)_([0-9]+)_branch([0-9]+)([a-z])_(.*)",
            r"s\1.pathway1_res\2.branch\3.\4_\5",
        ],
        # 'res_conv1_bn_' -> 's1.pathway0_stem.bn.'
        [r"^t_res_conv1_bn_(.*)", r"s1.pathway1_stem.bn.\1"],
        # 'conv1_w_momentum' -> 's1.pathway0_stem.conv.'
        [r"^t_conv1_(.*)", r"s1.pathway1_stem.conv.\1"],
        # 'res4_0_branch1_w' -> 'S4.pathway0_res0.branch1.weight'
        [
            r"^t_res([0-9]+)_([0-9]+)_branch([0-9]+)_(.*)",
            r"s\1.pathway1_res\2.branch\3_\4",
        ],
        # 'res_conv1_' -> 's1.pathway0_stem.conv.'
        [r"^t_res_conv1_(.*)", r"s1.pathway1_stem.conv.\1"],
        # ------------------------------------------------------------
        # pred_ -> head.projection.
        [r"pred_(.*)", r"head.projection.\1"],
        # '.b_bn_fc' -> '.se.fc'
        [r"(.*)b_bn_fc(.*)", r"\1se.fc\2"],
        # conv_5 -> head.conv_5.
        [r"conv_5(.*)", r"head.conv_5\1"],
        # conv_5 -> head.conv_5.
        [r"lin_5(.*)", r"head.lin_5\1"],
        # '.bn_b' -> '.weight'
        [r"(.*)bn.b\Z", r"\1bn.bias"],
        # '.bn_s' -> '.weight'
        [r"(.*)bn.s\Z", r"\1bn.weight"],
        # '_bn_rm' -> '.running_mean'
        [r"(.*)bn.rm\Z", r"\1bn.running_mean"],
        # '_bn_riv' -> '.running_var'
        [r"(.*)bn.riv\Z", r"\1bn.running_var"],
        # '_b' -> '.bias'
        [r"(.*)[\._]b\Z", r"\1.bias"],
        # '_w' -> '.weight'
        [r"(.*)[\._]w\Z", r"\1.weight"],
    ]

    def convert_caffe2_name_to_pytorch(caffe2_layer_name):
        """
        Convert the caffe2_layer_name to pytorch format by apply the list of
        regular expressions.
        Args:
            caffe2_layer_name (str): caffe2 layer name.
        Returns:
            (str): pytorch layer name.
        """
        for source, dest in pairs:
            caffe2_layer_name = re.sub(source, dest, caffe2_layer_name)
        return caffe2_layer_name

    return convert_caffe2_name_to_pytorch


def c2_normal_to_sub_bn(key, model_keys):
    """
    Convert BN parameters to Sub-BN parameters if model contains Sub-BNs.
    Args:
        key (OrderedDict): source dict of parameters.
        mdoel_key (OrderedDict): target dict of parameters.
    Returns:
        new_sd (OrderedDict): converted dict of parameters.
    """
    if "bn.running_" in key:
        if key in model_keys:
            return key

        new_key = key.replace("bn.running_", "bn.split_bn.running_")
        if new_key in model_keys:
            return new_key
        else:
            return key
    else:
        return key


def load_checkpoint(
    path_to_checkpoint,
    model
):
    """
    Load the checkpoint from the given file. If inflation is True, inflate the
    2D Conv weights from the checkpoint to 3D Conv.
    Args:
        path_to_checkpoint (string): path to the checkpoint to load.
        model (model): model to load the weights from the checkpoint.
        data_parallel (bool): if true, model is wrapped by
        torch.nn.parallel.DistributedDataParallel.
        optimizer (optim): optimizer to load the historical state.
        scaler (GradScaler): GradScaler to load the mixed precision scale.
        inflation (bool): if True, inflate the weights from the checkpoint.
        convert_from_caffe2 (bool): if True, load the model from caffe2 and
            convert it to pytorch.
        epoch_reset (bool): if True, reset #train iterations from the checkpoint.
        clear_name_pattern (string): if given, this (sub)string will be cleared
            from a layer name if it can be matched.
    Returns:
        (int): the number of training epoch of the checkpoint.
    """
    ms = model
    arch = model.cfg.MODEL.ARCH
    with open(path_to_checkpoint, "rb") as f:
        caffe2_checkpoint = pickle.load(f, encoding="latin1")
    state_dict = OrderedDict()
    name_convert_func = get_name_convert_func()
    for key in caffe2_checkpoint["blobs"].keys():
        converted_key = name_convert_func(key)
        if arch == 'slow':
            if 'pathway1' in converted_key:
                continue
        if arch == 'fast':
            if 'pathway0' in converted_key:
                continue
            elif 'pathway1' in converted_key:
                converted_key = converted_key.replace('pathway1', 'pathway0')
        converted_key = c2_normal_to_sub_bn(converted_key, ms.state_dict())
        if converted_key in ms.state_dict():
            c2_blob_shape = caffe2_checkpoint["blobs"][key].shape
            model_blob_shape = ms.state_dict()[converted_key].shape

            # expand shape dims if they differ (eg for converting linear to conv params)
            if len(c2_blob_shape) < len(model_blob_shape):
                c2_blob_shape += (1,) * (
                    len(model_blob_shape) - len(c2_blob_shape)
                )
                caffe2_checkpoint["blobs"][key] = np.reshape(
                    caffe2_checkpoint["blobs"][key], c2_blob_shape
                )
            if len(c2_blob_shape) == 5 and \
               c2_blob_shape[2] > 1 and \
               model_blob_shape[2] == 1:
               caffe2_checkpoint["blobs"][key] = caffe2_checkpoint["blobs"][key][:,:,:1]
               c2_blob_shape_origin = c2_blob_shape
               c2_blob_shape = list(c2_blob_shape)
               c2_blob_shape[2] = 1
               c2_blob_shape = tuple(c2_blob_shape)
               print(
                    "{}: {} => {}: {}".format(
                        key,
                        c2_blob_shape_origin,
                        converted_key,
                        tuple(model_blob_shape),
                    )
                )

            # Load BN stats to Sub-BN.
            if (
                len(model_blob_shape) == 1
                and len(c2_blob_shape) == 1
                and model_blob_shape[0] > c2_blob_shape[0]
                and model_blob_shape[0] % c2_blob_shape[0] == 0
            ):
                caffe2_checkpoint["blobs"][key] = np.concatenate(
                    [caffe2_checkpoint["blobs"][key]]
                    * (model_blob_shape[0] // c2_blob_shape[0])
                )
                c2_blob_shape = caffe2_checkpoint["blobs"][key].shape

            if c2_blob_shape == tuple(model_blob_shape):
                state_dict[converted_key] = torch.tensor(
                    caffe2_checkpoint["blobs"][key]
                ).clone()
                # print(
                #     "{}: {} => {}: {}".format(
                #         key,
                #         c2_blob_shape,
                #         converted_key,
                #         tuple(model_blob_shape),
                #     )
                # )
            elif c2_blob_shape != tuple(model_blob_shape):
                if c2_blob_shape[1] < model_blob_shape[1]:
                    state_dict[converted_key] = torch.cat(
                        [torch.tensor(caffe2_checkpoint["blobs"][key]).clone(),
                        ms.state_dict()[converted_key][:,c2_blob_shape[1]:]],
                        dim=1
                    )
                    print(
                        "{}: {} => {}: {}".format(
                            key,
                            c2_blob_shape,
                            converted_key,
                            tuple(model_blob_shape),
                        )
                    )
                elif c2_blob_shape[1] > model_blob_shape[1]:
                    state_dict[converted_key] = torch.tensor(caffe2_checkpoint["blobs"][key]).clone()[:,:model_blob_shape[1]]
                    print(
                        "{}: {} => {}: {}".format(
                            key,
                            c2_blob_shape,
                            converted_key,
                            tuple(model_blob_shape),
                        )
                    )
            else:
                print(
                    "!! {}: {} does not match {}: {}".format(
                        key,
                        c2_blob_shape,
                        converted_key,
                        tuple(model_blob_shape),
                    )
                )
        else:
            if not any(
                prefix in key for prefix in ["momentum", "lr", "model_iter", "nonlocal_conv4"]
            ):
                print(
                    "!! {}: can not be converted, got {}".format(
                        key, converted_key
                    )
                )
    diff = set(ms.state_dict()) - set(state_dict)
    diff = {d for d in diff if "num_batches_tracked" not in d}
    if len(diff) > 0:
        print("Not loaded {}".format(diff))
    ms.load_state_dict(state_dict, strict=False)