# Copyright 2020 InterDigital Communications, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn


def find_named_module(module, query):
    """Helper function to find a named module. Returns a `nn.Module` or `None`

    Args:
        module (nn.Module): the root module
        query (str): the module name to find

    Returns:
        nn.Module or None
    """

    return next((m for n, m in module.named_modules() if n == query), None)


def find_named_buffer(module, query):
    """Helper function to find a named buffer. Returns a `torch.Tensor` or `None`

    Args:
        module (nn.Module): the root module
        query (str): the buffer name to find

    Returns:
        torch.Tensor or None
    """
    return next((b for n, b in module.named_buffers() if n == query), None)


def _update_registered_buffer(
    module,
    buffer_name,
    state_dict_key,
    state_dict,
    policy="resize_if_empty",
    dtype=torch.int,
):
    new_size = state_dict[state_dict_key].size()
    registered_buf = find_named_buffer(module, buffer_name)

    if policy in ("resize_if_empty", "resize"):
        if registered_buf is None:
            raise RuntimeError(f'buffer "{buffer_name}" was not registered')

        if policy == "resize" or registered_buf.numel() == 0:
            registered_buf.resize_(new_size)

    elif policy == "register":
        if registered_buf is not None:
            raise RuntimeError(f'buffer "{buffer_name}" was already registered')

        module.register_buffer(buffer_name, torch.empty(new_size, dtype=dtype).fill_(0))

    else:
        raise ValueError(f'Invalid policy "{policy}"')


def update_registered_buffers(
    module,
    module_name,
    buffer_names,
    state_dict,
    policy="resize_if_empty",
    dtype=torch.int,
):
    """Update the registered buffers in a module according to the tensors sized
    in a state_dict.

    (There's no way in torch to directly load a buffer with a dynamic size)

    Args:
        module (nn.Module): the module
        module_name (str): module name in the state dict
        buffer_names (list(str)): list of the buffer names to resize in the module
        state_dict (dict): the state dict
        policy (str): Update policy, choose from
            ('resize_if_empty', 'resize', 'register')
        dtype (dtype): Type of buffer to be registered (when policy is 'register')
    """
    valid_buffer_names = [n for n, _ in module.named_buffers()]
    for buffer_name in buffer_names:
        if buffer_name not in valid_buffer_names:
            raise ValueError(f'Invalid buffer name "{buffer_name}"')

    for buffer_name in buffer_names:
        _update_registered_buffer(
            module,
            buffer_name,
            f"{module_name}.{buffer_name}",
            state_dict,
            policy,
            dtype,
        )


def conv(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=kernel_size // 2,
    )


def deconv(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        output_padding=stride - 1,
        padding=kernel_size // 2,
    )

def layer_store(model, x):
    output = {"enc": [], "gdn_gamma": [], "gdn_beta": [], "dec": [], "dec_round": []}
    output["enc"].append(x)
    gdn_index = [1,3,5]
    for i, layer in enumerate(model.g_a._modules.values()):
        x = layer(x)
        # print("Enc", i, x.shape)
        output["enc"].append(x)
    x_round = torch.round(x)
    for i, layer in enumerate(model.g_s._modules.values()):
        x = layer(x)
        output["dec"].append(x)
        # print("Dec,", i, x.shape)
    x = x_round
    for i, layer in enumerate(model.g_s._modules.values()):
        x = layer(x)
        output["dec_round"].append(x)
        # print("Dec,", i, x.shape)
    return x_round, output

def layer_compare(net, im_, im_s):
    x_round, layerout = layer_store(net, im_)
    x_s_round, layerout_s = layer_store(net, im_s)
    # compare
    print("Encoder:")
    for layer, layer_s in zip(layerout["enc"], layerout_s["enc"]):
        mean_error = torch.mean((layer-layer_s)**2).item()
        print(mean_error*0.5)
    x_error = torch.mean((x_round-x_s_round)**2)**0.5
    x_err_s = torch.mean((layer-x_s_round)**2)**0.5
    print("Quantization:", x_err_s.item(), x_error.item())
    print("Decoder:")
    for layer_r, layer, layer_s, layer_s_r, f in zip(layerout["dec_round"], layerout["dec"], layerout_s["dec"], layerout_s["dec_round"], net.g_s._modules.values()):
        mean_error, error_r = torch.mean((layer-layer_s_r)**2).item(), torch.mean((layer_r-layer_s_r)**2).item()
        print(mean_error**0.5, error_r**0.5)