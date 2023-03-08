from typing import List, Union

import torch as th
from torch import nn


def build_nn(structure: List) -> nn.Module:
    layers = [nn.Linear(structure[0], structure[1])]

    for i in range(1, len(structure) - 1):
        layers.append(nn.ReLU())
        layers.append(nn.Linear(structure[i], structure[i + 1]))

    return nn.Sequential(*layers)


def bound_loss(v: th.Tensor, lb: Union[float, th.Tensor], ub: Union[float, th.Tensor]) -> th.Tensor:
    relu = nn.ReLU()

    low_loss = relu(lb - v)
    high_loss = relu(v - ub)

    return th.mean(low_loss + high_loss)


def list_dict_to_dict_list(list_dict):
    dict_list = {}
    for _dict in list_dict:
        for k in _dict:
            if dict_list.get(k, None) is None:
                dict_list[k] = [_dict[k]]
            else:
                dict_list[k].append(_dict[k])
    return dict_list
