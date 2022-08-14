from abc import abstractmethod
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch as th
from torch import nn

from shrl.config import default_device
from shrl.learn.utils import bound_loss, build_nn


class InputAmplifierBase:
    """
    The state input's change can be very small (< 1e-3)
    this can cause small lie derivative loss,
    which makes us cannot effectively train the TCLF
    """

    @abstractmethod
    def __call__(self, x: th.Tensor) -> th.Tensor:
        pass


class TwinControlLyapunovFunction(nn.Module):
    def __init__(self,
                 lf_structure: List,
                 lqf_structure: List,
                 ub: float,
                 sink: List,
                 input_amplifier: Optional[InputAmplifierBase] = None,
                 lie_derivative_upper: float = 0.2,
                 device: str = default_device):
        super(TwinControlLyapunovFunction, self).__init__()

        self.lf = build_nn(lf_structure)
        self.lqf = build_nn(lqf_structure)
        self.sink = th.tensor(sink, dtype=th.float32, device=device)
        self.ub = ub
        self.input_amplifier = input_amplifier
        self.lie_derivative_upper = lie_derivative_upper
        self.device = device

        self.to(self.device)

    def forward(self, x: th.Tensor, a: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        return self.forward_lf(x), self.forward_lqf(x, a)

    def forward_lf(self, s: th.Tensor) -> th.Tensor:
        if self.input_amplifier is not None:
            inpt = self.input_amplifier(s)
        else:
            inpt = s

        return self.lf(inpt)

    def forward_lqf(self, s: th.Tensor, a: th.Tensor) -> th.Tensor:
        if self.input_amplifier is not None:
            inpt = self.input_amplifier(s)
        else:
            inpt = s

        return self.lqf(th.cat([inpt, a], axis=-1))

    def predict(self, obs: np.ndarray) -> np.ndarray:
        inpt = th.tensor(obs, dtype=th.float32, device=self.device)
        with th.no_grad():
            v = self.forward_lf(inpt)

        return v.cpu().detach().numpy()

    def loss(self, s0: th.Tensor, a: th.Tensor, s1: th.Tensor, current_q: th.Tensor) -> Dict[str, th.Tensor]:
        lf_v0, lqf_v0 = self.forward(s0, a)
        lf_v1 = self.forward_lf(s1)

        mse_loss = nn.MSELoss()
        lqf_loss = mse_loss(lqf_v0, lf_v1.detach())  # lqf should fit lf, thus detach lf

        # the layer derivative is related to distance to goal, this is more stable than simple lf_v1 - lf_v0
        lie_der = th.abs(lf_v0 - lf_v1 - th.norm(s0[:2], p=1, dim=-1).clip(0, self.lie_derivative_upper))
        lie_der_weight = (current_q - current_q.min()) / (current_q.max() - current_q.min())
        lie_der_loss = th.mean(lie_der * lie_der_weight)

        sink_loss = th.abs(self.forward_lf(self.sink.view(-1, self.sink.shape[-1])))

        lf_bound_loss0 = bound_loss(lf_v0, 0, self.ub)
        lf_bound_loss1 = bound_loss(lf_v1, 0, self.ub)
        lqf_bound_loss = bound_loss(lqf_v0, 0, self.ub)
        two_sides_bound_loss = lf_bound_loss0 + lf_bound_loss1 + lqf_bound_loss

        loss_sum = lqf_loss + lie_der_loss + sink_loss + two_sides_bound_loss

        return {"loss_sum": loss_sum,
                "lqf_loss": lqf_loss,
                "lie_der_loss": lie_der_loss,
                "sink_loss": sink_loss,
                "two_sides_bound_loss": two_sides_bound_loss}
