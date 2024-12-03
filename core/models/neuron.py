"""Source: https://github.com/qymeng94/SLTT/blob/main/modules/neuron.py"""

import torch
from spikingjelly.clock_driven.neuron import LIFNode
from spikingjelly.clock_driven import surrogate as surrogate_sj

class SLTTNeuron(LIFNode):
    def __init__(self, surrogate_function = surrogate_sj.PiecewiseQuadratic()):
        super().__init__(surrogate_function=surrogate_function)


    def neuronal_charge(self, x: torch.Tensor):
        if self.decay_input:
            x = x / self.tau

        if self.v_reset is None or self.v_reset == 0:
            if type(self.v) is float:
                self.v = x
            else:
                self.v = self.v.detach() * (1 - 1. / self.tau) + x
        else:
            if type(self.v) is float:
                self.v = self.v_reset * (1 - 1. / self.tau) + self.v_reset / self.tau + x
            else:
                self.v = self.v.detach() * (1 - 1. / self.tau) + self.v_reset / self.tau + x


class BPTTNeuron(LIFNode):
    def __init__(self):
        super().__init__()