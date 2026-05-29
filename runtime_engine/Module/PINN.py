# coding = utf-8
"""
Deterministic PINN (Teacher Network)

Standard Physics-Informed Neural Network used as the teacher
in the GBSD distillation framework.
"""
import torch
from collections import OrderedDict
from Module import PoissonTools as PT


class Net(torch.nn.Module):
    """
    Standard deterministic PINN with tanh activation.
    
    Args:
        layers: List of layer sizes [input, hidden1, ..., hiddenN, output]
    """
    def __init__(self, layers, use_fourier_features: bool = False,
                 fourier_modes: int = 4, hard_bc: bool = False):
        super(Net, self).__init__()
        self.depth = len(layers) - 1
        self.activation = torch.nn.Tanh
        self.use_fourier_features = use_fourier_features
        self.fourier_modes = int(fourier_modes)
        self.hard_bc = hard_bc

        layer_list = list()
        for i in range(self.depth - 1):
            layer_list.append(
                ('layer_%d' % i, torch.nn.Linear(layers[i], layers[i + 1]))
            )
            layer_list.append(('activation_%d' % i, self.activation()))

        layer_list.append(
            ('layer_%d' % (self.depth - 1), torch.nn.Linear(layers[-2], layers[-1]))
        )
        layerDict = OrderedDict(layer_list)

        self.layers = torch.nn.Sequential(layerDict)
        
        # Training history
        self.iter = 0
        self.iter_list = []
        self.loss_list = []
        self.loss_f_list = []
        self.loss_d_list = []
        self.loss_b_list = []
        self.loss_teach_list = []
        self.loss_rgl_list = []
        self.para_ud_list = []

    def forward(self, x):
        raw_x = x
        if self.use_fourier_features:
            x = PT.encode_fourier(x, modes=self.fourier_modes)
        out = self.layers(x)
        if self.hard_bc:
            out = PT.apply_zero_dirichlet_hard_bc(raw_x, out)
        return out
