"""
Open questions
- end to end framework?
- unique nn model for each action?
- Alternative to pytorch decomposition: https://scikit-learn.org/stable/modules/generated/sklearn.cross_decomposition.PLSRegression.html
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Tuple
from collections import OrderedDict

import sys

sys.path.append('.')  # needed for executing from top directory

from visual_features.utils import load_model_from_path


class AbstractVectorTransform(ABC, nn.Module):

    @abstractmethod
    def __init__(self, actions: set, vec_size: int):
        super(AbstractVectorTransform, self).__init__()
        self.actions = actions
        self.nr_actions = len(self.actions)
        self.vec_size = vec_size

    @abstractmethod
    def forward(self, v, action):
        pass

    @abstractmethod
    def process_batch(self, batch):
        pass


__supported_regressions__ = {'torch'}


class LinearVectorTransform(AbstractVectorTransform):
    """Module implementing vector transformation as matrix multiplication."""

    def __init__(self, actions: set, vec_size: int,
                 use_regression=False,
                 regression_type=None):
        super(LinearVectorTransform, self).__init__(actions, vec_size)
        self.regression_type = regression_type
        assert self.regression_type in __supported_regressions__

        self.use_regression = use_regression

        self.transforms = nn.ModuleDict({
            k: nn.Linear(self.vec_size, self.vec_size, bias=False) for k in self.actions
        })

    def forward(self, v, action: str):
        # Normalize vector?
        return self.transforms[action](v)

    def regression_init(self, data: Dict[str, Tuple], regression_type=None):
        # data is a mapping 'action' --> (samples matrix A, target vector B)
        # A, B: should be (num. samples, vec_size)
        if self.regression_type is None and regression_type is None:
            raise RuntimeError("unspecified modality of regression for initialization")

        if self.regression_type is None:
            self.regression_type = regression_type
            assert self.regression_type in __supported_regressions__

        if self.regression_type == 'torch':
            for action in data:
                sd = OrderedDict({
                    'weight': torch.linalg.lstsq(data[action][0], data[action][1]).solution
                })
                self.transforms[action].load_state_dict(sd)
        else:
            pass

        # Freezes parameters after performing regression
        for p in self.parameters():
            p.requires_grad_(False)

    # Use numpy or sklearn to solve partial least squares?
    # Separate initialization (solution of lsqs) and parameter assignment


__allowed_conditioning_methods__ = ['concat']
__allowed_activations__ = {
    'none': nn.Identity,
    'relu': nn.ReLU,
    'sigmoid': nn.Sigmoid,
    'tanh': nn.Tanh
}


class ConditionalNeuralVecT(AbstractVectorTransform):
    """Module implementing vector transform as a conditional feedforward network, where the input
    vector is combined with the action label embedding to provide conditionality."""

    def __init__(self, actions: set, vec_size: int,
                 hidden_sizes: list = None,
                 dropout_p: float = 0.0,
                 use_bn: bool = False,
                 activation='none',
                 conditioning_method='concat'):
        super(ConditionalNeuralVecT, self).__init__(actions, vec_size)
        self.dropout_p = dropout_p
        self.use_bn = use_bn

        assert activation in __allowed_activations__
        self.activation = activation

        assert conditioning_method in __allowed_conditioning_methods__
        self.conditioning_method = conditioning_method

        if hidden_sizes is None:
            h = int(self.nr_actions * 0.75)
            hidden_sizes = [h, h // 2, h]
        self.hidden_sizes = hidden_sizes

        layer_sizes = [self.vec_size] + hidden_sizes + [self.vec_size]

        if self.conditioning_method == 'concat':
            layer_sizes[0] += self.nr_actions
            self.embed_actions = nn.Embedding(self.nr_actions, self.nr_actions)

        # TODO: check ordering of BatchNorm and activation
        self.net = nn.Sequential(*[
                nn.Sequential(
                    OrderedDict([
                        ('dropout', nn.Dropout(self.dropout_p)),
                        ('linear', nn.Linear(layer_sizes[i], layer_sizes[i + 1])),
                        ('bnorm', nn.BatchNorm1d(layer_sizes[i + 1]) if (self.use_bn and i < len(layer_sizes) - 2) else nn.Identity()),
                        ('activation', __allowed_activations__[self.activation] if i < len(layer_sizes) - 2 else nn.Identity())
                    ])
                )
                for i in range(len(layer_sizes) - 1)]
            )

    def forward(self, v, action):
        return self.net(torch.cat([v, self.embed_actions(action)], dim=-1))

    def process_batch(self, batch):
        pass


class PolyNeuralVecT(AbstractVectorTransform):
    """Module implementing a vector transformation with a unique neural network for each action."""

    def __init__(self, actions: set, vec_size: int,
                 hidden_sizes: list = None,
                 dropout_p: float = 0.0,
                 activation='none',
                 use_bn: bool = False):
        super(PolyNeuralVecT, self).__init__(actions, vec_size)

        self.dropout_p = dropout_p
        self.use_bn = use_bn

        assert activation in __allowed_activations__
        self.activation = activation

        if hidden_sizes is None:
            h = int(self.nr_actions * 0.75)
            hidden_sizes = [h, h // 2, h]
        self.hidden_sizes = hidden_sizes

        layer_sizes = [self.vec_size] + hidden_sizes + [self.vec_size]

        self.transforms = {
            k: nn.Sequential(*[
                nn.Sequential(
                    OrderedDict([
                        ('dropout', nn.Dropout(self.dropout_p)),
                        ('linear', nn.Linear(layer_sizes[i], layer_sizes[i + 1])),
                        ('bnorm', nn.BatchNorm1d(layer_sizes[i + 1]) if (self.use_bn and i < len(layer_sizes) - 2) else nn.Identity()),
                        ('activation', __allowed_activations__[self.activation] if i < len(layer_sizes) - 2 else nn.Identity())
                    ])
                )
                for i in range(len(layer_sizes) - 1)]
            ) for k in self.actions
        }

        self.transforms = nn.ModuleDict(self.transforms)

    def forward(self, v, action):
        return self.transforms[action](v)

    def process_batch(self, batch):
        pass


if __name__ == '__main__':
    net = load_model_from_path('../bbox_results/clip-rn', nr=0)
    print(net)
