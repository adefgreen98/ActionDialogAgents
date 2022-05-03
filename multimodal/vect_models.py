from collections import OrderedDict

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Tuple

class AbstractVectorTransform(ABC, nn.Module):
    """Base class for all models of Vector Transformation (VecT)."""

    @abstractmethod
    def __init__(self, actions: set, vec_size: int,
                 device='cpu', **kwargs):
        super(AbstractVectorTransform, self).__init__()
        self.actions = actions
        self.nr_actions = len(self.actions)
        self.vec_size = vec_size
        self.device = device

    @abstractmethod
    def forward(self, v, action):
        pass

    @abstractmethod
    def process_batch(self, batch, loss_fn=None):
        pass

    def to(self, *args, **kwargs):
        self.device = args[0]
        res = super(AbstractVectorTransform, self).to(*args, **kwargs)
        # res.device = args[0]
        return res


__supported_regressions__ = {None, 'torch'}


class LinearVectorTransform(AbstractVectorTransform):
    """Module implementing vector transformation as matrix multiplication. Can also be initialized with a least-squares
    regression procedure."""

    # TODO implement this as a single tensor to allow batched learning
    def __init__(self, actions: set, vec_size: int,
                 use_regression=False,
                 regression_type=None,
                 **kwargs):
        super(LinearVectorTransform, self).__init__(actions, vec_size, **kwargs)
        self.regression_type = regression_type
        assert self.regression_type in __supported_regressions__

        self.use_regression = use_regression

        self.weights = nn.Parameter(torch.stack([nn.Linear(self.vec_size, self.vec_size, bias=False).weight.clone().detach() for i in range(self.nr_actions)], dim=0))


    def forward(self, v, action: str):
        # Normalize vector?
        pred = v.unsqueeze(1).bmm(self.weights.index_select(0, action))
        return pred.squeeze(1)

    def process_batch(self, batch, loss_fn=None):
        # batch --> tuple (before vectors, actions, after vectors)
        before, actions, after = batch
        before = before.to(self.device)
        after = after.to(self.device)
        actions = actions.to(self.device)
        predictions = self(before, actions)

        if self.training and loss_fn is not None:
            return loss_fn(predictions, after)
        elif not self.training and loss_fn is None:
            return predictions, after

    def regression_init(self, data: Tuple, regression_type=None):
        # data is a tuple of dicts ({'action' --> samples matrix A}, {'action' --> target vector B})
        # A, B: should be (num. samples, vec_size)
        if self.regression_type is None and regression_type is None:
            raise RuntimeError("unspecified modality of regression for initialization")

        if self.regression_type is None:
            self.regression_type = regression_type
            assert self.regression_type in __supported_regressions__

        train_mat, target_mat = data

        if self.regression_type == 'torch':
            sorted_actions = sorted(train_mat.keys(), key=lambda el: int(el))  # sort action indices
            weights = []
            for action in sorted_actions:
                sol = torch.linalg.lstsq(train_mat[action], target_mat[action]).solution.to(self.device)
                if sol.numel() == 0:
                    sol = torch.eye(self.vec_size, device=self.device)
                weights.append(sol)
            self.weights = nn.Parameter(torch.stack(weights, dim=0).to(self.device))
        else:
            pass  # TODO

        # Freezes parameters after performing regression
        for p in self.parameters():
            p.requires_grad_(False)

    # Use numpy or sklearn to solve partial least squares?
    # Separate initialization (solution of lsqs) and parameter assignment


class LinearConcatVecT(AbstractVectorTransform):
    def __init__(self, actions: set, vec_size: int, **kwargs):
        super().__init__(actions, vec_size, **kwargs)

        self.__eye_helper = torch.eye(self.nr_actions, device=self.device)

        def embed_action(action_batch: torch.LongTensor) -> torch.Tensor:
            return torch.stack([self.__eye_helper[i].detach().clone() for i in action_batch], dim=0).to(self.device)

        self.embed_action = embed_action
        self.net = nn.Linear(self.nr_actions + self.vec_size, self.vec_size, bias=False)

    def forward(self, v, action):
        embedded = self.embed_action(action)
        return self.net(torch.cat((v, embedded), dim=-1))

    def process_batch(self, batch, loss_fn=None):
        before, actions, after = batch
        before = before.to(self.device)
        actions = actions.to(self.device).long()
        after = after.to(self.device)

        predictions = self(before, actions)

        if self.training and loss_fn is not None:
            return loss_fn(predictions, after)
        elif not self.training and loss_fn is None:
            return predictions, after


__allowed_conditioning_methods__ = ['concat']
__allowed_activations__ = {
    'none': nn.Identity,
    'relu': nn.ReLU,
    'sigmoid': nn.Sigmoid,
    'tanh': nn.Tanh
}


class ConditionalFCNVecT(AbstractVectorTransform):
    """Module implementing vector transform as a conditional feedforward network, where the input
    vector is combined with the action label embedding to provide conditionality."""

    def __init__(self, actions: set, vec_size: int,
                 hidden_sizes=None,
                 dropout_p: float = 0.0,
                 use_bn: bool = False,
                 activation='none',
                 conditioning_method='concat',
                 **kwargs):
        super(ConditionalFCNVecT, self).__init__(actions, vec_size)
        self.dropout_p = dropout_p
        self.use_bn = use_bn

        assert activation in __allowed_activations__
        self.activation = activation

        assert conditioning_method in __allowed_conditioning_methods__
        self.conditioning_method = conditioning_method

        if hidden_sizes is None:
            h = int(self.vec_size * 0.75)
            hidden_sizes = [h]
        elif hidden_sizes == 'large':
            h = int(self.vec_size * 0.75)
            hidden_sizes = [h, h // 2, h]
        self.hidden_sizes = hidden_sizes

        layer_sizes = [self.vec_size] + hidden_sizes + [self.vec_size]

        if self.conditioning_method == 'concat':
            layer_sizes[0] += self.nr_actions
            # self.embed_actions = nn.Embedding(self.nr_actions, self.nr_actions)  # TODO understand if this works

            self.__eye_helper = torch.eye(self.nr_actions, device=self.device)

            def embed_action(action_batch: torch.LongTensor) -> torch.Tensor:
                return torch.stack([self.__eye_helper[i].detach().clone() for i in action_batch], dim=0).to(self.device)

            self.embed_action = embed_action

        # TODO: check ordering of BatchNorm and activation
        self.net = nn.Sequential(*[
                nn.Sequential(
                    OrderedDict([
                        ('dropout', nn.Dropout(self.dropout_p)),
                        ('linear', nn.Linear(layer_sizes[i], layer_sizes[i + 1])),
                        ('bnorm', nn.BatchNorm1d(layer_sizes[i + 1]) if (self.use_bn and i < len(layer_sizes) - 2) else nn.Identity()),  # up to penultimate layer
                        ('activation', __allowed_activations__[self.activation]() if i < len(layer_sizes) - 2 else nn.Identity())  # up to penultimate layer
                    ])
                )
                for i in range(len(layer_sizes) - 1)]  # because building with (i, i+1)
            )

    def forward(self, v, action):
        return self.net(torch.cat([v, self.embed_action(action)], dim=-1))

    def process_batch(self, batch, loss_fn=None):
        before, actions, after = batch
        before = before.to(self.device)
        actions = actions.to(self.device).long()
        after = after.to(self.device)

        predictions = self(before, actions)

        if self.training and loss_fn is not None:
            return loss_fn(predictions, after)
        elif not self.training and loss_fn is None:
            return predictions, after


class PolyFCNVecT(AbstractVectorTransform):
    """Module implementing a vector transformation with a unique neural network for each action."""

    def __init__(self, actions: set, vec_size: int,
                 hidden_sizes=None,
                 dropout_p: float = 0.0,
                 activation='none',
                 use_bn: bool = False,
                 **kwargs):
        super(PolyFCNVecT, self).__init__(actions, vec_size)

        self.dropout_p = dropout_p
        self.use_bn = use_bn

        assert activation in __allowed_activations__
        self.activation = activation

        if hidden_sizes is None:
            h = int(self.nr_actions * 0.75)
            hidden_sizes = [h]
        elif hidden_sizes == 'large':
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
                        ('activation', __allowed_activations__[self.activation] if i < len(layer_sizes) - 2 else nn.Identity())  # up to penultimate layer
                    ])
                )
                for i in range(len(layer_sizes) - 1)]
            ) for k in self.actions
        }

        self.transforms = nn.ModuleDict(self.transforms)

    def forward(self, v, action):
        return self.transforms[action](v)

    def process_batch(self, batch, loss_fn=None):
        pass  # TODO
