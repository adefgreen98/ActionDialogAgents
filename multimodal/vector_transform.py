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

from tqdm import tqdm

sys.path.append('.')  # needed for executing from top directory

from visual_features.utils import load_model_from_path
from visual_features.data import get_data


class AbstractVectorTransform(ABC, nn.Module):
    """Base class for all models of Vector Transformation (VecT)."""

    @abstractmethod
    def __init__(self, actions: set, vec_size: int,
                 device='cuda', **kwargs):
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


__supported_regressions__ = {'torch'}


class LinearVectorTransform(AbstractVectorTransform):
    """Module implementing vector transformation as matrix multiplication. Can also be initialized with a least-squares
    regression procedure."""

    def __init__(self, actions: set, vec_size: int,
                 use_regression=False,
                 regression_type=None,
                 **kwargs):
        super(LinearVectorTransform, self).__init__(actions, vec_size, **kwargs)
        self.regression_type = regression_type
        assert self.regression_type in __supported_regressions__

        self.use_regression = use_regression

        self.transforms = nn.ModuleDict({
            k: nn.Linear(self.vec_size, self.vec_size, bias=False) for k in self.actions
        })

    def forward(self, v, action: str):
        # Normalize vector?
        return self.transforms[action](v)

    def process_batch(self, batch, loss_fn=None):
        # batch --> tuple (before vectors, actions, after vectors)
        before, actions, after = batch
        before = before.to(self.device)
        after = after.to(self.device)
        pass

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
            return torch.stack([self.__eye_helper[i].detach().clone() for i in action_batch], dim=0)

        self.embed_action = embed_action
        self.net = nn.Linear(self.nr_actions + self.vec_size, self.vec_size)

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

    def process_batch(self, batch, loss_fn=None):
        pass  # TODO


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

    def process_batch(self, batch, loss_fn=None):
        pass  # TODO


def iterate(model: AbstractVectorTransform, dl, optimizer=None, loss_fn=None, mode='eval'):
    if mode == 'train':
        model.train()
        loss_history = []
        for batch in tqdm(dl, desc=mode.upper() + "..."):
            loss = model.process_batch(batch, loss_fn)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_history.append(loss.item())
        return torch.tensor(loss_history)
    else:
        model.eval()
        similarity_history = []
        with torch.no_grad():
            for batch in tqdm(dl, desc=mode.upper() + "..."):
                preds, gts = model.process_batch(batch)
                similarity_history.extend(torch.cosine_similarity(preds, gts).tolist())
        return torch.tensor(similarity_history)


def train(*args):
    return iterate(*args, mode='train')


def evaluate(*args):
    return iterate(*args, mode='eval')


if __name__ == '__main__':
    nr_epochs = 10
    device = 'cuda'
    full_dataset, train_dl, valid_dl = get_data('dataset/data-bbxs/pickupable-held', dataset_type='vect', transform='to_tensor')
    net = LinearConcatVecT(set(full_dataset.ids_to_actions.keys()), full_dataset.get_vec_size(), device=device).to(device)
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    loss_fn = torch.nn.MSELoss()
    ep_loss_mean = []
    ep_sim_mean = []

    for ep in range(nr_epochs):
        print("EPOCH", ep + 1, "/", nr_epochs)
        ep_loss_mean.append(train(net, train_dl, opt, loss_fn).mean(dim=0))
        print("Train loss: ", ep_loss_mean[-1])
        ep_sim_mean.append(evaluate(net, valid_dl).mean(dim=0))
        print("Eval sim: ", ep_sim_mean[-1])
