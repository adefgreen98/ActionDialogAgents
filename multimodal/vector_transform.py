"""
Open questions
- end to end framework?
- unique nn model for each action?
- Alternative to pytorch decomposition: https://scikit-learn.org/stable/modules/generated/sklearn.cross_decomposition.PLSRegression.html
"""

import json
import os
from argparse import ArgumentParser
from pathlib import Path
from pprint import pprint

import matplotlib.pyplot as plt
import pandas
import seaborn

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
from visual_features.utils import setup_argparser, get_optimizer
from visual_features.data import VecTransformDataset  # needed for hold-out procedures




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

    # TODO implement this as a single tensor to allow batched learning
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
        pass  # TODO

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
                return torch.stack([self.__eye_helper[i].detach().clone() for i in action_batch.to(self.device)], dim=0)

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
    elif mode == 'eval':
        model.eval()
        similarity_history = []
        with torch.no_grad():
            for batch in tqdm(dl, desc=mode.upper() + "..."):
                preds, gts = model.process_batch(batch)
                similarity_history.extend(torch.cosine_similarity(preds, gts).tolist())
        return torch.tensor(similarity_history)
    elif mode == 'eval-contrasts':
        assert isinstance(dl, torch.utils.data.Dataset), "in this branch it is assumed for a dataset to be passed"
        model.eval()
        accuracy_history = []
        with torch.no_grad():
            # Here dl should be the bare dataset, not DataLoader
            pbar = tqdm(total=len(dl), desc=mode.upper() + "...")
            for i in range(len(dl)):
                smp = dl[i]  # dict containing 'before', 'action', 'positive', 'negatives', 'neg_actions'

                pred = model(smp['before'].unsqueeze(0).to(model.device), torch.tensor(smp['action'], dtype=torch.long).unsqueeze(0).to(model.device)).to('cpu')

                contrast = torch.stack(smp['negatives'] + [smp['positive']], dim=0)

                # distances = torch.cdist(pred, contrast, p=2.0)
                # distances = ((pred - contrast)**2).sum(dim=-1).squeeze()

                distances = torch.norm(pred - contrast, dim=-1)
                if len(distances.shape) > 0:  # remove contrast with only 1 positive
                    correct = torch.argmin(distances, dim=-1).item() == (distances.shape[-1] - 1)  # correct if the positive (last one) is the nearest
                    accuracy_history.append(int(correct))
                pbar.update()
            pbar.close()
            return torch.tensor(accuracy_history).float()




def train(*args):
    return iterate(*args, mode='train')


def evaluate(model, dl, use_contrasts=False):
    sim = iterate(model, dl, mode='eval')
    acc = None
    if use_contrasts:
        acc = iterate(model, dl.dataset, mode='eval-contrasts')
    return sim, acc


_ALLOWED_MODELS = ['moca-rn', 'clip-rn']
_ALLOWED_TRANSFORM_MODULES = {
    'linear': LinearVectorTransform,
    'linear-concat': LinearConcatVecT,
    'fcn': ConditionalFCNVecT,
    'poly-fcn': PolyFCNVecT
}

__default_train_config__ = {
    # General settings
    'batch_size': (64, ),
    'learning_rate': (1e-4, ),
    'epochs': (20, ),
    'optimizer': ('adamw', ['adamw', 'adam', 'sgd']),
    'scheduler': ('none', ['step', 'none']),

    # VecT settings
    'vect_model': ('linear-concat', list(_ALLOWED_TRANSFORM_MODULES.keys())),
    'use_bn': False,
    'dropout_p': (0.0, [0.0, 0.3, 0.5, 0.7, 0.9]),
    'activation': ('none', __allowed_activations__),
    'hidden_sizes': (None, ),

    # Dataset settings
    'extractor_model': ('moca-rn', _ALLOWED_MODELS),
    'use_regression': False,
    'hold_out_procedure': ('none', VecTransformDataset.allowed_holdout_procedures),
    'hold_out_size': (4, ),

    # Statistics
    'statistical_iterations': (5, ),

    # Others
    'data_path': ('dataset/data-bbxs/pickupable-held',),
    'save_model': False,
    'save_path': ('VECT_results', ),

    'device': ('cuda' if torch.cuda.is_available() else 'cpu', ['cuda', 'cpu']),
    'dataparallel': False,
    'use_wandb': False
}

__tosave_hyperparams__ = [
    'batch_size',
    'learning_rate',
    'epochs',
    'optimizer',
    'scheduler',
    'vect_model',
    'use_bn',
    'dropout_p',
    'activation',
    'extractor_model',
    'use_regression'
]


def get_model(actions, vec_size, args, **kwargs):
    return _ALLOWED_TRANSFORM_MODULES[args.vect_model](actions, vec_size, **vars(args), **kwargs).to(args.device)


def run_training(args, save_results=True, plot=True):
    pprint(vars(args))

    nr_epochs = args.epochs
    full_dataset, train_dl, valid_dl, test_dl = get_data(**vars(args), dataset_type='vect')
    model = get_model(set(full_dataset.ids_to_actions.keys()), full_dataset.get_vec_size(), args)
    opt, sched = get_optimizer(args, model)
    loss_fn = torch.nn.MSELoss()
    ep_loss_mean = []
    ep_sim_mean = []
    ep_acc_mean = []

    # Prepares path for saving
    pth = Path(args.save_path, "+".join([args.vect_model, args.extractor_model]))
    new_idx = str(max([int(el) for el in os.listdir(pth) if el.isdigit()], default=-1) + 1) if os.path.exists(pth) else '0'
    pth = pth / new_idx

    best_acc = -1.0
    for ep in range(nr_epochs):
        print("EPOCH", ep + 1, "/", nr_epochs)

        # Training
        ep_loss_mean.append(round(train(model, train_dl, opt, loss_fn).mean(dim=0).item(), 4))
        print("Train loss: ", ep_loss_mean[-1])

        # Validation
        ev_result = evaluate(model, valid_dl, use_contrasts=True)
        ep_sim_mean.append(round(ev_result[0].mean(dim=0).item(), 4))
        print("Eval sim: ", ep_sim_mean[-1])
        ep_acc_mean.append(round(ev_result[1].mean(dim=0).item(), 4))
        print("Eval acc: ", ep_acc_mean[-1])

        if args.save_model:
            if ep_acc_mean[-1] > best_acc:
                best_acc = ep_acc_mean[-1]
                torch.save(model.state_dict(), pth / 'checkpoint.pth')

    # TESTING
    print()
    print("------ TEST ------")
    test_res = evaluate(model, test_dl, use_contrasts=True)
    test_sim, test_acc = round(test_res[0].mean(dim=-1).item(), 4), round(test_res[1].mean(dim=-1).item(), 4)
    print("Test sim: ", test_sim)
    print("Test acc: ", test_acc)


    if save_results:
        os.makedirs(pth, exist_ok=True)

        with open(pth / 'config.json', mode='wt') as f:
            json.dump(vars(args), f)

        # Saving metrics
        pandas.DataFrame({
            'Train Loss': ep_loss_mean,
            'Cosine Similarity': ep_sim_mean,
            'Contrast Accuracy': ep_acc_mean
        }).to_csv(str(pth / 'metrics.csv'), index=False)

        if plot:
            # Plotting
            plt.figure(figsize=(15, 6))
            plt.suptitle(args.vect_model + f" ({args.extractor_model})")
            for idx, title, metric in zip([1, 2, 3], ['Train Loss', 'Cosine Similarity', 'Contrast Accuracy'], [ep_loss_mean, ep_sim_mean, ep_acc_mean]):
                plt.subplot(1, 3, idx)
                plt.title(title)
                plt.plot(metric)
                plt.xlabel('Epoch')
            plt.savefig(str(pth / 'metrics.png'))
            plt.close()

        return model
    else:
        # needed for statistical evaluation
        return test_sim, test_acc, full_dataset  # to register hold-out items


def exp_hold_out(args):
    """Investigates model generalization capabilities by changing holding out procedure and running several tests,
    in order to build a statistical analysis with randomly sampled test set. At each iteration, a new dataset is
    initialized, and with it a new split of seen/unseen items (objects or scenes, depending on the procedure); within the
    'seen' items are created a training and a validation set, while from the 'unseen' ones it is built the test set.
    Supposing contrasts of hard-negatives (i.e. same object, same scene) each contrast set appears wholly either in the
    seen or in the unseen split; instead, for train and validation samples from the same contrast may be put in different splits."""

    st_path = Path(args.save_path, "statistics")
    os.makedirs(str(st_path), exist_ok=True)
    save_name = f"{max([int(name.split('_')[0]) for name in os.listdir(st_path) if 'hold_out' in name], default=-1) + 1}_hold_out@{args.statistical_iterations}"
    df = pandas.DataFrame(columns=['extractor_model', 'vect_model', 'hold_out_procedure', 'similarity', 'accuracy'])
    for hold_out_procedure in ['object_name', 'scene']:
        for extractor_model in ['moca-rn', 'clip-rn']:
            for vect_model in ['linear-concat', 'fcn']:
                args.hold_out_procedure = hold_out_procedure
                args.extractor_model = extractor_model
                args.vect_model = vect_model
                sim, acc, ds = run_training(args, save_results=False, plot=False)
                df.append({
                    'hold_out_procedure': hold_out_procedure,
                    'extractor_model': extractor_model,
                    'vect_model': vect_model,
                    'similarity': sim,
                    'accuracy': acc
                }, ignore_index=True)

                with open(st_path / ("" + save_name + ".txt"), mode='at') as f:
                    f.write(str(set(ds.after_vectors.loc[ds.hold_out_rows, ds.hold_out_procedure].to_list())) + "\n")

    # seaborn.barplot(x='extractor_model', hue='vect_model', data=df)

    g = seaborn.catplot(x='extractor_model', hue='vect_model', data=df, col="hold_out_procedure", kind="bar", height=10, aspect=.75)

    plt.savefig(Path(args.save_path) / save_name + '.png')
    df.to_csv(Path(args.save_path) / save_name + '.csv', index=False)



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--exp_hold_out', action='store_true')
    parser = setup_argparser(__default_train_config__, parser)
    args = parser.parse_args()

    if args.train:
        res = run_training(args)
    elif args.exp_hold_out:
        res = exp_hold_out(args)
    else:
        full_dataset, _, _, hold_dl = get_data('dataset/data-bbxs/pickupable-held', dataset_type='vect', hold_out_procedure='object_name', hold_out_size=4)
        model = LinearConcatVecT(set(full_dataset.ids_to_actions.keys()), full_dataset.get_vec_size(), device='cuda').to('cuda')
        print(evaluate(model, hold_dl, use_contrasts=True))




