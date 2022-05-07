
import json
import os
from argparse import ArgumentParser, Namespace
from pathlib import Path
from pprint import pprint

import matplotlib.pyplot as plt
import pandas
import seaborn

import torch
import torch.nn as nn
from torch.nn import functional as F
from abc import ABC, abstractmethod
from typing import Tuple
from collections import OrderedDict

import sys

from tqdm import tqdm

sys.path.append('.')  # needed for executing from top directory

from visual_features.utils import load_model_from_path
from visual_features.data import get_data
from visual_features.utils import setup_argparser, get_optimizer
from visual_features.data import VecTransformDataset  # needed for hold-out procedures

from multimodal.vect_models import *
from multimodal.vect_models import __allowed_activations__ as allowed_activations


def compute_embedding_distance(contrasts, pred):
    """
    Computes distance between embeddings of a contrast (in the form [negative1, negative2, ...., positive]) and a prediction.
    :param contrasts: Tensor of size (batch-size, contrast-size, vector size); note that contrast-size is equivalent to number of negatives plus 1 for the positive
    :param pred: Tensor of size (batch-size, vector size)
    """
    bs = contrasts.shape[0]
    contrast_size = contrasts.shape[1]
    assert len(pred.shape) == 2 and pred.shape[0] == bs, f"Prediction {pred.shape}, Contrast {contrasts.shape}"
    assert len(contrasts.shape) == 3 and (contrasts.shape[-1] == pred.shape[-1]), f"Prediction {pred.shape}, Contrast {contrasts.shape}"

    return torch.cdist(contrasts, pred.unsqueeze(-2), p=2.0).view(bs, contrast_size).to('cpu')


def contrastive_train_step(model: AbstractVectorTransform, batch, alpha=1.0):
    pred = model(batch['before'].to(model.device), batch['action'].to(model.device))
    contrasts = torch.cat((batch['negatives'], batch['positive'].unsqueeze(-2)), dim=-2).to(model.device)
    distances = compute_embedding_distance(contrasts, pred)
    distances = distances ** 2
    batch_loss = (alpha + (distances[:, -1].unsqueeze(-1) - distances[:, :-1]))
    # batch_loss = batch_loss * batch['mask']  # annihilates padded elemnents in contrast
    loss = (batch_loss * (batch_loss >= 0).int()).sum(dim=-1).mean(dim=0)
    return loss


def infonce_train_step(model: AbstractVectorTransform, batch):
    batch_size, contrast_size, vec_size = batch['negatives'].shape
    pred = model(batch['before'].to(model.device), batch['action'].to(model.device)).view(batch_size, 1, vec_size).to(batch['negatives'].device)

    # NOTE: we need to transpose prediction to perform MatMul
    logits_negatives = torch.bmm(batch['negatives'], pred.view(batch_size, vec_size, 1)).view(batch_size, contrast_size)
    logits_positive = torch.bmm(batch['positive'].view(batch_size, 1, vec_size), pred.view(batch_size, vec_size, 1)).view(batch_size, 1)

    logits = torch.cat((logits_negatives, logits_positive), dim=-1)
    # logits = logits * torch.cat((batch['mask'], torch.ones(batch_size, 1)), dim=-1)  # annihilates padded elements in contrast

    # 1. Compute random placement of positive for batch
    displacements = torch.stack([torch.randperm(contrast_size + 1) for _ in range(batch_size)], dim=0)
    pos_indices = torch.where(displacements == contrast_size)[-1]  # GT for this batch now is the position of the positive <--> last index in the contrast

    batch_indices = torch.tensor([i for i in range(batch_size) for _ in range(contrast_size + 1)])  # help in randomizing contrast position

    reworked_logits = logits[batch_indices, displacements.view(1, -1)].view(logits.shape)

    # 2. Compute crossentropy with correct positive placement
    loss = F.cross_entropy(reworked_logits, pos_indices)

    return loss


def iterate(model: AbstractVectorTransform, dl, optimizer=None, loss_fn=None, mode='eval', contrastive=False, infonce=False):
    assert not (contrastive and infonce)
    if mode == 'train':
        model.train()
        loss_history = []
        for batch in tqdm(dl, desc=mode.upper() + "..."):
            optimizer.zero_grad()
            if contrastive:
                loss = contrastive_train_step(model, batch)
            elif infonce:
                loss = infonce_train_step(model, batch)
            else:
                loss = model.process_batch(batch, loss_fn)
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
                # similarity_history.extend(compute_embedding_distance(gts, preds).flatten().tolist())  # list dimension needed for batch evaluation
                similarity_history.extend(torch.cosine_similarity(preds, gts, dim=-1).flatten().tolist())
        return torch.tensor(similarity_history).float().mean(dim=-1).item()
    elif mode == 'eval-contrasts':
        assert isinstance(dl, torch.utils.data.Dataset), "in this branch it is assumed for a dataset to be passed"
        model.eval()
        accuracy_history = []
        with torch.no_grad():
            # Here dl should be the bare dataset, not DataLoader
            pbar = tqdm(total=len(dl), desc=mode.upper() + "...")
            for i in range(len(dl)):
                smp = dl[i]  # dict containing 'before', 'action', 'positive', 'negatives', 'neg_actions'
                if len(smp['negatives']) > 0:  # remove contrast with only 1 positive
                    before_vec = smp['before'].unsqueeze(0).to(model.device)
                    action = torch.tensor(smp['action'], dtype=torch.long).unsqueeze(0).to(model.device)
                    pred = model(before_vec, action).to('cpu')

                    contrast = torch.stack(smp['negatives'] + [smp['positive']], dim=0).unsqueeze(0)  # needed unsqueeze to be consistent with model output (batch-size = 1)

                    distances = compute_embedding_distance(contrast, pred)
                    # distances = ((pred - contrast)**2).sum(dim=-1).squeeze()
                    # distances = torch.norm(pred - contrast, dim=-1)
                    # distances = (1 - torch.cosine_similarity(pred, contrast, dim=-1)) / 2

                    correct = torch.argmin(distances, dim=-1).item() == (distances.shape[-1] - 1)  # correct if the positive (last one) is the nearest
                    accuracy_history.append(int(correct))
                pbar.update()
            pbar.close()
            return torch.tensor(accuracy_history).float().mean(dim=-1).item()




def train(*args, contrastive=False, infonce=False):
    return round(iterate(*args, mode='train', contrastive=contrastive, infonce=infonce).mean(dim=-1).item(), 4)


def evaluate(model, dl, use_contrasts=False):
    sim = iterate(model, dl, mode='eval')
    acc = None
    if use_contrasts:
        acc = iterate(model, dl.dataset, mode='eval-contrasts')
    return round(sim, 4), round(acc, 4)


def baselines_evaluation(test_set):
    random_acc = []
    similarity_acc = []

    pbar = tqdm(total=len(test_set), desc="Evaluating baselines...")
    for i in range(len(test_set)):
        smp = test_set[i]  # dict containing 'before', 'action', 'positive', 'negatives', 'neg_actions'
        if len(smp['negatives']) > 0:  # remove contrast with only 1 positive
            before_vec = smp['before'].unsqueeze(0)
            # action = torch.tensor(smp['action'], dtype=torch.long).unsqueeze(0)  # not needed
            pred = before_vec

            contrast = torch.stack(smp['negatives'] + [smp['positive']], dim=0).unsqueeze(0)  # add fake batch dimension
            distances = compute_embedding_distance(contrast, pred)
            # distances = torch.cosine_similarity(contrast, pred, dim=-1)

            sim_correct = torch.argmin(distances, dim=-1).item() == (distances.shape[-1] - 1)  # correct if the positive (last one) is the nearest
            similarity_acc.append(int(sim_correct))

            random_acc.append(len(smp['negatives']))

        pbar.update()
    pbar.close()

    random_acc = 1 / (sum(random_acc) / len(random_acc))

    return {
        'random': round(random_acc, 4),
        'similarity': round(torch.tensor(similarity_acc).float().mean(dim=-1).item(), 4)
    }


_ALLOWED_MODELS = ['moca-rn', 'clip-rn']
_ALLOWED_TRANSFORM_MODULES = {
    'linear': LinearVectorTransform,
    'linear-concat': LinearConcatVecT,
    'fcn': ConditionalFCNVecT,
    'poly-fcn': PolyFCNVecT
}

__default_train_config__ = {
    # General settings
    'batch_size': (256, ),
    'learning_rate': (1e-4, ),
    'epochs': (20, ),
    'optimizer': ('adam', ['adamw', 'adam', 'sgd']),
    'scheduler': ('none', ['step', 'none']),

    # VecT settings
    'vect_model': ('linear-concat', list(_ALLOWED_TRANSFORM_MODULES.keys())),
    'use_bn': False,
    'dropout_p': (0.0, [0.0, 0.3, 0.5, 0.7, 0.9]),
    'activation': ('none', allowed_activations),
    'hidden_sizes': (None, ),
    'use_contrastive': False,
    'use_infonce': False,

    # Dataset settings
    'extractor_model': ('moca-rn', _ALLOWED_MODELS),
    'use_regression': False,
    'hold_out_procedure': ('samples', VecTransformDataset.allowed_holdout_procedures),
    'hold_out_size': (4, ),

    # Statistics
    'statistical_iterations': (5, ),

    # Others
    'data_path': ('new-dataset/data-improved-descriptions',),
    'save_model': False,
    'save_path': ('new-vect-results', ),

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
    'use_regression',
    'use_contrastive',
    'use_infonce'
]


def get_savepath(st_path, args, create_dir=True):
    os.makedirs(str(st_path), exist_ok=True)
    save_name = f"{max([int(name.split('_')[0] if name.split('_')[0].isdigit() else -1) for name in os.listdir(st_path)], default=-1) + 1}_@{args.statistical_iterations}"
    save_path = st_path / save_name
    if create_dir:
        os.makedirs(save_path, exist_ok=True)
    return save_path


def get_model(actions, vec_size, args, **kwargs):
    return _ALLOWED_TRANSFORM_MODULES[args.vect_model](actions, vec_size, **vars(args), **kwargs).to(args.device)


def run_training(args, fixed_dataset=None, save_results=True, plot=True):
    pprint(vars(args))

    nr_epochs = int(args.epochs)
    if fixed_dataset is None:
        full_dataset, train_dl, valid_dl, test_dl = get_data(**vars(args), dataset_type='vect')
    else:
        full_dataset, train_dl, valid_dl, test_dl = fixed_dataset

    model = get_model(set(full_dataset.ids_to_actions.keys()), full_dataset.get_vec_size(), args)
    opt, sched = get_optimizer(args, model)
    loss_fn = torch.nn.MSELoss()
    ep_loss_mean = []
    ep_sim_mean = []
    ep_acc_mean = []

    # Prepares path for saving
    if args.use_contrastive:
        args.vect_model = args.vect_model + '-contrastive'
    elif args.use_infonce:
        args.vect_model = args.vect_model + '-infonce'

    pth = Path(args.save_path, "models", "+".join([args.vect_model, args.extractor_model]))
    pth = get_savepath(pth, args, create_dir=save_results)

    best_acc = -1.0
    for ep in range(nr_epochs):
        print("EPOCH", ep + 1, "/", nr_epochs)

        # Training
        ep_loss_mean.append(train(model, train_dl, opt, loss_fn, contrastive=args.use_contrastive, infonce=args.use_infonce))
        print("Train loss: ", ep_loss_mean[-1])

        # Validation
        ev_sim, ev_acc = evaluate(model, valid_dl, use_contrasts=True)
        ep_sim_mean.append(ev_sim)
        print("Eval sim: ", ep_sim_mean[-1])
        ep_acc_mean.append(ev_acc)
        print("Eval acc: ", ep_acc_mean[-1])

        if args.save_model:
            if ep_acc_mean[-1] > best_acc:
                best_acc = ep_acc_mean[-1]
                torch.save(model.state_dict(), pth / 'checkpoint.pth')

    # TESTING
    print()
    print("------ TEST ------")
    test_sim, test_acc = evaluate(model, test_dl, use_contrasts=True)
    print("Test sim: ", test_sim)
    print("Test acc: ", test_acc)


    if save_results:
        os.makedirs(pth, exist_ok=True)

        with open(pth / 'config.json', mode='wt') as f:
            json.dump(vars(args), f, indent=2)

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

    # Hard-coded params
    args.activation = 'relu'
    args.use_bn = True

    st_path = Path(args.save_path, "statistics", "hold-out")
    save_path = get_savepath(st_path, args)

    # Here for debug
    # df = {
    #     'hold_out_procedure': ['a', 'b', 'c', 'd'],
    #     'extractor_model': [0, 1, 2, 3],
    #     'vect_model': ['a', 'b', 'c', 'd'],
    #     'similarity': [0, 1, 2, 3],
    #     'accuracy': [0, 1, 2, 3]
    # }

    tested_procedures = ['samples', 'object_name', 'scene']
    nr_samples_per_it = {k: [] for k in tested_procedures}

    tested_vect_models = ['linear', 'linear-concat', 'fcn']

    df = []
    for stat_it in range(args.statistical_iterations):
        for extractor_model in ['moca-rn', 'clip-rn']:
            for hold_out_procedure in tested_procedures:
                args.hold_out_procedure = hold_out_procedure
                args.extractor_model = extractor_model

                # needed both to fix the dataset
                fixed_ds = get_data(**vars(args), dataset_type='vect')

                with open(save_path / 'items.txt', mode='at') as f:
                    f.write(str(fixed_ds[0].get_hold_out_items()) + "\n")

                nr_samples_per_it[hold_out_procedure].append(fixed_ds[0].get_nr_hold_out_samples())

                for vect_model in tested_vect_models:
                    args.vect_model = vect_model
                    sim, acc, ds = run_training(args, fixed_dataset=fixed_ds, save_results=False, plot=False)
                    df.append({
                        'hold_out_procedure': hold_out_procedure,
                        'extractor_model': extractor_model,
                        'vect_model': args.vect_model,
                        'similarity': sim,
                        'accuracy': acc
                    })

                # Adds baselines for this dataset
                bsl_acc = baselines_evaluation(fixed_ds[-1].dataset)
                for k in bsl_acc:
                    df.append({
                        'hold_out_procedure': hold_out_procedure,
                        'extractor_model': extractor_model,
                        'vect_model': 'baseline-' + k,
                        'similarity': -1.0,
                        'accuracy': bsl_acc[k]
                    })

    with open(save_path / 'items.txt', mode='at') as f:
        f.write(str({
            'sizes': nr_samples_per_it,
            'averages': {k: ((sum(el) / len(el)) if len(el) > 0 else 0) for k, el in nr_samples_per_it.items()}
        }))

    with open(save_path / 'config.json', mode='at') as f:
        json.dump(vars(args), f, indent=2)

    df = pandas.DataFrame(df, columns=['extractor_model', 'vect_model', 'hold_out_procedure', 'similarity', 'accuracy'])
    df.to_csv(save_path / 'results.csv', index=False)

    seaborn.set(font_scale=1)

    # create column plot
    g = seaborn.catplot(x='extractor_model', y='accuracy', hue='vect_model', data=df[~df['vect_model'].isin({'baseline-random', 'baseline-similarity'})], col="hold_out_procedure", kind="bar", height=10, aspect=.75)
    [ax.axhline(df[df['vect_model'] == 'baseline-random']['accuracy'].mean(), linestyle='--') for el in g for ax in el.axes]  # random baseline
    [ax.axhline(df[df['vect_model'] == 'baseline-similarity']['accuracy'].mean(), linestyle='--') for el in g for ax in el.axes]  # random baseline
    g.savefig(save_path / 'stats.png')

    g = seaborn.catplot(hue='extractor_model', x='vect_model', y='accuracy', data=df, kind='swarm', col='hold_out_procedure')
    g.savefig(save_path / 'observations.png')

    return df


def exp_fcn_hyperparams(args):
    args.vect_model = 'fcn'
    args.statistical_iterations = 1

    hyperparam_options = {
        'batch_size': [256],
        'learning_rate': [1e-4, 1e-5],
        'optimizer': ['adam', 'adamw'],
        'use_bn': [True, False],
        'dropout_p': [0.0, 0.7],
        'extractor_model': ['moca-rn'],
        'activation': allowed_activations
    }
    df = []

    from itertools import product
    print("TOTAL CONFIGS:", len(list(product(*list(hyperparam_options.values())))), f"(* {args.statistical_iterations} iterations)")
    for i in range(args.statistical_iterations):
        for cfg in product(*list(hyperparam_options.values())):
            cfg = {k: cfg[i] for i, k in enumerate(hyperparam_options.keys())}
            args = vars(args)
            args.update(cfg)
            args = Namespace(**args)
            args.vect_model = 'fcn'  # needed for consistency with contrastive setup
            sim, acc, ds = run_training(args, save_results=False, plot=False)
            df.append({**cfg, **{'similarity': sim, 'accuracy': acc}})
    df = pandas.DataFrame(df, columns=list(hyperparam_options.keys()) + ['similarity', 'accuracy'])

    st_path = Path(args.save_path, "statistics", "hyperparams")
    save_name = get_savepath(st_path, args)
    df.to_csv(st_path / save_name / "result.csv", index=False)
    with open(st_path / save_name / 'best.txt', mode='wt') as f:
        bst = df.iloc[df['accuracy'].idxmax()].to_dict()
        pprint(bst, f)
    return df


def run_train_regression(args, fixed_dataset=None, save_results=False):

    # Prepares path for saving
    pth = Path(args.save_path, "regression", "+".join([args.vect_model, args.extractor_model]))
    pth = get_savepath(pth, args)

    if fixed_dataset is not None:
        full_dataset, reg_matrices, test_dl = fixed_dataset
    else:
        full_dataset, reg_matrices, test_dl = get_data(**vars(args), dataset_type='vect', use_regression=True)

    model = get_model(set(full_dataset.ids_to_actions.keys()), full_dataset.get_vec_size(), args)
    model.regression_init(reg_matrices, regression_type='torch')

    print(f"------ TEST ------")
    test_res = evaluate(model, test_dl, use_contrasts=True)
    test_sim, test_acc = round(test_res[0].mean(dim=-1).item(), 4), round(test_res[1].mean(dim=-1).item(), 4)
    print("Test sim: ", test_sim)
    print("Test acc: ", test_acc)
    print()

    if save_results:
        os.makedirs(pth, exist_ok=True)

        with open(pth / 'config.json', mode='wt') as f:
            json.dump(vars(args), f, indent=2)

        pandas.DataFrame({
            'similarity': test_sim,
            'accuracy': test_acc,
            # TODO: what other metrics?
            # 'time': -1.0
        }).to_csv(str(pth / 'metrics.csv'), index=False)

    return test_sim, test_acc, full_dataset  # to register hold-out items


def exp_regression(args):
    tested_procedures = ['object_name', 'scene']

    args.vect_model = 'linear'
    args.use_regression = True

    st_path = Path(args.save_path, "regression")

    save_path = get_savepath(st_path, args)

    nr_samples_per_it = {k: [] for k in tested_procedures}
    df = []
    for stat_it in range(args.statistical_iterations):
        for extractor_model in ['moca-rn', 'clip-rn']:
            for hold_out_procedure in tested_procedures:
                args.hold_out_procedure = hold_out_procedure
                args.extractor_model = extractor_model

                pprint({
                    'extractor_model': args.extractor_model,
                    'hold-out': args.hold_out_procedure
                })

                # needed both to fix the dataset
                fixed_ds = get_data(**vars(args), dataset_type='vect')

                with open(save_path / 'items.txt', mode='at') as f:
                    f.write(str(fixed_ds[0].get_hold_out_items()) + "\n")

                nr_samples_per_it[hold_out_procedure].append(fixed_ds[0].get_nr_hold_out_samples())

                sim, acc, ds = run_train_regression(args, fixed_dataset=fixed_ds, save_results=False)
                df.append({
                    'hold_out_procedure': hold_out_procedure,
                    'extractor_model': extractor_model,
                    'similarity': sim,
                    'accuracy': acc,
                    'vect_model': args.vect_model
                })

    with open(save_path / 'items.txt', mode='at') as f:
        f.write(str({
            'sizes': nr_samples_per_it,
            'averages': {k: ((sum(el) / len(el)) if len(el) > 0 else 0) for k, el in nr_samples_per_it.items()}
        }))

    df = pandas.DataFrame(df, columns=['extractor_model', 'vect_model', 'hold_out_procedure', 'similarity', 'accuracy'])
    df.to_csv(save_path / 'results.csv', index=False)

    seaborn.set(font_scale=2)
    g = seaborn.catplot(x='vect_model', y='accuracy', hue='extractor_model', data=df, col="hold_out_procedure", kind="bar", height=10, aspect=.75)
    g.savefig(save_path / 'stats.png')

    # g = seaborn.catplot(hue='extractor_model', x='vect_model', y='accuracy', data=df, kind='swarm', col='hold_out_procedure')
    # g.savefig(save_path / 'observations.png')



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--exp_hold_out', action='store_true')
    parser.add_argument('--exp_fcn_hyperparams', action='store_true')
    parser.add_argument('--exp_regression', action='store_true')
    parser = setup_argparser(__default_train_config__, parser)
    args = parser.parse_args()

    if args.train:
        res = run_training(args)
    elif args.exp_fcn_hyperparams:
        res = exp_fcn_hyperparams(args)
    elif args.exp_hold_out:
        res = exp_hold_out(args)
    elif args.exp_regression:
        res = exp_regression(args)
    else:
        full_dataset, _, _, hold_dl = get_data('dataset/data-bbxs', dataset_type='vect', hold_out_procedure='object_name', hold_out_size=4, extractor_model='clip-rn')
        # model = LinearConcatVecT(set(full_dataset.ids_to_actions.keys()), full_dataset.get_vec_size(), device='cuda').to('cuda')
        # print(model)
        print(full_dataset.after_vectors.loc[0, 'vector'].shape)
        # print(evaluate(model, hold_dl, use_contrasts=True))




