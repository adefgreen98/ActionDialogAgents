"""Other functions and checks performed during this part of the project"""
import argparse
import json
import os
from pathlib import Path

import torch
import torchvision

from torchinfo import summary
from argparse import Namespace, ArgumentParser


__model_summary_path__ = 'model_summaries'


"""
MOCA evaluation call stack:
1. EvalTask [models.eval.eval_seq2seq.py] --> instance of Eval [models.eval.eval.py]
2. Eval.__init__(args) --> import_module(args.model) 
    Imports the module corresponding to the selected model (default is models.model.seq2seq_im_mask.py)
3. Module.load(args.model_path) --> retrieves the enc/dec model
4. Eval.__init__() at line 48 --> self.resnet = 'resnet18' to load visual model for preprocessing
5. EvalTask.evaluate() [models/eval/eval_task.py] at line 70 --> loading MaskRCNN with 119 classes (unspecified if pretrained)
    Later will be used for predictions by selecting only pixels of predicted class (line 127)
"""


def check_moca_maskrcnn():
    tv = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    sd = torch.load('moca_models/weight_maskrcnn.pt')
    moca = torchvision.models.detection.mask_rcnn.maskrcnn_resnet50_fpn(pretrained=False)
    moca.roi_heads.box_predictor.cls_score = torch.nn.Linear(1024, 119)
    moca.roi_heads.box_predictor.bbox_pred = torch.nn.Linear(1024, 476)
    moca.roi_heads.mask_predictor.mask_fcn_logits = torch.nn.Conv2d(256, 119, kernel_size=(1, 1), stride=(1, 1))
    moca.load_state_dict(sd)

    assert len(list(moca.named_parameters())) == len(list(tv.named_parameters()))
    for t_tv, t_moca in zip(tv.named_parameters(), moca.named_parameters()):
        name_t, p_t = t_tv
        name_m, p_m = t_moca
        assert name_m == name_t
        if p_t.shape[0] == p_m.shape[0]:
            print(name_m, torch.nonzero((p_t == p_m).int()).shape[0], "/", "torchvision", p_t.numel(), ",", "moca", p_m.numel())
        else:
            print("Size mismatch for layer", name_m)


def count_model_params(model, only_train=False):
    if only_train:
        return sum([t.numel() for t in model.parameters() if t.requires_grad])
    else:
        return sum([t.numel() for t in model.parameters()])


def compute_last_layer_size(model):
    model.eval()
    with torch.no_grad():
        tmpb = torch.randn(1, 3, 224, 224).to(model.device)
        y = model(tmpb).squeeze()
    return y.numel()


def compute_last_layer_channels(model):
    model.eval()
    with torch.no_grad():
        device = next(model.parameters()).device
        tmpb = torch.randn(1, 3, 224, 224).to(device)
        y = model(tmpb).squeeze()
    return y.shape[0]


def model_summary(model):
    return str(summary(model, input_size=(1, 3, 224, 224), depth=3, col_names=["num_params", "input_size", "output_size"]))


def write_model_summary(model, args, fname=None):
    s = model_summary(model)
    if not os.path.exists(__model_summary_path__):
        os.makedirs(__model_summary_path__, exist_ok=True)
    fname = args.model_name if fname is None else fname
    fname = os.path.join(__model_summary_path__, fname) + '.txt'
    with open(fname, mode='at') as f:
        f.write("\n{:-^100}\n".format(args.model_name.upper() + "({})".format('unfrozen' if args.unfreeze else 'frozen')))
        f.write(s)

    return s


def load_model_from_path(parent_path, nr=None, name=None, device='cuda'):
    from visual_features.detection import get_model
    from visual_features.data import __nr_bbox_target_categories__ as nr_categories

    assert (name is not None and nr is None) or (name is None and nr is not None)

    arg_pth = None
    if name is not None:
        arg_pth = Path(parent_path) / (name + '.json')
        sd_pth = Path(parent_path) / (name + '.pth')
    if nr is not None:
        arg_pth = Path(parent_path) / (str(nr) + '.json')
        sd_pth = Path(parent_path) / (str(nr) + '.pth'
                                      )
    with open(arg_pth, mode='rt') as fp:
        args = json.load(fp)
        args = Namespace(**args)
        
    model = get_model(args, nr_categories).to(args.device)
    model.load_state_dict(torch.load(sd_pth))
    model = model.to(device)
    model.device = device

    return model


def setup_argparser(config: dict, parser=None):
    """
    Sets up a parser for training configurations, containing optional arguments with default values. Allows the possibility
    to modify an existing parser by passing it by name.

    Arguments are inferred from 3 different modalities, depending
    on their shape in the config dictionary:

        - booleans ( name : boolean value ), by default False
        - arguments without choices ( name : value )
        - arguments with choices ( name : tuple(value, list(choices)) )

    Therefore each element in config should assume one of these 3 shapes.
    """
    if parser is None:
        parser = ArgumentParser()

    for name, arg in config.items():
        if type(arg) == bool:
            parser.add_argument('--{}'.format(name), action='store_true', default=arg)
        else:
            if not isinstance(arg, list) and not isinstance(arg, tuple):
                # argument with standard default and no choices --> transform in tuple
                arg = (arg, )

            if len(arg) == 1:
                parser.add_argument('--{}'.format(name), default=arg[0])
            elif len(arg) == 2:
                parser.add_argument('--{}'.format(name), default=arg[0], choices=arg[-1])
            else:
                raise ValueError(f"incorrect format for argument '{name}'")

    return parser



if __name__ == '__main__':
    check_moca_maskrcnn()