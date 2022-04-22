
import os
import random
import pickle

import pandas
import torch

from pathlib import Path

from matplotlib import use
from torchvision import transforms
from collections import defaultdict
from PIL import Image
from torch.utils.data import Dataset, Subset
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm

import sys
sys.path.extend(['..', '.'])

from visual_features.vision_helper.utils import collate_fn as default_collate_fn

NEGATIVE_SAMPLING_METHODS = [
    'fixed_onlyhard',
    'random_onlysoft',
    # 'fixed_soft+hard', 'random_soft+hard'
]

# TODO: make the following two dynamic and loaded from an external file

__other_scene_action_id__ = 100
ACTION_CONVERSION_TABLE = {
    1: 'drop', 2: 'throw', 3: 'put', 4: 'push', 5: 'pull', 6: 'open', 7: 'close',
    8: 'slice', 9: 'dirty', 10: 'fill', 11: 'empty',
    12: 'toggle', 13: 'useUp', 14: 'break', 15: 'cook',
    __other_scene_action_id__: 'other'  # needed for soft negatives
}

# here not to be used dynamically but only as a reference
__bbox_target_categories__ = {'CellPhone', 'Pen', 'Towel', 'Candle', 'SoapBar', 'Footstool', 'BaseballBat', 'WateringCan',
                         'SoapBottle', 'Egg', 'DishSponge', 'Book', 'HandTowel', 'Ladle', 'Pencil', 'Plunger', 'Kettle',
                         'Lettuce', 'TeddyBear', 'TableTopDecor', 'Box', 'Bowl', 'AluminumFoil', 'Plate', 'Pillow',
                         'Vase', 'Mug', 'Pan', 'Pot', 'RemoteControl', 'KeyChain', 'SaltShaker', 'SprayBottle', 'Cup',
                         'TennisRacket', 'Boots', 'Bread', 'Bottle', 'Knife', 'CD', 'Potato', 'Tomato', 'Newspaper',
                         'Watch', 'CreditCard', 'Dumbbell', 'ButterKnife', 'TissueBox', 'Statue', 'AlarmClock',
                         'Spatula', 'ToiletPaper', 'Cloth', 'ScrubBrush', 'Fork', 'Laptop', 'BasketBall', 'WineBottle',
                         'PepperShaker', 'Spoon', 'Apple', 'PaperTowelRoll'}

__nr_bbox_target_categories__ = len(__bbox_target_categories__)


__default_dataset_stats__ = {
    'mean': torch.tensor([0.4689, 0.4682, 0.4712]),
    'std': torch.tensor([0.2060, 0.2079, 0.2052])
}  # to rework with full dataset


def convert_action(name: str):
    """
    Returns action extended description from action id.
    :param name: name (in numbers) representing the action
    :return: a string describing the action
    """
    # TODO: implement action conversion table
    return ACTION_CONVERSION_TABLE[int(name)]


def rework_annotations_path(df, basepath):
    # TODO: automatic handling of different path specifications
    df['after_image_path'] = df['after_image_path'].map(lambda pth: str(Path(basepath) / '../..' / pth))
    df['before_image_path'] = df['before_image_path'].map(lambda pth: str(Path(basepath) / '../..' / pth))
    return df


class ActionDataset(Dataset):
    def __init__(self,
                 path='dataset/data-bbxs/pickupable-held',
                 negative_sampling_method='fixed_onlyhard',
                 soft_negatives_nr=2,
                 image_extension='png',
                 transform=None,
                 **kwargs):

        self.path = Path(path)

        self._stats = __default_dataset_stats__

        # TODO: drop the (train-)split column and select which (train-)split to use
        # annotations dataframe
        self.annotation = pandas.read_csv(self.path / '..' / 'bbox-data.csv', index_col=0)
        self.annotation = rework_annotations_path(self.annotation, self.path)
        self.annotation.index = pandas.Series(range(len(self.annotation)))  # rework index for consistency

        self.image_extension = image_extension

        # defining method for sampling negatives
        assert negative_sampling_method in NEGATIVE_SAMPLING_METHODS
        self.negative_sampling_method = negative_sampling_method
        self.soft_negatives_nr = soft_negatives_nr

        # TODO: some action indices are missing (8, 13), so I had to workaround by using a new enumeration
        # self.actions_map = dict(set(zip(self.annotation['action_name'].to_list(), self.annotation['action_id'].to_list())))
        self.actions_map = {ac: i for i, ac in enumerate(list(set(self.annotation['action_name'].to_list())))}

        self.objects = dict(set(zip(
            self.annotation['object_name'].to_list(),
            [int(self.annotation.loc[i, 'image_name'].split("_")[0]) for i in range(len(self.annotation))]
        )))

        # populates contrasts
        # self.samples = self._load_samples()
        self._load_negatives()

        # defines default transformation of torchvision models (if not provided)
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ]) if transform is None else transform

    def __len__(self):
        return len(self.annotation)

    def _read_image_from_annotation_path(self, path: str):
        pth = self.path / Path(*Path(path).parts[-2:])
        return Image.open(pth)

    def __getitem__(self, i) -> dict:
        # smp = self.samples[i]
        # res = {
        #     'positive': self.transform(Image.open(str(self.path / smp['scene'] / smp['positive']))),
        #     'before': self.transform(Image.open(str(self.path / smp['scene'] / smp['before']))),
        #     'action': smp['action'],
        #     'neg_actions': smp['neg_actions'],
        #     'negatives': [self.transform(Image.open(str(self.path / neg))) for neg in smp['negatives']]  # since negatives may be from a different scene their path should already contain it
        # }
        smp = self.annotation.iloc[i]
        res = {
            'positive': self.transform(self._read_image_from_annotation_path(smp['after_image_path'])),
            'before': self.transform(self._read_image_from_annotation_path(smp['before_image_path'])),
            'action': self.actions_map[smp['action_name']],
            'neg_actions': [self.actions_map[self.annotation.loc[neg_i, 'action_name']] for neg_i in smp['contrast']],
            'negatives': [self.transform(self._read_image_from_annotation_path(self.annotation.loc[neg_i, 'after_image_path'])) for neg_i in smp['contrast'][0]],
        }
        return res

    def _load_negatives(self):
        """Computes contrast sets by adding a new column ('contrast') to the annotation DataFrame.

        Possible methods (determined by the 'negative_sampling_method' field of this class) are:
        * fixed, same environment
        * fixed, different environments plus same environment (hard distractors) (TODO)
        * randomized different environments plus fixed of same environment (TODO)
        * randomized from different scenes, while the positive is the only one from the actual scene
        """
        self.annotation['contrast'] = None
        if self.negative_sampling_method == 'fixed_onlyhard':
            contrasts = [
                self.annotation[lambda df: df['before_image_path'] == before_img].index.to_list()
                for before_img in set(self.annotation['before_image_path'].to_list())
            ]
            for c in contrasts:
                for img_idx in c:
                    self.annotation.at[img_idx, 'contrast'] = list(set(c) - {img_idx})
        elif self.negative_sampling_method == 'random_onlysoft':
            indices = self.annotation.index.to_list()
            for i in range(len(self.annotation)):
                c = []
                while len(c) < self.soft_negatives_nr:
                    neg_i = random.sample(indices, k=1)
                    neg_smp = self.annotation[neg_i]
                    if neg_i != i and \
                            neg_smp['contrast_set'] == self.annotation.iloc[i]['contrast_set'] and \
                            neg_smp['scene'] != self.annotation.iloc[i]['scene']:
                        c.append(neg_i)
                self.annotation.at[i, 'contrast'] = c

    @staticmethod
    def _scene_from_path(p) -> str:
        raise RuntimeError("this was used before migrating to scene-as-directory file structure, it is not intended to be used.")

    @staticmethod
    def obj_from_path(p) -> str:
        return str(Path(p).parts[-1])[:-4].split("_")[0]

    @staticmethod
    def action_from_path(p) -> str:
        return str(Path(p).parts[-1])[:-4].split("_")[-1]

    def get_stats(self):
        if self._stats is None:
            tmp = self.annotation.copy()
            tmp[lambda df: df['after_image_path'].isnull()]['after_image_path'] = tmp[lambda df: df['after_image_path'].isnull()]['before_image_path']
            acc = torch.stack([
                to_tensor(Image.open(pth))
                for pth in set(self.annotation['after_image_path'].tolist() + self.annotation['before_image_path'])
            ]).view(3, -1)
            self._stats = {'mean': acc.mean(dim=-1), 'std': acc.std(dim=-1)}

        return self._stats


class VecTransformDataset(Dataset):

    allowed_holdout_procedures = ['none', 'object_name', 'scene', 'object_type', 'sample']

    # NOTE: for regression matrix, each visual vector in float32 takes
    #                   0.000374510884 GB   ((1 * 100352 * 4) / 2^10)
    # of memory. Supposing 4800 samples (which are very few) we end up in occupying ~1.8 GB.

    def __init__(self,
                 extractor_model='moca-rn',
                 override_transform=None,
                 hold_out_size=1,
                 hold_out_procedure='none',
                 hold_out_indices=None,
                 randomize_hold_out=True,
                 **kwargs):
        """
        Initializes the dataset for vector transformation. Eventually, this also extracts visual vectors, if these
        are not present at the specified path (as additional keyword argument).

        There are 5 possible hold out modalities (with their string code in brackets):

        * None [none]
        * Object [object_name]: instances of the specified nr. of objects will be selected and excluded from each scene's training data, meaning that before and after vectors from all of their scenes will be placed in the hold-out dataset
        * Scene [scene]: all vectors from a scene will be placed in the hold-out set
        * Object type [object_type]: all instances of a specific object type (e.g. 'Mug' of different colors) will be held out from all the scenes
        * Sample [sample]: specific instances of an object will be held-out by picking them from a specific scene (useful when there are more than 1 object instance per scene)

        ------------------------------------------------------------------------

        :param extractor_model: name for convolutional model that will be used for exracting features
        :param override_transform: torchvision.transform that will replace the one provided by the model (default: None); it can also be a string "to_tensor", automatically initializing it as a ToTensor transform.
        :param hold_out_size: size of the held-out set, in terms of items selected to be excluded (respectively: object instances, scenes, object types, object instances of a specific scene)
        :param hold_out_procedure: type of hold out method used (default: None)
        :param hold_out_indices: allows specifying indices of items of the hold-out group that will be selected
        :param randomize_hold_out: whether to randomize creation of the held-out set at the first time, used in combination with hold_out_indices
        :param use_contrast_set: decide to keep and use contrast sets of the internal ActionDataset (negative sampling can be controlled with additional keyword arguments)
        :param kwargs: keyword arguments for the internal ActionDataset structure
        """

        hold_out_size = int(hold_out_size)

        self._action_dataset = ActionDataset(**kwargs, transform=transforms.ToTensor())
        self.extractor_model = extractor_model

        self.path = (self._action_dataset.path / 'visual-vectors' / self.extractor_model)

        self.actions_to_ids = self._action_dataset.actions_map
        self.ids_to_actions = {v: k for k, v in self.actions_to_ids.items()}

        if isinstance(override_transform, str):
            if override_transform == 'to_tensor':
                self.override_transform = transforms.ToTensor()
        else:
            self.override_transform = override_transform

        if not os.path.exists(self.path):
            os.makedirs(self.path, exist_ok=True)
            self.before_vectors, self.after_vectors = self.preprocess()
            with open(self.path / 'before.pkl', mode='wb') as bpth:
                pickle.dump(self.before_vectors, bpth)
            with open(self.path / 'after.pkl', mode='wb') as apth:
                pickle.dump(self.after_vectors, apth)
        else:
            """
            samples = {
                'before_vectors' --> {before_path --> v}},
                'after_vectors' ---> DataFrame['scene', 'object', 'action', 'vector'] (+ other columns from ActionDataset)
            }
            """
            with open(self.path / 'before.pkl', mode='rb') as bpth:
                self.before_vectors = pickle.load(bpth)

            with open(self.path / 'after.pkl', mode='rb') as apth:
                self.after_vectors = pickle.load(apth)

        assert hold_out_procedure in self.allowed_holdout_procedures
        self.hold_out_procedure = hold_out_procedure

        if self.hold_out_procedure != 'none':
            # only supports randomization when no predefined indices are selected
            assert (randomize_hold_out and (hold_out_indices is None)) or \
                   ((not randomize_hold_out) and (hold_out_indices is not None))
            self.hold_out_indices = hold_out_indices

            assert hold_out_size > 0
            self.hold_out_size = hold_out_size

            # prepares the set where to choose hold-out items (according to hold-out methods)
            self.hold_out_item_group = {
                'object_name': list(set(self._action_dataset.objects.keys())),
                'scene': list(set(self._action_dataset.annotation['scene'].to_list())),
                'object_type': None,
                'sample': self.after_vectors.index.to_list()
            }[self.hold_out_procedure]

            if self.hold_out_indices is None:
                self.hold_out_indices = list(range(len(self.hold_out_item_group)))
                if randomize_hold_out:
                    random.shuffle(self.hold_out_indices)
                self.hold_out_indices = self.hold_out_indices[-self.hold_out_size:]

            # creates list of dataset row indices to exclude according to selected hold-out items
            # msk = self._action_dataset.annotation[self.hold_out_procedure].map(lambda el: el in {self.hold_out_item_group[idx] for idx in self.hold_out_indices})
            msk = self._action_dataset.annotation[self.hold_out_procedure].isin({self.hold_out_item_group[idx] for idx in self.hold_out_indices})
            self.train_rows = self.after_vectors[~msk].index
            self.hold_out_rows = self.after_vectors[msk].index
        else:
            self.hold_out_size = 0
            self.hold_out_rows = pandas.Series([])


    def __len__(self):
        return len(self.after_vectors)

    def __getitem__(self, item):
        after_smp = self.after_vectors.iloc[item]
        res = {
            'before': self.before_vectors[after_smp['before_image_path']],
            'action': self.actions_to_ids[after_smp['action_name']],  # TODO fix issue in ActionDataset to solve also here
            'positive': after_smp['vector'],
            'negatives': [self.after_vectors.loc[neg_i, 'vector'] for neg_i in after_smp['contrast']],
            'neg_actions': [self.actions_to_ids[self.after_vectors.loc[neg_i, 'action_name']] for neg_i in after_smp['contrast']]
        }
        return res

    def preprocess(self):
        from argparse import Namespace
        from visual_features.visual_baseline import load_model_and_transform

        extractor, transform = load_model_and_transform(
            Namespace(model_name=self.extractor_model, device='cuda' if torch.cuda.is_available() else 'cpu'),
            keep_pooling=self.extractor_model != 'clip-rn',
            add_flatten=False
        )
        extractor.eval()

        # adjusts transform with dataset statistics
        if isinstance(transform, transforms.Compose):
            for t in transform.transforms:
                if isinstance(t, transforms.Normalize):
                    t.mean = self._action_dataset.get_stats()['mean']
                    t.std = self._action_dataset.get_stats()['std']

        # excludes 'contrast' column from the action categorization task and adds the 'vector' column
        # self.after_vectors = self._action_dataset.annotation[[c for c in self._action_dataset.annotation.columns if c not in {'contrast'}]].copy()

        self.after_vectors = self._action_dataset.annotation.copy()  # also include 'contrast' column (containing contrast set for each sample)
        self.before_vectors = None

        def get_vec(pth):
            return extractor(transform(Image.open(pth)).unsqueeze(0).to(extractor.device)).squeeze().cpu()

        with torch.no_grad():
            self.after_vectors['vector'] = pandas.Series([
                get_vec(pth)
                for pth in tqdm(self.after_vectors['after_image_path'], desc=f"Extracting 'after' visual vectors using {self.extractor_model}...")
            ])
            self.before_vectors = {
                bf_path: get_vec(bf_path) for bf_path in tqdm(set(self.after_vectors['before_image_path'].to_list()), desc=f"Extracting 'before' visual vectors using {self.extractor_model}...")
            }

        return self.before_vectors, self.after_vectors

    def split(self, valid_ratio=-1.0, for_regression=False):
        """Splits dataset with the current hold-out settings (defined in initialization). An additional parameter
        allows controlling whether data should be prepared for regression or not.
        Returns two torch Subset objects that contain samples with the training and the held-out sets."""

        if self.hold_out_procedure == 'none':
            raise RuntimeError('trying to make the hold out set but no holding out procedure was specified')

        hold_out = Subset(self, self.hold_out_rows.to_list())
        if not for_regression:
            assert (0 < valid_ratio < 1) or valid_ratio == -1.0, "choose for validation ratio a value in (0, 1)"
            train = Subset(self, self.train_rows.to_list())
            valid_indices = []
            if valid_ratio > 0:
                sep = int(len(train) * (1 - valid_ratio))
                train_indices = list(range(len(train)))[:sep]
                valid_indices = list(range(len(train)))[sep:]

                train_set = Subset(train, train_indices)
                valid_set = Subset(train, valid_indices)
                return train_set, valid_set, hold_out
            else:
                valid = Subset(train, [])
                return train, valid, hold_out
        else:
            train_after_df = self.after_vectors.iloc[self.train_rows]
            train_after = {action: [] for action in set(self.actions_to_ids.values())}
            train_before = {action: [] for action in set(self.actions_to_ids.values())}
            for after_row in train_after_df.iterrows():
                train_after[self.actions_to_ids[after_row['action_name']]].append(after_row['vector'])
                train_before[self.actions_to_ids[after_row['action_name']]].append(self.before_vectors[after_row['before_image_path']])

            train_after = {action: torch.stack(train_after[action], dim=0) for action in train_after}
            train_before = {action: torch.stack(train_before[action], dim=0) for action in train_after}
            return (train_before, train_after), hold_out

    def get_vec_size(self):
        return self.after_vectors.loc[0, 'vector'].shape[-1]


class BBoxDataset(torch.utils.data.Dataset):

    def __init__(self,
                 path='dataset/data-bbxs/pickupable-held',
                 transform=None,
                 image_extension='png',
                 object_conversion_table=None):

        # TODO: add split information and separate train/valid/test splits
        # TODO: add info about pickupable vs non-pickupable (up to now it is supposed to be either one of the two)
        self.path = Path(path)
        self.image_extension = image_extension

        # TODO: update these values with newer ones
        self._stats = __default_dataset_stats__

        self.transform = transforms.ToTensor() if transform is None else transform

        self._annotations = pandas.read_csv(self.path / '..' / 'bbox-data.csv', index_col=0)
        self._annotations = rework_annotations_path(self._annotations, self.path)

        added_before_samples = {}
        for idx, row in self._annotations.iterrows():
            if added_before_samples.get(row['before_image_path'], None) is None:
                tmp = row.copy()
                tmp['after_image_path'] = None
                tmp['after_image_bb'] = None
                tmp['image_name'] = tmp['before_image_name']  # needed for image_name_from_idx
                added_before_samples[row['before_image_path']] = tmp

        self._annotations = self._annotations.append(list(added_before_samples.values()), ignore_index=True)
        self._annotations = self._annotations.sample(frac=1.0, random_state=42)  # just needed to shuffle before and after rows, more stochasticity may come in DataLoaders

        self.objs_to_ids = {name: i for i, name in enumerate(list(set(self._annotations['object_name'])))} if object_conversion_table is None else object_conversion_table

        self.ids_to_objs = {v: k for k, v in self.objs_to_ids.items()}

        self.excluded_files = []
        for i, row in self._annotations.iterrows():
            name = row['after_image_path'] if row['after_image_path'] is not None else row['before_image_path']
            boxes = row['after_image_bb'] if row['after_image_bb'] is not None else row['before_image_bb']
            boxes = self.tensorize_bbox_from_str(boxes)
            if not torch.logical_and(boxes[0] < boxes[2], boxes[1] < boxes[3]):
                self.excluded_files.append(name)
                self._annotations.drop(labels=i, axis=0, inplace=True)

        self._annotations.index = pandas.Series(range(len(self._annotations)))  # rework for better consistency

    def __len__(self):
        return len(self._annotations)

    def __getitem__(self, item):
        smp = self._annotations.iloc[item]
        if smp['after_image_path'] is None:
            # is a before image
            imgpath = smp['before_image_path']
            obj_id = self.convert_object_to_ids(self.obj_name_from_df_row(smp))
            boxes = self.tensorize_bbox_from_str(smp['before_image_bb']).unsqueeze(0)
        else:
            imgpath = smp['after_image_path']
            obj_id = self.convert_object_to_ids(self.obj_name_from_df_row(smp))
            boxes = self.tensorize_bbox_from_str(smp['after_image_bb']).unsqueeze(0)

        assert torch.logical_and(boxes[:, 0] < boxes[:, 2], boxes[:, 1] < boxes[:, 3]), \
            f"wrong box for image {imgpath} ({boxes.int().squeeze().tolist()})"

        img = self.transform(Image.open(imgpath))

        # convert to COCO annotation
        ann = {
            'image_id': torch.tensor([item]),
            'boxes': boxes,
            'labels': torch.tensor([obj_id]),
            'area': (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
            'iscrowd': torch.zeros(1, dtype=torch.int)
        }

        return img, ann

    def convert_object_to_ids(self, obj_name):
        return self.objs_to_ids[obj_name]

    def convert_id_to_obj(self, _id):
        return self.ids_to_objs[_id]

    def get_object_set(self):
        return set(self.objs_to_ids.keys())

    @classmethod
    def obj_name_from_df_row(cls, row):
        return row['object_name']

    @classmethod
    def tensorize_bbox_from_str(cls, s):
        return torch.tensor([int(el) for el in s.strip('()').split(',')], dtype=torch.float16)

    def get_stats(self):
        if self._stats is None:
            tmp = self._annotations.copy()
            tmp[lambda df: df['after_image_path'].isnull()]['after_image_path'] = tmp[lambda df: df['after_image_path'].isnull()]['before_image_path']
            acc = torch.stack([to_tensor(Image.open(pth)) for pth in tmp['after_image_path']]).view(3, -1)
            self._stats = {'mean': acc.mean(dim=-1), 'std': acc.std(dim=-1)}

        return self._stats

    def image_name_from_idx(self, i):
        return self._annotations.loc[i, 'image_name']


# Other utilities
def vect_collate(batch):
    """
    :param batch: list of tuples (before-tensor, action-id, after-tensor)
    :return: a tuple (stacked inputs, stacked actions, stacked outputs)
    """
    before = torch.stack([batch[i]['before'] for i in range(len(batch))], dim=0)
    actions = torch.tensor([batch[i]['action'] for i in range(len(batch))], dtype=torch.long)
    after = torch.stack([batch[i]['positive'] for i in range(len(batch))], dim=0)
    return before, actions, after


def get_data(data_path, batch_size=32, dataset_type=None, obj_dict=None, transform=None, valid_ratio=0.2, **kwargs):
    if dataset_type == 'bboxes':
        # Transforms should be not none because FastRCNN require PIL images
        dataset = BBoxDataset(data_path, transform=transform, object_conversion_table=obj_dict)
        indices = list(range(len(dataset)))
        sep = int(len(dataset) * (1 - valid_ratio))
        train_set = Subset(dataset, indices[:sep])
        valid_set = Subset(dataset, indices[sep:])
        train_dl = torch.utils.data.DataLoader(train_set, batch_size=batch_size, collate_fn=default_collate_fn)
        valid_dl = torch.utils.data.DataLoader(valid_set, batch_size=1, collate_fn=default_collate_fn)
        test_set = valid_set
    elif dataset_type == 'actions':
        raise NotImplementedError
    elif dataset_type == 'vect':
        dataset = VecTransformDataset(path=data_path, override_transform=transform, **kwargs)
        if dataset.hold_out_procedure == 'none':
            indices = list(range(len(dataset)))
            sep = int(len(dataset) * (1 - valid_ratio))
            train_set = Subset(dataset, indices[:sep])
            valid_set = Subset(dataset, indices[sep:])
            test_set = valid_set
        else:
            train_set, valid_set, test_set = dataset.split(valid_ratio=valid_ratio, for_regression=kwargs.get('use_regression', False))
        train_dl = torch.utils.data.DataLoader(train_set, batch_size=batch_size, collate_fn=vect_collate)
        valid_dl = torch.utils.data.DataLoader(valid_set, batch_size=1, collate_fn=vect_collate)
        test_dl = torch.utils.data.DataLoader(test_set, batch_size=1, collate_fn=vect_collate)  # here for compatibility, but dataset can always be accessed with test_dl.dataset
    else:
        raise ValueError(f"unsupported type of dataset '{dataset_type}'")
    return dataset, train_dl, valid_dl, test_dl


if __name__ == "__main__":
    from pprint import pprint
    from visual_features.data import *
    ds = VecTransformDataset()
    pprint(ds[0])
