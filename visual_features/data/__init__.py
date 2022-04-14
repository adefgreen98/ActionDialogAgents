
import os
import random
import pickle

import pandas
import torch

from pathlib import Path

from torchvision import transforms
from collections import defaultdict
from PIL import Image
from torch.utils.data import Dataset, Subset
from tqdm import tqdm

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


def convert_action(name: str):
    """
    Returns action extended description from action id.
    :param name: name (in numbers) representing the action
    :return: a string describing the action
    """
    # TODO: implement action conversion table
    return ACTION_CONVERSION_TABLE[int(name)]


class ActionDataset(Dataset):
    def __init__(self,
                 path='../dataset/pickupable-held',
                 negative_sampling_method='fixed_onlyhard',
                 soft_negatives_nr=2,
                 image_extension='png',
                 transform=None,
                 **kwargs):

        self.path = Path(path)

        self.image_extension = image_extension

        # defining method for sampling negatives
        assert negative_sampling_method in NEGATIVE_SAMPLING_METHODS
        self.negative_sampling_method = negative_sampling_method
        self.soft_negatives_nr = soft_negatives_nr

        # populated by self._load_samples()
        self._scenes_objs_dict = None
        self.actions = None
        self.objects = None

        # populates dataset
        self.samples = self._load_samples()

        assert self.actions is not None and self.objects is not None

        # defines default transformation of torchvision models (if not provided)
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ]) if transform is None else transform



    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i) -> dict:
        smp = self.samples[i]
        res = {
            'positive': self.transform(Image.open(str(self.path / smp['scene'] / smp['positive']))),
            'before': self.transform(Image.open(str(self.path / smp['scene'] / smp['before']))),
            'action': smp['action'],
            'neg_actions': smp['neg_actions'],
            'negatives': [self.transform(Image.open(str(self.path / neg))) for neg in smp['negatives']]  # since negatives may be from a different scene their path should already contain it
        }
        return res

    def _load_samples(self):
        """
        Precomputes samples' dictionaries (containing: before frame path, after frame path, action id, wrong-after-frames paths, wrong-action-ids)
         by sampling negatives according to the specified method.
        Possible methods (determined by the 'negative_sampling_method' field of this class) are:
            * fixed, same environment
            * fixed, different environments plus same environment (hard distractors) (TODO)
            * randomized different environments plus fixed of same environment (TODO)
            * randomized from different scenes, while the positive is the only one from the actual scene
        """
        scenes_objs_dict = defaultdict(lambda: defaultdict(list))  # needed for fast creation of negatives
        for scene in os.listdir(self.path):
            if (self.path / scene).is_dir():
                for fname in os.listdir(self.path / scene):
                    if fname.endswith(self.image_extension):
                        scenes_objs_dict[scene][self.obj_from_path(fname)].append(fname)
        self._scenes_objs_dict = scenes_objs_dict  # needed for VecTransformDataset

        self.actions = set([self.action_from_path(name) for d in self._scenes_objs_dict.values() for v in d.values() for name in v])
        self.objects = set([obj for d in self._scenes_objs_dict.values() for obj in d.keys()])

        res = []
        for scene in os.listdir(self.path):
            if (self.path / scene).is_dir():
                for fname in os.listdir(self.path / scene):
                    if fname.endswith(self.image_extension) and self.action_from_path(fname) != '0':
                        action = self.action_from_path(fname)
                        obj = self.obj_from_path(fname)
                        before = "_".join([obj, "0"]) + '.' + self.image_extension

                        neg_paths = None
                        neg_actions = None

                        if self.negative_sampling_method == 'fixed_onlyhard':
                            neg_paths = [os.path.join(scene, neg) for neg in set(scenes_objs_dict[scene][obj]) - {before, fname}]
                            neg_actions = [self.action_from_path(neg) for neg in neg_paths]
                        elif self.negative_sampling_method == 'random_onlysoft':
                            other_samples = [os.path.join(sc, other_path) for sc, ob in scenes_objs_dict.items()
                                             for other_path in [el for obj_list in ob.values() for el in obj_list]
                                             if sc != scene]
                            neg_paths = random.sample(other_samples, k=self.soft_negatives_nr)
                            neg_actions = [__other_scene_action_id__ for _ in neg_paths]

                        res.append({
                            'scene': scene,
                            'action': action,  # TODO: convert this into sentence action
                            'before': before,
                            'positive': fname,
                            'negatives': neg_paths,
                            'neg_actions': neg_actions
                        })

        return res

    def get_filenames_dict(self):
        """Returns the dictionary associating scenes to objects, and these to all filenames containing that object."""
        return self._scenes_objs_dict

    @staticmethod
    def _scene_from_path(p) -> str:
        raise RuntimeError("this was used before migrating to scene-as-directory file structure, it is not intended to be used.")

    @staticmethod
    def obj_from_path(p) -> str:
        return str(Path(p).parts[-1])[:-4].split("_")[0]

    @staticmethod
    def action_from_path(p) -> str:
        return str(Path(p).parts[-1])[:-4].split("_")[-1]


class VecTransformDataset(Dataset):

    allowed_holdout_procedures = ['none', 'object', 'scene', 'object_type', 'sample']

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

        There are 5 possible hold out modalities:

        * None
        * Object: instances of the specified nr. of objects will be selected and excluded from each scene's training data, meaning that before and after vectors from all of their scenes will be placed in the hold-out dataset
        * Scene: all vectors from a scene will be placed in the hold-out set
        * Object type: all instances of a specific object type (e.g. 'Mug' of different colors) will be held out from all the scenes
        * Sample: specific instances of an object will be held-out by picking them from a specific scene (useful when there are more than 1 object instance per scene)

        ------------------------------------------------------------------------

        :param extractor_model: name for convolutional model that will be used for exracting features
        :param override_transform: torchvision.transform that will replace the one provided by the model
        :param hold_out_size: size of the held-out set, in terms of items selected to be excluded (respectively: object instances, scenes, object types, object instances of a specific scene)
        :param hold_out_procedure: type of hold out method used (default: None)
        :param hold_out_indices: allows specifying indices of items of the hold-out group that will be selected
        :param randomize_hold_out: whether to randomize creation of the held-out set at the first time, used in combination with hold_out_indices
        :param kwargs: keyword arguments for the internal ActionDataset structure
        """

        self._action_dataset = ActionDataset(**kwargs)
        self.extractor_model = extractor_model
        self.override_transform = override_transform
        self.path = (self._action_dataset.path / 'visual-vectors' / self.extractor_model).with_suffix('.pkl')

        if not os.path.exists(self.path):
            self.samples = self.preprocess()
        else:
            with open(self.path) as pth:
                """
                samples = {
                    'before_vectors' --> {scene --> {object --> v}},
                    'after_vectors' ---> DataFrame['scene', 'object', 'action', 'vector']
                }
                """
                self.samples = pickle.load(pth)

        self.actions = sorted(list(set(self.samples['action'].tolist())))
        self.objects = sorted(list(set(self.samples['object'].tolist())))
        self.scenes = sorted(list(set(self.samples['scene'].tolist())))

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
                'object': self.objects,
                'scene': self.scenes,
                'object_type': None,
                'sample': self.samples }[self.hold_out_procedure]

            if self.hold_out_indices is None:
                self.hold_out_indices = list(range(len(self.hold_out_item_group)))
                if randomize_hold_out:
                    random.shuffle(self.hold_out_indices)
                self.hold_out_indices = self.hold_out_indices[-self.hold_out_size:]

            # creates list of dataset row indices to exclude according to selected hold-out items
            msk = self.samples['after_vectors'][self.hold_out_procedure] in {self.hold_out_item_group[idx] for idx in self.hold_out_indices}
            self.hold_out_rows = self.samples['after_vectors'][msk].index.tolist()


    def __len__(self):
        return len(self.samples['after_vectors'])

    def __getitem__(self, item):
        after_smp = self.samples['after_vectors'][item]
        res = (
            self.samples['before_vectors'][after_smp['scene'].item()][after_smp['object'].item()],
            after_smp['vector'].item(),
            after_smp['action'].item()
        )
        return res

    def preprocess(self):
        from argparse import Namespace
        from visual_features.visual_baseline import load_model_and_transform

        extractor, transform = load_model_and_transform(Namespace(model_name=self.extractor_model, device='cuda' if torch.cuda.is_available() else 'cpu'))
        extractor.eval()

        # Overrides image transformation if needed (e.g. different normalization factors)
        if self.override_transform is not None:
            transform = self.override_transform

        # Adds tensorization if not present
        if isinstance(transform, transforms.Compose):
            if not isinstance(transform.transforms[0], transforms.ToTensor):
                transform = transforms.Compose([transforms.ToTensor(), *transform.transforms])

        # see note in __init__ for explanation
        self.samples = {
            'before_vectors': {},
            'after_vectors': []
        }

        def get_vec(s, n):
            return extractor(transform(Image.open(Path(s, n))).unsqueeze(0).to(extractor.device)).squeeze().cpu().to_numpy()

        for smp in tqdm(self._action_dataset.samples, desc=f'Extracting visual vectors using {self.extractor_model}...'):
            scene = smp['scene']
            before = smp['before']
            action = smp['action']
            positive = smp['positive']

            obj = self._action_dataset.obj_from_path(before)

            self.samples['after_vectors'].append({
                'action': action,
                'object': obj,
                'scene': scene,
                'vector': get_vec(scene, positive)
            })
            self.samples['before_vectors'][scene][obj] = get_vec(scene, before)

        self.samples['after_vectors'] = pandas.DataFrame(self.samples['after_vectors'])  # needed for holding out

        return self.samples


    def split(self, for_regression=False):
        """Splits dataset with the current hold-out settings (defined in initialization). An additional parameter
        allows controlling whether data should be prepared for regression or not.
        Returns two torch Subset objects that contain samples with the training and the held-out sets."""

        if self.hold_out_procedure == 'none':
            raise RuntimeError('trying to make the hold out set but no holding out procedure was specified')

        train_indices = list(set(range(len(self))) - set(self.hold_out_rows))
        train = Subset(self, train_indices)
        hold_out = Subset(self, self.hold_out_rows)

        if for_regression:
            col = self.samples['vector']
            train = torch.stack(col[train_indices].tolist(), )

        return train, hold_out




class BBoxDataset(torch.utils.data.Dataset):

    def __init__(self,
                 path='../dataset/bboxes/train',
                 transform=None,
                 image_extension='png',
                 object_conversion_table=None):

        self.path = Path(path)
        self.image_extension = image_extension

        self._stats = {
            'mean': torch.tensor([0.4689, 0.4682, 0.4712]),
            'std': torch.tensor([0.2060, 0.2079, 0.2052])
        }

        self.transform = transforms.ToTensor() if transform is None else transform

        self._annotations = pandas.read_csv(self.path / '..' / 'labels.csv', index_col=0)

        self._samples = []
        for s in os.listdir(self.path):
            if s.endswith(self.image_extension):
                coords = self.tensorize_bbox_from_str(self._annotations[lambda df: df['image_name'] == s]['bbox'].item())
                if coords[0] >= coords[2] or coords[1] >= coords[3]:
                    pass  # invalid coordinates
                else:
                    self._samples.append(s)

        self.objs_to_ids = {
                name: i for i, name in enumerate(sorted(list({row['object_id'].split("|")[0] for _, row in self._annotations.iterrows()})))
        } if object_conversion_table is None else object_conversion_table

        self.ids_to_objs = {v: k for k, v in self.objs_to_ids.items()}

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, item):
        smp = self._samples[item]
        ann = self._annotations[lambda df: df['image_name'] == smp]
        obj_id = self.convert_object_to_ids(self.obj_name_from_df_row(ann))

        img = self.transform(Image.open(self.path / smp))
        boxes = self.tensorize_bbox_from_str(ann['bbox'].item()).unsqueeze(0)
        # convert to COCO annotation
        ann = {
            'image_id': torch.tensor([item]),
            'boxes': boxes,
            'labels': torch.tensor([obj_id]),
            'area': (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
            'iscrowd': torch.zeros(1, dtype=torch.int64)
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
        return (row['object_id']).item().split("|")[0]

    @classmethod
    def tensorize_bbox_from_str(cls, s):
        return torch.tensor([int(el) for el in s.strip('()').split(',')], dtype=float)

    def retrieve_object_set_from_csv(self):
        return {self.obj_name_from_df_row(row) for row in pandas.read_csv(self.path / ".." / "labels.csv").iterrows()}

    def get_stats(self):
        if self._stats is None:
            acc = torch.stack([self[i] for i in range(len(self))], dim=0).view(3,-1)
            self._stats = {'mean': acc.mean(dim=-1), 'std': acc.std(dim=-1)}

        return self._stats

    def image_name_from_idx(self, i):
        return self._samples[i]


def get_data(data_path, batch_size=32, dataset_type=None, obj_dict=None, transform=None, **kwargs):
    if dataset_type == 'bboxes':
        # Transforms should be not none because FastRCNN require PIL images
        train_set = BBoxDataset(os.path.join(data_path, 'train'), transform=transform, object_conversion_table=obj_dict)
        valid_set = BBoxDataset(os.path.join(data_path, 'valid'), transform=transform, object_conversion_table=obj_dict)
        test_set = BBoxDataset(os.path.join(data_path, 'test'), transform=transform, object_conversion_table=obj_dict)
        dataset = {'train': train_set, 'valid': valid_set, 'test': test_set}
        train_dl = torch.utils.data.DataLoader(train_set, batch_size=batch_size, collate_fn=default_collate_fn)
        valid_dl = torch.utils.data.DataLoader(valid_set, batch_size=1, collate_fn=default_collate_fn)
        return dataset, train_dl, valid_dl
    else:
        raise ValueError(f"unsupported type of dataset '{dataset_type}'")



if __name__ == "__main__":
    from pprint import pprint

    ds = BBoxDataset('../bboxes/train')
    pprint(ds[0])
    pprint(ds[0][0].shape)

