import io
import json
import os
import pickle
import random
from typing import Optional

import numpy as np
import torch
from PIL import Image
from pytorch_lightning import LightningDataModule
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from utils.zipreader import ZipReader


def mu_law_encode(x, mu=1, m=0):
    sign = torch.sign(x)
    numerator = torch.log(torch.abs(x) * mu + 1.0)

    # denominator = torch.log(torch.tensor())
    return (numerator / (m * mu + 1.0)) * sign


class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p

    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)


def tokenize_continuous_values(x, mu=1.7, m=0, bins=1024, shift=None):
    # > Finally, they are discretized using bins of uniform width on the domain [-1, 1].
    c = mu_law_encode(x, mu, m)
    # > We use 256 bins and shift the resulting integers
    # > so they are not overlapping with the ones used for text tokens.
    c = (c + 1) * (bins / 2)

    c = c.int()
    if shift is not None:
        c += shift
    return c


def mu_law_decode(x, mu=1, m=0):
    sign = torch.sign(x)
    numerator = (torch.exp((x / sign) * (m * mu + 1.0)) - 1.0) / mu
    return numerator * sign


def decode_action_tokenize(x, mu=1.7, m=0, bins=1024, shift=None):
    if shift is not None:
        c = x - shift
    c = (c / (bins / 2.0)) - 1.0
    c = mu_law_decode(c, mu, m)
    c = torch.where(
        torch.isnan(c),
        torch.full_like(c, 0),
        c)
    return c


class HabitatDataset(Dataset):
    """ Imitation learning (IL) video dataset in Habitat RGB FPV for ImageNav & PointNav &  Objectnav. """

    # def __init__(self, dataset_dir='/mnt/shuang/Data/gibson_data/data', ann_file_path='pointnav_gib_mp_13300.json' , transform=DataAugmentationForMultiMAE(), context_length=6, max_clips=-1, mod='rgb',
    #              max_count=-1, data_mode=None):
    def __init__(self, dataset_dir, ann_file_path, transform, context_length=6, max_clips=-1, mod='rgb',
                 max_count=-1, data_mode=None, cont_action=False, action_bins=256):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.context_length = context_length
        self.max_clips = max_clips
        self.data_mode = data_mode
        self.json_path = os.path.join(dataset_dir, ann_file_path)
        self.clip_indices = []
        self.mod = mod
        self.max_count = max_count
        self.transform = transform
        self.cont_action = cont_action
        self.action_bins = action_bins
        self.construct_data()
        if self.max_clips != -1:
            self.clip_indices = self.clip_indices[:self.max_clips]
        # while len(self.clip_indices)<2000*64:
        #     self.clip_indices+= self.clip_indices

    def construct_data(self):
        with open(self.json_path, 'r') as f:
            data = json.load(f)['data']
        # print(self.json_path)
        # data_slice content: 
        # d[0] eposides name
        # d[1] eposides length 
        # d[2] is_success
        # d[3] task_type
        # d[4] task describe

        for data_slice in data:
            zip_file = data_slice[0].replace('/home/COMPASS-user/workspace/blobfuse/shuang/Data/gibson_data/data/', '')
            zip_file = os.path.join(self.dataset_dir, zip_file)
            video_len = data_slice[1] - 1
            if self.context_length <= video_len <= 500:
                for start_frame_index in range(0, video_len):
                    frame_index = min(start_frame_index + self.context_length, video_len) - self.context_length
                    self.clip_indices.append(
                        (zip_file, frame_index, video_len, 1, 'pointnav'))
                    if start_frame_index + self.context_length > video_len:
                        break

    def __len__(self):

        return len(self.clip_indices)

    def __getitem__(self, index):

        zip_file, start_frame_index, video_len, task_type, task_desc = self.clip_indices[index]
        ann = json.load(io.BytesIO(
            ZipReader.read(zip_file, 'data.json')))
        for video_name in ann:
            frames = ann[video_name]['data']
        image_goal = self.transform(
            Image.open(io.BytesIO(ZipReader.read(zip_file, os.path.join('rgb', str(video_len - 1) + ".png")))))
        object_goal = frames[0]['objectgoal']

        acts = []
        pg_goals = []
        gps_goals = []
        # object_goals=[]
        timesteps = []
        rgb_list = []
        # depth_list = []
        reward = []

        pg_goal = frames[0]['point_goal']
        for frame_index in range(start_frame_index, start_frame_index + self.context_length):
            frame = frames[frame_index]
            action = frames[frame_index]['action']
            rgb = self.transform(Image.open(
                io.BytesIO(ZipReader.read(zip_file, os.path.join('rgb', str(frame_index) + ".png")))))
            rw = 0
            reward.append(np.array(rw))

            rgb_list.append(rgb)
            gps_goal = frame['gps_goal']
            gps_goal = np.array(gps_goal, dtype=np.float32)
            gps_goals.append(gps_goal)
            times = frame['img_number']

            act = np.array(action, dtype=int)
            acts.append(act)
            times = np.array(times, dtype=int)
            timesteps.append(times)

        rgbs = torch.stack(rgb_list, dim=0)
        # dps=[]
        # if self.mod == 'rgbd':
        #     dps = torch.stack(depth_list, dim=0)

        timesteps_seq = [torch.from_numpy(item) for item in timesteps]
        timesteps_seq = torch.stack(timesteps_seq, dim=0).float()
        act_seq = [torch.from_numpy(item) for item in acts]
        act_seq = torch.stack(act_seq, dim=0)

        if self.cont_action:
            _act = torch.LongTensor(act_seq).reshape(-1, 1)
            # print(_act)
            act_seq = torch.zeros(self.context_length, 10).scatter_(1, _act, 1).float()
        else:
            act_seq = (act_seq - 0) / 5.0
            act_seq = act_seq.reshape(self.context_length, -1)
            act_seq = torch.cat([act_seq, torch.zeros((self.context_length, 4))], 1)
            act_seq = tokenize_continuous_values(act_seq, bins=self.action_bins)
        # obj_goal_seq = torch.tensor(object_goals)
        # obj_goal_seq = torch.stack(obj_goal_seq, dim=0)
        gps_goal_seq = [torch.from_numpy(item) for item in gps_goals]
        gps_goal_seq = torch.stack(gps_goal_seq, dim=0).float()
        reward_seq = [torch.from_numpy(item) for item in reward]
        reward_seq = torch.stack(reward_seq, dim=0).float()
        obss_seq = torch.zeros((timesteps_seq.shape[0], 44))
        pg_goal = torch.from_numpy(np.array(pg_goal))
        # _item = {'rgb': rgbs, 'depth': dps, 'act': act_seq, 'image_goal': image_goal,
        #         "gps_goal": gps_goal_seq, 'pg_goal': pg_goal_seq, 'timesteps': timesteps_seq}
        # return rgbs,obss_seq,act_seq,timesteps_seq,image_goal,None,None,None

        return rgbs, obss_seq, act_seq, timesteps_seq, image_goal, gps_goal_seq, pg_goal, object_goal, task_type, 'habitat', reward_seq
    # except (Exception) as e:
    #     print(e)
    #     print(self.clip_indices[index])
    #     self.__getitem__(index+1)


class MetaworldDataset(Dataset):
    """ Imitation learning (IL) video dataset in Habitat RGB FPV for ImageNav & PointNav &  Objectnav. """

    # def __init__(self, dataset_dir='/mnt/shuang/Data/gibson_data/data', ann_file_path='pointnav_gib_mp_13300.json' , transform=DataAugmentationForMultiMAE(), context_length=6, max_clips=-1, mod='rgb',
    #              max_count=-1, data_mode=None):
    def __init__(self, dataset_dir, ann_file_path, transform, context_length=6, max_clips=-1, mod='corner2',
                 max_count=-1, data_mode=None, cont_action=False, action_bins=256):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.context_length = context_length
        self.max_clips = max_clips
        self.data_mode = data_mode
        self.json_path = os.path.join(dataset_dir, ann_file_path)
        self.clip_indices = []
        self.mod = mod
        self.max_count = max_count
        self.transform = transform
        self.cont_action = cont_action
        self.action_bins = action_bins
        self.construct_data()
        if self.max_clips != -1:
            self.clip_indices = self.clip_indices[:self.max_clips]
        while len(self.clip_indices) < 2000 * 64:
            self.clip_indices += self.clip_indices

    def construct_data(self):
        with open(self.json_path, 'r') as f:
            data = json.load(f)

        for data_slice in data:
            zip_file = data_slice[0]

            zip_file = os.path.join(self.dataset_dir, 'metaworld_data', zip_file)
            video_len = data_slice[1]
            if self.context_length <= video_len <= 500:

                for start_frame_index in range(0, video_len):
                    frame_index = min(start_frame_index + self.context_length, video_len) - self.context_length
                    self.clip_indices.append(
                        (zip_file, frame_index, video_len, data_slice[3], data_slice[5]))
                    if start_frame_index + self.context_length > video_len:
                        break

    def __len__(self):

        return len(self.clip_indices)

    def get_rtgs(self, frames):
        rtgs = []
        rtg = 0
        for f in frames:
            rtgs.append(rtg)
            rtg += f['reward']
        return [rtg - r for r in rtgs]

    def __getitem__(self, index):
        # try:
        zip_file, start_frame_index, video_len, task_type, task_desc = self.clip_indices[index]
        frames = pickle.load(io.BytesIO(
            ZipReader.read(zip_file, 'param/param.pickle')))

        image_goal = self.transform(Image.open(io.BytesIO(
            ZipReader.read(zip_file, os.path.join('rgb', self.mod, 'image_save_' + str(video_len - 1) + ".jpg")))))

        acts = []
        obss = []
        timesteps = []
        rgb_list = []
        reward = []
        rtgs = self.get_rtgs(frames)
        # print(rtgs)
        for frame_index in range(start_frame_index, start_frame_index + self.context_length):
            # action 4-d reward 1-d obs 39-d isdone True or False
            # infor success near_object  grasp_success  grasp_reward in_place_reward obj_to_target unscaled_reward
            frame = frames[frame_index]
            # print(frame)
            action = frame['action']
            rw = rtgs[frame_index]
            obs = frame['obs']
            obs = np.array(obs).reshape((39,))
            obs = np.concatenate([obs, np.zeros((5,))], 0)
            obss.append(obs)
            reward.append(np.array(rw))
            rgb = self.transform(Image.open(
                io.BytesIO(ZipReader.read(zip_file,
                                          os.path.join('rgb', self.mod, 'image_save_' + str(frame_index) + ".jpg")))))

            rgb_list.append(rgb)

            times = frame_index

            act = np.array(action)
            acts.append(act)
            times = np.array(times, dtype=int)
            timesteps.append(times)

        rgbs = torch.stack(rgb_list, dim=0)
        obss_seq = [torch.from_numpy(item) for item in obss]
        obss_seq = torch.stack(obss_seq, dim=0).float()
        timesteps_seq = [torch.from_numpy(item) for item in timesteps]
        timesteps_seq = torch.stack(timesteps_seq, dim=0).float()
        act_seq = [torch.from_numpy(item) for item in acts]
        reward_seq = [torch.from_numpy(item) for item in reward]
        reward_seq = torch.stack(reward_seq, dim=0).float()
        act_seq = torch.stack(act_seq, dim=0).float()
        _gps = torch.zeros(self.context_length, 2)
        _pg = torch.zeros(2)
        # return rgbs,obss_seq,act_vect,timesteps_seq,image_goal,gps_goal_seq,pg_goal_seq,object_goal,task_type,task_desc
        if self.cont_action:
            act_seq = torch.cat([torch.zeros((self.context_length, 6)), act_seq], 1).float()
            act_seq = torch.clamp(act_seq, -1, 1)

        else:
            act_seq = torch.cat([torch.zeros((self.context_length, 1)), act_seq], 1)
            act_seq = torch.clamp(act_seq, -1, 1)

            act_seq = tokenize_continuous_values(act_seq, bins=self.action_bins)
        # print(act_seq[0])
        # print(reward_seq)
        return rgbs, obss_seq, act_seq, timesteps_seq, image_goal, _gps, _pg, 'null', 3, task_desc, reward_seq
    # except (Exception) as e:
    #     print(e)
    #     print(self.clip_indices[index])


class MultiDomainDataModule(LightningDataModule):
    """Example of LightningDataModule for pretraining in multiple tasks (same domain) on DMC."""

    def __init__(
            self,
            dataset_dir,
            habitat_file_path,
            metaworld_file_path,
            context_length=6,
            cont_action=False,
            max_clips=-1,
            mode='rgb',
            meta_view='corner2',
            data_mode=None,
            batch_size=128,
            num_workers=1,
            action_bins=256,
            is_aug=False,
            seed=42,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None

    @property
    def context_length(self) -> int:
        return self.context_length

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning when doing `trainer.fit()` and `trainer.test()`,
        so be careful not to execute the random split twice! The `stage` can be used to
        differentiate whether it's called before trainer.fit()` or `trainer.test()`.
        """
        print('is_aug', self.hparams.is_aug)
        if self.hparams.is_aug:
            transform_train = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0), interpolation=3),  # 3 is bicubic
                transforms.ColorJitter(0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            # transform_train = transforms.Compose([
            #     RandomApply(
            #     transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            #     p = 0.8
            #     ),
            #     transforms.RandomGrayscale(p=0.5),
            #     RandomApply(
            #         transforms.GaussianBlur((3, 3), (1.0, 2.0)),
            #         p = 0.5
            #     ),
            #     transforms.RandomResizedCrop(224, scale=(0.8, 1.0), interpolation=3),  # 3 is bicubic
            #     transforms.ToTensor(),
            #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        else:
            transform_train = transforms.Compose([
                transforms.Resize(224),  # 3 is bicubic
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        if not self.train_dataset:
            if self.hparams.habitat_file_path:
                habitat_dataset = HabitatDataset(
                    self.hparams.dataset_dir, self.hparams.habitat_file_path, transform_train,
                    self.hparams.context_length, max_clips=self.hparams.max_clips, mod=self.hparams.mode,
                    data_mode=self.hparams.data_mode, cont_action=self.hparams.cont_action,
                    action_bins=self.hparams.action_bins
                )
            if self.hparams.metaworld_file_path:
                meta_dataset = MetaworldDataset(
                    self.hparams.dataset_dir, self.hparams.metaworld_file_path, transform_train,
                    self.hparams.context_length, max_clips=self.hparams.max_clips, mod=self.hparams.mode,
                    data_mode=self.hparams.data_mode, cont_action=self.hparams.cont_action,
                    action_bins=self.hparams.action_bins
                )
            if self.hparams.habitat_file_path and self.hparams.metaworld_file_path:
                self.train_dataset = habitat_dataset + meta_dataset
            elif self.hparams.habitat_file_path:
                self.train_dataset = habitat_dataset
            elif self.hparams.metaworld_file_path:
                self.train_dataset = meta_dataset
        print(len(self.train_dataset))
        # if not self.val_dataset:

        #     self.val_dataset = HabitatDataset(
        #             self.hparams.dataset_dir,self.hparams.test_file_path,transform_val, 
        #             self.hparams.context_length, max_clips=10000,mod=self.hparams.mode,data_mode=self.hparams.data_mode
        #         )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            pin_memory=True,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            # persistent_workers=True,
        )

    # def val_dataloader(self):
    #     return DataLoader(
    #         self.val_dataset,
    #         shuffle=False,
    #         pin_memory=True,
    #         batch_size=self.hparams.batch_size,
    #         num_workers=self.hparams.num_workers,
    #         persistent_workers=True,
    #     )

    def test_dataloader(self):
        return None
