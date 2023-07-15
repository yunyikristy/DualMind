from typing import Dict

import habitat
import numpy as np
import torch
from PIL import Image
from habitat.core.simulator import Observations
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from numpy import bool_, int64
from torchvision import transforms


# from TE.ViT_eg import LRP
# from TE.ViT_LRP import vit_base_patch16_224 as vit_LRP


def mu_law_encode(x, mu=1, m=0):
    # Appendix B. Agent Data Tokenization Details
    sign = torch.sign(x)
    numerator = torch.log(torch.abs(x) * mu + 1.0)

    # denominator = torch.log(torch.tensor())
    return (numerator / (m * mu + 1.0)) * sign


def tokenize_continuous_values(x, mu=1.7, m=0, bins=256, shift=None):
    # Appendix B. Agent Data Tokenization Details
    # > Finally, they are discretized using bins of uniform width on the domain [-1, 1].
    c = mu_law_encode(x, mu, m)
    # > We use 1024 bins and shift the resulting integers
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


def decode_action_tokenize(x, mu=1.7, m=0, bins=256, shift=None):
    if shift is not None:
        c = x - shift
    c = (c / (bins / 2.0)) - 1.0
    c = mu_law_decode(c, mu, m)
    c = torch.where(
        torch.isnan(c),
        torch.full_like(c, 0),
        c)
    return c


gibson_objects = [
    "null",
    "chair",
    "potted plant",
    "sink",
    "vase",
    "book",
    "couch",
    "bed",
    "bottle",
    "dining table",
    "toilet",
    "refrigerator",
    "tv",
    "clock",
    "oven",
    "bowl",
    "cup",
    "bench",
    "microwave",
    "suitcase",
    "umbrella",
    "teddy bear"
]
# start_from_zero
mp3d_objects = [
    "chair",
    "table",
    "picture",
    "cabinet",
    "cushion",
    "sofa",
    "bed",
    "chest_of_drawers",
    "plant",
    "sink",
    "toilet",
    "stool",
    "towel",
    "tv_monitor",
    "shower",
    "bathtub",
    "counter",
    "fireplace",
    "gym_equipment",
    "seating",
    "clothes"
]
hm3d_objects = [
    "chair",
    "bed",
    "plant",
    "toilet",
    "tv_monitor",
    "sofa"
]
objects = {
    'gibson': gibson_objects,
    'mp3d': mp3d_objects,
    'hm3d': hm3d_objects
}


class DecisionTransformerAgent(habitat.Agent):
    def __init__(self, success_distance, model, task_type, args) -> None:
        self.dist_threshold_to_stop = success_distance
        self.device = args.device
        self.model = model

        self.task_type = task_type
        self.model.to(self.device)
        self.scene_type = args.scene_type
        self.transform = transforms.Compose([
            transforms.Resize(224),  # 3 is bicubic
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.step = 0  # count agent step
        self.actions = []
        self.rgbs = []
        self.depths = []
        self.goals = []
        self.epsiode = 0
        self.args = args
        self.count = 0
        self.succ_ct = 0

    def reset(self) -> None:
        self.step = 0  # count agent step
        self.epsiode += 1

        self.goals = []
        self.actions = []
        self.rgbs = []
        self.depths = []

    def is_goal_reached(self, observations: Observations) -> bool_:
        dist = observations['pointgoal_with_gps_compass'][0]
        # print(dist)
        return 0 <= dist <= self.dist_threshold_to_stop

    def act(self, observations: Observations) -> Dict[str, int64]:
        goals = {
            'image': None,
            'object': None,
            'point': None,
            'meta': None
        }

        if self.task_type == 'imagenav':
            goal = observations['imagegoal']
            goal = Image.fromarray(goal)
            goal = self.transform(goal).reshape(1, 3, 224, 224).to(self.device)
            goals["image"] = goal
            task_type = torch.tensor([1])
        elif self.task_type == 'objectnav':
            goal = int(observations['objectgoal'])
            goals["object"] = [objects['mp3d'][goal]]
            task_type = torch.tensor([2])
        elif self.task_type == 'pointnav':
            goal = observations['pointgoal']
            goals["point"] = torch.tensor(goal).reshape(1, 2).to(self.device)
            task_type = torch.tensor([0])

        rgb = observations['rgb']
        # depth=observations['depth']
        rgb = Image.fromarray(rgb)
        rgb = self.transform(rgb)
        # depth=Image.fromarray((depth * 255).astype(np.uint8).reshape(256,256))
        # depth=self.transform(depth,'depth')
        B = 1
        N = 1
        self.step += 1
        # print(self.step)
        self.rgbs.append(rgb)

        block_size = 6
        if self.step >= block_size:
            self.rgbs = self.rgbs[-block_size:]
            # self.depths=self.depths[-(block_size+2):]
            self.actions = self.actions[-block_size:]
            start_index = self.step - block_size
        else:
            start_index = 0

        timesteps = np.arange(start_index, start_index + len(self.rgbs), dtype=np.int64)
        t = torch.tensor(timesteps).reshape(1, -1, 1)
        t = t.to(self.device)
        rgb_input = torch.cat(self.rgbs, 0).reshape(-1, 3, 224, 224)
        # depth_input=torch.cat(self.depths,0).reshape(-1,1,224,224)

        state = {
            'rgb': rgb_input.to(self.device),
            # 'depth':depth_input.to(self.device)
        }
        if len(self.actions) > 0:
            if self.args.cont_action:
                act = torch.tensor(self.actions).reshape(-1, 1).long()

                act = torch.zeros(len(self.actions), 10).scatter_(1, act, 1).float().reshape(1, -1, 10).to(self.device)
            else:
                act = torch.tensor(self.actions).reshape(1, -1, 1).long()

                act_seq = (act - 0) / 5.0
                act_seq = act_seq.reshape(len(self.actions), -1)
                act_seq = torch.cat([act_seq, torch.zeros((len(self.actions), 4))], 1)
                act = tokenize_continuous_values(act_seq).float().to(self.device).reshape(1, -1, 5)

        else:
            act = None
        obss_seq = torch.zeros((1, t.shape[1], 44)).to(self.device)

        task_types = task_type.to(self.device)
        with torch.no_grad():
            logit = self.model.get_action(state, act, None, goals, t, obss_seq, task_types)
            if self.args.cont_action:
                if self.task_type == 'imagenav':
                    logit = (logit[:, -1, :]).reshape((10,))[:4]
                    # logit=logit-min(logit)
                    logit = torch.softmax(logit, dim=0)
                    # print(logit)
                    D = torch.distributions.Categorical(logit)
                    # a=int(np.array(D.sample().to('cpu')))
                    a = int(np.array(torch.argmax(logit).to("cpu")))
                else:
                    logit = (logit[:, -1, :]).reshape((10,))[:6]
                    # logit=logit-min(logit)
                    logit = torch.softmax(logit, dim=0)
                    # print(logit)
                    D = torch.distributions.Categorical(logit)
                    # a=int(np.array(D.sample().to('cpu')))
                    a = int(np.array(torch.argmax(logit).to("cpu")))
            else:
                # logit=(logit[:,-1,500:850]).reshape(1,-1)
                logit = (logit[:, -1, [128, 165, 194, 217, 237, 255]]).reshape(1, -1)[:, :4]
                # print(logit)

                logit = torch.softmax(logit, dim=1)
                # print(logit)
                D = torch.distributions.Categorical(logit)
                # D=torch.distributions.Categorical( logit)
                a = int(np.array(D.sample().to('cpu')))
                # a=int(np.array(torch.argmax(logit).to("cpu")))
                # print(D.argmax())
                # a=torch.tensor(int(np.array(D.sample().to('cpu'))))+500
                # # print(act)
                # a=int(np.floor(decode_action_tokenize(a)*5+0.1))
                # print(a)
        # act=int(np.array(torch.argmax(logit).to("cpu")))

        action = [HabitatSimActions.STOP,
                  HabitatSimActions.MOVE_FORWARD,
                  HabitatSimActions.TURN_LEFT,
                  HabitatSimActions.TURN_RIGHT,
                  ]

        self.actions.append(a)
        # if a ==0:
        #     if self.is_goal_reached(observations):
        #         self.succ_ct+=1
        #         print('succ',self.succ_ct)

        return {"action": action[a]}
