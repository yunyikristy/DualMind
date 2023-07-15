import numpy as np
import torch
from PIL import Image
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
    c = x
    if shift is not None:
        c = x - shift
    c = (c / (bins / 2.0)) - 1.0
    c = mu_law_decode(c, mu, m)
    c = torch.where(
        torch.isnan(c),
        torch.full_like(c, 0),
        c)
    return c


class DecisionTransformerAgent():
    def __init__(self, model, args) -> None:

        self.device = args.device
        self.model = model
        # print(self.model.state_encoder)

        self.model.to(self.device)

        self.transform = self.transform = transforms.Compose([
            transforms.Resize(224),  # 3 is bicubic
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        self.step = 0  # count agent step
        self.actions = []
        self.rgbs = []
        self.obs_vecs = []
        self.epsiode = 0
        self.args = args
        self.count = 0
        self.succ_ct = 0

    def reset(self) -> None:
        self.step = 0  # count agent step
        self.epsiode += 1
        self.obs_vecs = []
        self.actions = []
        self.rgbs = []

    def act(self, observations, goal, obs):
        goals = {
            'image': None,
            'object': None,
            'point': None,
            'meta': goal
        }

        rgb = Image.fromarray(observations)
        rgb = self.transform(rgb)

        B = 1
        N = 1
        self.step += 1
        self.rgbs.append(rgb)
        obs_vec = torch.tensor(obs).reshape(1, 39)
        obs_vec = torch.cat([obs_vec, torch.zeros((1, 5))], 1).to(self.device)
        self.obs_vecs.append(obs_vec)

        block_size = 6
        if self.step >= block_size:
            self.rgbs = self.rgbs[-block_size:]
            self.obs_vecs = self.obs_vecs[-block_size:]
            # self.depths=self.depths[-(block_size+2):]
            self.actions = self.actions[-block_size:]
            start_index = self.step - block_size
        else:
            start_index = 0

        timesteps = np.arange(start_index, start_index + len(self.rgbs), dtype=np.int64)
        t = torch.tensor(timesteps).reshape(1, -1, 1)
        t = t.to(self.device)
        rgb_input = torch.cat(self.rgbs, 0).reshape(-1, 3, 224, 224)
        obs_vecss = torch.cat(self.obs_vecs, 0).reshape(1, -1, 44)

        state = {
            'rgb': rgb_input.to(self.device),
            # 'depth':depth_input.to(self.device)
        }
        if len(self.actions) > 0:
            act = torch.cat(self.actions, 0).reshape(1, -1, 5)
            # act=torch.tensor(self.actions).reshape(1,-1,5)

            act_seq = act.reshape(len(self.actions), -1)
            # act_seq=torch.cat([torch.zeros((len(self.actions),1)),act_seq],1)
            act = tokenize_continuous_values(act_seq).float().to(self.device).reshape(1, -1, 5)

        else:
            act = None

        task_types = torch.tensor([3]).to(self.device)
        with torch.no_grad():
            logit = self.model.get_action(state, act, None, goals, t, obs_vecss, task_types)
            logit = logit.reshape(1, -1, 5, 256)
            logit = logit[:, -1, :, :]
            logit = torch.softmax(logit, dim=2)

            D = torch.distributions.Categorical(logit)
            # D=torch.distributions.Categorical( logit)
            a = torch.argmax(logit, -1).to("cpu")
            # b=a[:,1:]
            a = decode_action_tokenize(a)
        self.actions.append(a)
        a = a[:, 1:]

        return a
