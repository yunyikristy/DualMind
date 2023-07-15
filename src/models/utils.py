import random

import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float("Inf")
    return out


def get_min_action_dmc(domain):
    domain_to_actions = {
        "cheetah": 6,
        "walker": 6,
        "hopper": 4,
        "cartpole": 1,
        "acrobot": 1,
        "pendulum": 1,
        "finger": 2,
    }
    if domain in domain_to_actions.keys():
        return domain_to_actions[domain]
    else:
        raise NotImplementedError()


def get_exp_return_dmc(domain, task):
    game_to_returns = {
        "cheetah_run": 850,
        "walker_stand": 980,
        "walker_walk": 950,
        "walker_run": 700,
        "hopper_stand": 900,
        "hopper_hop": 200,
        "cartpole_swingup": 875,
        "cartpole_balance": 1000,
        "pendulum_swingup": 1000,
        "finger_turn_easy": 1000,
        "finger_turn_hard": 1000,
        "finger_spin": 800,
    }
    if domain + "_" + task in game_to_returns.keys():
        return game_to_returns[domain + "_" + task]
    else:
        raise NotImplementedError()
