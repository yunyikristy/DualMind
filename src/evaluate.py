# import blosc
import argparse
import random

import habitat
import numpy as np
import torch
from habitat.config.default import get_config

from agents.dt_agent import DecisionTransformerAgent
from models.components.mingpt_multidomain import GPT, GPTConfig
from models.components.modules import FrozenClipImageEmbedder, FrozenCLIPTextEmbedder, PointEmbedder
from multimae.input_adapters import PatchedInputAdapter
from multimae.output_adapters import TokenLearnerAdapter, LinearOutputAdapter
from utils import create_model


def get_args():
    parser = argparse.ArgumentParser('m3-pact', add_help=False)

    parser.add_argument("--action_bins", type=int, default=256)

    parser.add_argument('--device', default='cuda', help='device to use for training / testing')

    parser.add_argument("--load_model_from", type=str, default=None)

    parser.add_argument("--seed", default=1111, type=int, help="random seed")

    # Decision Transformer parameters
    parser.add_argument("--log_dir", type=str, default='output')
    parser.add_argument("--output_dir", type=str, default='output')

    parser.add_argument('--encoder', default='multimae-b_98_rgb+-depth-semseg_1600e_multivit-afff3f8c.pth', type=str)

    parser.add_argument("--context_length", type=int, default=6)

    parser.add_argument("--max_timestep", type=int, default=1000)
    parser.add_argument("--n_layer", type=int, default=8)
    parser.add_argument("--n_head", type=int, default=16)
    parser.add_argument("--n_embd", type=int, default=512)
    parser.add_argument("--rtg_layers", type=int, default=1)
    parser.add_argument("--bc_layers", type=int, default=1)
    parser.add_argument("--pred_layers", type=int, default=1)
    parser.add_argument("--token_decoder", type=str, default='tokenlearner')

    # parser.add_argument( "--task_type",type=str,default='pretrain')
    parser.add_argument("--vocab_size", type=int, default=4)

    parser.add_argument("--difficulty", type=str, default='easy')
    parser.add_argument("--task_type", type=str, default='imagenav')
    parser.add_argument("--scene_type", type=str, default='gibson')
    parser.add_argument("--finetune", default=False, action="store_true")
    parser.add_argument("--prompt", default=False, action="store_true")
    parser.add_argument("--pretrain_type", type=str, default="xattn_gpt")
    parser.add_argument("--cont_action", default=False, action="store_true")
    parser.add_argument("--auto_stop", default=False, action="store_true")

    return parser.parse_args()


def test_model(model, task_type, experiment, args, epoch_num=0):
    if task_type == "pointnav":

        task_config = f"configs/tasks/{task_type}_{args.scene_type}.yaml"
        success_distance = 0.36
    elif task_type == "imagenav":
        success_distance = 1
        task_config = f"configs/tasks/imagenav_{args.difficulty}.yaml"
    elif task_type == 'objectnav':
        success_distance = 0.1
        task_config = f"configs/tasks/{task_type}_{args.scene_type}.yaml"
    config = get_config(task_config)
    agent = DecisionTransformerAgent(
        success_distance=success_distance,
        model=model,
        task_type=task_type,
        args=args
    )
    benchmark = habitat.Benchmark(config_paths=task_config)
    metrics = benchmark.evaluate(agent)
    for k, v in metrics.items():
        habitat.logger.info("{}: {:.3f}".format(k, v))
        if experiment is not None:
            experiment.log_metric(k, v, epoch_num)


experiment = None


def main(args):
    """
    This is a single process that is linked to a single GPU
    :param local_rank: The id of the GPU on the current node
    :param world_size: Total number of processes across nodes
    :param args:
    :return:
    """
    device = torch.device(args.device)

    seed = args.seed

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    block_size = args.context_length * 10

    mconf = GPTConfig(
        args.vocab_size,
        block_size,
        max_timestep=args.max_timestep,
        training_phase=2,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        cont_action=args.cont_action,
        pred_layers=args.pred_layers,
        rtg_layers=args.rtg_layers,
        bc_layers=args.bc_layers,
        token_decoder=args.token_decoder,
        obs_vector_dim=44,
        action_bins=args.action_bins,
        supervised=False,
        reward_conditioned=False,
        prompt=args.prompt
    )
    input_adapters = {
        'rgb': PatchedInputAdapter(
            num_channels=3, stride_level=1,
            patch_size_full=16,
            image_size=224
        )
    }
    if args.token_decoder == 'tokenlearner':
        output_adapters = {
            'rgb': TokenLearnerAdapter(
                num_tokens=8, dim_tokens_enc=768,
                token_learner_type='RGB',
            )}
    else:
        output_adapters = {
            'rgb': LinearOutputAdapter(
                num_classes=768, dim_tokens_enc=768,
            )}

    multimae = create_model(
        'multivit_base',
        input_adapters=input_adapters,
        output_adapters=output_adapters,
        drop_path_rate=0.0,
        checkpoint_path=args.encoder,
        pretrained=True,
    )

    image_embedder = FrozenClipImageEmbedder('ViT-B/16')
    image_embedder.freeze()
    object_embedder = FrozenCLIPTextEmbedder('ViT-B/16')
    object_embedder.freeze()
    point_embedder = PointEmbedder(2, 512)
    goal_encoder = [point_embedder, image_embedder, object_embedder]

    model = GPT(multimae, goal_encoder, mconf)
    pretrain_weight = torch.load(args.load_model_from)
    model_dict = model.state_dict()
    if 'state_dict' in pretrain_weight.keys():
        pretrain_weight = pretrain_weight['state_dict']
    pw = {}
    for e in pretrain_weight:
        _e = e.replace('net.', '')
        if 'inverse_pred_head' not in _e and 'forward_pred_head' not in _e:
            pw[_e] = pretrain_weight[e]

    pw = {k: v for k, v in pw.items() if k in model_dict}
    for k in pw.keys():
        print(k)
    model_dict.update(pw)

    model.load_state_dict(model_dict, strict=False)

    test_model(model, args.task_type, experiment, args=args)


if __name__ == "__main__":
    opts = get_args()

    main(opts)
