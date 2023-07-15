import argparse
import random

import metaworld
import numpy as np
import torch

from agents.dt_agent_mw import DecisionTransformerAgent
from models.components.mingpt_multidomain import GPT, GPTConfig
from models.components.modules import FrozenClipImageEmbedder, FrozenCLIPTextEmbedder, PointEmbedder
from multimae.input_adapters import PatchedInputAdapter
from multimae.output_adapters import TokenLearnerAdapter, LinearOutputAdapter
from utils import create_model

all_v2_pol_instance = {'assembly-v2': ['assemble nut', 'Pick up a nut and place it onto a peg'],
                       'basketball-v2': ['basketball', 'Dunk the basketball into the basket'],
                       'bin-picking-v2': ['pick bin', 'Grasp the puck from one bin and place it into another bin'],
                       'box-close-v2': ['close box', 'Grasp the cover and close the box with it'],
                       'button-press-topdown-v2': ['press button top', 'Press a button from the top'],
                       'button-press-topdown-wall-v2': ['press button top wall',
                                                        'Bypass a wall and press a button from the top'],
                       'button-press-v2': ['press button', 'Press a button'],
                       'button-press-wall-v2': ['press button wall', 'Bypass a wall and press a button'],
                       'coffee-button-v2': ['get coffee', 'Push a button on the coffee machine'],
                       'coffee-pull-v2': ['pull mug', 'Pull a mug from a coffee machine'],
                       'coffee-push-v2': ['push mug', 'Push a mug under a coffee machine'],
                       'dial-turn-v2': ['turn dial', 'Rotate a dial 180 degrees'],
                       'disassemble-v2': ['disassemble nut', 'pick a nut out of the a peg'],
                       'door-close-v2': ['close door', 'Close a door with a revolving joint'],
                       'door-lock-v2': ['lock door', 'Lock the door by rotating the lock clockwise'],
                       'door-open-v2': ['open door', 'Open a door with a revolving joint'],
                       'door-unlock-v2': ['unlock door', 'Unlock the door by rotating the lock counter-clockwise'],
                       'hand-insert-v2': ['insert hand', 'Insert the gripper into a hole'],
                       'drawer-close-v2': ['close drawer', 'Push and close a drawer'],
                       'drawer-open-v2': ['open drawer', 'Open a drawer'],
                       'faucet-open-v2': ['turn on faucet', 'Rotate the faucet counter-clockwise'],
                       'faucet-close-v2': ['Turn off faucet', 'Rotate the faucet clockwise'],
                       'hammer-v2': ['hammer', 'Hammer a screw on the wall'],
                       'handle-press-side-v2': ['press handle side', 'Press a handle down sideways'],
                       'handle-press-v2': ['press handle', 'Press a handle down'],
                       'handle-pull-side-v2': ['pull handle side', 'Pull a handle up sideways'],
                       'handle-pull-v2': ['pull handle', 'Pull a handle up'],
                       'lever-pull-v2': ['Pull lever', 'Pull a lever down 90 degrees'],
                       'peg-insert-side-v2': ['insert peg side', 'Insert a peg sideways'],
                       'pick-place-wall-v2': ['pick and place wall', 'Pick a puck, bypass a wall and place the puck'],
                       'pick-out-of-hole-v2': ['pick out of hole', 'Pick up a puck from a hole'],
                       'reach-v2': ['reach', 'reach a goal position'],
                       'push-back-v2': ['push back', 'push back'],
                       'push-v2': ['Push', 'Push the puck to a goal'],
                       'pick-place-v2': ['pick and place', 'Pick and place a puck to a goal'],
                       'plate-slide-v2': ['slide plate', 'Slide a plate into a cabinet'],
                       'plate-slide-side-v2': ['slide plate side', 'Slide a plate into a cabinet sideways'],
                       'plate-slide-back-v2': ['retrieve plate', 'Get a plate from the cabinet'],
                       'plate-slide-back-side-v2': ['retrieve plate side', 'Get a plate from the cabinet sideways'],
                       'peg-unplug-side-v2': ['unplug peg', 'Unplug a peg sideways'],
                       'soccer-v2': ['soccer', 'Kick a soccer into the goal'],
                       'stick-push-v2': ['push with stick', 'Grasp a stick and push a box using the stick'],
                       'stick-pull-v2': ['pull with stick', 'Grasp a stick and pull a box with the stick'],
                       'push-wall-v2': ['push with wall', 'Bypass a wall and push a puck to a goal'],
                       'reach-wall-v2': ['reach with wall', 'Bypass a wall and reach a goal'],
                       'shelf-place-v2': ['place onto shelf', 'pick and place a puck onto a shelf'],
                       'sweep-into-v2': ['sweep into hole', 'Sweep a puck into a hole'],
                       'sweep-v2': ['Sweep', 'Sweep a puck off the table'],
                       'window-open-v2': ['open window', 'Push and open a window'],
                       'window-close-v2': ['close window', 'Push and close a window']}


def get_args():
    parser = argparse.ArgumentParser('m3-pact', add_help=False)

    parser.add_argument("--action_bins", type=int, default=256)

    parser.add_argument('--device', default='cuda', help='device to use for training / testing')

    parser.add_argument("--load_model_from", type=str, default=None)

    parser.add_argument("--seed", default=1111, type=int, help="random seed")

    # Decision Transformer parameters
    parser.add_argument("--log_dir", type=str, default='output')
    parser.add_argument("--output_dir", type=str, default='output')

    parser.add_argument('--encoder',
                        default='/data/weiyao/VL_Control/ICCV2023/m3-pact/src/models/multimae-b_98_rgb+-depth-semseg_1600e_multivit-afff3f8c.pth',
                        type=str)

    parser.add_argument("--context_length", type=int, default=6)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--agent_type", type=str, default="gpt")

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
    parser.add_argument("--finetune", default=True, action="store_true")
    parser.add_argument("--ft_type", type=str, default="meta")
    parser.add_argument("--pretrain_type", type=str, default="gpt")
    parser.add_argument("--cont_action", default=False, action="store_true")
    parser.add_argument("--meta_type", type=str, default="MT10")

    return parser.parse_args()


def test_model(model, experiment, args, epoch_num=0):
    if args.meta_type == 'MT10':
        ml10 = metaworld.MT10()  # Construct the benchmark, sampling tasks
    elif args.meta_type == "ML10":
        ml10 = metaworld.ML10()  # C
    elif args.meta_type == "ML45":
        ml10 = metaworld.ML45()  #
    elif args.meta_type == 'MT50':
        ml10 = metaworld.MT50()  #
    training_envs = []
    # for name, env_cls in ml10.test_classes.items():
    #     env = env_cls()
    #     task = random.choice([task for task in ml10.test_tasks
    #                             if task.env_name == name])
    #     env.set_task(task)
    #     training_envs.append([name,env])
    for name, env_cls in ml10.train_classes.items():
        env = env_cls()
        task = random.choice([task for task in ml10.train_tasks
                              if task.env_name == name])
        env.set_task(task)
        training_envs.append([name, env])
    agent = DecisionTransformerAgent(
        model=model,
        args=args)
    metrics = {}
    for d in training_envs:
        env = d[1]
        caption = all_v2_pol_instance[d[0]][1]

        success = 0
        rtgs = 0
        for j in range(args.episodes):
            obs = env.reset()  # Reset environment
            goal = [caption]
            rtg = 0
            for i in range(500):

                rgb = env.render('rgb_array', camera_name='corner2', resolution=(256, 256))
                a = agent.act(rgb, goal, obs)
                a = np.array(a)[0]
                obs, reward, done, info = env.step(a)
                rtg += reward
                if info['success']:
                    success += 1
                    break
            rtgs += rtg
            agent.reset()

        print(d[0] + "/success:", success / args.episodes, " ", d[0] + "/rtg:", rtgs / args.episodes)
        metrics[d[0]] = {
            'success': success / args.episodes,
            'rtg': rtgs / args.episodes
        }
    # save_name=args.load_model_from.split('/')[-2]+'_'+args.load_model_from.split('/')[-1].replace('.ckpt','')+'_'+args.meta_type+"_30.json"
    # with open(os.path.join('result/',save_name),'w') as f:
    #     json.dump(metrics, f)
    #     # print(,metrics[d[0]])
    #     # rgb = env.render('rgb_array', camera_name='corner2', resolution=(256, 256))

    # # metrics = benchmark.evaluate(agent)
    # # for k, v in metrics.items():
    # #     habitat.logger.info("{}: {:.3f}".format(k, v))
    # #     if experiment is not None:
    # #         experiment.log_metric(k, v, epoch_num)


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
        finetune=True,
        ft_type='meta',
        pretrain_type=args.pretrain_type,
        action_bins=args.action_bins,
        reward_conditioned=False,
        prompt=False
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
        if 'inverse_pred_head' not in _e and 'orward_pred_head' not in _e:
            pw[_e] = pretrain_weight[e]

    pw = {k: v for k, v in pw.items() if k in model_dict}
    for p in pw:
        print(p)
    model_dict.update(pw)

    model.load_state_dict(model_dict, strict=False)
    model.to(args.device)
    test_model(model, experiment, args=args)


if __name__ == "__main__":
    opts = get_args()

    main(opts)
