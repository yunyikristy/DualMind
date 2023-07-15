import argparse

parser = argparse.ArgumentParser()

## program and path
parser.add_argument("--amlt", default=False, action="store_true", help="whether runing by amlt")
parser.add_argument("--data_dir_prefix", type=str, default="./data/")
parser.add_argument("--output_dir", type=str, default="./outputs/")
parser.add_argument("--load_model_from", type=str, default=None)
parser.add_argument("--no_load_action", default=False, action="store_true")
parser.add_argument("--no_strict", default=False, action="store_true")
parser.add_argument("--no_action_head", default=False, action="store_true")
parser.add_argument("--stat_file", type=str, default="stat.csv")
parser.add_argument("--reset_path", default=None, type=str)
parser.add_argument('--dist_on_itp', action='store_true')

## trainer
parser.add_argument("--seed", type=int, default=123)
parser.add_argument("--accelerator", type=str, default="gpu")
parser.add_argument("--devices", type=int, default=1)
parser.add_argument("--nodes", type=int, default=1)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--save_k", type=int, default=5, help="how many checkpoints to save in finetuning")

## evaluate
parser.add_argument("--eval_epochs", type=int, default=50)

## data
parser.add_argument("--context_length", type=int, default=6)
parser.add_argument("--batch_size", type=int, default=128)

parser.add_argument("--timestep", type=int, default=1000)
parser.add_argument("--cont_action", default=False, action="store_true")
parser.add_argument("--finetune", default=False, action="store_true")

parser.add_argument('--layer_decay', type=float, default=0.75)

parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR',
                    help='warmup learning rate (default: 1e-6)')
parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                    help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                    help='epochs to warmup LR, if scheduler supports')
parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                    help='num of steps to warmup LR, will overload warmup_epochs if set > 0')
parser.add_argument('--device', default='cuda',
                    help='device to use for training / testing')
# Random Erase params
parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                    help='Random erase prob (default: 0.25)')
parser.add_argument('--remode', type=str, default='pixel',
                    help='Random erase mode (default: "pixel")')
parser.add_argument('--recount', type=int, default=1,
                    help='Random erase count (default: 1)')
parser.add_argument('--resplit', action='store_true', default=False,
                    help='Do not random erase first (clean) augmentation split')

# Dataset parameters
parser.add_argument('--num_workers', default=10, type=int)
parser.add_argument('--pin_mem', action='store_true',
                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
parser.set_defaults(pin_mem=True)

# Finetuning params
parser.add_argument('--model_key', default='model|module', type=str)
parser.add_argument('--model_prefix', default='', type=str)
parser.add_argument('--init_scale', default=0.001, type=float)
parser.add_argument('--use_mean_pooling', action='store_true')
parser.set_defaults(use_mean_pooling=True)
parser.add_argument('--use_cls', action='store_false', dest='use_mean_pooling')

# Decision Transformer parameters
parser.add_argument("--dataset_dir", type=str, default='/VL_Control/data/expert_data_fixed')

parser.add_argument('--encoder',
                    default='/home/COMPASS-user/workspace/blobfuse/shuang/Data/gibson_data/data/multimae-b_98_rgb+-depth-semseg_1600e_multivit-afff3f8c.pth',
                    type=str)
parser.add_argument('--aug_type', default='wo', type=str)
parser.add_argument('--mode', default='corner2', type=str)

parser.add_argument("--habitat_file_path", type=str, default=None)
parser.add_argument("--metaworld_file_path", type=str, default=None)
parser.add_argument("--max_timestep", type=int, default=1000)
parser.add_argument("--task_type", type=str, default='multimae')
parser.add_argument("--vocab_size", type=int, default=10)
parser.add_argument('--max_clips', default=-1, type=int)
parser.add_argument("--is_aug", default=False, action="store_true")
parser.add_argument("--training_phase", default=1, type=int)
