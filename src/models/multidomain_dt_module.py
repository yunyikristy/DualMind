import re
from typing import Any

import torch
from models.components.mingpt_multidomain import GPT, GPTConfig
from models.components.modules import FrozenClipImageEmbedder, FrozenCLIPTextEmbedder, PointEmbedder
from multimae.input_adapters import PatchedInputAdapter
from multimae.output_adapters import TokenLearnerAdapter, LinearOutputAdapter
from pytorch_lightning import LightningModule

from utils import create_model


def interpolate_pos_embed_multimae(model, checkpoint_model):
    pattern = "input_adapters\.(.*)\.pos_emb"
    matched_keys = [k for k in checkpoint_model if bool(re.match(pattern, k))]

    for key in matched_keys:
        domain = re.match(pattern, key).group(1)  # group(0) is entire matched regex
        if getattr(model.input_adapters, domain, None) is not None:
            pos_embed_checkpoint = checkpoint_model[key]
            _, _, orig_H, orig_W = pos_embed_checkpoint.shape
            _, _, new_H, new_W = getattr(model.input_adapters, domain).pos_emb.shape
            if (orig_H != new_H) or (orig_W != new_W):
                print(f"Key {key}: Position interpolate from {orig_H}x{orig_W} to {new_H}x{new_W}")
                pos_embed_checkpoint = torch.nn.functional.interpolate(
                    pos_embed_checkpoint, size=(new_H, new_W), mode='bicubic', align_corners=False)
                checkpoint_model[key] = pos_embed_checkpoint


class MultiTaskDTLitModule(LightningModule):
    """Example of LightningModule for MNIST classification.

    A LightningModule organizes your PyTorch code into 5 sections:
        - Computations (init).
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Optimizers (configure_optimizers)

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
            self,
            model_type,
            encoder,
            token_decoder='linear',
            timestep=10000,
            n_embd=128,
            lr=6e-4,
            forward=False,
            inverse=False,
            reward=False,
            rand_inverse=False,
            rand_mask_size=1,
            context_length=30,
            betas=(0.9, 0.95),
            weight_decay=0.1,
            n_layer=6,
            n_head=8,
            pred_layers=1,
            ft_params=0,
            action_bins=256,
            supervised=False,
            reward_conditioned=False,
            prompt=False,
            **kwargs,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters()
        self._epoch = 0
        vocab_size = self.hparams.vocab_size
        block_size = self.hparams.context_length * 10

        mconf = GPTConfig(
            vocab_size,
            block_size,
            max_timestep=self.hparams.timestep,
            training_phase=self.hparams.training_phase,
            n_layer=self.hparams.n_layer,
            n_head=self.hparams.n_head,
            n_embd=self.hparams.n_embd,
            cont_action=self.hparams.cont_action,
            pred_layers=self.hparams.pred_layers,
            rtg_layers=self.hparams.rtg_layers,
            bc_layers=self.hparams.bc_layers,
            token_decoder=self.hparams.token_decoder,
            obs_vector_dim=44,
            action_bins=self.hparams.action_bins,
            supervised=self.hparams.supervised,
            reward_conditioned=self.hparams.reward_conditioned,
            prompt=self.hparams.prompt
        )

        # load MutliMAE encoder

        input_adapters = {
            'rgb': PatchedInputAdapter(
                num_channels=3, stride_level=1,
                patch_size_full=16,
                image_size=224
            )
        }
        if self.hparams.token_decoder == 'tokenlearner':
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

        )

        checkpoint = torch.load(self.hparams.encoder, map_location='cpu')

        checkpoint_model = checkpoint['model']

        for k in list(checkpoint_model.keys()):
            if "output_adapters" in k:
                del checkpoint_model[k]

        # Interpolate position embedding
        interpolate_pos_embed_multimae(multimae, checkpoint_model)

        # Load pre-trained model
        msg = multimae.load_state_dict(checkpoint_model, strict=False)

        if self.hparams.training_phase == 2:
            image_embedder = FrozenClipImageEmbedder('ViT-B/16')
            image_embedder.freeze()
            object_embedder = FrozenCLIPTextEmbedder('ViT-B/16')
            object_embedder.freeze()
            point_embedder = PointEmbedder(2, 512)
            goal_encoder = [point_embedder, image_embedder, object_embedder]

        elif self.hparams.training_phase == 1:
            goal_encoder = torch.nn.Sequential()
        self.net = GPT(multimae, goal_encoder, mconf)

        if self.hparams.load_model_from:
            # print
            pretrain_weight = torch.load(self.hparams.load_model_from, map_location='cpu')

            model_dict = self.net.state_dict()
            # print('-+'*100)
            if 'state_dict' in pretrain_weight.keys():
                pretrain_weight = pretrain_weight['state_dict']
                # print(pretrain_weight)
            pw = {}
            for pn in pretrain_weight:

                _pn = pn.replace('net.', '')
                if not 'mask' in _pn:
                    pw[_pn] = pretrain_weight[pn]

            pw = {k: v for k, v in pw.items() if k in model_dict}
            # # 2. overwrite entries in the existing state dict
            # print('-+'*100)
            for w in pw:
                print(w)
                # print(pw)
            msg = model_dict.update(pw)
            print(msg)
            msg = self.net.load_state_dict(model_dict, strict=False)

        else:
            assert "agent type not supported"

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("DTModel")

        parser.add_argument(
            "--model_type", type=str, default="gpt",
            choices=["multitask", "meta", 'objectnav', 'imagenav', 'pointnav', "gpt"]
        )
        parser.add_argument("--n_embd", type=int, default=128)
        parser.add_argument("--lr", type=float, default=6e-4)
        parser.add_argument("--forward", default=False, action="store_true")
        parser.add_argument("--token_decoder", type=str, default='tokenlearner')
        parser.add_argument("--inverse", default=False, action="store_true")
        parser.add_argument("--reward_conditioned", default=False, action="store_true")
        parser.add_argument("--reward", default=False, action="store_true")
        parser.add_argument("--rand_inverse", default=False, action="store_true")
        parser.add_argument("--rand_mask_size", type=int, default=30)
        parser.add_argument("--mask_obs_size", type=int, default=0)
        parser.add_argument("--n_layer", type=int, default=6)
        parser.add_argument("--n_head", type=int, default=8)
        parser.add_argument("--action_bins", type=int, default=256)
        parser.add_argument("--supervised", default=False, action="store_true")
        parser.add_argument("--prompt", default=False, action="store_true")
        parser.add_argument("--model_save_path", type=str, default="checkpoints")

        # weights
        parser.add_argument("--forward_weight", type=float, default=1.0)

        # layers
        parser.add_argument("--rtg_layers", type=int, default=1)
        parser.add_argument("--bc_layers", type=int, default=1)
        parser.add_argument("--pred_layers", type=int, default=1)
        # STGPT

        parser.add_argument("--ft_params", type=int, default=0)
        parser.add_argument("--curriculum", type=int, default=0)

        # DMC

        return parent_parser

    def calc_topk_accuracy(self, output, target, topk=(1,)):
        """
        Modified from: https://gist.github.com/agermanidis/275b23ad7a10ee89adccf021536bb97e
        Given predicted and ground truth labels, 
        calculate top-k accuracies.
        """
        maxk = max(topk)
        batch_size, seq_len = target.size(0), target.size(1)

        # reshape
        output = output.view(batch_size * seq_len, -1)
        target = target.view(batch_size * seq_len, -1)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            # correct_k = correct[:k].view(-1).float().sum(0)
            correct_k = correct[:k].flatten().float().sum(0)
            # res.append(correct_k.mul_(1 / batch_size))
            res.append(correct_k.mul_(1 / (batch_size * seq_len)))
        return res

    def training_step(self, batch: Any, batch_idx: int):
        # obs, actions, rtg, ts, rewards, task_ids = batch
        # obs = batch['rgb']
        # actions=batch['act']
        # ts=batch['timesteps']
        obs, obs_vector, actions, ts, image_goal, gps_goal, pg_goal, object_goal, task_type, task_desc, rward = batch
        B, T, C, H, W = obs.shape

        ts = ts.view(B, T, 1)
        rward = rward.view(B, T, 1)
        obs_vector = obs_vector.view(B, T, -1)

        obs = {'rgb': obs.view(-1, C, H, W).float()}
        targets = actions
        goals = None
        if self.hparams.training_phase == 2:
            goals = {
                'image': None,
                'object': None,
                'meta': None
            }

            imagenav = []
            objectnav = []
            meta = []
            for i in range(len(task_type)):

                if task_type[i] == 1:
                    imagenav.append(image_goal[i,])
                elif task_type[i] == 2:
                    objectnav.append(object_goal[i])
                elif task_type[i] == 3:
                    meta.append(task_desc[i])
            if len(imagenav) > 0:
                goals["image"] = torch.stack(imagenav)
            if len(objectnav) > 0:
                goals["object"] = objectnav
            if len(meta) > 0:
                goals["meta"] = meta

        rand_mask_size = self.hparams.rand_mask_size
        mask_obs_size = self.hparams.mask_obs_size
        if self.hparams.rand_inverse:
            if self.hparams.rand_mask_size < 0:
                rand_mask_size = max(1,
                                     int(self.hparams.context_length * (self.current_epoch + 1) / self.hparams.epochs))

                self.log(f"train/mask_size", rand_mask_size, on_step=False, on_epoch=True, prog_bar=False,
                         sync_dist=True)
            else:
                rand_mask_size = self.hparams.rand_mask_size

            if self.hparams.mask_obs_size < 0:
                mask_obs_size = min(
                    self.hparams.context_length // 2,
                    max(1, int(self.hparams.context_length * (self.current_epoch + 1) / self.hparams.epochs)),
                )
                self.log(f"train/mask_obs_size", mask_obs_size, on_step=False, on_epoch=True, prog_bar=False,
                         sync_dist=True)
            else:
                mask_obs_size = self.hparams.mask_obs_size

        logits, all_losses = self.net(
            obs,
            actions.float(),
            targets,
            goals,
            ts,
            obs_vector,
            task_type,
            rewards=rward,
            pred_forward=self.hparams.forward,
            pred_inverse=self.hparams.inverse,
            pred_reward=self.hparams.reward,
            pred_rand_inverse=self.hparams.rand_inverse,
            rand_mask_size=rand_mask_size,
            mask_obs_size=mask_obs_size,
            forward_weight=self.hparams.forward_weight,
        )

        avg_loss = 0
        # print(all_losses)
        for name, loss in all_losses.items():
            self.log(f"train/{name}", loss, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
            avg_loss += loss

        avg_loss /= len(all_losses.keys())
        # log train metrics
        self.log("train/avg_loss", avg_loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

        return {"loss": avg_loss, "logits": logits}

    def training_epoch_end(self, outputs) -> None:

        # remove multi_node version
        # if self.hparams.dist_on_itp:
        #     if 'OMPI_COMM_WORLD_RANK' in os.environ:
        #             output_dir = os.path.join(AMLT_MODEL_DIR, self.hparams.output_dir,'checkpoints')
        #             os.makedirs(output_dir, exist_ok=True)
        #             rank = int(os.environ['OMPI_COMM_WORLD_RANK'])

        #             if rank==0:
        #                 torch.save(self.net.state_dict(),os.path.join(output_dir,'checkpoint_'+str(self._epoch)+'.ckpt'))
        #                 self._epoch+=1
        # else:
        #     self._epoch+=1

        return super().training_epoch_end(outputs)

    def configure_optimizers(self):
        if self.hparams.ft_params == 1:
            return self.net.configure_finetune_optimizer(self.hparams)
        elif self.hparams.ft_params == 2:
            return self.net.configure_finetune_4567_optimizer(self.hparams)
        elif self.hparams.ft_params == 3:
            return self.net.configure_xattn_optimizer(self.hparams)
        elif self.hparams.ft_params == 4:
            return self.net.configure_finetune_freeze_enc_optimizer(self.hparams)
        elif self.hparams.ft_params == 5:
            return self.net.configure_finetune_0123_optimizer(self.hparams)
        elif self.hparams.training_phase == 2:
            return self.net.configure_finetune_optimizer(self.hparams)
        elif self.hparams.training_phase == 1:
            return self.net.configure_optimizers(self.hparams)
        else:
            print('configure_xattn_optimizer')
            return self.net.configure_xattn_optimizer(self.hparams)
