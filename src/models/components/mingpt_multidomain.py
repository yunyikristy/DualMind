import logging
import math
from collections import defaultdict

import torch
import torch.nn as nn
from models.utils import soft_update_params
from torch.nn import functional as F

logger = logging.getLogger(__name__)

import numpy as np


def mask_inverse(config):
    # start, unit_size, remove = 2, 2, 1

    # mask = torch.tril(torch.ones(config.block_size + 1, config.block_size + 1))
    # for i in range(start, config.block_size, unit_size):
    #     for j in range(1, remove + 1):
    #         mask[i, i - j] = 0
    mask = torch.tril(torch.ones(config.block_size + 1, config.block_size + 1))

    if config.token_decoder == 'tokenlearner':
        start, unit_size, remove = 10, 10, 1
        for i in range(config.block_size):
            idx = i // unit_size
            for j in range(unit_size - 1):
                mask[i, idx * unit_size + j] = 1
    else:
        start, unit_size, remove = 3, 3, 1
        for i in range(config.block_size):
            idx = i // unit_size
            for j in range(unit_size - 1):
                mask[i, idx * unit_size + j] = 1

    for i in range(start, config.block_size - 1, unit_size):
        for k in range(i, i + unit_size - 1):
            # print(i)
            for j in range(i - 1, i + remove - 1):
                mask[k, j] = 0
    return mask


def init_mask(config):
    mask = torch.tril(torch.ones(config.block_size + 1, config.block_size + 1))
    if config.token_decoder == 'tokenlearner':

        unit_size = 10
        for i in range(config.block_size):
            idx = i // unit_size
            for j in range(unit_size - 1):
                mask[i, idx * unit_size + j] = 1

    if config.prompt:
        mask = torch.cat([torch.ones(config.block_size + 1, 1), mask], 1)

        mask = torch.cat([torch.cat([torch.ones(1, 10), torch.zeros(1, config.block_size + 1 + 1 - 10)], 1), mask], 0)

    return mask


class GELU(nn.Module):
    def forward(self, input):
        return F.gelu(input)


class GPTConfig:
    """base GPT config, params common to all GPT versions."""

    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k, v in kwargs.items():
            setattr(self, k, v)


class GPT1Config(GPTConfig):
    """GPT-1 like network roughly 125M params."""

    n_layer = 12
    n_head = 12
    n_embd = 768


def build_mlp(
        input_size,
        output_size,
        n_layers,
        size=512,
        activation=nn.ReLU(),
        output_activation=nn.Identity(),
        init_method=None,
        bias=True,
):
    layers = []
    in_size = input_size
    for _ in range(n_layers - 1):
        curr_layer = nn.Linear(in_size, size)
        if init_method is not None:
            curr_layer.apply(init_method)
        layers.append(curr_layer)
        layers.append(activation)
        in_size = size

    last_layer = nn.Linear(in_size, output_size, bias=bias)
    if init_method is not None:
        last_layer.apply(init_method)

    layers.append(last_layer)
    layers.append(output_activation)

    return nn.Sequential(*layers)


class CausalSelfAttention(nn.Module):
    """A vanilla multi-head masked self-attention layer with a projection at the end.

    It is possible to use torch.nn.MultiheadAttention here but I am including an explicit
    implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        # self.register_buffer("mask", torch.tril(torch.ones(config.block_size + 1, config.block_size + 1))
        #                              .view(1, 1, config.block_size + 1, config.block_size + 1))
        # self.register_buffer("inverse_mask", mask_inverse(config)
        #                              .view(1, 1, config.block_size + 1, config.block_size + 1))
        self.n_head = config.n_head

    def forward(self, x, mask):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y


class Block(nn.Module):
    """an unassuming Transformer block."""

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, inputs):
        x, mask = inputs

        x = x + self.attn(self.ln1(x), mask)
        x = x + self.mlp(self.ln2(x))
        outputs = x, mask
        return outputs


class XAttention(nn.Module):
    def __init__(
            self,
            config,
            detach_qk: bool = False,
            fp32_logits: bool = True,
            use_geglu: bool = False,
    ):
        super().__init__()
        self.config = config
        self.dim = self.config.n_embd
        dim = self.dim
        ff_expanding = 4
        self.num_heads = self.config.n_head
        num_heads = self.num_heads
        if self.dim % self.num_heads != 0:
            raise ValueError(
                f"dim ({self.dim}) must be divisible by num_heads ({self.num_heads})."
            )
        self.dim_per_head = self.dim // self.num_heads

        # Layer normalization
        self.layernorm = nn.LayerNorm(dim)
        # Projection matrices
        self.query = nn.Linear(dim, dim, bias=False)
        self.key_value = nn.Linear(dim, 2 * dim, bias=False)
        self.attention_out = nn.Linear(dim, dim, bias=False)

        inner_dim = int(dim * ff_expanding)
        self.ln = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, inner_dim, bias=False)
        self.act = nn.GELU()
        self.linear2 = nn.Linear(inner_dim, dim, bias=False)
        if use_geglu:
            self.gated_layer = nn.Linear(dim, inner_dim, bias=False)
        else:
            self.gated_layer = None

        self._detach_qk = detach_qk
        self._fp32_logits = fp32_logits

    def transpose_for_scores(self, x, channels_per_head):
        new_x_shape = x.size()[:-1] + (self.num_heads, channels_per_head)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3).contiguous()

    def forward(
            self,
            *,
            q,
            kv,
            attention_mask=None,
    ):
        queries = self.layernorm(q)
        queries = self.query(queries)

        keys, values = self.key_value(kv).chunk(2, dim=-1)

        # Reshape channels for multi-head attention.
        # We reshape from (batch_size, time, channels) to (batch_size, num_heads, time, channels per head)
        queries = self.transpose_for_scores(queries, self.dim_per_head)

        keys = self.transpose_for_scores(keys, self.dim_per_head)
        values = self.transpose_for_scores(values, self.dim_per_head)

        # Take the dot product between the queries and keys to get the raw attention scores.
        if self._fp32_logits:
            queries = queries.to(torch.float32)
            keys = keys.to(torch.float32)
        attention_scores = torch.matmul(
            queries, keys.transpose(-1, -2)
        )  # (B, NH, T_q, T_k)

        batch_size, num_heads, q_len, q_head_dim = queries.shape
        _, _, kv_len, v_head_dim = values.shape
        hiddens = self.num_heads * v_head_dim

        attention_scores = attention_scores / math.sqrt(q_head_dim)

        if self._detach_qk:
            attention_scores = attention_scores.detach()
        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = attention_probs.to(values.dtype)

        context_layer = torch.matmul(attention_probs, values)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (hiddens,)
        context_layer = context_layer.view(*new_context_layer_shape)

        # Output projection
        attention_output = self.attention_out(context_layer)
        attention_output = attention_output + q

        ff_output = self.ln(attention_output)
        ff_output = self.linear1(ff_output)
        ff_output = self.act(ff_output)
        if self.gated_layer is not None:
            ff_output = ff_output * self.gated_layer(attention_output)
        ff_output = self.linear2(ff_output)

        output = ff_output + attention_output
        return output

    def invert_attention_mask(self, encoder_attention_mask):
        """
        Invert an attention mask (e.g., switches 0. and 1.).
        Args:
            encoder_attention_mask (`torch.Tensor`): An attention mask.
        Returns:
            `torch.Tensor`: The inverted attention mask.
        """
        if encoder_attention_mask.dim() == 3:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
        if encoder_attention_mask.dim() == 2:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
        # T5 has a mask that can compare sequence ids, we can simulate this here with this transposition
        # Cf. https://github.com/tensorflow/mesh/blob/8d2465e9bc93129b913b5ccc6a59aa97abd96ec6/mesh_tensorflow
        # /transformer/transformer_layers.py#L270
        # encoder_extended_attention_mask = (encoder_extended_attention_mask ==
        # encoder_extended_attention_mask.transpose(-1, -2))
        encoder_extended_attention_mask = encoder_extended_attention_mask.to(
            dtype=self.dtype
        )  # fp16 compatibility
        encoder_extended_attention_mask = (
                                                  1.0 - encoder_extended_attention_mask
                                          ) * torch.finfo(self.dtype).min

        return encoder_extended_attention_mask

    @property
    def dtype(self) -> torch.dtype:
        """
        `torch.dtype`: The dtype of the module (assuming that all the module parameters have the same dtype).
        """
        return get_parameter_dtype(self)


def get_parameter_dtype(parameter):
    """
    Returns the first found floating dtype in parameters if there is one, otherwise returns the last dtype it found.
    """
    last_dtype = None
    for t in parameter.parameters():
        last_dtype = t.dtype
        if t.is_floating_point():
            return t.dtype

    if last_dtype is not None:
        # if no floating dtype was found return whatever the first dtype is
        return last_dtype


class MultiMlpNet(nn.Module):
    def __init__(
            self,
            input_dim: int,
            vocab_size: int,
            output_dim: int,
            pred_layers: int,
    ):
        super().__init__()
        self.mlps = nn.ModuleList()
        for _ in range(vocab_size):
            net = build_mlp(input_dim, output_dim, pred_layers, bias=False)
            self.mlps.append(net)

    def forward(self, x):
        return torch.cat([mlp(x) for mlp in self.mlps], dim=-1)


class GPT(nn.Module):
    """the full GPT language model, with a context size of block_size."""

    def __init__(self, state_encoder, goal_encoder, config):
        super().__init__()

        self.config = config
        self.block_size = config.block_size
        self.training_phase = config.training_phase  # act based on rtgs ('reward_conditioned') or not ('naive')
        self.ct = 0

        # input embedding stem
        # self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)

        # pos embedding
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size + 1, config.n_embd))
        self.global_pos_emb = nn.Parameter(torch.zeros(1, config.max_timestep + 1, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)

        # transformer
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        if not self.config.training_phase == 1:
            if not self.config.prompt:
                self.xattns = nn.ModuleList([XAttention(config) for _ in range(config.n_layer)])
            self.goal_fc = nn.Linear(512, config.n_embd)

        # normalization
        self.ln_f = nn.LayerNorm(config.n_embd)

        if config.token_decoder == 'tokenlearner':
            obs_size = 8 + 1
        else:
            obs_size = 1 + 1
        if self.config.cont_action:
            self.naive_head = build_mlp(config.n_embd, config.vocab_size, config.bc_layers, bias=False)
            self.inverse_pred_head = build_mlp(config.n_embd * 2 * obs_size, config.vocab_size, config.pred_layers,
                                               bias=False)
            self.rand_inverse_pred_head = build_mlp(config.n_embd, config.vocab_size, config.pred_layers, bias=True)
        else:
            self.naive_head = MultiMlpNet(config.n_embd, config.vocab_size, self.config.action_bins, config.bc_layers)
            self.inverse_pred_head = MultiMlpNet(config.n_embd * 2 * obs_size, config.vocab_size,
                                                 self.config.action_bins, config.pred_layers)
            self.rand_inverse_pred_head = MultiMlpNet(config.n_embd, config.vocab_size, self.config.action_bins,
                                                      config.pred_layers)
        # forward prediction head
        self.forward_pred_head = build_mlp(config.n_embd * (obs_size + 1), config.n_embd * obs_size, config.pred_layers,
                                           bias=True)
        # inverse prediction head
        # if config.use_rand_inverse:

        self.state_fc = nn.Linear(768, config.n_embd)
        self.apply(self._init_weights)

        self.obs_vector_encoder = build_mlp(config.obs_vector_dim, config.n_embd, config.pred_layers, bias=False)

        self.state_encoder = state_encoder
        self.target_state_encoder = state_encoder
        self.target_state_encoder.load_state_dict(self.state_encoder.state_dict())
        self.target_obs_vector_encoder = self.obs_vector_encoder
        self.target_obs_vector_encoder.load_state_dict(self.obs_vector_encoder.state_dict())

        if self.config.reward_conditioned:
            self.ret_encoder = nn.Sequential(nn.Linear(1, config.n_embd), nn.Tanh())
            # self.reward_conditioned_head = build_mlp(config.n_embd * 2, config.vocab_size, config.rtg_layers, bias=False)
            self.reward_conditioned_head = MultiMlpNet(config.n_embd * 2, config.vocab_size, self.config.action_bins,
                                                       config.rtg_layers)

        # action encoder
        if self.config.cont_action:
            self.action_encoder = nn.Sequential(nn.Linear(config.vocab_size, config.n_embd), nn.Tanh())
        else:
            # self.action_encoder = nn.Sequential(nn.Linear(config.vocab_size, config.n_embd), nn.Tanh())
            self.action_encoder = torch.nn.Sequential(torch.nn.Embedding(self.config.action_bins, config.n_embd),
                                                      torch.nn.Tanh(), torch.nn.Flatten(2),
                                                      torch.nn.Linear(config.n_embd * config.vocab_size, config.n_embd))

        if self.config.training_phase == 2:
            self.image_goal_encoder = goal_encoder[1]
            self.image_goal_encoder.freeze()
            self.object_goal_encoder = goal_encoder[2]
            self.object_goal_encoder.freeze()

        nn.init.normal_(self.action_encoder[0].weight, mean=0.0, std=0.02)

        self.register_buffer(
            "mask",
            init_mask(config).view(
                1, 1, config.block_size + 1 + int(self.config.prompt), config.block_size + 1 + int(self.config.prompt)
            ),
        )

        self.register_buffer(
            "inverse_mask", mask_inverse(config).view(1, 1, config.block_size + 1, config.block_size + 1)
        )

        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def encode_goal(self, goals, task_type):
        if goals['image'] is not None:
            image_goal_embeddings = self.image_goal_encoder(goals['image'])

        if goals['object'] is not None:
            l = []
            for g in goals['object']:
                l.append('navigate to a ' + g)
            object_goal_embeddings = self.object_goal_encoder(l)
        if goals['meta'] is not None:
            l = []
            for g in goals['meta']:
                l.append(g)
            meta_prompt_embeddings = self.object_goal_encoder(l)
        i_idx = 0
        o_idx = 0
        m_idx = 0
        goal_embeddings = []
        for t in task_type:
            if t == 1:
                goal_embeddings.append(image_goal_embeddings[i_idx, :])
                i_idx += 1
            elif t == 2:
                goal_embeddings.append(object_goal_embeddings[o_idx, :])
                o_idx += 1
            elif t == 3:
                goal_embeddings.append(meta_prompt_embeddings[m_idx, :])
                m_idx += 1
        goal_embeddings = torch.stack(goal_embeddings)
        return goal_embeddings

    def forward(
            self,
            states,
            actions,
            targets=None,
            goals=None,
            timesteps=None,
            obs_vector=None,
            task_types=None,
            rewards=None,
            pred_forward=False,
            pred_inverse=False,
            pred_reward=False,
            pred_rand_inverse=False,
            rand_mask_size=1,
            mask_obs_size=0,
            forward_weight=1,
            rand_attn_only=False,
            naive=False
    ):
        # states: (batch, context_length, 4*84*84)
        # actions: (batch, context_length, n_actions)
        # targets: (batch, context_length, 1)
        # obs_vector: (batch, context_length, obs_dim)
        # timesteps: (batch, 1, 1)

        self.ct += 1  # for debug

        D = self.config.n_embd
        B, T, _ = timesteps.shape
        states['rgb'] = states['rgb'].type(torch.float32).contiguous()
        # print(states.shape)
        state_embeddings = self.state_encoder(states)
        state_embeddings = state_embeddings['rgb'].contiguous()
        state_embeddings = self.state_fc(state_embeddings)

        state_embeddings = state_embeddings.reshape(B, T, -1, D)
        obs_vector_embeddings = self.obs_vector_encoder(obs_vector)
        obs_vector_embeddings = obs_vector_embeddings.reshape(B, T, -1, D)

        # (batch * context_length, n_embd)
        # _actions = torch.zeros(B, T, 1, D).to(state_embeddings.device)

        token_embeddings = torch.cat(
            [state_embeddings, obs_vector_embeddings, torch.zeros(B, T, 1, D).to(actions.device)], 2)
        embdding_length = token_embeddings.shape[2]
        token_embeddings = token_embeddings.reshape(B, -1, D)
        token_embeddings = token_embeddings[:, :token_embeddings.shape[1] - int(targets is None), :]
        if not self.config.training_phase == 1:
            goal_embeddings = self.encode_goal(goals, task_types)
            goal_embeddings = self.goal_fc(goal_embeddings)
            goal_embeddings = goal_embeddings.reshape(B, -1, D)
        if actions is not None:
            if self.config.cont_action:
                action_embeddings = self.action_encoder(actions)
            else:
                action_embeddings = self.action_encoder(
                    actions.type(torch.long).squeeze(-1)
                )  # (batch, context_length, n_embd)
                # print(action_embeddings.shape)
            token_embeddings[:, embdding_length - 1::embdding_length, :] = action_embeddings[:,
                                                                           -T + int(targets is None):, :]

        timesteps = timesteps.reshape(B, T, 1)

        ## position embedding 
        all_global_pos_emb = torch.repeat_interleave(self.global_pos_emb, B, dim=0)  # batch_size, traj_length, n_embd
        position_embeddings = torch.gather(all_global_pos_emb, 1,
                                           torch.repeat_interleave(timesteps.long(), self.config.n_embd, dim=-1))
        position_embeddings = torch.repeat_interleave(position_embeddings.reshape(B, T, 1, self.config.n_embd),
                                                      embdding_length,
                                                      dim=2).reshape(B, -1, self.config.n_embd)[:,
                              :token_embeddings.shape[1], :]
        if not self.config.prompt:
            final_embeddings = self.drop(token_embeddings + position_embeddings)
        else:

            final_embeddings = token_embeddings + position_embeddings
            final_embeddings = self.drop(torch.cat([goal_embeddings, final_embeddings], 1))

        if not self.config.training_phase == 1 and not self.config.prompt:
            x = final_embeddings
            for self_attn, xattn in zip(self.blocks, self.xattns):
                x = xattn(q=x, kv=goal_embeddings)
                x, _ = self_attn((x, self.mask))
        else:
            x, _ = self.blocks((final_embeddings, self.mask))
        x = self.ln_f(x)
        if self.config.prompt:
            # pass
            x = x[:, 1:, :]

        if actions is not None:

            state_output = x.reshape(B, -1, embdding_length, D)[:, :, :-1, :]
            state_last_output = x[:, embdding_length - 2::embdding_length, :]  # predictions from action embeddings

            # state_output = x[:, ::2, :]  # predictions from state embeddings
            action_output = x[:, embdding_length - 1::embdding_length, :]  # predictions from action embeddings
        else:
            state_output = x  # for completeness
            action_output = None

        ## act
        naive_action_logits = None
        ## compute losses
        losses = defaultdict(float)

        naive_action_logits = self.naive_head(state_last_output)
        logits = naive_action_logits

        if naive or self.config.supervised:
            if targets is not None:
                if self.config.reward_conditioned:
                    rtg_embeddings = self.ret_encoder(rewards.type(torch.float32))
                    rtg_action_logits = self.reward_conditioned_head(
                        torch.cat((state_last_output, rtg_embeddings), dim=2))
                    logits = rtg_action_logits

                    rtg_action_logits = rtg_action_logits.reshape(-1, self.config.action_bins).float()

                    if self.config.cont_action:
                        losses["acton_rtg"] = F.mse_loss(rtg_action_logits, targets)
                    else:
                        losses["acton_rtg"] = F.cross_entropy(
                            rtg_action_logits.reshape(-1, rtg_action_logits.size(-1)), targets.long().reshape(-1)
                        )
                else:
                    if self.config.cont_action:
                        losses["acton_naive"] = F.mse_loss(naive_action_logits, targets)
                    else:

                        naive_action_logits = naive_action_logits.reshape(-1, self.config.action_bins).float()
                        losses["acton_naive"] = F.cross_entropy(
                            naive_action_logits.reshape(-1, naive_action_logits.size(-1)), targets.long().reshape(-1)
                        )

        if pred_reward:
            reward_pred = self.reward_pred_head(torch.cat((state_output, action_output), dim=2))
            losses["reward_pred"] = nn.BCEWithLogitsLoss()(reward_pred, rewards)
        if pred_forward:
            next_state_embeddings = self.target_state_encoder(
                states
            )  # (batch, context_length, n_embd)
            next_state_embeddings = next_state_embeddings['rgb'].contiguous()
            next_state_embeddings = self.state_fc(next_state_embeddings)
            next_state_embeddings = next_state_embeddings.reshape(B, T, -1, D)
            next_obs_vector_embeddings = self.target_obs_vector_encoder(obs_vector)
            next_obs_vector_embeddings = next_obs_vector_embeddings.reshape(B, T, -1, D)
            next_state_embeddings = torch.cat([next_state_embeddings, next_obs_vector_embeddings], 2)
            next_state_embeddings = next_state_embeddings.reshape(B, T, -1)

            _action_output = action_output.reshape(B, T, 1, D)[:, : -1, :, :]
            next_state_embeddings = next_state_embeddings[:, 1:, :]  # (batch, context_length-1, n_embd)

            forward_pred = self.forward_pred_head(
                torch.cat((state_output[:, :-1, :, :], _action_output), dim=2).reshape(B, T - 1, -1)
            )
            losses["forward_pred"] = (
                    F.mse_loss(
                        forward_pred.reshape(-1, forward_pred.size(-1)).float(),
                        next_state_embeddings.reshape(-1, next_state_embeddings.size(-1)),
                    )
                    * forward_weight
            )
            soft_update_params(self.state_encoder, self.target_state_encoder, 0.005)
            soft_update_params(self.obs_vector_encoder, self.target_obs_vector_encoder, 0.005)

        if pred_inverse:
            inv_x, _ = self.blocks((final_embeddings, self.inverse_mask))
            inv_x = self.ln_f(inv_x)
            cur_state_output = state_output[:, :-1, :, :].reshape(B, T - 1, -1)  # predictions from cur-state embeddings
            inv_state_output = inv_x.reshape(B, -1, embdding_length, D)[:, :, :-1, :].reshape(B, T, -1)
            next_state_output = inv_state_output[:, 1:, :]  # predictions from next-state embeddings
            inverse_action_logits = self.inverse_pred_head(torch.cat((cur_state_output, next_state_output), dim=2))
            inverse_target = actions[:, : -1 + int(targets is None)]
            if self.config.cont_action:
                losses["inverse_pred"] = F.mse_loss(inverse_action_logits, inverse_target)
            else:
                _inverse_action_logits = inverse_action_logits.reshape(-1, self.config.action_bins).float()

                losses["inverse_pred"] = F.cross_entropy(
                    _inverse_action_logits.reshape(-1, _inverse_action_logits.size(-1)),
                    inverse_target.long().reshape(-1)
                )

        if pred_rand_inverse:
            obs_size = embdding_length - 1
            # randomly mask past actions and predict them
            rand_mask_idx = np.random.choice(actions.shape[1], rand_mask_size, replace=False)
            masked_token = token_embeddings.clone()  # .detach()

            if rand_attn_only:
                masked_token = masked_token.detach()

            for j in range(rand_mask_size):
                # print( obs_size + (obs_size+1)* rand_mask_idx[j])
                masked_token[:, obs_size + (obs_size + 1) * rand_mask_idx[j], :] = -1

            if mask_obs_size > 0:
                assert actions.shape[1] > 2
                rand_mask_obs_idx = np.random.choice(list(range(1, actions.shape[1] - 1)), mask_obs_size, replace=False)
                for j in range(mask_obs_size):
                    start_idx = rand_mask_obs_idx[j]
                    for i in range((obs_size + 1) * (start_idx - 1), (obs_size + 1) * start_idx - 1):
                        masked_token[:, i, :] = -1

            final_masked_embeddings = self.drop(masked_token + position_embeddings)

            temp_mask = (
                torch.ones((self.config.block_size + 1, self.config.block_size + 1))
                .view(1, 1, self.config.block_size + 1, self.config.block_size + 1)
                .to(masked_token.device)
            )
            masked_x, _ = self.blocks((final_masked_embeddings, temp_mask))
            x = self.ln_f(masked_x)
            token_rand_mask_idx = [obs_size + i * (obs_size + 1) for i in rand_mask_idx]
            rand_inverse_logits = self.rand_inverse_pred_head(x[:, token_rand_mask_idx, :])
            rand_inverse_action_targets = actions[:, rand_mask_idx]
            if self.config.cont_action:
                losses["rand_inverse_pred"] = F.mse_loss(rand_inverse_logits, rand_inverse_action_targets)
            else:
                _rand_inverse_logits = rand_inverse_logits.reshape(-1, self.config.action_bins).float()

                losses["rand_inverse_pred"] = F.cross_entropy(
                    _rand_inverse_logits.reshape(-1, _rand_inverse_logits.size(-1)),
                    rand_inverse_action_targets.long().reshape(-1),
                )

        return logits, losses

    def get_action(self,
                   states,
                   actions,
                   targets=None,
                   goals=None,
                   timesteps=None,
                   obs_vector=None,
                   task_types=None):
        # states: (batch, context_length, 4*84*84)
        # actions: (batch, context_length, n_actions)
        # targets: (batch, context_length, 1)
        # rtgs: (batch, context_length, 1)
        # timesteps: (batch, 1, 1)

        self.ct += 1  # for debug

        D = self.config.n_embd
        B, T, _ = timesteps.shape
        states['rgb'] = states['rgb'].type(torch.float32).contiguous()
        # print(states.shape)
        state_embeddings = self.state_encoder(states)
        state_embeddings = state_embeddings['rgb'].contiguous()
        state_embeddings = self.state_fc(state_embeddings)

        obs_vector_embeddings = self.obs_vector_encoder(obs_vector.float())
        state_embeddings = state_embeddings.reshape(B, T, -1, D)

        obs_vector_embeddings = obs_vector_embeddings.reshape(B, T, -1, D)

        _actions = torch.zeros(B, T, 1, D).to(state_embeddings.device)

        token_embeddings = torch.cat([state_embeddings, obs_vector_embeddings, _actions], 2)

        goal_embeddings = self.encode_goal(goals, task_types)

        goal_embeddings = self.goal_fc(goal_embeddings.float())
        goal_embeddings = goal_embeddings.reshape(B, -1, D)

        embdding_length = token_embeddings.shape[2]
        token_embeddings = token_embeddings.reshape(B, -1, D)
        token_embeddings = token_embeddings[:, :token_embeddings.shape[1] - int(targets is None), :]

        if actions is not None:
            if self.config.cont_action:
                action_embeddings = self.action_encoder(actions)
            else:
                action_embeddings = self.action_encoder(
                    actions.type(torch.long).squeeze(-1)
                )  # (batch, context_length, n_embd)

            token_embeddings[:, embdding_length - 1::embdding_length, :] = action_embeddings[:,
                                                                           -T + int(targets is None):, :]

        timesteps = timesteps.reshape(B, T, 1)

        all_global_pos_emb = torch.repeat_interleave(self.global_pos_emb, B, dim=0)  # batch_size, traj_length, n_embd
        position_embeddings = torch.gather(all_global_pos_emb, 1,
                                           torch.repeat_interleave(timesteps.long(), self.config.n_embd, dim=-1))
        position_embeddings = torch.repeat_interleave(position_embeddings.reshape(B, T, 1, self.config.n_embd),
                                                      embdding_length,
                                                      dim=2).reshape(B, -1, self.config.n_embd)[:,
                              :token_embeddings.shape[1], :]
        final_embeddings = self.drop(token_embeddings + position_embeddings)

        for self_attn, xattn in zip(self.blocks, self.xattns):
            x = xattn(q=final_embeddings, kv=goal_embeddings)
            x, _ = self_attn((x, self.mask))

        x = self.ln_f(x)

        naive_action_logits = self.naive_head(x[:, embdding_length - 2::embdding_length, :])

        logits = naive_action_logits

        return logits

    def configure_optimizers(self, hparams):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        # whitelist_weight_modules = (torch.nn.Linear, )
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = f"{mn}.{pn}" if mn else pn  # full param name

                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add("pos_emb")
        no_decay.add("global_pos_emb")
        no_decay.add("state_encoder.global_tokens")
        no_decay.add("state_encoder.input_adapters.rgb.pos_emb")

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, f"parameters {str(inter_params)} made it into both decay/no_decay sets!"
        assert (
                len(param_dict.keys() - union_params) == 0
        ), "parameters {} were not separated into either decay/no_decay set!".format(
            str(param_dict.keys() - union_params)
        )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": hparams.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=hparams.lr, betas=hparams.betas)
        return optimizer

    def configure_xattn_optimizer(self, hparams):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        # whitelist_weight_modules = (torch.nn.Linear, )
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            if not (mn.startswith('object_goal_encoder') or mn.startswith('image_goal_encoder')):
                for pn, p in m.named_parameters():
                    fpn = f"{mn}.{pn}" if mn else pn  # full param name

                    if pn.endswith("bias"):
                        # all biases will not be decayed
                        no_decay.add(fpn)
                    elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                        # weights of whitelist modules will be weight decayed
                        decay.add(fpn)
                    elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                        # weights of blacklist modules will NOT be weight decayed
                        no_decay.add(fpn)
        l = []
        for n in no_decay:
            l.append(n)
        for n in l:
            if n.startswith('object_goal_encoder') or n.startswith('image_goal_encoder'):
                no_decay.remove(n)
        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add("pos_emb")
        no_decay.add("global_pos_emb")
        no_decay.add("state_encoder.global_tokens")
        no_decay.add("state_encoder.input_adapters.rgb.pos_emb")

        # validate that we considered every parameter
        # param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {}
        for name, p in self.named_parameters():
            # print(name)
            if "object_goal_encoder" in name:
                # print("update only",name)
                p.requires_grad = False
            elif 'image_goal_encoder' in name:
                # print("update only",name)
                p.requires_grad = False
            elif 'pos_emb' in name:
                p.requires_grad = False
                param_dict[name] = p
            elif 'global_pos_emb' in name:
                p.requires_grad = False
                param_dict[name] = p
            else:
                param_dict[name] = p

        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, f"parameters {str(inter_params)} made it into both decay/no_decay sets!"
        assert (
                len(param_dict.keys() - union_params) == 0
        ), "parameters {} were not separated into either decay/no_decay set!".format(
            str(param_dict.keys() - union_params)
        )
        for name, p in self.named_parameters():
            # print(name)

            print(name, " ", p.requires_grad)
        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": hparams.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=hparams.lr, betas=hparams.betas)
        return optimizer

    def configure_finetune_optimizer(self, hparams):

        paras = []
        for name, p in self.named_parameters():
            # print(name)
            if "xattns" in name:
                # print("update only",name)
                p.requires_grad = True
                paras.append(p)
            elif 'goal_fc' in name:
                p.requires_grad = True
                paras.append(p)
            elif 'naive_head' in name:
                # print("update only",name)
                p.requires_grad = True
                paras.append(p)
            elif name.startswith('ret_encoder'):
                p.requires_grad = True
                paras.append(p)
            elif name.startswith('reward_conditioned_head'):
                p.requires_grad = True
                paras.append(p)
            else:
                p.requires_grad = False

        for name, p in self.named_parameters():
            # print(name)

            print(name, " ", p.requires_grad)
        optimizer = torch.optim.AdamW(paras, lr=hparams.lr, betas=hparams.betas)
        return optimizer

    def configure_finetune_4567_optimizer(self, hparams):
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        # whitelist_weight_modules = (torch.nn.Linear, )
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            if not (mn.startswith('object_goal_encoder') or mn.startswith('image_goal_encoder')):
                for pn, p in m.named_parameters():
                    fpn = f"{mn}.{pn}" if mn else pn  # full param name

                    if pn.endswith("bias"):
                        # all biases will not be decayed
                        no_decay.add(fpn)
                    elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                        # weights of whitelist modules will be weight decayed
                        decay.add(fpn)
                    elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                        # weights of blacklist modules will NOT be weight decayed
                        no_decay.add(fpn)
        l = []
        for n in no_decay:
            l.append(n)
        for n in l:
            if not (n.startswith('blocks.4') or n.startswith('goal_fc') or n.startswith('blocks.5') or n.startswith(
                    'blocks.6') or n.startswith('blocks.7') or n.startswith('naive_head') or n.startswith('xattns')):
                no_decay.remove(n)
        l = []
        for n in decay:
            l.append(n)
        for n in l:
            if not (n.startswith('blocks.4') or n.startswith('goal_fc') or n.startswith('blocks.5') or n.startswith(
                    'blocks.6') or n.startswith('blocks.7') or n.startswith('naive_head') or n.startswith('xattns')):
                decay.remove(n)

        param_dict = {}
        for name, p in self.named_parameters():
            if name.startswith("xattns"):
                p.requires_grad = True
                param_dict[name] = p
            elif name.startswith('naive_head'):
                p.requires_grad = True
                param_dict[name] = p
            elif name.startswith('ret_encoder'):
                p.requires_grad = True
                param_dict[name] = p
            elif name.startswith('reward_conditioned_head'):
                p.requires_grad = True
                param_dict[name] = p
            elif name.startswith('blocks.4'):
                p.requires_grad = True
                param_dict[name] = p
            elif name.startswith('blocks.5'):
                p.requires_grad = True
                param_dict[name] = p
            elif name.startswith('blocks.6'):
                p.requires_grad = True
                param_dict[name] = p
            elif name.startswith('blocks.7'):
                p.requires_grad = True
                param_dict[name] = p
            elif 'goal_fc' in name:
                p.requires_grad = True
                param_dict[name] = p
            else:
                p.requires_grad = False

        for name, p in self.named_parameters():
            # print(name)

            print(name, " ", p.requires_grad)
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": hparams.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(optim_groups, lr=hparams.lr, betas=hparams.betas)
        return optimizer

    def configure_finetune_0123_optimizer(self, hparams):
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        # whitelist_weight_modules = (torch.nn.Linear, )
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            if not (mn.startswith('object_goal_encoder') or mn.startswith('image_goal_encoder')):
                for pn, p in m.named_parameters():
                    fpn = f"{mn}.{pn}" if mn else pn  # full param name

                    if pn.endswith("bias"):
                        # all biases will not be decayed
                        no_decay.add(fpn)
                    elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                        # weights of whitelist modules will be weight decayed
                        decay.add(fpn)
                    elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                        # weights of blacklist modules will NOT be weight decayed
                        no_decay.add(fpn)
        l = []
        for n in no_decay:
            l.append(n)
        for n in l:
            if not (n.startswith('blocks.0') or n.startswith('goal_fc') or n.startswith('blocks.1') or n.startswith(
                    'blocks.2') or n.startswith('blocks.3') or n.startswith('naive_head') or n.startswith('xattns')):
                no_decay.remove(n)
        l = []
        for n in decay:
            l.append(n)
        for n in l:
            if not (n.startswith('blocks.0') or n.startswith('goal_fc') or n.startswith('blocks.1') or n.startswith(
                    'blocks.2') or n.startswith('blocks.3') or n.startswith('naive_head') or n.startswith('xattns')):
                decay.remove(n)

        param_dict = {}
        for name, p in self.named_parameters():
            if name.startswith("xattns"):
                p.requires_grad = True
                param_dict[name] = p
            elif name.startswith('naive_head'):
                p.requires_grad = True
                param_dict[name] = p
            elif name.startswith('ret_encoder'):
                p.requires_grad = True
                param_dict[name] = p
            elif name.startswith('reward_conditioned_head'):
                p.requires_grad = True
                param_dict[name] = p
            elif name.startswith('blocks.0'):
                p.requires_grad = True
                param_dict[name] = p
            elif name.startswith('blocks.1'):
                p.requires_grad = True
                param_dict[name] = p
            elif name.startswith('blocks.2'):
                p.requires_grad = True
                param_dict[name] = p
            elif name.startswith('blocks.3'):
                p.requires_grad = True
                param_dict[name] = p
            elif 'goal_fc' in name:
                p.requires_grad = True
                param_dict[name] = p
            else:
                p.requires_grad = False

        for name, p in self.named_parameters():
            # print(name)

            print(name, " ", p.requires_grad)
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": hparams.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(optim_groups, lr=hparams.lr, betas=hparams.betas)
        return optimizer

    def configure_finetune_freeze_enc_optimizer(self, hparams):
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        # whitelist_weight_modules = (torch.nn.Linear, )
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            if not (mn.startswith('object_goal_encoder') or mn.startswith('image_goal_encoder')):
                for pn, p in m.named_parameters():
                    fpn = f"{mn}.{pn}" if mn else pn  # full param name

                    if pn.endswith("bias"):
                        # all biases will not be decayed
                        no_decay.add(fpn)
                    elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                        # weights of whitelist modules will be weight decayed
                        decay.add(fpn)
                    elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                        # weights of blacklist modules will NOT be weight decayed
                        no_decay.add(fpn)
        l = []
        for n in no_decay:
            l.append(n)
        for n in l:
            if not (n.startswith('blocks') or n.startswith('naive_head') or n.startswith('xattns') or n.startswith(
                    'goal_fc')):
                no_decay.remove(n)
        l = []
        for n in decay:
            l.append(n)
        for n in l:
            if not (n.startswith('blocks') or n.startswith('naive_head') or n.startswith('xattns') or n.startswith(
                    'goal_fc')):
                decay.remove(n)

        param_dict = {}
        for name, p in self.named_parameters():
            # print(name)

            if name.startswith("xattns"):
                # print("update only",name)
                p.requires_grad = True
                param_dict[name] = p
            elif name.startswith('naive_head'):
                # print("update only",name)
                p.requires_grad = True
                param_dict[name] = p
            elif name.startswith('ret_encoder'):
                p.requires_grad = True
                param_dict[name] = p
            elif name.startswith('reward_conditioned_head'):
                p.requires_grad = True
                param_dict[name] = p
            elif name.startswith('blocks'):
                p.requires_grad = True
                param_dict[name] = p
            elif 'goal_fc' in name:
                p.requires_grad = True
                param_dict[name] = p
            else:
                p.requires_grad = False

        for name, p in self.named_parameters():
            # print(name)

            print(name, " ", p.requires_grad)

        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": hparams.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(optim_groups, lr=hparams.lr, betas=hparams.betas)
        return optimizer
