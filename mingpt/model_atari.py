"""
The MIT License (MIT) Copyright (c) 2020 Andrej Karpathy
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

https://github.com/karpathy/minGPT/blob/master/LICENSE
"""

import torch
import math
import torch.nn as nn
from torch.nn import functional as F

# replace NewGELU as nn.GELU()
# class NewGELU(nn.Module):
#     """
#     Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
#     Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
#     """
#     def forward(self, x):
#         return 0.5 * x (1.0 + torch.tanh(math.sqrt(2 / math.pi) * (x+0.044715 * torch.pow(x, 3))))

class GPTConfig:
    attn_pdrop = 0.1
    resid_pdrop = 0.1
    embd_pdrop = 0.1
    def __init__(self, step_size, max_timestep, 
                 vocab_size, n_head, n_layer, n_embd, device='cuda'):
        self.block_size = step_size * 3
        self.max_timestep = max_timestep
        self.vocab_size = vocab_size
        self.n_head = n_head
        self.n_layer = n_layer
        self.n_embd = n_embd  # each head has n_embd / n_head
        self.device = device

class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head
        self.n_embd = config.n_embd
    
    def forward(self, x):
        B, T, C = x.size() # batch, sequence length, embeding dimension(n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class Block(nn.Module):
    """
    An unassuming Transformer block.
    """
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd),
            act     = nn.GELU(),
            c_proj  = nn.Linear(4 * config.n_embd, config.n_embd),
            dropout = nn.Dropout(config.resid_pdrop)
        ))
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x)))) # MLP forward

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlpf(self.ln_2(x))
        return x
    
# embedding action, state, and rtg
class Embeddings_Atari(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.state_embedding = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, config.n_embd),
            nn.Tanh()
        )

        self.action_embedding = nn.Sequential(
            # using embedding layer for action, since it's discrete
            nn.Embedding(config.vocab_size, config.n_embd),
            nn.Tanh()
        )

        self.rtgs_embedding = nn.Sequential(
            nn.Linear(1, config.n_embd),
            nn.Tanh()
        )

        self.apply(self._init_weights)
    
    # see karpathy/minGPT for weight's initilization in OpenAI GPT
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, states, actions, rtgs):
        ### inputs
        # rtgs        : (batch_size, step_size, 1)
        # states      : (batch_size, step_size, 4, 84, 84)
        # actions     : (batch_size, step_size)
        rtgs_emb = self.rtgs_embedding(rtgs)
        
        states_shp = states.reshape(-1, 4, 84, 84)
        states_emb = self.state_embedding(states_shp)
        states_emb = states_emb.reshape(states.shape[0], states.shape[1], states_emb.shape[1])

        if actions is None:
            actions_emb = None
        else:
            actions_emb = self.action_embedding(actions)
        
        return rtgs_emb, states_emb, actions_emb


class GPT(nn.Module): # for Decision Transformer
    def __init__(self, config):
        super().__init__()

        self.block_size = config.block_size
        self.n_embd = config.n_embd

        # embedding action, state, and rtg
        self.embedding_atari = Embeddings_Atari(config).to(config.device)

        # build modules
        # official DT method
        # self.global_timestep_encoding = nn.Parameter(torch.zeros(1, config.max_timestep+1, config.n_embd))
        # self.context_position_encoding = nn.Parameter(torch.zeros(1, config.block_size + 1, config.n_embd))

        # minGPT method
        self.global_timestep_encoding = nn.Embedding(config.max_timestep, config.n_embd)
        self.context_position_encoding = nn.Embedding(config.block_size, config.n_embd)
        self.dropout = nn.Dropout(config.embd_pdrop)

        # transformer
        self.block_loop = nn.ModuleList([Block(config) for _ in range(config.n_layer)])

        # decoder head
        self.norm = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # initialize weights
        self.apply(self._init_weights)

    # see karpathy/minGPT for weight's initilization in OpenAI GPT
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                 # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)
        
        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )
        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer
    
    def forward(self, states, actions, rtgs, timesteps):
        ### inputs
        # rtgs        : (batch_size, step_size, 1)
        # states      : (batch_size, step_size, 4, 84, 84)
        # actions     : (batch_size, step_size)
        # timesteps   : (batch_size, step_size)  <-- but only the first step is used (other steps are ignored)

        rtgs_emb, states_emb, actions_emb = self.embedding_atari(states, actions, rtgs)
        # rtgs_emb    : (batch_size, step_size, n_embd)
        # states_emb  : (batch_size, step_size, n_embd)
        # actions_emb : (batch_size, step_size, n_embd)

        batch_size = states_emb.shape[0]
        actual_step_size = states_emb.shape[1]

        #
        # Generate a sequence of tokens :
        # [s], [a], [R] --> [R, s, a, R, s, a, ...]
        #

        token_emb = torch.zeros(
            (batch_size, actual_step_size*3, self.n_embd),
            dtype=torch.float32,
            device=states_emb.device) # (batch_size, step_size*3, n_embd)
        token_emb[:,::3,:] = rtgs_emb
        token_emb[:,1::3,:] = states_emb
        if actions_emb is not None:
            token_emb[:,2::3,:] = actions_emb
        
        #
        # Position encoding
        #

        timestep_start = torch.repeat_interleave(timesteps[:,0].unsqueeze(dim=-1), actual_step_size*3, dim=-1) # (batch_size, actual_step_size*3)
        pos_global = self.global_timestep_encoding(timestep_start)
        context_position = torch.arange(actual_step_size*3, device=states_emb.device).repeat(batch_size,1) # (batch_size, actual_step_size*3)
        pos_relative = self.context_position_encoding(context_position)
        pos_emb = pos_global + pos_relative

        x = self.dropout(token_emb + pos_emb)
    
        #
        # Apply multi-layered MHA (multi-head attentions)
        #

        for block in self.block_loop:
            x = block(x)

        x = self.norm(x)

        #
        # Apply Feed-Forward and Return
        #

        logits = self.lm_head(x)
        # only get predictions from states
        logits = logits[:,1::3,:]

        return logits

    def select_action(self, states, actions, rtgs, timesteps):
        """
        Select an action based on the current state, action, and rtg.
        Corresponds to the "sample" function in mingpt/utils of the original DT code.
        https://github.com/kzl/decision-transformer/blob/master/atari/mingpt/utils.py
        Because of the parameterization of the "sample" function, 
            this code functions the same as the "sample" of riginal code
        """
        logits = self.forward(states, actions, rtgs, timesteps)
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        sampled_action = torch.multinomial(probs, num_samples=1)
        return sampled_action






