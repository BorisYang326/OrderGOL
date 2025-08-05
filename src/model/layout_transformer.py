"""GPT model:
- the initial stem consists of a combination of token encoding and a positional encoding
- the meat of it is a uniform sequence of Transformer blocks
    - each Transformer is a sequential combination of a 1-hidden-layer MLP block and a self-attention block
    - all blocks feed into a central residual pathway similar to resnets
- the final decoder is a linear projection into a vanilla Softmax classifier.
"""

import logging
import math
from typing import Any, Dict, List, Optional

from einops import rearrange
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.nn import functional as F
from torch import Tensor
import torch.nn.init as init

logger = logging.getLogger(__name__)


class LayoutTransformerConfig:
    """base GPT config, params common to all GPT versions."""

    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k, v in kwargs.items():
            setattr(self, k, v)


class GPT1Config(LayoutTransformerConfig):
    """GPT-1 like network roughly 125M params."""

    n_layer = 12
    n_head = 12
    n_embd = 768


class CausalSelfAttention(nn.Module):
    """A vanilla multi-head masked self-attention layer with a projection at the end."""

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        block_size: int,
        attn_pdrop: float,
        resid_pdrop: float,
    ):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size)).view(
                1, 1, block_size, block_size
            ),
        )
        self.n_head = n_head

    def forward(self, x, key_padding_mask=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = (
            self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        )  # (B, nh, T, hs)
        q = (
            self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        )  # (B, nh, T, hs)
        v = (
            self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        )  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        # Apply the causal mask
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))

        # Apply the key_padding_mask
        if key_padding_mask is not None:
            # key_padding_mask: (B, T) -> (B, 1, 1, T) for broadcasting
            att = att.masked_fill(
                key_padding_mask[:, None, None, :], float("-inf")
            )

        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y

class PositionalEncoding(nn.Module):

    def __init__(self, pe_mode:str,d_model: int,block_size:int, dropout: float = 0.1):
        super().__init__()
        logger.debug(f"Current pe_mode: <{pe_mode}>")
        self.pe_mode = pe_mode
        self.dropout = nn.Dropout(p=dropout)
        if pe_mode == "param":
            self.pe = nn.Parameter(torch.zeros(1, block_size, d_model))
        elif pe_mode == "cosine":
            position = torch.arange(block_size).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
            pe = torch.zeros(block_size, 1, d_model)
            pe[:, 0, 0::2] = torch.sin(position * div_term)
            pe[:, 0, 1::2] = torch.cos(position * div_term)
            # change to batch first
            pe = rearrange(pe, 't b e -> b t e')
            self.register_buffer('pe', pe)
        elif pe_mode == "none":
            self.pe = None
        else:
            raise ValueError(f"pe_mode={pe_mode} not supported")

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        t = x.size(1)
        if self.pe is not None:
            x = x + self.pe[:, :t, :]
        return self.dropout(x)


class Block(nn.Module):
    """an unassuming Transformer block."""

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        block_size: int,
        attn_pdrop: float,
        resid_pdrop: float,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(
            n_embd, n_head, block_size, attn_pdrop, resid_pdrop
        )
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x,key_padding_mask=None):
        x = x + self.attn(self.ln1(x),key_padding_mask)
        x = x + self.mlp(self.ln2(x))
        return x


class LayoutTransformer(nn.Module):
    """the full GPT language model, with a context size of block_size."""

    def __init__(
        self,
        vocab_size: int,
        n_embd: int,
        n_head: int,
        n_layer: int,
        pe_mode: str,
        block_size: int,
        embd_pdrop: float,
        resid_pdrop: float,
        attn_pdrop: float,
    ):
        super().__init__()

        # input embedding stem
        # self.tok_emb = nn.Embedding(vocab_size, n_embd)
        # DEPRECATED: wrong replacement of nn.Linear to nn.Embedding
        # self.tok_ln = nn.Linear(1, n_embd)
        self.pos_emb = PositionalEncoding(pe_mode, n_embd, block_size, dropout=embd_pdrop)
        self.drop = nn.Dropout(embd_pdrop)
        # transformer blocks
        self.blocks = nn.ModuleList([
            Block(n_embd, n_head, block_size, attn_pdrop, resid_pdrop)
            for _ in range(n_layer)
        ])
        # decoder head
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        # self.img_token_range = img_token_range
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_heads = n_head
        self.n_layers = n_layer
        self.n_embd = n_embd
        # self.apply(self._init_weights)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)  # Xavier initialization
                if module.bias is not None:
                    init.constant_(module.bias, 0)  # Initialize bias to 0
            elif isinstance(module, nn.LayerNorm):
                init.constant_(module.weight, 1)  # Initialize LayerNorm weight to 1
                init.constant_(module.bias, 0)  # Initialize LayerNorm bias to 0
            elif isinstance(module, nn.Embedding):
                init.normal_(module.weight, mean=0, std=0.01)  # Initialize Embedding with normal distribution

    def get_param_group(self, train_config: DictConfig) -> List[Dict[str, Any]]:
        """This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name

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
        if self.pos_emb.pe_mode == "param":
            no_decay.add("pos_emb.pe")

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
            len(inter_params) == 0
        ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
            len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (
            str(param_dict.keys() - union_params),
        )

        # create the pytorch optimizer object
        optim_groups = [
            {
                "name": "model_decay_group",
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": train_config.weight_decay,
                "lr": train_config.learning_rate,
            },
            {
                "name": "model_no_decay_group",
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
                "lr": train_config.learning_rate,
            },
        ]
        # optimizer = torch.optim.AdamW(
        #     optim_groups, lr=learning_rate, betas=train_config.betas
        # )
        return optim_groups

    def forward(
        self,
        token_embeddings,
        pad_mask=None,
    ):
        b, t, emb = token_embeddings.size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."
        x = self.pos_emb(token_embeddings)
        for block in self.blocks:
            x = block(x, key_padding_mask=~pad_mask)  # Manually pass pad_mask to each block
        x = self.ln_f(x)
        x = self.head(x)
        return x

    @torch.no_grad()
    def extract_embedding(
        self,
        token_embeddings,
        pad_mask=None,
        # mask:Optional[torch.Tensor]=None,
    ) -> torch.Tensor:
        b, t, emb = token_embeddings.size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."

        # forward the GPT model

        # NOTE: Use original transformer
        # token_embeddings = idx
        x = self.pos_emb(token_embeddings)
        for block in self.blocks:
            x = block(x, key_padding_mask=~pad_mask)  # Manually pass pad_mask to each block
        x = self.ln_f(x)
        return self.masked_avg_pooling(x, ~pad_mask)

    def _set_requires_grad(self, requires_grad: bool) -> None:
        for param in self.parameters():
            param.requires_grad = requires_grad
            
    def masked_avg_pooling(self, encoder_output, key_padding_mask):
        key_padding_mask = torch.logical_not(key_padding_mask).unsqueeze(-1)
        masked_output = encoder_output * key_padding_mask
        token_count = key_padding_mask.sum(dim=1)
        token_count = token_count.where(token_count != 0, torch.ones_like(token_count))
        summed = masked_output.sum(dim=1)
        averaged = summed / token_count
        return averaged