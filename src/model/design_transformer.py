import logging
import math

from einops import rearrange
import torch
import torch.nn as nn
from torch import Tensor, BoolTensor
from typing import Any, Dict, List, Optional
from omegaconf import DictConfig
import torch.nn.init as init
from einops import repeat

logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):

    def __init__(
        self, pe_mode: str, d_model: int, block_size: int, dropout: float = 0.1
    ):
        super().__init__()
        logger.debug(f"Current pe_mode: <{pe_mode}>")
        self.pe_mode = pe_mode
        self.dropout = nn.Dropout(p=dropout)
        if pe_mode == "param":
            self.pe = nn.Parameter(torch.zeros(1, block_size, d_model))
        elif pe_mode == "cosine":
            position = torch.arange(block_size).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
            )
            pe = torch.zeros(block_size, 1, d_model)
            pe[:, 0, 0::2] = torch.sin(position * div_term)
            pe[:, 0, 1::2] = torch.cos(position * div_term)
            # change to batch first
            pe = rearrange(pe, "t b e -> b t e")
            self.register_buffer("pe", pe)
        elif pe_mode == "none":
            self.pe = None
        else:
            raise ValueError(f"pe_mode={pe_mode} not supported")

    def forward(self, x: Tensor) -> Tensor:
        t = x.size(1)
        if self.pe is not None:
            x = x + self.pe[:, :t, :]
        return self.dropout(x)


class DesignTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_embd: int,
        n_head: int,
        n_layer: int,
        block_size: int,
        pe_mode: str,
    ):
        super().__init__()

        self.pos_emb = PositionalEncoding(pe_mode, n_embd, block_size)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=n_embd,
                nhead=n_head,
                dim_feedforward=n_embd * 4,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            num_layers=n_layer,
        )
        self.head = nn.Linear(n_embd, vocab_size)
        self.ln_f = nn.LayerNorm(n_embd)
        self.n_heads = n_head
        self.n_layers = n_layer
        self.n_embd = n_embd

    def forward(self, tok_emb: Tensor, pad_mask: BoolTensor) -> Tensor:
        x = self.pos_emb(tok_emb)
        seq_len = x.size(1)
        # auto-regressive generation
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x)
        x = self.transformer_encoder(x, src_key_padding_mask=~pad_mask,mask=causal_mask,is_causal = True)
        x = self.ln_f(x)
        x = self.head(x)
        return x

    @torch.no_grad()
    def extract_embedding(
        self, tok_emb: Tensor, pad_mask: BoolTensor
    ) -> torch.Tensor:
        x = self.pos_emb(tok_emb)
        seq_len = x.size(1)
        # auto-regressive generation
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x)
        x = self.transformer_encoder(x, src_key_padding_mask=~pad_mask,mask=causal_mask,is_causal = True)
        # return x[mask].mean(dim=0)
        x = self.ln_f(x)
        return self.masked_avg_pooling(x, ~pad_mask)

    def get_param_group(self, train_config: DictConfig) -> List[Dict[str, Any]]:
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
                elif pn.endswith("in_proj_weight") or pn.endswith("out_proj.weight"):
                    # weights of in_proj and out_proj in decoder will be decayed
                    decay.add(fpn)

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
        return optim_groups
    
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