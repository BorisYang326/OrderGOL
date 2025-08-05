import torch
import torch.nn as nn
from typing import Any, Dict, List
from omegaconf import DictConfig
class Codebook(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_embd: int,
        pad_tok_idx:int
    ):
        super().__init__()
        self._vocab_size = vocab_size
        self._n_embd = n_embd
        self.emb = nn.Embedding(vocab_size, n_embd, padding_idx=pad_tok_idx)
        
    def forward(self, x):
        assert torch.all(x < self._vocab_size), "token index overflow."
        return self.emb(x)
    
    
    def get_param_group(self,trainer_config: DictConfig,) -> List[Dict[str, Any]]:
        # separate out all parameters to those that will and won't experience regularizing weight decay
        optim_groups = [
            {
                "name": "codebook_decay_group",
                "params": self.emb.parameters(),
                "weight_decay": trainer_config.weight_decay,
                "lr": trainer_config.learning_rate,
                "betas": trainer_config.betas,
            },
        ]
        return optim_groups
    
    @property
    def vocab_size(self):
        return self._vocab_size
    
    @property
    def n_embd(self):
        return self._n_embd
    
    def _set_requires_grad(self, requires_grad: bool):
        for param in self.parameters():
            param.requires_grad = requires_grad