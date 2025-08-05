import logging
from typing import Any, List, Optional, Tuple
import torch
import torch.nn as nn
from einops import rearrange, reduce, repeat
from omegaconf import DictConfig
from torchvision import models
import src.model.clip_backbone as clip_backbone
from src.configs import NON_VISUAL_NUM

logger = logging.getLogger(__name__)


# Using PyTorch Transformer encoder class
class Scorer(nn.Module):
    """Main Scorer Module."""

    def __init__(
        self, scorer_config: DictConfig, vocab_size: int, mix_visual_feat: bool = False
    ):
        """Initialize Scorer."""
        super().__init__()

        self.learning_rate = scorer_config.learning_rate
        self.ratio_img_backbone = scorer_config.ratio_img_backbone
        self._scorer_config = scorer_config
        assert (
            scorer_config.attr_dim == 5
        ), "invalide attribute dim for layout annotation"
        self.vocab_size = vocab_size
        self.mix_visual_feat = mix_visual_feat
        self.attr_dim = scorer_config.attr_dim
        self.device = scorer_config.device
        self.transform = None
        self.trans_enc_heads = scorer_config.trans_enc_heads
        self.trans_enc_layers = scorer_config.trans_enc_layers
        self.drop_out = scorer_config.drop_out
        self.img_backbone_name = scorer_config.img_backbone_name
        self.channel_mode = scorer_config.channel_mode
        if not self._scorer_config.shared_layout_codebook:
            self.layout_codebook = nn.Embedding(vocab_size, scorer_config.embed_size)
        assert self.channel_mode in ["both", "visual", "layout"], "Invalid channel mode."
        assert self._scorer_config.visual_emb_mode in ["both", "clip", "tok"], "Invalid visual emb mode."
        # default case
        self.l_embed_dim = scorer_config.embed_size
        self.i_embed_dim = scorer_config.embed_size
        self.trans_enc_dim = scorer_config.embed_size
        if self._scorer_config.use_fuse_v2:
            self.ln_concat = nn.LayerNorm(self.trans_enc_dim)
        else:
            self.lay_1x1conv = nn.Conv1d(scorer_config.embed_size, self.l_embed_dim, 1)
            self.ln_layout = nn.LayerNorm(self.l_embed_dim)
            self.ln_visual = nn.LayerNorm(self.i_embed_dim)
        self.optimizer: torch.optim.Optimizer
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.trans_enc_dim,
                nhead=self.trans_enc_heads,
                dropout=self.drop_out,
                activation="gelu",
                batch_first=True,
                norm_first=scorer_config.norm_first,
            ),
            num_layers=self.trans_enc_layers,
        )

        # Using output of transformer for classification:
        self.cls_head = nn.Sequential(
                nn.Linear(self.trans_enc_dim, 256),
                nn.LeakyReLU(),
                nn.Dropout(self.drop_out),
                nn.Linear(256, 128),
                nn.LeakyReLU(),
                nn.Dropout(self.drop_out),
                nn.Linear(128, 1),
            )

    def _freeze_except(
        self, model: nn.Module, exception_layers: Tuple[str, ...]
    ) -> None:
        """Freeze all submodules except those in given tuple.

        Args:
            model (nn.Module): current model to freeze.
            exception_layers (Tuple[str,...]): submodules name which we don't want to freeze.
        """
        for param in model.parameters():
            param.requires_grad = False

        for name, param in model.named_parameters():
            if any(layer in name for layer in exception_layers):
                param.requires_grad = True

    def get_param_group(self) -> List[dict]:
        """Return customized optimizer."""
        decay, no_decay = self._config_decay()
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # create the pytorch optimizer object
        optim_groups = [
            {
                "name": "scorer_decay_group",
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": self._scorer_config.weight_decay,
                "betas": self._scorer_config.betas,
                "lr": self.learning_rate,
            },
            {
                "name": "scorer_no_decay_group",
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
                "betas": self._scorer_config.betas,
                "lr": self.learning_rate,
            },
        ]
        return optim_groups

    def forward(
        self,
        seq: torch.LongTensor,
        mask: torch.BoolTensor,
        patch_tensors: torch.Tensor,
        shared_coodebook: nn.Embedding,
    ) -> Any:
        """Forward pass of the scorer."""
        if self._scorer_config.use_fuse_v2:
            concat_embed = self._fuse_v2(
                seq, mask, shared_coodebook
            )
        else:
            concat_embed = self._fuse(
                seq, mask, shared_coodebook
        )
        per_elem_mask = mask[..., 0]
        e_output = self.transformer_encoder(
            concat_embed, src_key_padding_mask=~per_elem_mask
        )
        score = self.cls_head(e_output)
        assert isinstance(score, torch.Tensor)  # Add this to pass mypy check.
        return score

    def _fuse(
        self,
        seq: torch.LongTensor,
        mask: torch.BoolTensor,
        shared_coodebook: Optional[nn.Embedding] = None,
    ) -> torch.Tensor:
        B, S = seq.shape[:2]
        layout_toks = seq[..., 1:6]
        # layout part
        if self._scorer_config.shared_layout_codebook:
            layout_embed_ = shared_coodebook(layout_toks)
        else:
            layout_embed_ = self.layout_codebook(layout_toks)
        layout_embed = self.ln_layout(
            self.lay_1x1conv(
                reduce(layout_embed_, "B S A H -> B S H", "mean").permute(0, 2, 1)
            ).permute(0, 2, 1)
        )
        if self.channel_mode == "layout":
            return layout_embed, None
        # visual part
        visual_tok_embed_ = self._get_tok_embed(seq, mask, shared_coodebook)
        visual_embed = self.ln_visual(visual_tok_embed_)
        # concat visual and layout embedding
        concat_embed = layout_embed + visual_embed
        return concat_embed
    
    def _fuse_v2(self,
        seq: torch.LongTensor,
        mask: torch.BoolTensor,
        shared_coodebook: Optional[nn.Embedding] = None,
    ) -> torch.Tensor:
        embedding_ = shared_coodebook(seq)
        weight_mask = self._create_weight_mask(seq,mask)
        # Apply weights to embedding
        weighted_embedding = embedding_ * weight_mask.unsqueeze(-1)
        # Sum along the attribute dimension
        concat_embed = reduce(weighted_embedding, 'B S A H -> B S H', 'sum')
        # Apply layer normalization
        concat_embed = self.ln_concat(concat_embed)
        return concat_embed
    
    def _config_decay(self):
        # whitelist_weight_modules = torch.nn.Linear
        no_decay = set()
        decay = set()
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name
                # prevent add the same parameter twice
                if fpn in decay | no_decay:
                    continue
                if self._check_nodecay(m, pn):
                    no_decay.add(fpn)
                else:
                    decay.add(fpn)
        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = (decay) & no_decay
        union_params = decay | no_decay 
        assert (
            len(inter_params) == 0
        ), "parameters %s made it into both decay/no_decay/frozen sets!" % (
            str(inter_params),
        )
        assert len(param_dict.keys() - union_params) == 0, (
            "parameters %s were not separated into either decay/no_decay/frozen set!"
            % (str(param_dict.keys() - union_params),)
        )
        return decay, no_decay

    def _check_nodecay(self, m, pn):
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        if pn.endswith("bias"):
            return True
        if pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
            return True
        if pn.endswith("positional_embedding"):
            return True
        return False
    
    def _get_tok_embed(self, seq: torch.LongTensor, mask: torch.BoolTensor, shared_coodebook: nn.Embedding):
        visual_mask = mask.clone()
        visual_mask[..., :NON_VISUAL_NUM] = False
        B, S = seq.shape[:2]
        visual_tok_embed_with_pad = shared_coodebook(seq[..., NON_VISUAL_NUM:])
        H = visual_tok_embed_with_pad.size(-1)
        visual_tok_embed = visual_tok_embed_with_pad.masked_fill(
            repeat(
                ~mask[..., NON_VISUAL_NUM:],
                "B S A-> B S A H",
                H=H,
            ),
            0,
        )
        
        weight_mask = self._create_weight_mask(seq,mask)
        weight_mask = weight_mask[:, :, NON_VISUAL_NUM:]
        
        visual_tok_embed = visual_tok_embed * weight_mask.unsqueeze(-1)
        visual_tok_embed = reduce(visual_tok_embed, "B S A H -> B S H", "sum")
        
        return visual_tok_embed
    
    def _create_weight_mask(self, seq: torch.LongTensor, mask: torch.BoolTensor) -> torch.Tensor:
        B, S, A = seq.shape
        weight_mask = torch.ones(B, S, A, device=seq.device)
        # Identify text elements (assuming the last column of mask indicates text)
        is_text_row = repeat(mask[..., -1], "B S -> B S A", A=A)
        # Set non-visual attributes to False, keeping visual attributes as they are
        is_text = is_text_row.clone()
        is_text[:, :, :NON_VISUAL_NUM] = False
        # Apply weights
        weight_mask = weight_mask.masked_fill(is_text, self._scorer_config.font_img_balance_factor)
        # Set weights for non-text (image) elements
        is_img_row = torch.logical_and(mask, torch.logical_not(is_text))
        is_img = is_img_row.clone()
        is_img[:, :, :NON_VISUAL_NUM] = False
        weight_mask = weight_mask.masked_fill(is_img, 3.0 * self._scorer_config.font_img_balance_factor)
        # Apply padding mask
        return weight_mask.masked_fill(~mask, 0)
