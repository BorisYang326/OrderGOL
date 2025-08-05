from typing import Optional
import torch
from torch import nn, Tensor
import logging
from einops import repeat, rearrange
import math
import os
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
        """Arguments:
        x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        t = x.size(1)
        if self.pe is not None:
            x = x + self.pe[:, :t, :]
        return self.dropout(x)


class TransformerWithToken(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, num_layers):
        super().__init__()

        self.token = nn.Parameter(torch.randn(1, 1, d_model))
        token_mask = torch.zeros(1, 1, dtype=torch.bool)
        self.register_buffer("token_mask", token_mask)

        self.core = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                batch_first=True,
            ),
            num_layers=num_layers,
        )

    def forward(self, x, src_key_padding_mask):
        # x: [B, S, H]
        # padding_mask: [B, S]
        #   `False` for valid values
        #   `True` for padded values

        B, S, H = x.size()

        token = self.token.expand(B, -1, -1)
        x = torch.cat([token, x], dim=1)

        token_mask = self.token_mask.expand(B, -1)
        padding_mask = torch.cat([token_mask, src_key_padding_mask], dim=1)

        x = self.core(x, src_key_padding_mask=padding_mask)

        return x


class DesignAE(nn.Module):
    def __init__(
        self, vocab_size, pad_token, block_size, d_model=512, nhead=8, num_layers=6
    ):
        super().__init__()

        # encoder
        self.tok_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token)
        self.enc_transformer = TransformerWithToken(
            d_model=d_model,
            dim_feedforward=d_model // 4,
            nhead=nhead,
            num_layers=num_layers,
        )

        # decoder
        self.pos_embedding = PositionalEncoding("cosine", d_model, block_size)

        te = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model // 2, batch_first=True
        )
        self.dec_transformer = nn.TransformerEncoder(te, num_layers=num_layers)

        self.mlp_vocab = nn.Linear(d_model, vocab_size)

        self.decoder_type = "nat"

    def extract_features(self, src, padding_mask):
        tok = self.tok_embedding(src)
        x = self.enc_transformer(tok, ~padding_mask)
        # for cls token
        return x[:, 0]

    def forward(self, src, padding_mask):
        B, S = src.size()
        feat = self.extract_features(src, padding_mask)
        x = self.pos_embedding(repeat(feat, "B H -> B S H", S=S))
        out = self.dec_transformer(x, src_key_padding_mask=~padding_mask)
        # logit: [B, S, V]
        logits = self.mlp_vocab(out)
        return logits

    def get_loss(self, src, src_pad_mask, tgt, tgt_pad_mask, writer=None, weight_mask=None):
        logits = self(src, src_pad_mask)
        logits_log_prob = nn.functional.log_softmax(logits, dim=-1)
        if weight_mask is not None:
            logits_log_prob = logits_log_prob * weight_mask
        loss = nn.functional.nll_loss(
            logits_log_prob.transpose(1, 2),
            tgt,
            ignore_index=self.tok_embedding.padding_idx
        )
        if writer is not None:
            writer.log({"train_loss": loss.item()})
        return loss

    def load_model(self, cache_path: str, visual_balance_factor: float, postfix: Optional[str]=None):
        # Try loading with postfix first if provided
        if postfix is not None:
            path = os.path.join(
                cache_path, f"fid_weights/{postfix}_design_ae_w{int(visual_balance_factor)}.pth"
            )
            try:
                self.load_state_dict(torch.load(path), strict=True)
                logger.info("load model from {}".format(path))
                return
            except Exception as e:
                logger.warning(f"Failed to load model with postfix from {path}: {e}")
        
        # Fallback to loading without postfix
        path = os.path.join(
            cache_path, f"fid_weights/design_ae_w{int(visual_balance_factor)}.pth"
        )
        try:
            self.load_state_dict(torch.load(path), strict=True)
            logger.info("load model from {}".format(path))
        except Exception as e:
            logger.error("load model from {} failed".format(path))
