import torch
from torch import nn,Tensor
from torch.nn import functional as F
import logging
from torch.nn.modules.module import Module
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm
from torch.nn.modules.activation import MultiheadAttention
from torch.nn.modules.transformer import (
    _get_activation_fn,
)
from einops import repeat,rearrange
import os
import math

logger = logging.getLogger(__name__)

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

class TransformerDecoderLayerGlobalImproved(Module):
    def __init__(
        self,
        d_model,
        d_global,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        d_global2=None,
        batch_first=False,
    ):
        super(TransformerDecoderLayerGlobalImproved, self).__init__()
        self.self_attn = MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )

        self.linear_global = Linear(d_global, d_model)

        if d_global2 is not None:
            self.linear_global2 = Linear(d_global2, d_model)

        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if "activation" not in state:
            state["activation"] = F.relu
        super(TransformerDecoderLayerGlobalImproved, self).__setstate__(state)

    def forward(
        self, tgt, memory, tgt_mask=None, tgt_key_padding_mask=None, *args, **kwargs
    ):
        tgt2 = self.self_attn(
            tgt, tgt, tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        if memory is not None:
            tgt2 = self.linear_global(memory)
            tgt = tgt + self.dropout3(tgt2)
        return tgt


class DesignEncoder(nn.Module):
    def __init__(
        self,
        tok_embedding,
        block_size,
        d_model,
        n_heads,
        n_layers,
        dim_feedforward=2048,
        d_latent=512,
    ):
        super(DesignEncoder, self).__init__()
        self.tok_embedding = tok_embedding
        self.pos_embedding = PositionalEncoding('param',d_model,block_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward, activation="gelu", batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, n_layers)
        self.mlp_mu = nn.Linear(d_model, d_latent)
        self.mlp_logvar = nn.Linear(d_model, d_latent)

    def forward(self, src_, src_pad_mask):
        src = self.tok_embedding(src_)
        src_single = self.extract_features(src, src_pad_mask)
        mu = self.mlp_mu(src_single)
        logvar = self.mlp_logvar(src_single)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return z, mu, logvar
    
    def extract_features(self, src, src_pad_mask):
        out = self.encoder(self.pos_embedding(src), src_key_padding_mask=~src_pad_mask)
        return self.masked_avg_pooling(out, src_pad_mask)

    def masked_avg_pooling(self, encoder_output, src_mask):
        src_mask = src_mask.unsqueeze(-1)
        masked_output = encoder_output * src_mask
        token_count = src_mask.sum(dim=1)
        token_count = token_count.where(token_count != 0, torch.ones_like(token_count))
        summed = masked_output.sum(dim=1)
        averaged = summed / token_count
        return averaged


class DesignNATDecoder(nn.Module):
    def __init__(self, d_model, d_global, nhead, num_layers, vocab_size, seq_len):
        super(DesignNATDecoder, self).__init__()
        self.learnable_embed = nn.Parameter(torch.randn(seq_len, d_model))
        self.decoder_layers = nn.ModuleList(
            [
                TransformerDecoderLayerGlobalImproved(
                    d_model, d_global, nhead, activation="gelu", batch_first=True
                )
                for _ in range(num_layers)
            ]
        )
        self.mlp_vocab = nn.Linear(d_model, vocab_size)

    def forward(self, z_, tgt_pad_mask):
        B, H = z_.size()
        S = self.learnable_embed.size(0)
        z = repeat(z_, "B H -> B S H", S=S)
        tgt = repeat(self.learnable_embed, "S H -> B S H", B=B)
        for layer in self.decoder_layers:
            tgt = layer(tgt, z, tgt_key_padding_mask=~tgt_pad_mask)
        logits = self.mlp_vocab(tgt)
        return logits


class DesignATDecoder(nn.Module):
    def __init__(
        self,
        tok_embedding: nn.Embedding,
        d_model,
        nhead,
        num_layers,
        dim_feedforward=2048,
        dropout=0.1,
    ):
        super(DesignATDecoder, self).__init__()
        self.tok_embedding = tok_embedding
        self.decoder_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model, nhead, dim_feedforward, dropout, batch_first=True
                )
                for _ in range(num_layers)
            ]
        )
        self.mlp_vocab = nn.Linear(d_model, tok_embedding.num_embeddings)
        self.pad_tok = tok_embedding.padding_idx

    def forward(self, z, seq_len, target_seq=None):
        if target_seq is not None:  # Teacher forcing during training
            batch_size, seq_len = target_seq.size(0), target_seq.size(1)
            tgt_emb = self.tok_embedding(target_seq)
            tgt_emb[:, 0, :] = z  # Replace the first token with z

            tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(
                z.device
            )
            tgt_key_padding_mask = target_seq == self.pad_tok

            output = tgt_emb
            for layer in self.decoder_layers:
                output = layer(
                    output, src_mask=tgt_mask, src_key_padding_mask=tgt_key_padding_mask
                )
            logits = self.mlp_vocab(output)
        else:  # Autoregressive decoding during inference
            batch_size, d_model = z.size()
            tgt = z.unsqueeze(1)  # Start with z as the first token
            outputs = []

            for _ in range(seq_len):
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                    tgt.size(1)
                ).to(z.device)
                output = tgt
                for layer in self.decoder_layers:
                    output = layer(output, src_mask=tgt_mask)
                output = self.mlp_vocab(output[:, -1, :])
                outputs.append(output.unsqueeze(1))
                next_input = self.tok_embedding(output.argmax(dim=-1))
                tgt = torch.cat((tgt, next_input.unsqueeze(1)), dim=1)

            logits = torch.cat(outputs, dim=1)

        return logits


class DesignVAE(nn.Module):
    def __init__(
        self,
        vocab_size,
        pad_tok,
        block_size,
        d_model=512,
        nhead=8,
        num_layers=4,
        d_latent=8,
        decoder_type="nat",
        beta: float = 1e-2,
        weight_mask=None,
    ):
        super(DesignVAE, self).__init__()
        self.tok_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_tok)
        self.encoder = DesignEncoder(self.tok_embedding,block_size, d_model, nhead, num_layers, d_latent=d_latent)
        if decoder_type == "nat":
            self.decoder = DesignNATDecoder(
                d_model, d_latent, nhead, num_layers, vocab_size, block_size
            )
        elif decoder_type == "at":
            self.decoder = DesignATDecoder(
                self.tok_embedding, d_model, nhead, num_layers
            )
        else:
            raise ValueError(f"decoder_type {decoder_type} is not supported")
        self.decoder_type = decoder_type
        self.beta = beta
        self.weight_mask = torch.ones(vocab_size) if weight_mask is None else weight_mask

    def forward(self, src, src_pad_mask, tgt=None, tgt_pad_mask=None,is_sampling=False):
        B, S = src.size()
        z_, mu, logvar = self.encoder(src, src_pad_mask)
        if isinstance(self.decoder, DesignNATDecoder):
            z = z_
            logits = self.decoder(z, tgt_pad_mask)
        elif isinstance(self.decoder, DesignATDecoder):
            z = self.encoder.masked_avg_pooling(z_, src_pad_mask)
            tgt_ = None if is_sampling else tgt
            logits = self.decoder(z, S, target_seq=tgt_)
        return logits, mu, logvar

    def loss_function(self, logits, tgt, mu, logvar, pad_tok,tgt_pad_mask):
        logits_log_prob = F.log_softmax(logits, dim=-1)
        weighted_log_prob = logits_log_prob * self.weight_mask
        loss_ce = F.nll_loss(
            weighted_log_prob.reshape(-1, logits.size(-1)),
            tgt.reshape(-1),
            ignore_index=pad_tok,
        )
        # kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        # Define the normal distribution for mu and logvar
        q_z = torch.distributions.Normal(mu, torch.exp(0.5 * logvar))
        p_z = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(logvar))
        kl_div = torch.distributions.kl_divergence(q_z, p_z)  # shape: (B, S, H)
        ## For single z, we do not need consider the padding token
        # kl_div = kl_div_ * tgt_pad_mask.unsqueeze(-1)
        # kl_div = kl_div.sum(dim=-1)  # (B, S)
        # kl_div = kl_div.sum(dim=1) / tgt_pad_mask.sum(dim=1)  # (B)
        kl_div = kl_div.mean()  # scalar
        # beta_scale = loss_ce.detach() / (1e-6 + torch.abs(kl_div.detach()))
        loss = loss_ce + self.beta * kl_div
        return loss, loss_ce, kl_div

    def get_loss(self, src, src_pad_mask, tgt, tgt_pad_mask,writer=None):
        logits, mu, logvar = self(src, src_pad_mask, tgt, tgt_pad_mask)
        loss, loss_ce, kl_div = self.loss_function(logits, tgt, mu, logvar, self.tok_embedding.padding_idx,tgt_pad_mask)
        if writer is not None:
            writer.log({"train_loss": loss.item()})
            writer.log({"train_loss_ce": loss_ce.item()})
            writer.log({"train_loss_kl": kl_div.item()})
        return loss, loss_ce, kl_div
    
    def load_pretrained(self, cache_path: str, visual_balance_factor: float):
        path = os.path.join(
            cache_path, f"fid_weights/design_vae_w{int(visual_balance_factor)}_{self.decoder_type}.pth"
        )
        try:
            self.load_state_dict(torch.load(path))
            logger.debug("load model from {}".format(path))
        except Exception as e:
            logger.error(e)
            logger.error("load model from {} failed".format(path))

    @torch.no_grad()
    def extract_features(self, src, src_pad_mask):
        z, mu, _ = self.encoder(src, src_pad_mask)
        # return self.encoder.masked_avg_pooling(z, src_pad_mask)
        # https://stackoverflow.com/a/72131257 use mu as deterministic embedding.
        return self.encoder.masked_avg_pooling(mu, src_pad_mask)
