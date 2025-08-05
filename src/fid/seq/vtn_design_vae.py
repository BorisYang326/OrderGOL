
import torch
from torch import nn
from torch.nn import functional as F
import logging
from einops import rearrange
from torch.nn import functional as F
from torch.nn.modules.module import Module
from torch.nn.modules.container import ModuleList
from torch.nn.init import xavier_uniform_
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm
from src.utils import masked_avg_pooling

logger = logging.getLogger(__name__)

class DesignVAE(nn.Module):
    def __init__(self, vocab_size,pad_tok, d_model=256, nhead=4, num_layers=4, max_seq_length=50):
        super(DesignVAE, self).__init__()
        
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model,padding_idx=pad_tok)
        self.position_embedding = nn.Embedding(max_seq_length, d_model)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 4,
                activation="gelu",
                batch_first=True
            ),
            num_layers=num_layers
        )
        self.fc_mu = nn.Linear(d_model, d_model)
        self.fc_logvar = nn.Linear(d_model, d_model)
        self.decoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 4,
                activation="gelu"
            ),
            num_layers=num_layers
        )
        self.fc_out = nn.Linear(d_model, vocab_size)

    def encode(self, src, src_mask):
        embedded = self.embedding(src) + self.position_embedding(torch.arange(src.size(1), device=src.device))
        encoder_output_ = self.encoder(embedded, src_key_padding_mask=~src_mask)
        encoder_output  = masked_avg_pooling(encoder_output_, src_mask)
        return encoder_output

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        causal_mask = torch.triu(torch.ones(z.size(1), z.size(1), device=z.device), diagonal=1).bool()
        decoder_output = self.decoder(z, mask=causal_mask,is_causal=True)
        return self.fc_out(decoder_output)

    def forward(self, src, src_pad_mask,tgt, tgt_pad_mask):
        encoder_output = rearrange(self.encode(src, src_pad_mask),'B H -> B 1 H')
        mu = self.fc_mu(encoder_output)
        logvar = self.fc_logvar(encoder_output)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar

    def loss_function(self, logits, y, mu, logvar,weight_mask,pad_token, beta=1.0):
        # recon_loss = F.cross_entropy(recon_x.view(-1, recon_x.size(-1)), x.view(-1), reduction='mean')
        logits_log_prob = F.log_softmax(logits, dim=-1)
        weighted_log_prob = logits_log_prob * weight_mask
        recon_loss = F.nll_loss(
            weighted_log_prob.reshape(-1, logits.size(-1)),
            y.reshape(-1),
            ignore_index=pad_token,
        )
        kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + beta * kl_divergence