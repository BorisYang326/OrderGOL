from typing import Optional
import torch
import torch.nn as nn
from lightly.models import utils
import logging
from PIL import Image
import os
import timm
import torchvision.transforms as T
from lightly.models.modules import MAEDecoderTIMM, MaskedVisionTransformerTIMM
from lightly.transforms import MAETransform

logger = logging.getLogger(__name__)


class MAE(nn.Module):
    def __init__(
        self,
        vit: Optional[nn.Module] = None,
        mask_ratio: float = 0.75,
        pretrained_path: Optional[str] = None,
        vit_pretrained: bool = False,
        drop_path_rate: Optional[float] = None,
        input_size: int = 224,
    ):
        super().__init__()

        decoder_dim = 512
        if vit is None:
            # vit_large_patch16_siglip_384.webli,vit_large_patch16_224,vit_large_patch16_224.mae
            # vit = timm.create_model('vit_base_patch32_clip_224.openai', pretrained=vit_pretrained)
            vit = timm.create_model(
                "vit_large_patch16_224.mae", pretrained=vit_pretrained
            )
            vit.drop_path_rate = drop_path_rate if vit_pretrained else 0
        self._mask_ratio = mask_ratio
        self._patch_size = vit.patch_embed.patch_size[0]
        self._input_size = input_size
        self.backbone = MaskedVisionTransformerTIMM(vit=vit)
        self._sequence_length = self.backbone.sequence_length
        self.decoder = MAEDecoderTIMM(
            num_patches=vit.patch_embed.num_patches,
            patch_size=self._patch_size,
            embed_dim=vit.embed_dim,
            decoder_embed_dim=decoder_dim,
            decoder_depth=8,
            decoder_num_heads=16,
            mlp_ratio=4.0,
            proj_drop_rate=0.0,
            attn_drop_rate=0.0,
        )
        self._transform = MAETransform(input_size=self._input_size)
        if pretrained_path is None and vit_pretrained is False:
            logger.error("No pretrained path provided, using random initialized model.")
        elif pretrained_path is None and vit_pretrained is True:
            logger.error("No pretrained path provided, using timm pretrained vit.")
        else:
            if not pretrained_path.endswith(".pth"):
                pretrained_path = os.path.join(
                    pretrained_path, f"fid_weights/visual_mae.pth"
                )
            try:
                self.load_state_dict(torch.load(pretrained_path))
                logger.info("load model from {}".format(pretrained_path))
            except FileNotFoundError:
                logger.error("load model from {} failed".format(pretrained_path))
            except Exception as e:
                logger.error(e)

    def forward_encoder(self, images, idx_keep=None):
        return self.backbone.encode(images, idx_keep=idx_keep)

    def forward_decoder(self, x_encoded, idx_keep, idx_mask, is_return_visible=False):
        # build decoder input
        batch_size = x_encoded.shape[0]
        x_decode = self.decoder.embed(x_encoded)
        x_masked = utils.repeat_token(
            self.decoder.mask_token, (batch_size, self._sequence_length)
        )
        x_masked = utils.set_at_index(x_masked, idx_keep, x_decode.type_as(x_masked))

        # decoder forward pass
        x_decoded = self.decoder.decode(x_masked)

        # predict pixel values for masked tokens
        if not is_return_visible:
            x_pred = utils.get_at_index(x_decoded, idx_mask)
            x_pred = self.decoder.predict(x_pred)
        else:
            x_pred = self.decoder.predict(x_decoded)[:, 1:, :]
        return x_pred

    def forward(self, images):
        batch_size = images.shape[0]
        idx_keep, idx_mask = utils.random_token_mask(
            size=(batch_size, self._sequence_length),
            mask_ratio=self._mask_ratio,
            device=images.device,
        )

        x_encoded = self.forward_encoder(images, idx_keep)
        x_pred = self.forward_decoder(x_encoded, idx_keep, idx_mask)

        # get image patches for masked tokens
        patches = utils.patchify(images, self._patch_size)
        # must adjust idx_mask for missing class token
        target = utils.get_at_index(patches, idx_mask - 1)
        return x_pred, target

    def reconstruct(self, images):
        batch_size = images.shape[0]
        idx_keep, idx_mask = utils.random_token_mask(
            size=(batch_size, self._sequence_length),
            mask_ratio=self._mask_ratio,
            device=images.device,
        )

        x_encoded = self.forward_encoder(images, idx_keep)
        x_pred_pa = self.forward_decoder(
            x_encoded, idx_keep, idx_mask, is_return_visible=True
        )

        # Patchify the input images
        x_input_pa = utils.patchify(images, self._patch_size)

        # Create a mask tensor
        mask = torch.ones_like(images)
        mask_patches_ = utils.patchify(mask, self._patch_size)
        pseudo_cls_token = torch.zeros(
            (batch_size, 1, x_input_pa.shape[2]), device=x_input_pa.device
        )
        mask_patches_with_cls = torch.cat([pseudo_cls_token, mask_patches_], dim=1)
        # 0 is masked, 1 is visible
        mask_patches = utils.set_at_index(
            mask_patches_with_cls, idx_mask, torch.zeros_like(x_pred_pa)
        )
        mask_patches_no_cls = mask_patches[:, 1:, :]
        mask = utils.unpatchify(mask_patches_no_cls, self._patch_size)

        # Masked image
        mask_input = images * mask

        # Reconstruction only
        recon_only = utils.unpatchify(x_pred_pa, self._patch_size)

        # Reconstruction + Visible
        recon_visible = images * mask + recon_only * (1 - mask)
        return images, mask_input, recon_only, recon_visible

    def extract_embedding(
        self, image: Image.Image, device: str, is_transformed: bool = False
    ):
        if not is_transformed:
            _transform = T.Compose([T.Resize([self._input_size,self._input_size]), T.ToTensor()])
            x = _transform(image).to(device).unsqueeze(0)
        else:
            x = image.to(device).unsqueeze(0)

        # Temporarily set mask_ratio to 0
        original_mask_ratio = self.mask_ratio
        self.mask_ratio = 0

        try:
            # Use the encoder without masking
            # output feature shape (B, 1, H),cls_token feature.
            x_encoded = self.backbone.encode(x)[:, 0, :]
        finally:
            # Restore the original mask_ratio
            self._mask_ratio = original_mask_ratio

        return x_encoded

    def save_model(self, path, encoder_only=False):
        if encoder_only:
            torch.save(self.backbone.state_dict(), path)
        else:
            torch.save(self.state_dict(), path)

    def load_model(self, path, encoder_only=False):
        if encoder_only:
            self.backbone.load_state_dict(torch.load(path))
        else:
            self.load_state_dict(torch.load(path))

    @property
    def mask_ratio(self):
        return self._mask_ratio

    @mask_ratio.setter
    def mask_ratio(self, mask_ratio):
        assert 0 <= mask_ratio <= 1, "mask_ratio must be between 0 and 1"
        self._mask_ratio = mask_ratio
