from PIL import Image
from torchvision.transforms import (
    CenterCrop,
    Compose,
    InterpolationMode,
    Normalize,
    Resize,
    ToTensor,
    ToPILImage,
)
from torchvision.transforms.functional import pad


def _convert_image_to_rgb(image: Image.Image) -> Image.Image:
    return image.convert("RGB")


class PadToSquare:
    """Adaptively pad the image to make it square."""

    def __init__(self, fill: int = 0) -> None:
        """Initialize the PadToSquare class."""
        self.fill = fill

    def __call__(self, img: Image.Image) -> pad:
        """Return the callable adaptive padding funtion to square."""
        w, h = img.size
        max_dim = max(w, h)
        padding = (0, 0, max_dim - w, max_dim - h)
        return pad(img, padding, fill=self.fill)


def preprocess_pad_to_square(n_px: int = 224) -> Compose:
    """Preprocess the image to make it square by padding.

    Args:
        n_px (int, optional): Target size. Defaults to 224.

    Returns:
        Compose: The preprocessing pipeline.
    """
    # Calculate padding

    return Compose(
        [
            # Padding image to make it square
            PadToSquare(),
            Resize(n_px, interpolation=InterpolationMode.BICUBIC),
            _convert_image_to_rgb,
            ToTensor(),
            Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )


def preprocess_reshape(n_px: int = 224) -> Compose:
    """Preprocess the image to make it square by reshaping.

    Args:
        n_px (int, optional): Target size. Defaults to 224.

    Returns:
        Compose: The preprocessing pipeline.
    """
    return Compose(
        [
            Resize((n_px, n_px), interpolation=InterpolationMode.BICUBIC),
            _convert_image_to_rgb,
            ToTensor(),
            Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )


def preprocess_clip_original(n_px: int = 224) -> Compose:
    """Preprocess the image to make it square provided from CLIP official repo.

    Args:
        n_px (int, optional): Target size. Defaults to 224.

    Returns:
        Compose: The preprocessing pipeline.
    """
    return Compose(
        [
            Resize(n_px, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(n_px),
            _convert_image_to_rgb,
            ToTensor(),
            Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )


def preprocess_tensor_crop(n_px: int = 221) -> Compose:
    return Compose([ToPILImage(), CenterCrop((n_px, n_px)), ToTensor()])


def get_preprocess(process_mode: str, model_in_res: int = 224) -> Compose:
    """Return preprocess function for given process mode.

    Args:
        process_mode (str): process mode, ['padding', 'reshape']
        model_in_res (int): model input resolution. Defaults to 224.

    Raises:
        NotImplementedError: process mode not implemented.

    Returns:
        Compose: preprocess function.
    """
    if process_mode == "padding":
        return preprocess_pad_to_square(model_in_res)
    elif process_mode == "reshape":
        return preprocess_reshape(model_in_res)
    elif process_mode == "clip":
        return preprocess_clip_original(model_in_res)
    else:
        raise NotImplementedError(f"process_mode {process_mode} not implemented.")
