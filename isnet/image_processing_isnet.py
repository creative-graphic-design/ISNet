from typing import Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from PIL.Image import Image as PilImage
from torchvision import transforms
from torchvision.transforms.functional import normalize
from transformers.image_processing_utils import BaseImageProcessor, BatchFeature
from transformers.image_utils import ImageInput


def apply_transform(data):
    transform = transforms.ToTensor()
    return transform(data)


class ISNetImageProcessor(BaseImageProcessor):
    def __init__(self, model_in_size: Tuple[int, int] = (1024, 1024), **kwargs) -> None:
        super().__init__(**kwargs)
        self.model_in_size = model_in_size

    def preprocess(self, images: ImageInput, **kwargs) -> BatchFeature:
        if not isinstance(images, PilImage):
            raise ValueError(f"Expected PIL Image, got {type(images)}")

        image_pil = images
        image_tensor = apply_transform(image_pil)

        # shape: (3, h, w) -> (1, 3, h, w)
        image_tensor = image_tensor.unsqueeze(dim=0)

        image_tensor = F.interpolate(
            image_tensor, size=self.model_in_size, mode="bilinear", align_corners=False
        )
        image_tensor = normalize(
            image_tensor, mean=[0.5, 0.5, 0.5], std=[1.0, 1.0, 1.0]
        )
        return BatchFeature(data={"pixel_values": image_tensor}, tensor_type="pt")

    def postprocess(
        self, prediction: torch.Tensor, width: int, height: int, **kwargs
    ) -> PilImage:
        def _norm_prediction(d: torch.Tensor) -> torch.Tensor:
            ma, mi = torch.max(d), torch.min(d)

            # division while avoiding zero division
            dn = (d - mi) / ((ma - mi) + torch.finfo(torch.float32).eps)
            return dn

        prediction = _norm_prediction(prediction)
        prediction = prediction.squeeze()
        prediction = prediction * 255 + 0.5
        prediction = prediction.clamp(0, 255)

        prediction_np = prediction.cpu().numpy()
        image = Image.fromarray(prediction_np).convert("RGB")
        image = image.resize((width, height), resample=Image.Resampling.BILINEAR)
        return image
