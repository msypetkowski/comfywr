from collections import namedtuple
import torch
from PIL import Image
import numpy as np
import comfy.model_management

import torchvision.transforms as T
from torchvision.transforms import functional as F 

SEG = namedtuple("SEG",
                 ['cropped_image', 'cropped_mask', 'confidence', 'crop_region', 'bbox', 'label', 'control_net_wrapper'],
                 defaults=[None])

class GroundingDINOBBoxDetectorWrapper:
    def __init__(self, grounding_dino_model, prompt):

        self.model = grounding_dino_model  # Assumed to be from SAM2 pack's GroundingDinoModelLoader
        self.prompt = prompt

        self.transform = T.Compose(
            [
                T.Resize(800, max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def load_dino_image(self,image_pil: Image):
        image = self.transform(image_pil)  # 3, h, w
        return image

    def get_grounding_output(self, image, caption, min_threshold):
        caption = caption.lower()
        caption = caption.strip()
        if not caption.endswith("."):
            caption = caption + "."
        device = comfy.model_management.get_torch_device()
        image = image.to(device)
        with torch.no_grad():
            outputs = self.model(image[None], captions=[caption])
        logits = outputs["pred_logits"].sigmoid()[0]  # (nq, 256)
        boxes = outputs["pred_boxes"][0]  # (nq, 4)
        # filter output
        logits_filt = logits.clone()
        boxes_filt = boxes.clone()
        filt_mask = logits_filt.max(dim=1)[0] > min_threshold
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
        confidence_filt = logits_filt.max(dim=1)[0] # num_filt, 
        return boxes_filt.cpu(), confidence_filt.cpu()
    
    # .detect(image, threshold, dilation, crop_factor, drop_size, detailer_hook)

    def detect(self, image, threshold, dilatation, crop_factor, drop_size, detailer_hook):
        """
        Arguments:
            image: A PIL.Image or numpy.ndarray in RGB format
            prompt: Text prompt to detect
            threshold: Confidence threshold

        Returns:
            detections: List of dictionaries, each with keys: 'bbox', 'confidence', 'label'
        """
        
        if isinstance(image, torch.Tensor):
            image_pil = (image.squeeze(0).cpu().numpy() * 255).astype('uint8')
            image_pil = Image.fromarray(image_pil)
        elif isinstance (image, Image):
            image_pil = image

        W, H = image_pil.size
        image_tensor = self.load_dino_image(image_pil)
        labels = self.prompt.split(',')
        labels = [label.strip() for label in labels]
        segs = []
        
        for label in labels:
            boxes_filt, logits_filt = self.get_grounding_output(image=image_tensor, caption=label, min_threshold=threshold)

            boxes_filt = boxes_filt * torch.tensor([W, H, W, H], device=boxes_filt.device)
            boxes_filt[:, :2] -= boxes_filt[:, 2:] / 2  # x_center, y_center to x0, y0
            boxes_filt[:, 2:] += boxes_filt[:, :2]      # x0, y0 to x1, y1

            for box, confidence in zip(boxes_filt, logits_filt):
                x0, y0, x1, y1 = box.int().tolist()
                x0b = max(x0, 0)
                y0b = max(y0, 0)
                x1b = min(x1, W)
                y1b = min(y1, H)

                # Apply crop factor
                width = x1b - x0b
                height = y1b - y0b
                x0 = max(int(x0b - (crop_factor - 1) * width / 2), 0)
                y0 = max(int(y0b - (crop_factor - 1) * height / 2), 0)
                x1 = min(int(x1b + (crop_factor - 1) * width / 2), W)
                y1 = min(int(y1b + (crop_factor - 1) * height / 2), H)

                cropped_image = F.to_tensor(image_pil.crop((x0, y0, x1, y1))).permute(1,2,0).unsqueeze(0)
                cropped_mask = torch.ones_like(cropped_image[0,...,0]).cpu().numpy()

                seg = SEG(
                    cropped_image=cropped_image,
                    cropped_mask=cropped_mask,
                    confidence=confidence.item(),
                    crop_region=(x0, y0, x1, y1),
                    bbox=(x0b, y0b, x1b, y1b),
                    label=label,
                    control_net_wrapper=None
                )
                segs.append(seg)
        final_segs = ((H,W),segs)
        return final_segs