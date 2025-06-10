from custom_nodes.comfywr_nodes.utils.sam_impact_brigde import GroundingDINOBBoxDetectorWrapper
import numpy as np
import torch
import comfy.model_management

class GroundingDINOBBoxDetectorProviderNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "grounding_dino_model": ("GROUNDING_DINO_MODEL",),
                "labels": ("STRING", {"default": ""})
            }
        }

    RETURN_TYPES = ("BBOX_DETECTOR",)
    RETURN_NAMES = ("bbox_detector",)
    FUNCTION = "create"
    CATEGORY = "utils/detectors"

    def create(self, grounding_dino_model, labels):
        return (GroundingDINOBBoxDetectorWrapper(grounding_dino_model, labels),)
    

class BboxDetectorSEGSBatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
                        "bbox_detector": ("BBOX_DETECTOR", ),
                        "image": ("IMAGE", ),
                        "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                        "dilation": ("INT", {"default": 10, "min": -512, "max": 512, "step": 1}),
                        "crop_factor": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 100, "step": 0.1}),
                        "drop_size": ("INT", {"min": 1, "max": 1024, "step": 1, "default": 10}),
                      },
                "optional": {"detailer_hook": ("DETAILER_HOOK",), }
                }

    RETURN_TYPES = ("SEGS",)
    RETURN_NAMES = ("segs",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "dobatch"
    CATEGORY = "Impact/Detection"

    def dobatch(self, bbox_detector, image, threshold, dilation, crop_factor, drop_size, detailer_hook=None):
        segs_batch = []
    

        for img in image:

            segs = bbox_detector.detect(img.unsqueeze(0), threshold, dilation, crop_factor, drop_size, detailer_hook)
            
            segs_batch.append(segs)

        return (segs_batch, )
    

def combined_weighted_mask(segs, aggregate = np.maximum):
    shape = segs[0]
    h = shape[0]
    w = shape[1]

    detections = segs[1]

    mask = np.zeros((h, w), dtype=np.float32)

    for seg in detections:
        cropped_mask = seg.cropped_mask
        crop_region = seg.crop_region
        confidence = seg.confidence
        temp_mask = np.zeros_like(mask)

        x1, y1, x2, y2 = crop_region
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        temp_mask[ y1:y2, x1:x2] = cropped_mask * confidence
        mask = aggregate(mask, temp_mask)

    torch_mask = torch.from_numpy(mask.astype(np.float32))

    return torch_mask



class SEGSWeightedMaskCombined:

    MASK_AGGREGATES = {"maximum": np.maximum, "probabilistic-sum": lambda x,y: x+y-x*y}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "segs": ("SEGS",),  # A single SEGS input (shape, detections)
                "aggregate_method": (list(cls.MASK_AGGREGATES.keys()),),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("weighted_mask",)
    INPUT_IS_LIST = True

    FUNCTION = "combine_weighted"
    CATEGORY = "Impact/Segmentation"

    def combine_weighted(self, segs, aggregate_method):

        if len(aggregate_method) == 1:
            aggregate_method = aggregate_method * len(segs)

        # aggregate_method = self.MASK_AGGREGATES[aggregate]
        device = comfy.model_management.get_torch_device()
        masks = tuple(combined_weighted_mask(segs, aggr) for segs,aggr in zip(segs, aggregate_method))

        return (torch.stack(masks, axis=0).to(device=device),)
