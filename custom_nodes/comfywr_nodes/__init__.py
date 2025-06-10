from .alignment_node import AlignMeshToMutliviewMasks
from .impact_nodes import BboxDetectorSEGSBatch, GroundingDINOBBoxDetectorProviderNode, SEGSWeightedMaskCombined
from .nodes import ApplyMeshTransform, ImageBatchDiagnoser, ImageAligner, AlignMeshToMasks, NormalizeMeshBBox, CreateSimpleBuildingMesh

NODE_CLASS_MAPPINGS = {"Image Diagnoser": ImageBatchDiagnoser,
                       "Image Aligner": ImageAligner,
                       "Align Mesh to Masks": AlignMeshToMasks,
                       "Align Mesh To Mutliview Masks": AlignMeshToMutliviewMasks,
                       "Normalize Mesh Bounding Box": NormalizeMeshBBox,
                       "Create Simple Building Mesh": CreateSimpleBuildingMesh,
                       "GroundingDINO BBox Detector Provider": GroundingDINOBBoxDetectorProviderNode,
                       "SEGS Weighted Mask Combined": SEGSWeightedMaskCombined,
                       "BBox Detector SEGS Batch": BboxDetectorSEGSBatch,
                       "Apply Mesh Transform": ApplyMeshTransform,
                       }
NODE_DISPLAY_NAME_MAPPINGS = {}
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
