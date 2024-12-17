from .nodes import ImageBatchDiagnoser, ImageAligner, AlignMeshToMasks, NormalizeMeshBBox

NODE_CLASS_MAPPINGS = {"Image Diagnoser": ImageBatchDiagnoser,
                       "Image Aligner": ImageAligner,
                       "Align Mesh to Masks": AlignMeshToMasks,
                       "Normalize Mesh Bounding Box": NormalizeMeshBBox,
                       }
NODE_DISPLAY_NAME_MAPPINGS = {}
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
