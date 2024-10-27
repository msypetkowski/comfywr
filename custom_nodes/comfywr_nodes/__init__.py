from .nodes import ImageBatchDiagnoser, ImageAligner, AlignMeshToMasks

NODE_CLASS_MAPPINGS = {"Image Diagnoser": ImageBatchDiagnoser,
                       "Image Aligner": ImageAligner,
                       "Align Mesh to Masks": AlignMeshToMasks}
NODE_DISPLAY_NAME_MAPPINGS = {}
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
