from .nodes import ImageBatchDiagnoser, ImageAligner
NODE_CLASS_MAPPINGS = {"Image Diagnoser": ImageBatchDiagnoser,
                       "Image Aligner": ImageAligner}
NODE_DISPLAY_NAME_MAPPINGS = {}
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
