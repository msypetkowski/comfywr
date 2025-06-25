from .nodes.meshes_nodes import RoofedBuildingMeshNode, CuboidMeshNode, ParametricMeshToMesh
from .nodes.misc_nodes import DepthToNormalMap, ReprojectFromImageToMesh
from .nodes.optimization_nodes import NormalMapGradientOptimization

NODE_CLASS_MAPPINGS = {"Parametric Roofed Building": RoofedBuildingMeshNode,
                       "Parametric Cuboid": CuboidMeshNode,
                       "Parametric Mesh To Mesh": ParametricMeshToMesh,
                       "Depth To Normal Map": DepthToNormalMap,
                       "NormalMap Gradient Optimization": NormalMapGradientOptimization,
                       "Re-project From Image To Mesh": ReprojectFromImageToMesh,
                       }
NODE_DISPLAY_NAME_MAPPINGS = {}
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']