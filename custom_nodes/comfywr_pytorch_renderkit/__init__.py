from .nodes.meshes_nodes import RoofedBuildingMeshNode, CuboidMeshNode, ParametricMeshToMesh
from .nodes.misc_nodes import DepthToNormalMap
from .nodes.optimization_nodes import NormalMapGradientOptimization

NODE_CLASS_MAPPINGS = {"Parametric Roofed Building": RoofedBuildingMeshNode,
                       "Parametric Cuboid": CuboidMeshNode,
                       "ParametricMeshToMesh": ParametricMeshToMesh,
                       "Depth To Normal Map": DepthToNormalMap,
                       "NormalMap Gradient Optimization": NormalMapGradientOptimization,
                       }
NODE_DISPLAY_NAME_MAPPINGS = {}
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']