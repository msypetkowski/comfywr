import importlib
import torch
from ..meshes.cuboid import Cuboid
from ..meshes.roofed_building import RoofedBuilding

class CuboidMeshNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("FLOAT", {"default": 1.0, "min": 0.0}),
                "height": ("FLOAT", {"default": 1.0, "min": 0.0}),
                "depth": ("FLOAT", {"default": 1.0, "min": 0.0}),
            },
            "optional": {
                "transform": ("TRANSFORM",),
            }
        }

    RETURN_TYPES = ("PARAMETRIC_MESH",)
    RETURN_NAMES = ("cuboid_mesh",)
    FUNCTION = "create"

    CATEGORY = "comfy3dgen/primitives"

    def create(self, width, height, depth, transform=None):
        sizes = torch.tensor([width, height, depth])
        sizes /= sizes.max() # TODO make it optional
        width, height, depth = sizes
        return (Cuboid(width, height, depth, transform=transform),)
    

class RoofedBuildingMeshNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("FLOAT", {"default": 1.0, "min": 0.0}),
                "height": ("FLOAT", {"default": 1.0, "min": 0.0}),
                "depth": ("FLOAT", {"default": 1.0, "min": 0.0}),
                "roof_inset_x": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0}),
                "roof_inset_z": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0}),
                "roof_height": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0}),
            },
            "optional": {
                "transform": ("TRANSFORM",),
            }
        }

    RETURN_TYPES = ("PARAMETRIC_MESH",)
    RETURN_NAMES = ("roofed_building",)
    FUNCTION = "create"

    CATEGORY = "comfy3dgen/primitives"

    def create(self, width, height, depth, roof_inset_x, roof_inset_z, roof_height, transform=None):

        sizes = torch.tensor([width, height, depth])
        sizes /= sizes.max() # TODO make it optional
        width, height, depth = sizes

        roof_size = (roof_inset_x, roof_height, roof_inset_z)
        return (RoofedBuilding(width, height, depth, roof_size=roof_size, transform=transform),)
    

class ParametricMeshToMesh:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "parametric_mesh": ("PARAMETRIC_MESH",),
                "apply_transform": ("BOOLEAN", {"default": True}),
                "texture_res": ("INT", {"default": 512, "min": 4, "max": 4096, "step": 4}),
            }
        }
    
    RETURN_TYPES = ("MESH",)
    RETURN_NAMES = ("mesh",)
    FUNCTION = "export"

    def export(self, parametric_mesh, apply_transform, texture_res):

        trimesh_mesh = parametric_mesh.export_trimesh(apply_transform=apply_transform)

        # Convert to 3D-pack Mesh
        Mesh = importlib.import_module('custom_nodes.ComfyUI-3D-Pack.mesh_processer.mesh').Mesh
        mesh = Mesh.load_trimesh(given_mesh=trimesh_mesh)
        mesh.auto_normal()
        mesh.auto_uv()
        mesh.set_new_albedo(texture_res, texture_res)
        return (mesh,)
