import numpy as np
import torch
import nerfstudio.utils.math 
#from nerfstudio.utils.math import intersect_plane
import itertools
from dataclasses import dataclass
from typing import Literal, Tuple

import torch
from jaxtyping import Bool, Float
from torch import Tensor

from nerfstudio.data.scene_box import OrientedBox
import plotly.graph_objects as go

def safe_normalize(
    vectors: Float[Tensor, "*batch_dim N"],
    eps: float = 1e-10,
) -> Float[Tensor, "*batch_dim N"]:
    """Normalizes vectors.

    Args:
        vectors: Vectors to normalize.
        eps: Epsilon value to avoid division by zero.

    Returns:
        Normalized vectors.
    """
    return vectors / (torch.norm(vectors, dim=-1, keepdim=True) + eps)

# def intersect_plane(
#     plane_coefficients,     
#     origins: torch.Tensor,
#     directions: torch.Tensor,
# ):
#     a = plane_coefficients[0]
#     b = plane_coefficients[1]
#     c = plane_coefficients[2]
#     d = plane_coefficients[3]
#     normal = torch.Tensor([a,b,c])
#     normal = safe_normalize(normal)     #get plane normal
#     point = torch.Tensor([0,0,-d/c])
#     t = torch.inner((point - origins),normal)/torch.inner(directions,normal)
#     return t

object_tensor_aabb= [1,2,3]
rays_o = torch.Tensor([[1,2,3],[0,0,0],[0,1,2]])
rays_d = torch.Tensor([[0,-1,0],[3,1,2],[-1,0,1]])
plane_coefficients = np.load('plane_coefficients.npy')
#t_plane = intersect_plane(plane_coefficients, rays_o, rays_d)
t_plane = nerfstudio.utils.math.intersect_plane(plane_coefficients,rays_o, rays_d)

points = (rays_o.T + t_plane.T * rays_d).T
scatter = go.Scatter3d(x=points[:,0], y=points[:,1], z=points[:,2], mode='markers', marker=dict(size=2))
fig = go.Figure(data=[scatter])