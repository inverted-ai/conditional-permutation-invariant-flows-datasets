from torchdrivesim.lanelet2 import road_mesh_from_lanelet_map, load_lanelet_map
from torchdrivesim.infractions import offroad_infraction_loss, iou_differentiable
from torchdrivesim.mesh import BaseMesh
from util import recenter_offsets
import torch
import os

def get_mesh_for_map(location, dataset_path):
    map_path = os.path.join(dataset_path, 'maps', location + '.osm')
    lanelet_map = load_lanelet_map(map_path)
    origin = recenter_offsets[location]
    mesh = road_mesh_from_lanelet_map(lanelet_map)
    mesh.verts -= torch.tensor(origin).float()
    return mesh

def collate_meshes(meshes):
    return BaseMesh.collate(meshes)

def calc_offroad(x, road_mesh, threshold=2.0, mask=None):
    """Calculate the offroad amount per agent
    Args:
        x: tensor of [B, N, 5]
        road_mesh: batch of road meshes as provided by "get_mesh_for_map" and optionally collated by "collate_meshes"
        threshold: float describing the size of the buffer in which offroad is allowed
        mask: Optional mask indicating which x are valid (mask[i,j] = 1) or not (mask[i,j] = 0)
    Returns:
        size (B,N) tensor with offroad amount for each agent
    """

    driving_surface_mesh = road_mesh.to(x.device)

    states = torch.cat([x[...,0:2],x[...,4:5], torch.zeros_like(x[...,4:5])], dim=-1)
    lenwid = x[...,2:4]

    if mask is None:
        mask = torch.ones_like(x).mean(dim=-1)

    return mask * offroad_infraction_loss(states, lenwid, driving_surface_mesh, threshold)

def calc_offroad_infractions(x, road_mesh, mask=None):
    """Calculate the offroad infraction per example
    Args:
        x: tensor of [B, N, 5]
        road_mesh: batch of road meshes as provided by "get_mesh_for_map" and optionally collated by "collate_meshes"
        threshold: float describing the size of the buffer in which offroad is allowed
        mask: Optional mask indicating which x are valid (mask[i,j] = 1) or not (mask[i,j] = 0)
    Returns:
        size (B) tensor with 1. for infracting examples and 0 for non infracting examples
    """
    offroad_loss = calc_offroad(x, road_mesh, mask=mask)
    return offroad_loss.sum(dim=-1).bool().float()

def calc_collisions(x, mask=None):
    """ Calculate the collision loss for all x, except mask

    Args:
        x:  size (B, N, 5) tensor
        mask:  size (B, N) tensor of 0. and 1., masking the original x

    Returns:
        size (B,N) tensor with intersection over union for each x with all other x

    """

    if mask is None:
        mask = torch.ones(x.shape[0:2]).to(x.device)

    mask = mask.unsqueeze(1) * mask.unsqueeze(2)

    nmax = mask.shape[1]
    offdiagonal = ((torch.ones((nmax,nmax)) - torch.diag(torch.ones(nmax))).unsqueeze(0)).to(mask.device)
    mask = mask * offdiagonal

    x2, x1 = torch.broadcast_tensors(x.unsqueeze(1), x.unsqueeze(2))

    orig_shape = x2.shape
    shape = (orig_shape[0], orig_shape[1] * orig_shape[2], orig_shape[3])

    iou = (iou_differentiable(x1.reshape(shape), x2.reshape(shape)) * mask.reshape((shape[0], -1))).reshape(orig_shape[0:3])
    # iou: multiply by .5 because all pairs have been calculated twice
    total_iou = 0.5 * iou.mean(dim=(2,))

    return total_iou

def calc_collision_infractions(x, mask=None):
    """ Calculate the collision infractions for all x, except mask

    Args:
        x:  size (B, N, 5) tensor
        mask:  size (B, N) tensor of 0. and 1., masking the original x

    Returns:
        size (B) tensor with 1. for infracting examples and 0 for non infracting examples

    """
    collision_loss = calc_collisions(x, mask=mask)
    return collision_loss.sum(dim=-1).bool().float()

def calc_all_infractions(collision_infractions, offroad_infractions):
    """Calculate the overall infraction rate

    Args:
        collision_infractions: size (B,) tensor
        offroad_infractions: size (B,) tensor
    Returns:
        size (B,) tensor with overall infractions
    """
    return torch.logical_or(collision_infractions, offroad_infractions)
