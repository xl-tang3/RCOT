import math
import open3d
import numpy as np
import torch
import pytorch3d.loss
import pytorch3d.ops
import pytorch3d.structures
from pytorch3d.loss.point_mesh_distance import point_face_distance
from torch_cluster import fps
from loss.emd.emd_module import emdFunction
from models.net_utils import normalize_sphere, normalize_pcl


def chamfer_distance_unit_sphere(gen, ref, batch_reduction='mean', point_reduction='mean'):
    # ref, center, scale = normalize_sphere(ref)
    # gen = normalize_pcl(gen, center, scale)
    return pytorch3d.loss.chamfer_distance(gen, ref, batch_reduction=batch_reduction, point_reduction=point_reduction)

def get_repulsion_loss(pred, k=4, eps=1e-12, h=0.03, radius=0.07):
    _, idx, grouped_points = pytorch3d.ops.knn_points(pred, pred, K=k, return_nn=True) ## idx:(B, N, K); grouped_points(B,N,K,3)
    # idx = idx[:, :, 1:].to(torch.int32) # remove first one
    # idx = idx.contiguous() # B, N, nn
    #
    pred = pred.permute(0, 2, 1).contiguous() # B, 3, N
    grouped_points = grouped_points.permute(0, 3, 1, 2).contiguous()  # B, 3, N, K
    # grouped_points = pytorch3d.ops.knn_gather(pred, idx) # (B, 3, N), (B, N, nn) => (B, 3, N, nn)

    grouped_points = grouped_points - pred.unsqueeze(-1)
    dist2 = torch.sum(grouped_points ** 2, dim=1)
    dist2 = torch.max(dist2, torch.tensor(eps).cuda())
    dist = torch.sqrt(dist2)
    weight = torch.exp(- dist2 / h ** 2)

    uniform_loss = torch.mean((torch.tensor(radius).cuda() - dist) * weight)
    # uniform_loss = torch.mean(self.radius - dist * weight) # punet
    # print("uniform_loss", uniform_loss, uniform_loss.shape)
    return uniform_loss

# def point_to_mesh_distance_single_unit_sphere(pcl, verts, faces):
#     """
#     Args:
#         pcl:    (N, 3).
#         verts:  (M, 3).
#         faces:  LongTensor, (T, 3).
#     Returns:
#         Squared pointwise distances, (N, ).
#     """

#     assert pcl.dim() == 2 and verts.dim() == 2 and faces.dim() == 2, 'Batch is not supported.'
    
#     # Normalize mesh
#     verts, center, scale = normalize_sphere(verts.unsqueeze(0))
#     verts = verts[0]
#     # Normalize pcl
#     pcl = normalize_pcl(pcl.unsqueeze(0), center=center, scale=scale)
#     pcl = pcl[0]

#     # print('%.6f %.6f' % (verts.abs().max().item(), pcl.abs().max().item()))

#     # Convert them to pytorch3d structures
#     pcls = pytorch3d.structures.Pointclouds([pcl])
#     meshes = pytorch3d.structures.Meshes([verts], [faces])
    
#     N = len(meshes)

#     # packed representation for pointclouds
#     points = pcls.points_packed()  # (P, 3)
#     points_first_idx = pcls.cloud_to_packed_first_idx()
#     max_points = pcls.num_points_per_cloud().max().item()

#     # packed representation for faces
#     verts_packed = meshes.verts_packed()
#     faces_packed = meshes.faces_packed()
#     tris = verts_packed[faces_packed]  # (T, 3, 3)
#     tris_first_idx = meshes.mesh_to_faces_packed_first_idx()
#     max_tris = meshes.num_faces_per_mesh().max().item()

#     # point to face distance: shape (P,)
#     point_to_face = point_face_distance(
#         points, points_first_idx, tris, tris_first_idx, max_points
#     )

#     return point_to_face

def Point2Face_distance(meshes, pcls):
    """
    Computes the distance between a pointcloud and a mesh within a batch.
    Given a pair `(mesh, pcl)` in the batch, we define the distance to be the
    sum of two distances, namely `point_face(mesh, pcl) + face_point(mesh, pcl)`

    `point_face(mesh, pcl)`: Computes the squared distance of each point p in pcl
        to the closest triangular face in mesh and averages across all points in pcl
    `face_point(mesh, pcl)`: Computes the squared distance of each triangular face in
        mesh to the closest point in pcl and averages across all faces in mesh.

    The above distance functions are applied for all `(mesh, pcl)` pairs in the batch
    and then averaged across the batch.

    Args:
        meshes: A Meshes data structure containing N meshes
        pcls: A Pointclouds data structure containing N pointclouds

    Returns:
        loss: The `point_face(mesh, pcl)` distance
            between all `(mesh, pcl)` in a batch averaged across the batch.
    """

    if len(meshes) != len(pcls):
        raise ValueError("meshes and pointclouds must be equal sized batches")
    N = len(meshes)

    # packed representation for pointclouds
    points = pcls.points_packed()  # (P, 3)
    points_first_idx = pcls.cloud_to_packed_first_idx()
    max_points = pcls.num_points_per_cloud().max().item()

    # packed representation for faces
    verts_packed = meshes.verts_packed()
    faces_packed = meshes.faces_packed()
    tris = verts_packed[faces_packed]  # (T, 3, 3)
    tris_first_idx = meshes.mesh_to_faces_packed_first_idx()
    max_tris = meshes.num_faces_per_mesh().max().item()

    # point to face distance: shape (P,)
    point_to_face = point_face_distance(
        points, points_first_idx, tris, tris_first_idx, max_points
    )

    point_to_face = torch.sqrt(point_to_face).mean(0)

    return point_to_face

def point_mesh_bidir_distance_single_unit_sphere(pcl, verts, faces):
    """
    Args:
        pcl:    (N, 3).
        verts:  (M, 3).
        faces:  LongTensor, (T, 3).
    Returns:
        Squared pointwise distances, (N, ).
    """
    assert pcl.dim() == 2 and verts.dim() == 2 and faces.dim() == 2, 'Batch is not supported.'
    
    # Normalize mesh
    verts, center, scale = normalize_sphere(verts.unsqueeze(0))
    verts = verts[0]
    # Normalize pcl
    pcl = normalize_pcl(pcl.unsqueeze(0), center=center, scale=scale)
    pcl = pcl[0]

    # print('%.6f %.6f' % (verts.abs().max().item(), pcl.abs().max().item()))

    # Convert them to pytorch3d structures
    pcls = pytorch3d.structures.Pointclouds([pcl])
    meshes = pytorch3d.structures.Meshes([verts], [faces])
    return pytorch3d.loss.point_mesh_face_distance(meshes, pcls)
    # return Point2Face_distance(meshes, pcls)


def hausdorff_distance_unit_sphere(gen, ref):
    """
    Args:
        gen:    (B, N, 3)
        ref:    (B, N, 3)
    Returns:
        (B, )
    """
    # ref, center, scale = normalize_sphere(ref)
    # gen = normalize_pcl(gen, center, scale)

    dists_ab, _, _ = pytorch3d.ops.knn_points(ref, gen, K=1)
    dists_ab = dists_ab[:,:,0].max(dim=1, keepdim=True)[0]  # (B, 1)
    # print(dists_ab)

    dists_ba, _, _ = pytorch3d.ops.knn_points(gen, ref, K=1)
    dists_ba = dists_ba[:,:,0].max(dim=1, keepdim=True)[0]  # (B, 1)
    # print(dists_ba)
    
    dists_hausdorff = torch.max(torch.cat([dists_ab, dists_ba], dim=1), dim=1)[0]

    return dists_hausdorff

def emd_loss(preds, gts, eps=0.005, iters=100):
    loss, assignment = emdFunction.apply(preds, gts, eps, iters)
    return loss, assignment

def get_knn_idx_dist(pos:torch.FloatTensor, query:torch.FloatTensor, k, offset=0):
    """
    :param  pos:     (B, N, F)
    :param  query:   (B, M, F)
    :return knn_idx: (B, M, k)
    """
    B, N, F = tuple(pos.size())
    M = query.size(1)

    pos = pos.unsqueeze(1).expand(B, M, N, F)
    query = query.unsqueeze(2).expand(B, M, N, F)   # B * M * N * F
    dist = torch.sum((pos - query) ** 2, dim=3, keepdim=False)   # B * M * N
    knn_idx = torch.argsort(dist, dim=2)[:, :, offset:k+offset]   # B * M * k
    knn_dist = torch.gather(dist, dim=2, index=knn_idx)           # B * M * k

    return knn_idx, knn_dist

def RepulsionLoss(pc, knn=4, h=0.03):

    knn_idx, knn_dist = get_knn_idx_dist(pc, pc, k=knn, offset=1)  # (B, N, k)
    weight = torch.exp(- knn_dist / (h ** 2))
    loss = torch.sum(- knn_dist * weight)
    return loss

def point_density(pcl, knn):
    """

    Args:
        pcl: B,N,3

    Returns:
        point-wise density: B, N, 1

    """
    knn_dst, knn_idx, _ = pytorch3d.ops.knn_points(pcl, pcl, K=knn, return_nn=True)  # [B N K]
    # knn_dst_max = torch.max(knn_dst, dim=-1, keepdim=True)[0]   # [B N 1]
    density = knn_dst[:,:,1:].mean(-1, keepdim=True)  # [B N 1]
    return density

def patch_density(pcl_low, pcl_high, pcl_up, knn):
    """

    pc_low: B,N,3
        pcl_high: B,R*N,3
        pcl_up: B,R*N,3
        knn: the nearest points
    Returns:
        loss

    """
    knn_dst_h, knn_idx_h, _ = pytorch3d.ops.knn_points(pcl_low, pcl_high, K=knn, return_nn=True)  # [B N K]
    knn_dst_up, knn_idx_up, _ = pytorch3d.ops.knn_points(pcl_low, pcl_up, K=knn, return_nn=True)  # [B N K]
    # weight = torch.exp(torch.abs(knn_dst_h.mean(dim=-1) - knn_dst_up.mean(dim=-1))) #2022.4.12
    # weight = torch.abs(knn_dst_h.mean(dim=-1) - knn_dst_up.mean(dim=-1))  #2022-4-14
    dst_h_mean = knn_dst_h[:,:,1:].mean(dim=-1, keepdim=True)
    dst_up_mean = knn_dst_up[:,:,1:].mean(dim=-1, keepdim=True)
    # density1 = torch.sum(torch.abs(dst_h_mean - dst_up_mean) + (knn_dst_h - dst_h_mean) ** 2 + (knn_dst_up - dst_up_mean) ** 2)
    density = torch.abs(dst_h_mean - dst_up_mean).mean() + torch.abs((knn_dst_h - dst_h_mean) ** 2 - (knn_dst_up - dst_up_mean) ** 2).mean()
    return density, knn_idx_h, knn_idx_up

# 2022-4-21
# def DensityLoss(pcl_low, pcl_high, pcl_up, knn=4, alpha=100):
#     """
#     Args:
#         pc_low: B,N,3
#         pcl_high: B,R*N,3
#         pcl_up: B,R*N,3
#         knn: the nearest points
#     Returns:
#         loss
#     """
#     pcl_low = pcl_low[:, :, :3]
#     pcl_high = pcl_high[:, :, :3]
#     pcl_up = pcl_up[:, :, :3]
#     B, N, _ = pcl_low.shape
#     pcl_up_density = point_density(pcl_up, knn) # [B N 1]
#     pcl_high_density = point_density(pcl_high, knn) # [B N 1]
#     patch_dense, knn_idx_h, knn_idx_up = patch_density(pcl_low, pcl_high, pcl_up, knn)
#     up_density_knn = pytorch3d.ops.knn_gather(pcl_up_density, knn_idx_up)
#     high_density_knn = pytorch3d.ops.knn_gather(pcl_high_density, knn_idx_h)
#     point_dense = torch.abs(up_density_knn - high_density_knn).mean()
#     density = patch_dense + point_dense
#     return density

# def DensityLoss(pcl_low, pcl_high, pcl_up, knn=4, alpha=100):
#     """
#     Args:
#         pc_low: B,N,3
#         pcl_high: B,R*N,3
#         pcl_up: B,R*N,3
#         knn: the nearest points
#     Returns:
#         loss
#     """
#     pcl_low = pcl_low[:, :, :3] # B,N,3
#     pcl_high = pcl_high[:, :, :3] # B,RN,3
#     pcl_up = pcl_up[:, :, :3] # B,RN,3
#     B, N, _ = pcl_low.shape
#     _, knn_idx_low, knn_low = pytorch3d.ops.knn_points(pcl_low, pcl_low, K=int(knn/3), return_nn=True)  # [B N K]
#     _, knn_idx_up, knn_up = pytorch3d.ops.knn_points(knn_low.reshape(B, -1, 3), pcl_up, K=knn, return_nn=True)  # [B N K]
#     _, knn_idx_high, knn_high = pytorch3d.ops.knn_points(knn_low.reshape(B, -1, 3), pcl_high, K=knn, return_nn=True)  # [B N K]
#     knn_up = knn_up.reshape(B, N, -1, 3)
#     knn_high = knn_high.reshape(B, N, -1, 3)
#     dst_low_up = (knn_up - pcl_low.unsqueeze(-2)).norm(p=2,dim=-1)
#     dst_low_high = (knn_high - pcl_low.unsqueeze(-2)).norm(p=2,dim=-1)
#     density = torch.abs(dst_low_up - dst_low_high).sum(dim=-1).mean()
#     return density

# def DensityLoss(pcl_low, pcl_high, knn=4, alpha=1000):
#     """
#     Args:
#         pc_low: B,N,3
#         pcl_high: B,R*N,3
#         pcl_up: B,R*N,3
#         knn: the nearest points
#     Returns:
#         loss
#     """
#     pcl_low = pcl_low[:, :, :3]
#     pcl_high = pcl_high[:, :, :3]
#     B, N, _ = pcl_low.shape
#     knn_dst_h, knn_idx_h, knn_point = pytorch3d.ops.knn_points(pcl_low, pcl_high, K=knn, return_nn=True)  # [B N K]
#     # weight = torch.exp( - alpha * torch.norm(knn_point - pcl_low.unsqueeze(-2).repeat(1,1,knn,1), p=2, dim=-1, keepdim=True))
#     weight = torch.exp(- alpha * knn_dst_h)
#     # weight = weight / (weight.sum(-1, keepdim=True) + 1e-7)
#     proj_point = torch.matmul(weight.unsqueeze(-1).permute(0,1,3,2), knn_point).squeeze(-2) / (weight.sum(-1, keepdim=True) + 1e-7)
#     density = torch.norm(pcl_low - proj_point, p=2, dim=-1).mean(-1)
#     return density.mean()

def square_distance(src, dst):
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def DensityLoss(pcl_high, pcl_up, knn=4, t=0.01, alpha=100):
    """
    Args:
        pcl_high: B,R*N,3
        pcl_up: B,R*N,3
        knn: the nearest points
    Returns:
        loss
    """
    pcl_high = pcl_high[:, :, :3] # B,RN,3
    pcl_up = pcl_up[:, :, :3] # B,RN,3
    B, N, C = pcl_up.shape
    B, M, _ = pcl_high.shape
    knn_high_up_dst, knn_idx_high_up, knn_high_up = pytorch3d.ops.knn_points(pcl_high, pcl_up, K=knn, return_nn=True)  # [B N K]
    knn_high_high_dst, knn_idx_high_high, knn_high_high = pytorch3d.ops.knn_points(pcl_high, pcl_high, K=knn, return_nn=True)  # [B N K]
    # weight_high_up = torch.exp( -knn_high_up_dst / t ).unsqueeze(-1).repeat(1,1,1,C)
    # weight_high_high = torch.exp( -knn_high_high_dst / t ).unsqueeze(-1).repeat(1,1,1,C)
    # mean_knn_high_up = torch.mul(knn_high_up, weight_high_up).mean(-2)
    # mean_knn_up_high = torch.mul(knn_high_high, weight_high_high).mean(-2)
    mean_point = torch.abs(knn_high_up.mean(-2) - knn_high_high.mean(-2)).sum(-1).mean()
    # mean_dst = (torch.abs(knn_high_up_dst.sum(-1)-knn_high_high_dst.sum(-1)) ** 2).mean()
    knn_high_up = knn_high_up.reshape(B*N, knn, C)
    knn_high_high = knn_high_high.reshape(B * N, knn, C)
    mean_dst = torch.abs(square_distance(knn_high_high, knn_high_high).sum(-1).sum(-1) - square_distance(knn_high_up, knn_high_up).sum(-1).sum(-1)).mean()
    density = mean_point + mean_dst   #2022_08_08__11_32_57
    ###xinzeng/2022_08_08__23_28_50
    # loss_dst = torch.abs(knn_high_high_dst - knn_high_up_dst).mean()
    # density = mean_point + mean_dst + loss_dst
    # density = mean_dst
    return density

def get_normal_vector(points):
    B, N, _ = points.shape
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(points.reshape(-1, 3).detach().cpu().numpy())
    pcd.estimate_normals(search_param=open3d.geometry.KDTreeSearchParamKNN(knn=4))
    normal_vec = torch.FloatTensor(pcd.normals).cuda().reshape(B, N, -1)
    return normal_vec

def NormalLoss(pcl_low, pcl_high, pcl_up, knn=4):
    """
    Args:
        pc_low: B,N,3
        pcl_high: B,R*N,3
        pcl_up: B,R*N,3
        knn: the nearest points
    Returns:
        loss
    """
    pcl_low = pcl_low[:, :, :3]  # [B N K]
    pcl_high = pcl_high[:, :, :3]  # [B R*N K]
    pcl_up = pcl_up[:, :, :3]  # [B R*N K]
    nor_high = get_normal_vector(pcl_high)  # [B R*N 3]
    nor_up = get_normal_vector(pcl_up)  # [B R*N 3]
    # nor_up_knn = pytorch3d.ops.knn_gather(nor_up, knn_idx_h)
    _, knn_idx_h, _ = pytorch3d.ops.knn_points(pcl_low, pcl_high, K=knn, return_nn=True)  # [B N K]
    _, knn_idx_up, _ = pytorch3d.ops.knn_points(pcl_low, pcl_up, K=knn, return_nn=True)  # [B N K]
    nor_high_knn = pytorch3d.ops.knn_gather(nor_high, knn_idx_h)  # [B N K 3]
    nor_up_knn = pytorch3d.ops.knn_gather(nor_up, knn_idx_up)  # [B N K 3]
    nor_0 = torch.sum((nor_high_knn - nor_up_knn) ** 2, dim=-1)  # [B N K]
    nor_1 = torch.sum((nor_high_knn + nor_up_knn) ** 2, dim=-1)  # [B N K]
    normal = torch.max(nor_0, nor_1)
    return torch.sum(normal)

def angle_point(pc1, pc2, k):
    """
    Args:
        pc1: B, N, 3
        pc2: B, N, 3
        k: KNN

    Returns:
        angle: the angle of each point, B, N
    """
    INNER_PRODUCT_THRESHOLD = math.pi / 2
    _, idx, pc1_knn = pytorch3d.ops.knn_points(pc1, pc2, K=k, return_nn=True)

    inner_prod = (pc1_knn * pc1.unsqueeze(-2)).sum(dim=-1)
    pc1_knn_norm = torch.norm(pc1_knn,p=2,dim=-1)
    pc1_norm = torch.norm(pc1.unsqueeze(-2).repeat(1,1,k,1), p=2, dim=-1)
    inner_prod = inner_prod / (pc1_knn_norm * pc1_norm)
    # inner_prod[inner_prod > 1] = 1
    # inner_prod[inner_prod < -1] = -1
    angle = torch.acos(inner_prod)
    # angle[angle > INNER_PRODUCT_THRESHOLD] = math.pi - angle[angle > INNER_PRODUCT_THRESHOLD]
    # angle = angle.sum(dim=-1)
    return angle

def density_point(pc1, pc2, knn, t):
    """
    Args:
        pc1: B, N, 3
        pc2: B, N, 3
        k: KNN

    Returns:
        angle: the angle of each point, B, N
    """
    B, N, C = pc1.shape
    _, M, _ = pc2.shape
    knn_dst, knn_idx, knn_point = pytorch3d.ops.knn_points(pc1, pc2, K=knn, return_nn=True)  # [B N K]
    weight = torch.exp(-knn_dst / t).unsqueeze(-1).repeat(1, 1, 1, C)
    mean_point = torch.mul(knn_point, weight).mean(-2)
    return mean_point

# def Density_Angle(pcl_high, pcl_up, knn=4, t=0.01, alpha=100):
#     """
#     Args:
#         pcl_high: B,R*N,3
#         pcl_up: B,R*N,3
#         knn: the nearest points
#     Returns:
#         loss
#     """
#     pcl_high = pcl_high[:, :, :3] # B,RN,3
#     pcl_up = pcl_up[:, :, :3] # B,RN,3
#     # knn_high_up_dst, knn_idx_high_up, knn_high_up = pytorch3d.ops.knn_points(pcl_high, pcl_up, K=knn, return_nn=True)  # [B N K]
#     # knn_high_high_dst, knn_idx_high_high, knn_high_high = pytorch3d.ops.knn_points(pcl_high, pcl_high, K=knn, return_nn=True)  # [B N K]
#     # weight_high_up = torch.exp( -knn_high_up_dst / t ).unsqueeze(-1).repeat(1,1,1,C)
#     # weight_high_high = torch.exp( -knn_high_high_dst / t ).unsqueeze(-1).repeat(1,1,1,C)
#     # mean_knn_high_up = torch.mul(knn_high_up, weight_high_up).mean(-2)
#     # mean_knn_high_high = torch.mul(knn_high_high, weight_high_high).mean(-2)
#     # mean_point = ((knn_high_up.mean(-2) - knn_high_high.mean(-2)) ** 2).sum(-1).mean()
#     mean_knn_high_high = density_point(pcl_high, pcl_high, knn, t)
#     mean_knn_high_up = density_point(pcl_high, pcl_up, knn, t)
#     # mean_point = (torch.abs(mean_knn_high_high - mean_knn_high_up)).sum(-1).mean()
#     mean_point = (torch.abs(mean_knn_high_up - pcl_high).sum(-1)).mean()
#     # mean_dst = (torch.abs(knn_high_up_dst.sum(-1)-knn_high_high_dst.sum(-1)) ** 2).mean()
#     angle_high_up = angle_point(pcl_high, pcl_high, knn)
#     angle_high_high = angle_point(pcl_high, pcl_up, knn)
#     loss_angle = torch.abs(angle_high_high - angle_high_up).mean()
#     loss = mean_point
#     return loss

def Density_Angle(pcl_high, pcl_up, knn=4):
    """
    Args:
        pcl_high: B,R*N,3
        pcl_up: B,R*N,3
        knn: the nearest points
    Returns:
        loss
    """
    pcl_high = pcl_high[:, :, :3] # B,RN,3
    pcl_up = pcl_up[:, :, :3] # B,RN,3
    knn_high_up_dst, knn_idx_high_up, knn_high_up = pytorch3d.ops.knn_points(pcl_high, pcl_up, K=knn, return_nn=True)  # [B N K]
    knn_high_high_dst, knn_idx_high_high, knn_high_high = pytorch3d.ops.knn_points(pcl_high, pcl_high, K=knn, return_nn=True)  # [B N K]
    vec_high_up = pcl_high.unsqueeze(-2) - knn_high_up
    vec_high_high = pcl_high.unsqueeze(-2) - knn_high_high
    loss_norm = torch.norm(vec_high_high - vec_high_up, p=2, dim=-1).mean(-1)
    loss_angle = torch.mul(vec_high_high,vec_high_up).sum(-1) / (torch.norm(vec_high_high, p=2, dim=-1) * torch.norm(vec_high_up, p=2, dim=-1) + 0.00001)
    loss_angle = loss_angle.mean(-1)
    loss = loss_norm.mean() + loss_angle.mean()
    return loss

if __name__ == '__main__':
    pcl_high = torch.randn(2,8,3)
    pcl_up = torch.randn(2,8,3)
    loss = Density_Angle(pcl_high, pcl_up, knn=8)
    print(loss)