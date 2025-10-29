import os
import torch
import numpy as np


def extract_mesh_aligned_features(trg, img_tensor, K_img, bbox_info, rf_i=None, device='cuda', out_list=None, vis_feat_list=None):
    """Extract mesh-aligned features from a TRG model for a single-batch input.

    Args:
        trg: TRG model instance (eval mode recommended)
        img_tensor: torch.Tensor [B,3,H,W] (normalized)
        K_img: torch.Tensor or None, intrinsic (transposed) as used by model
        bbox_info: torch.Tensor [B,7] or [B,5] as used by model
        rf_i: int or None, iteration index to extract features for. If None, uses last iter.
        device: device string or torch.device

    Returns:
        ref_feature: torch.Tensor [B, C_p * N] (mesh-aligned concatenated features)
        sampling_points: torch.Tensor [B, N, 2] or None (image-space 2D points), if available
    """
    trg.to(device)
    trg.eval()

    # If out_list/vis_feat_list are provided (from a prior forward pass), reuse them to avoid re-running the model
    if out_list is None or vis_feat_list is None:
        with torch.no_grad():
            out_list, vis_feat_list = trg(img_tensor.to(device), K_img=K_img.to(device) if K_img is not None else None, bbox_info=bbox_info.to(device))

    if rf_i is None:
        rf_i = int(trg.cfg.MODEL.TRG.N_ITER) - 1

    # s_feat_i is stored in vis_feat_list: vis_feat_list[0] is base, vis_feat_list[1] -> rf_i==0, etc.
    s_feat_i = vis_feat_list[rf_i + 1].to(device)

    # pred_face and pred_R_t used when rf_i > 0
    pred_out = out_list['output'][rf_i]
    pred_face = pred_out.get('pred_face_world', None)
    pred_R_t = pred_out.get('pred_R_t', None)

    laf = trg.laf_extractor[rf_i]

    if rf_i == 0:
        # TRG uses a grid for the first iteration
        batch_size = img_tensor.shape[0]
        sample_points = torch.transpose(trg.points_grid.expand(batch_size, -1, -1), 1, 2).to(device)
        ref_feature = laf.sampling(sample_points, s_feat_i)
        sampling_points = None
    else:
        ref_feature, sampling_points = laf(pred_face.to(device), pred_R_t.to(device), K_img.to(device), bbox_info.to(device), s_feat_i)

    return ref_feature, sampling_points


def save_mesh_features(out_dir, frame_idx, face_idx, ref_feature, trg, sampling_points=None):
    """Save per-frame mesh-aligned features to disk.

    Saves a NumPy file with shape [B, N, C_p] (B usually 1).
    Also saves sampling points if provided.
    """
    os.makedirs(out_dir, exist_ok=True)
    # C_p is last dim in MLP_DIM
    C_p = int(trg.cfg.MODEL.TRG.MLP_DIM[-1])
    B = ref_feature.shape[0]
    N = int(ref_feature.shape[1] // C_p)

    per_point = ref_feature.view(B, N, C_p).cpu().numpy()

    feat_path = os.path.join(out_dir, f'frame_{frame_idx:05d}_face_{face_idx:03d}_feats.npy')
    np.save(feat_path, per_point)

    if sampling_points is not None:
        spath = os.path.join(out_dir, f'frame_{frame_idx:05d}_face_{face_idx:03d}_samp_pts.npy')
        np.save(spath, sampling_points.cpu().numpy())
        return feat_path, spath

    return feat_path, None
