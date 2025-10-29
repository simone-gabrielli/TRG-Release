import os
import sys
import cv2
import numpy as np
from pdb import set_trace
import torch
import torch.nn as nn
from util.pkl import read_pkl, write_pkl
from util.vis import plotting_points2d_on_img_cv, render_mesh, draw_axis
import copy
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import trimesh
from util.path import get_file_dir_name
import scipy.io as sio
import face_alignment
import torchvision.transforms as transforms
from models.trg_model.core.cfgs import parse_args as parse_args_trg
from models.trg_model import load_trg
from data.preprocessing import update_after_crop, update_after_resize
from tqdm import tqdm
from util.img_video_conversion import video2image, image2video
import argparse
import time
from util.feature_utils import extract_mesh_aligned_features, save_mesh_features

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--video_path", type=str, default="./example/demo.mp4", help="Path, where logs will be stored")
    parser.add_argument("--fiter_threshold", type=float, default=0.9, help="Path, where logs will be stored")
    parser.add_argument("--frame_rate", type=int, default=30, help="Path, where logs will be stored")
    parser.add_argument('--do_render', action='store_true', default=False)
    parser.add_argument("--model_path", type=str, default="checkpoint/trg_240717/checkpoint-30/state_dict.bin", help="trg_240717, trg_single_240717")
    parser.add_argument('--show_features', action='store_true', default=False, help='Overlay and save mesh-aligned features on full image')
    args = parser.parse_args()
    return args

def main(args):
    ############################################
    # Load video
    ############################################
    demo_video_path = args.video_path

    ############################################
    # video -> images
    ############################################
    # make output directory
    save_img_dir = "output"
    os.makedirs(save_img_dir, exist_ok=True)

    ###############################################
    # Video to image conversion
    ###############################################
    video2image(demo_video_path, save_img_dir)

    # load images
    img_path_list = get_file_dir_name(save_img_dir, target_name='.jpg')[-1]

    # make render output directory
    # detector = 'sfd'
    tmp_render_out_dir = os.path.join(save_img_dir, f'rendered')
    os.makedirs(tmp_render_out_dir, exist_ok=True)
    # tmp_crop_render_out_dir = os.path.join(save_img_dir, f'rendered_crop')
    # os.makedirs(tmp_crop_render_out_dir, exist_ok=True)

    # load data
    mesh_faces = np.load('./data/triangles.npy')  # [2304, 3]

    ############################################
    model_fan = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False)
    # face detector 수정
    model_fan.face_detector.fiter_threshold = args.fiter_threshold
    ############################################
    # Load TRG
    ############################################
    device = torch.device('cuda')
    init_face_pose_dict = read_pkl('data/init_vtx_cam.pkl')
    init_face, init_pose = init_face_pose_dict['t_vtx_1220'], init_face_pose_dict['R_t']

    # get config
    cfg_path = 'models/trg_model/configs/trg_face_config.yaml'
    config = parse_args_trg(cfg_path)

    # load model
    trg = load_trg(config, init_face, init_pose).to(device)

    # load weight file
    state_dict_path = 'checkpoint/trg_240717/checkpoint-30/state_dict.bin'

    cpu_device = torch.device('cpu')
    state_dict = torch.load(state_dict_path, map_location=cpu_device)
    trg.load_state_dict(state_dict, strict=True)
    trg.eval()

    ############################################
    # for preprocessing image
    ############################################
    img_mean = np.array([0.485, 0.456, 0.406])
    img_std = np.array([0.229, 0.224, 0.225])

    tfm_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=img_mean, std=img_std)
    ])

    mesh_total_frame = []
    ################################################
    # Face detector, TRG inferece start
    ################################################
    for frame_i in tqdm(range(len(img_path_list))):
        mesh_each_frame = []
        img_path = img_path_list[frame_i]
        img_full = cv2.imread(img_path)

        ############################################
        # Get landmark from FAN for cropping
        ############################################
        # head_bbox_cur = head_bboxes[frame_i]
        head_bbox_cur = None
        lmk_preds = model_fan.get_landmarks(img_full, head_bbox_cur)  # list, [68,2]

        pred_verts_cam_list = []

        try:
            if len(lmk_preds) > 0:
                pass
        except TypeError:
            continue

        ############################################
        # Save head bbox
        ############################################
        for lmk_pred in lmk_preds:
            bbox_cur = []
            roi_cnt = 0.5 * (np.max(lmk_pred, 0)[0:2] + np.min(lmk_pred, 0)[0:2])
            size = np.max(np.max(lmk_pred, 0)[0:2] - np.min(lmk_pred, 0)[0:2])
            x_center = roi_cnt[0]
            y_center = roi_cnt[1]
            ss = np.array([0.75, 0.75, 0.75, 0.75])

            left = int(x_center - ss[0] * size)
            right = int(x_center + ss[1] * size)
            top = int(y_center - ss[2] * size)
            bottom = int(y_center + ss[3] * size)

            bbox_cur.append([left, top, right, bottom])

        bbox_cur = np.array(bbox_cur)

        ################################################
        # TRG inferece part
        ################################################
        for i, lmk_pred in enumerate(lmk_preds):
            ############################################
            # Get bounding box information
            ############################################
            roi_cnt = 0.5 * (np.max(lmk_pred, 0)[0:2] + np.min(lmk_pred, 0)[0:2])
            size = np.max(np.max(lmk_pred, 0)[0:2] - np.min(lmk_pred, 0)[0:2])
            x_center = roi_cnt[0]
            y_center = roi_cnt[1]
            ss = np.array([0.75, 0.75, 0.75, 0.75])

            left = int(x_center - ss[0] * size)
            right = int(x_center + ss[1] * size)
            top = int(y_center - ss[2] * size)
            bottom = int(y_center + ss[3] * size)

            ############################################
            # build bbox info and intrinsic matrix
            ############################################
            img_size = 192
            focal_length = 5000.0
            img_h, img_w, _ = img_full.shape
            bbox_info = np.array([left, top, right, bottom, focal_length, img_h, img_w], dtype=np.float32)

            bbox_size = right - left

            intrinsic_uncrop = np.array([
                [focal_length, 0, 0],
                [0, focal_length, 0],
                [img_w / 2, img_h / 2, 1],
                [0, 0, 0],
            ])

            intrinsic_crop = intrinsic_uncrop.copy()
            K_tmp = update_after_crop(intrinsic_uncrop[:3, :3].T, bbox_info[:4])  # [3,3]
            K_tmp = update_after_resize(K_tmp, [bbox_size, bbox_size], [img_size, img_size])  # [3,3]
            intrinsic_crop[:3, :3] = K_tmp.T
            intrinsic_crop_copy = intrinsic_crop.copy()

            # np.ndarray -> torch.tensor
            intrinsic_crop = torch.FloatTensor(intrinsic_crop[None, :, :]).cuda()
            bbox_info = torch.FloatTensor(bbox_info[None, :]).cuda()
            ############################################
            # Get tform matrix
            ############################################
            src_pts = np.float32([
                [left, top],
                [left, bottom],
                [right, top]
            ])

            dst_pts = np.float32([
                [0, 0],
                [0, img_size - 1],
                [img_size - 1, 0]
            ])

            tform = cv2.getAffineTransform(src_pts, dst_pts)
            tform_inv = cv2.getAffineTransform(dst_pts, src_pts)

            ############################################
            # Crop image
            ############################################
            img_crop = cv2.warpAffine(img_full, tform, (img_size,) * 2)
            img = img_crop[:, :, ::-1].copy()
            ############################################
            # TRG: forward
            ############################################
            with torch.no_grad():
                img = tfm_test(img)
                img = img[None, :, :, :].cuda()
                out_list, vis_feat_list = trg(img, intrinsic_crop, bbox_info)

                preds = out_list['output'][-1]
                pred_vtx_world, pred_R_t, pred_vtx_cam = \
                    preds['pred_face_world'], \
                    preds['pred_R_t'], \
                    preds['pred_face_cam']

                # Save all prediction
                pred_verts_cam_list.append(pred_vtx_cam)

                # --- extract and save mesh-aligned features ---
                try:
                    # choose iteration index (use last iteration by default)
                    rf_i = int(trg.cfg.MODEL.TRG.N_ITER) - 1
                    # reuse already computed out_list/vis_feat_list to avoid double forward
                    ref_feature, sampling_points = extract_mesh_aligned_features(trg, img, intrinsic_crop, bbox_info, rf_i=rf_i, device=device, out_list=out_list, vis_feat_list=vis_feat_list)

                    feat_out_dir = os.path.join(save_img_dir, 'features')
                    feat_path, spath = save_mesh_features(feat_out_dir, frame_i, i, ref_feature, trg, sampling_points=sampling_points)

                    # If requested, overlay features on full image and save
                    if args.show_features:
                        try:
                            # per-point descriptors: [B, N, C]
                            C_p = int(trg.cfg.MODEL.TRG.MLP_DIM[-1])
                            B = ref_feature.shape[0]
                            N = int(ref_feature.shape[1] // C_p)
                            per_point = ref_feature.view(B, N, C_p).cpu().numpy()

                            # get sampling points in crop pixel coords
                            if sampling_points is None:
                                # reconstruct grid sampling positions for rf_i == 0
                                grid_norm = trg.points_grid.squeeze(0).t().cpu().numpy()  # (N,2) normalized [-1,1]
                                # map to crop pixels (img_size used earlier)
                                img_size_local = img.shape[-1]
                                samp = (grid_norm + 1.0) / 2.0 * (img_size_local - 1)
                            else:
                                if isinstance(sampling_points, torch.Tensor):
                                    samp = sampling_points.cpu().numpy()[0]
                                else:
                                    samp = sampling_points[0]

                            # map crop coords back to full image coordinates using tform_inv
                            pts = samp.reshape(-1, 2)
                            ones = np.ones((pts.shape[0], 1), dtype=np.float32)
                            pts_h = np.concatenate([pts, ones], axis=1)  # Nx3
                            tform_inv_np = np.array(tform_inv, dtype=np.float32)
                            full_pts = (pts_h @ tform_inv_np.T)  # Nx2

                            # color by mean descriptor per point
                            scalars = per_point[0].mean(axis=1)
                            smin, smax = float(scalars.min()), float(scalars.max())
                            if smax - smin < 1e-8:
                                norm_s = np.zeros_like(scalars)
                            else:
                                norm_s = (scalars - smin) / (smax - smin)
                            colors = (cm.get_cmap('jet')(norm_s)[:, :3] * 255).astype(np.uint8)

                            # draw on a copy of original full image
                            overlay_img = img_full.copy()
                            for (x_f, y_f), col in zip(full_pts, colors):
                                xi, yi = int(round(x_f)), int(round(y_f))
                                if xi < 0 or yi < 0 or xi >= overlay_img.shape[1] or yi >= overlay_img.shape[0]:
                                    continue
                                # col is RGB; convert to BGR for cv2
                                cv2.circle(overlay_img, (xi, yi), 2, (int(col[2]), int(col[1]), int(col[0])), -1)

                            overlays_dir = os.path.join(feat_out_dir, 'overlays')
                            os.makedirs(overlays_dir, exist_ok=True)
                            overlay_path = os.path.join(overlays_dir, f'frame_{frame_i:05d}_face_{i:03d}_overlay.jpg')
                            cv2.imwrite(overlay_path, overlay_img)
                        except Exception as e_overlay:
                            print(f'Warning: failed to create overlay for frame {frame_i} face {i}: {e_overlay}')
                except Exception as e:
                    # don't break inference if feature save fails
                    print(f'Warning: failed to extract/save features for frame {frame_i} face {i}: {e}')

            mesh_each_frame.append(pred_vtx_cam[0].cpu().numpy())

        mesh_total_frame.append(mesh_each_frame)
        ################################################
        # Rendering part
        ################################################
        if args.do_render:
            rendered_img = img_full.copy()
            for pred_verts_cam in pred_verts_cam_list:
                # render mesh on image
                tmp_verts_cam = pred_verts_cam[0].cpu().numpy()

                # uncropped intrinsics setting
                focal_length = [intrinsic_uncrop[0, 0], intrinsic_uncrop[1, 1]]
                princpt = intrinsic_uncrop[2, :2]
                rendered_img = render_mesh(rendered_img.copy(), tmp_verts_cam, mesh_faces, focal_length, princpt, alpha=1.0)
                out_img_path = os.path.join(tmp_render_out_dir, 'frame_%05d_rgb.jpg' % frame_i)
                cv2.imwrite(out_img_path, rendered_img)

    ###############################################
    # Save mesh, faces obj
    ###############################################
    path = os.path.join(save_img_dir, '1_mesh.pkl')
    write_pkl(path, mesh_total_frame)
    print(f'save mesh: {path}')

    ###############################################
    # Image to video conversion
    ###############################################
    if args.do_render:
        frame_rate = args.frame_rate
        output_video_path = demo_video_path.replace(".mp4", "_rendered.mp4")
        image2video(tmp_render_out_dir, output_video_path, frame_rate, width=img_w, height=img_h, img_ext='jpg')
        # image2video(tmp_crop_render_out_dir, output_video_path, frame_rate, width=img_w, height=img_h, img_ext='jpg')

if __name__ == "__main__":
    args = parse_args()
    print("args: {}".format(args))
    main(args)
