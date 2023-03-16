from PIL import Image
import os
import time
import numpy as np
import torch
import torch.nn.functional as F

import data
from utils import frame_utils
from utils.flow_viz import save_vis_flow_tofile

from utils.utils import InputPadder, compute_out_of_boundary_mask
from glob import glob
from gmflow.geometry import forward_backward_consistency_check


def calculate_great_circle_distance(input_flow, target_flow, grid):
    input = torch.clone(input_flow)
    gt = torch.clone(target_flow)
    input[0, :, :] = input[0, :, :] / (1024/2) * 3.1415926
    input[1, :, :] = input[1, :, :] / (512/2) * 3.1415926
    gt[0, :, :] = gt[0, :, :] / (1024/2) * 3.1415926
    gt[1, :, :] = gt[1, :, :] / (512/2) * 3.1415926
    input = grid + input
    gt = grid + gt
    theta = input[0, :, :]
    phi = input[1, :, :]

    _x = torch.cos(phi) * torch.cos(theta)
    _y = torch.sin(phi)
    _z = torch.cos(phi) * torch.sin(theta)
    gt_theta = gt[0, :, :]
    gt_phi = gt[1, :, :]

    gt_x = torch.cos(gt_phi) * torch.cos(gt_theta)
    gt_y = torch.sin(gt_phi)
    gt_z = torch.cos(gt_phi) * torch.sin(gt_theta)
    dot_result = _x * gt_x + _y * gt_y + _z * gt_z
    dot_result[dot_result < -1] = -1
    dot_result[dot_result > 1] = 1
    angle = torch.arccos(dot_result)

    arc = angle * 1024.0 / (2 * 3.1415926)

    ret = arc.mean()
    return ret


@torch.no_grad()
def create_sintel_submission(model,
                             output_path='sintel_submission',
                             padding_factor=8,
                             save_vis_flow=False,
                             no_save_flo=False,
                             attn_splits_list=None,
                             corr_radius_list=None,
                             prop_radius_list=None,
                             ):
    """ Create submission for the Sintel leaderboard """
    model.eval()
    for dstype in ['clean', 'final']:
        test_dataset = data.MpiSintel(split='test', aug_params=None, dstype=dstype)

        flow_prev, sequence_prev = None, None
        for test_id in range(len(test_dataset)):
            image1, image2, (sequence, frame) = test_dataset[test_id]
            if sequence != sequence_prev:
                flow_prev = None

            padder = InputPadder(image1.shape, padding_factor=padding_factor)
            image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())

            results_dict = model(image1, image2,
                                 attn_splits_list=attn_splits_list,
                                 corr_radius_list=corr_radius_list,
                                 prop_radius_list=prop_radius_list,
                                 )

            flow_pr = results_dict['flow_preds'][-1]  # [B, 2, H, W]

            flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()

            output_dir = os.path.join(output_path, dstype, sequence)
            output_file = os.path.join(output_dir, 'frame%04d.flo' % (frame + 1))

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            if not no_save_flo:
                frame_utils.writeFlow(output_file, flow)
            sequence_prev = sequence

            # Save vis flow
            if save_vis_flow:
                vis_flow_file = output_file.replace('.flo', '.png')
                save_vis_flow_tofile(flow, vis_flow_file)


@torch.no_grad()
def create_kitti_submission(model,
                            output_path='kitti_submission',
                            padding_factor=8,
                            save_vis_flow=False,
                            attn_splits_list=None,
                            corr_radius_list=None,
                            prop_radius_list=None,
                            ):
    """ Create submission for the Sintel leaderboard """
    model.eval()
    test_dataset = data.KITTI(split='testing', aug_params=None)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for test_id in range(len(test_dataset)):
        image1, image2, (frame_id,) = test_dataset[test_id]
        padder = InputPadder(image1.shape, mode='kitti', padding_factor=padding_factor)
        image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())

        results_dict = model(image1, image2,
                             attn_splits_list=attn_splits_list,
                             corr_radius_list=corr_radius_list,
                             prop_radius_list=prop_radius_list,
                             )

        flow_pr = results_dict['flow_preds'][-1]

        flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()

        output_filename = os.path.join(output_path, frame_id)

        if save_vis_flow:
            vis_flow_file = output_filename
            save_vis_flow_tofile(flow, vis_flow_file)
        else:
            frame_utils.writeFlowKITTI(output_filename, flow)


@torch.no_grad()
def validate_chairs(model,
                    with_speed_metric=False,
                    attn_splits_list=False,
                    corr_radius_list=False,
                    prop_radius_list=False,
                    ):
    """ Perform evaluation on the FlyingChairs (test) split """
    model.eval()
    epe_list = []
    results = {}

    if with_speed_metric:
        s0_10_list = []
        s10_40_list = []
        s40plus_list = []

    val_dataset = data.FlyingChairs(split='validation')

    print('Number of validation image pairs: %d' % len(val_dataset))

    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, _ = val_dataset[val_id]

        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        results_dict = model(image1, image2,
                             attn_splits_list=attn_splits_list,
                             corr_radius_list=corr_radius_list,
                             prop_radius_list=prop_radius_list,
                             )

        flow_pr = results_dict['flow_preds'][-1]  # [B, 2, H, W]

        assert flow_pr.size()[-2:] == flow_gt.size()[-2:]

        epe = torch.sum((flow_pr[0].cpu() - flow_gt) ** 2, dim=0).sqrt()
        epe_list.append(epe.view(-1).numpy())

        if with_speed_metric:
            flow_gt_speed = torch.sum(flow_gt ** 2, dim=0).sqrt()
            valid_mask = (flow_gt_speed < 10)
            if valid_mask.max() > 0:
                s0_10_list.append(epe[valid_mask].cpu().numpy())

            valid_mask = (flow_gt_speed >= 10) * (flow_gt_speed <= 40)
            if valid_mask.max() > 0:
                s10_40_list.append(epe[valid_mask].cpu().numpy())

            valid_mask = (flow_gt_speed > 40)
            if valid_mask.max() > 0:
                s40plus_list.append(epe[valid_mask].cpu().numpy())

    epe_all = np.concatenate(epe_list)
    epe = np.mean(epe_all)
    px1 = np.mean(epe_all > 1)
    px3 = np.mean(epe_all > 3)
    px5 = np.mean(epe_all > 5)
    print("Validation Chairs EPE: %.3f, 1px: %.3f, 3px: %.3f, 5px: %.3f" % (epe, px1, px3, px5))
    results['chairs_epe'] = epe
    results['chairs_1px'] = px1
    results['chairs_3px'] = px3
    results['chairs_5px'] = px5

    if with_speed_metric:
        s0_10 = np.mean(np.concatenate(s0_10_list))
        s10_40 = np.mean(np.concatenate(s10_40_list))
        s40plus = np.mean(np.concatenate(s40plus_list))

        print("Validation Chairs s0_10: %.3f, s10_40: %.3f, s40+: %.3f" % (
            s0_10,
            s10_40,
            s40plus))

        results['chairs_s0_10'] = s0_10
        results['chairs_s10_40'] = s10_40
        results['chairs_s40+'] = s40plus

    return results


@torch.no_grad()
def validate_things(model,
                    padding_factor=8,
                    with_speed_metric=False,
                    max_val_flow=400,
                    val_things_clean_only=True,
                    attn_splits_list=False,
                    corr_radius_list=False,
                    prop_radius_list=False,
                    ):
    """ Peform validation using the Things (test) split """
    model.eval()
    results = {}

    for dstype in ['frames_cleanpass', 'frames_finalpass']:
        if val_things_clean_only:
            if dstype == 'frames_finalpass':
                continue

        val_dataset = data.FlyingThings3D(dstype=dstype, test_set=True, validate_subset=True,
                                          )
        print('Number of validation image pairs: %d' % len(val_dataset))
        epe_list = []

        if with_speed_metric:
            s0_10_list = []
            s10_40_list = []
            s40plus_list = []

        for val_id in range(len(val_dataset)):
            image1, image2, flow_gt, valid_gt = val_dataset[val_id]
            image1 = image1[None].cuda()
            image2 = image2[None].cuda()

            padder = InputPadder(image1.shape, padding_factor=padding_factor)
            image1, image2 = padder.pad(image1, image2)

            results_dict = model(image1, image2,
                                 attn_splits_list=attn_splits_list,
                                 corr_radius_list=corr_radius_list,
                                 prop_radius_list=prop_radius_list,
                                 )
            flow_pr = results_dict['flow_preds'][-1]

            flow = padder.unpad(flow_pr[0]).cpu()

            # Evaluation on flow <= max_val_flow
            flow_gt_speed = torch.sum(flow_gt ** 2, dim=0).sqrt()
            valid_gt = valid_gt * (flow_gt_speed < max_val_flow)
            valid_gt = valid_gt.contiguous()

            epe = torch.sum((flow - flow_gt) ** 2, dim=0).sqrt()
            val = valid_gt >= 0.5
            epe_list.append(epe[val].cpu().numpy())

            if with_speed_metric:
                valid_mask = (flow_gt_speed < 10) * (valid_gt >= 0.5)
                if valid_mask.max() > 0:
                    s0_10_list.append(epe[valid_mask].cpu().numpy())

                valid_mask = (flow_gt_speed >= 10) * (flow_gt_speed <= 40) * (valid_gt >= 0.5)
                if valid_mask.max() > 0:
                    s10_40_list.append(epe[valid_mask].cpu().numpy())

                valid_mask = (flow_gt_speed > 40) * (valid_gt >= 0.5)
                if valid_mask.max() > 0:
                    s40plus_list.append(epe[valid_mask].cpu().numpy())

        epe_list = np.mean(np.concatenate(epe_list))

        epe = np.mean(epe_list)

        if dstype == 'frames_cleanpass':
            dstype = 'things_clean'
        if dstype == 'frames_finalpass':
            dstype = 'things_final'

        print("Validation Things test set (%s) EPE: %.3f" % (dstype, epe))
        results[dstype + '_epe'] = epe

        if with_speed_metric:
            s0_10 = np.mean(np.concatenate(s0_10_list))
            s10_40 = np.mean(np.concatenate(s10_40_list))
            s40plus = np.mean(np.concatenate(s40plus_list))

            print("Validation Things test (%s) s0_10: %.3f, s10_40: %.3f, s40+: %.3f" % (
                dstype, s0_10,
                s10_40,
                s40plus))

            results[dstype + '_s0_10'] = s0_10
            results[dstype + '_s10_40'] = s10_40
            results[dstype + '_s40+'] = s40plus

    return results


@torch.no_grad()
def validate_sintel(model,
                    count_time=False,
                    padding_factor=8,
                    with_speed_metric=False,
                    evaluate_matched_unmatched=False,
                    attn_splits_list=False,
                    corr_radius_list=False,
                    prop_radius_list=False,
                    ):
    """ Peform validation using the Sintel (train) split """
    model.eval()
    results = {}

    if count_time:
        total_time = 0
        num_runs = 100

    for dstype in ['clean', 'final']:
        val_dataset = data.MpiSintel(split='training', dstype=dstype,
                                     load_occlusion=evaluate_matched_unmatched,
                                     )

        print('Number of validation image pairs: %d' % len(val_dataset))
        epe_list = []

        if evaluate_matched_unmatched:
            matched_epe_list = []
            unmatched_epe_list = []

        if with_speed_metric:
            s0_10_list = []
            s10_40_list = []
            s40plus_list = []

        for val_id in range(len(val_dataset)):
            if evaluate_matched_unmatched:
                image1, image2, flow_gt, valid, noc_valid = val_dataset[val_id]

                # compuate in-image-plane valid mask
                in_image_valid = compute_out_of_boundary_mask(flow_gt.unsqueeze(0)).squeeze(0)  # [H, W]

            else:
                image1, image2, flow_gt, _ = val_dataset[val_id]

            image1 = image1[None].cuda()
            image2 = image2[None].cuda()

            padder = InputPadder(image1.shape, padding_factor=padding_factor)
            image1, image2 = padder.pad(image1, image2)

            if count_time and val_id >= 5:  # 5 warmup
                torch.cuda.synchronize()
                time_start = time.perf_counter()

            results_dict = model(image1, image2,
                                 attn_splits_list=attn_splits_list,
                                 corr_radius_list=corr_radius_list,
                                 prop_radius_list=prop_radius_list,
                                 )

            # useful when using parallel branches
            flow_pr = results_dict['flow_preds'][-1]

            if count_time and val_id >= 5:
                torch.cuda.synchronize()
                total_time += time.perf_counter() - time_start

                if val_id >= num_runs + 4:
                    break

            flow = padder.unpad(flow_pr[0]).cpu()

            epe = torch.sum((flow - flow_gt) ** 2, dim=0).sqrt()
            epe_list.append(epe.view(-1).numpy())

            if evaluate_matched_unmatched:
                matched_valid_mask = (noc_valid > 0.5) & (in_image_valid > 0.5)

                if matched_valid_mask.max() > 0:
                    matched_epe_list.append(epe[matched_valid_mask].cpu().numpy())
                    unmatched_epe_list.append(epe[~matched_valid_mask].cpu().numpy())

            if with_speed_metric:
                flow_gt_speed = torch.sum(flow_gt ** 2, dim=0).sqrt()
                valid_mask = (flow_gt_speed < 10)
                if valid_mask.max() > 0:
                    s0_10_list.append(epe[valid_mask].cpu().numpy())

                valid_mask = (flow_gt_speed >= 10) * (flow_gt_speed <= 40)
                if valid_mask.max() > 0:
                    s10_40_list.append(epe[valid_mask].cpu().numpy())

                valid_mask = (flow_gt_speed > 40)
                if valid_mask.max() > 0:
                    s40plus_list.append(epe[valid_mask].cpu().numpy())

        epe_all = np.concatenate(epe_list)
        epe = np.mean(epe_all)
        px1 = np.mean(epe_all > 1)
        px3 = np.mean(epe_all > 3)
        px5 = np.mean(epe_all > 5)

        dstype_ori = dstype

        print("Validation Sintel (%s) EPE: %.3f, 1px: %.3f, 3px: %.3f, 5px: %.3f" % (dstype_ori, epe, px1, px3, px5))

        dstype = 'sintel_' + dstype

        results[dstype + '_epe'] = np.mean(epe_list)
        results[dstype + '_1px'] = px1
        results[dstype + '_3px'] = px3
        results[dstype + '_5px'] = px5

        if with_speed_metric:
            s0_10 = np.mean(np.concatenate(s0_10_list))
            s10_40 = np.mean(np.concatenate(s10_40_list))
            s40plus = np.mean(np.concatenate(s40plus_list))

            print("Validation Sintel (%s) s0_10: %.3f, s10_40: %.3f, s40+: %.3f" % (
                dstype_ori, s0_10,
                s10_40,
                s40plus))

            results[dstype + '_s0_10'] = s0_10
            results[dstype + '_s10_40'] = s10_40
            results[dstype + '_s40+'] = s40plus

        if count_time:
            print('Time: %.6fs' % (total_time / num_runs))
            break  # only the clean pass when counting time

        if evaluate_matched_unmatched:
            matched_epe = np.mean(np.concatenate(matched_epe_list))
            unmatched_epe = np.mean(np.concatenate(unmatched_epe_list))

            print('Validatation Sintel (%s) matched epe: %.3f, unmatched epe: %.3f' % (
                dstype_ori, matched_epe, unmatched_epe))

            results[dstype + '_matched'] = matched_epe
            results[dstype + '_unmatched'] = unmatched_epe

    return results


@torch.no_grad()
def validate_kitti(model,
                   padding_factor=8,
                   with_speed_metric=False,
                   average_over_pixels=True,
                   attn_splits_list=False,
                   corr_radius_list=False,
                   prop_radius_list=False,
                   ):
    """ Peform validation using the KITTI-2015 (train) split """
    model.eval()

    val_dataset = data.KITTI(split='training')
    print('Number of validation image pairs: %d' % len(val_dataset))

    out_list, epe_list = [], []
    results = {}

    if with_speed_metric:
        if average_over_pixels:
            s0_10_list = []
            s10_40_list = []
            s40plus_list = []
        else:
            s0_10_epe_sum = 0
            s0_10_valid_samples = 0
            s10_40_epe_sum = 0
            s10_40_valid_samples = 0
            s40plus_epe_sum = 0
            s40plus_valid_samples = 0

    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, mode='kitti', padding_factor=padding_factor)
        image1, image2 = padder.pad(image1, image2)

        results_dict = model(image1, image2,
                             attn_splits_list=attn_splits_list,
                             corr_radius_list=corr_radius_list,
                             prop_radius_list=prop_radius_list,
                             )

        # useful when using parallel branches
        flow_pr = results_dict['flow_preds'][-1]

        flow = padder.unpad(flow_pr[0]).cpu()

        epe = torch.sum((flow - flow_gt) ** 2, dim=0).sqrt()
        mag = torch.sum(flow_gt ** 2, dim=0).sqrt()

        if with_speed_metric:
            # flow_gt_speed = torch.sum(flow_gt ** 2, dim=0).sqrt()
            flow_gt_speed = mag

            if average_over_pixels:
                valid_mask = (flow_gt_speed < 10) * (valid_gt >= 0.5)  # note KITTI GT is sparse
                if valid_mask.max() > 0:
                    s0_10_list.append(epe[valid_mask].cpu().numpy())

                valid_mask = (flow_gt_speed >= 10) * (flow_gt_speed <= 40) * (valid_gt >= 0.5)
                if valid_mask.max() > 0:
                    s10_40_list.append(epe[valid_mask].cpu().numpy())

                valid_mask = (flow_gt_speed > 40) * (valid_gt >= 0.5)
                if valid_mask.max() > 0:
                    s40plus_list.append(epe[valid_mask].cpu().numpy())

            else:
                valid_mask = (flow_gt_speed < 10) * (valid_gt >= 0.5)  # note KITTI GT is sparse
                if valid_mask.max() > 0:
                    s0_10_epe_sum += (epe * valid_mask).sum() / valid_mask.sum()
                    s0_10_valid_samples += 1

                valid_mask = (flow_gt_speed >= 10) * (flow_gt_speed <= 40) * (valid_gt >= 0.5)
                if valid_mask.max() > 0:
                    s10_40_epe_sum += (epe * valid_mask).sum() / valid_mask.sum()
                    s10_40_valid_samples += 1

                valid_mask = (flow_gt_speed > 40) * (valid_gt >= 0.5)
                if valid_mask.max() > 0:
                    s40plus_epe_sum += (epe * valid_mask).sum() / valid_mask.sum()
                    s40plus_valid_samples += 1

        epe = epe.view(-1)
        mag = mag.view(-1)
        val = valid_gt.view(-1) >= 0.5

        out = ((epe > 3.0) & ((epe / mag) > 0.05)).float()

        if average_over_pixels:
            epe_list.append(epe[val].cpu().numpy())
        else:
            epe_list.append(epe[val].mean().item())

        out_list.append(out[val].cpu().numpy())

    if average_over_pixels:
        epe_list = np.concatenate(epe_list)
    else:
        epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    f1 = 100 * np.mean(out_list)

    print("Validation KITTI EPE: %.3f, F1-all: %.3f" % (epe, f1))
    results['kitti_epe'] = epe
    results['kitti_f1'] = f1

    if with_speed_metric:
        if average_over_pixels:
            s0_10 = np.mean(np.concatenate(s0_10_list))
            s10_40 = np.mean(np.concatenate(s10_40_list))
            s40plus = np.mean(np.concatenate(s40plus_list))
        else:
            s0_10 = s0_10_epe_sum / s0_10_valid_samples
            s10_40 = s10_40_epe_sum / s10_40_valid_samples
            s40plus = s40plus_epe_sum / s40plus_valid_samples

        print("Validation KITTI s0_10: %.3f, s10_40: %.3f, s40+: %.3f" % (
            s0_10,
            s10_40,
            s40plus))

        results['kitti_s0_10'] = s0_10
        results['kitti_s10_40'] = s10_40
        results['kitti_s40+'] = s40plus

    return results


def convert_360_gt(flow_gt):
    '''Convert gt to 360 flow'''
    flow_gt = flow_gt.unsqueeze(dim=0)
    flow_gt[:, 0] = torch.where(flow_gt[:, 0] > (flow_gt.shape[3] // 2),
                                flow_gt[:, 0] - flow_gt.shape[3],
                                flow_gt[:, 0])
    flow_gt[:, 0] = torch.where(flow_gt[:, 0] < -(flow_gt.shape[3] // 2),
                                flow_gt.shape[3] + flow_gt[:, 0],
                                flow_gt[:, 0])
    return flow_gt.squeeze()


@torch.no_grad()
def validate_omni(model,
                  padding_factor=8,
                  with_speed_metric=False,
                  max_val_flow=400,
                  val_things_clean_only=True,
                  attn_splits_list=False,
                  corr_radius_list=False,
                  prop_radius_list=False,
                  cfe_activate=False
                  ):
    """ Peform validation using the Omni (test) split """
    model.eval()
    results = {}
    epe_any = []
    epe_m = []
    

    for dstype in ['CartoonTree', 'Forest', 'LowPolyModels']:
        val_dataset = data.OmniDataset(root="/media/cartizzu/DATA/DATASETS/OMNIFLOWNET_DATASET/", dstype=dstype)
        
        sum_time = 0

        print('Number of validation image pairs: %d' % len(val_dataset))
        epe_list = []

        for val_id in range(len(val_dataset)):
            image1, image2, flow_gt, valid_gt = val_dataset[val_id]
            image1 = image1[None].cuda()
            image2 = image2[None].cuda()

            padder = InputPadder(image1.shape, padding_factor=padding_factor)
            image1, image2 = padder.pad(image1, image2)

            # convert gt to 360 flow
            if cfe_activate:
                flow_gt = convert_360_gt(flow_gt)

                # check if is 360 flow gt
                if flow_gt[0, :, :].max() > flow_gt.shape[2]//2:
                    raise "Not 360 Flow"
            start_time = time.time()
            results_dict = model(image1, image2,
                                 attn_splits_list=attn_splits_list,
                                 corr_radius_list=corr_radius_list,
                                 prop_radius_list=prop_radius_list,
                                 )
            loc_time = 100*(time.time() - start_time)
            if val_id != 0:
                sum_time = sum_time + float(loc_time)
            # print("Total execution time: {0:.3f} ms".format(sum_time))
            # print("Total per item: {0:.3f} ms".format(loc_time))

            flow_pr = results_dict['flow_preds'][-1]

            flow = padder.unpad(flow_pr[0]).cpu()

            # Evaluation on flow <= max_val_flow
            flow_gt_speed = torch.sum(flow_gt ** 2, dim=0).sqrt()
            valid_gt = valid_gt * (flow_gt_speed < max_val_flow)
            valid_gt = valid_gt.contiguous()

            epe = torch.sum((flow - flow_gt) ** 2, dim=0).sqrt()
            val = valid_gt >= 0.5
            epe_list.append(epe[val].cpu().numpy())

        epe_any.append(epe_list)
        epe_all = np.concatenate(epe_list)
        epe = np.mean(epe_all)
        epe_m.append(epe)

        print(len(val_dataset))
        print("Total execution time: {0:.3f} ms".format(sum_time))
        print("Total per item: {0:.3f} ms".format(sum_time/len(val_dataset)))

        print("Validation Omni {} EPE : {}".format(dstype, epe))
        results[dstype + '_epe'] = epe
        # import pdb
        # pdb.set_trace()

    # epe_final_all = np.concatenate(epe_any)
    epe_final = np.mean(epe_m)
    print("Validation Omni (final) EPE : {}".format(epe_final))
    results['final'] = epe_final

    if cfe_activate:
        print("CFE is activated: Flow is 360")

    return results


@torch.no_grad()
def validate_flow360(model,
                     padding_factor=8,
                     with_speed_metric=False,
                     max_val_flow=400,
                     val_things_clean_only=True,
                     attn_splits_list=False,
                     corr_radius_list=False,
                     prop_radius_list=False,
                     cfe_activate=False
                     ):
    """ Peform validation using the Flow360 (test) split """
    model.eval()
    results = {}
    epe_any = []
    epe_m = []

    for dstype in ['cloud', 'fog', 'rain', 'sunny']:
        val_dataset = data.Flow360(split='test', root="/media/cartizzu/DATA/DATASETS/Flow360_release/", dstype=dstype)

        print('Number of validation image pairs: %d' % len(val_dataset))
        epe_list = []

        for val_id in range(len(val_dataset)):
            image1, image2, flow_gt, valid_gt = val_dataset[val_id]
            image1 = image1[None].cuda()
            image2 = image2[None].cuda()

            padder = InputPadder(image1.shape, padding_factor=padding_factor)
            image1, image2 = padder.pad(image1, image2)

            # convert gt to 360 flow
            if cfe_activate:
                flow_gt = convert_360_gt(flow_gt)

                # check if is 360 flow gt
                if flow_gt[0, :, :].max() > flow_gt.shape[2]//2:
                    raise "Not 360 Flow"

            results_dict = model(image1, image2,
                                 attn_splits_list=attn_splits_list,
                                 corr_radius_list=corr_radius_list,
                                 prop_radius_list=prop_radius_list,
                                 )

            flow_pr = results_dict['flow_preds'][-1]

            flow = padder.unpad(flow_pr[0]).cpu()

            # Evaluation on flow <= max_val_flow
            flow_gt_speed = torch.sum(flow_gt ** 2, dim=0).sqrt()
            valid_gt = valid_gt * (flow_gt_speed < max_val_flow)
            valid_gt = valid_gt.contiguous()

            # if val_id == 0:
            #     print("WARNING TOP BOTTOM OF IMAGE CROPPED!")
            # valid_gt[255:, :] = 0

            epe = torch.sum((flow - flow_gt) ** 2, dim=0).sqrt()
            val = valid_gt >= 0.5
            epe_list.append(epe[val].cpu().numpy())

        epe_any.append(epe_list)
        epe_all = np.concatenate(epe_list)
        epe = np.mean(epe_all)
        epe_m.append(epe)

        print("Validation Flow360 {} EPE : {}".format(dstype, epe))
        results[dstype + '_epe'] = epe

    # epe_final_all = np.concatenate(epe_any)
    epe_final = np.mean(epe_m)
    print("Validation Flow360 (final) EPE : {}".format(epe_final))
    results['final'] = epe_final

    if cfe_activate:
        print("CFE is activated: Flow is 360")

    return results


@torch.no_grad()
def validate_cityscene(model,
                       padding_factor=8,
                       with_speed_metric=False,
                       max_val_flow=400,
                       val_things_clean_only=True,
                       attn_splits_list=False,
                       corr_radius_list=False,
                       prop_radius_list=False,
                       cfe_activate=False
                       ):
    """ Peform validation using the Cityscene split """
    model.eval()
    results = {}
    epe_any = []
    epe_m = []

    for dstype in ['CityScene_100_r', 'CityScene_200_r', 'CityScene_2000_r']:
        val_dataset = data.CityScene(split='', root="/media/cartizzu/DATA/DATASETS/CityScene/", dstype=dstype)

        print('Number of validation image pairs: %d' % len(val_dataset))
        epe_list = []

        for val_id in range(len(val_dataset)):
            image1, image2, flow_gt, valid_gt = val_dataset[val_id]
            image1 = image1[None].cuda()
            image2 = image2[None].cuda()

            padder = InputPadder(image1.shape, padding_factor=padding_factor)
            image1, image2 = padder.pad(image1, image2)

            # # convert gt to 360 flow
            # if cfe_activate:
            #     flow_gt = convert_360_gt(flow_gt)

            #     # check if is 360 flow gt
            #     if flow_gt[0, :, :].max() > flow_gt.shape[2]//2:
            #         raise "Not 360 Flow"

            results_dict = model(image1, image2,
                                 attn_splits_list=attn_splits_list,
                                 corr_radius_list=corr_radius_list,
                                 prop_radius_list=prop_radius_list,
                                 )

            flow_pr = results_dict['flow_preds'][-1]

            flow = padder.unpad(flow_pr[0]).cpu()

            # Evaluation on flow <= max_val_flow
            flow_gt_speed = torch.sum(flow_gt ** 2, dim=0).sqrt()
            valid_gt = valid_gt * (flow_gt_speed < max_val_flow)
            valid_gt = valid_gt.contiguous()

            # if val_id == 0:
            #     print("WARNING TOP BOTTOM OF IMAGE CROPPED!")
            if cfe_activate:
                if val_id == 0:
                    print("WARNING TOP and BOTTOM OF IMAGE CROPPED!")
                valid_gt[:100, :] = 0
                valid_gt[412:, :] = 0

            epe = torch.sum((flow - flow_gt) ** 2, dim=0).sqrt()
            val = valid_gt >= 0.5
            epe_list.append(epe[val].cpu().numpy())

        epe_any.append(epe_list)
        epe_all = np.concatenate(epe_list)
        epe = np.mean(epe_all)
        epe_m.append(epe)

        print("Validation CityScene {} EPE : {}".format(dstype, epe))
        results[dstype + '_epe'] = epe

    # epe_final_all = np.concatenate(epe_any)
    epe_final = np.mean(epe_m)
    print("Validation CityScene (final) EPE : {}".format(epe_final))
    results['final'] = epe_final

    if cfe_activate:
        print("CFE is activated: Flow is 360")

    return results


@torch.no_grad()
def validate_EFT(model,
                 padding_factor=8,
                 with_speed_metric=False,
                 max_val_flow=400,
                 val_things_clean_only=True,
                 attn_splits_list=False,
                 corr_radius_list=False,
                 prop_radius_list=False,
                 cfe_activate=False
                 ):
    """ Peform validation using the EFT split """
    model.eval()
    results = {}
    epe_any = []
    epe_m = []

    for dstype in ['EFTs_Car100', 'EFTs_Car200', 'EFTs_Car2000']:
        val_dataset = data.EFT(split='', root="/media/cartizzu/DATA/DATASETS/EquirectFlyingThings/", dstype=dstype)

        print('Number of validation image pairs: %d' % len(val_dataset))
        epe_list = []

        for val_id in range(len(val_dataset)):
            image1, image2, flow_gt, valid_gt = val_dataset[val_id]
            image1 = image1[None].cuda()
            image2 = image2[None].cuda()

            padder = InputPadder(image1.shape, padding_factor=padding_factor)
            image1, image2 = padder.pad(image1, image2)

            # convert gt to 360 flow
            if cfe_activate:
                flow_gt = convert_360_gt(flow_gt)

                # check if is 360 flow gt
                if flow_gt[0, :, :].max() > flow_gt.shape[2]//2:
                    raise "Not 360 Flow"

            results_dict = model(image1, image2,
                                 attn_splits_list=attn_splits_list,
                                 corr_radius_list=corr_radius_list,
                                 prop_radius_list=prop_radius_list,
                                 )

            flow_pr = results_dict['flow_preds'][-1]

            flow = padder.unpad(flow_pr[0]).cpu()

            # Evaluation on flow <= max_val_flow
            flow_gt_speed = torch.sum(flow_gt ** 2, dim=0).sqrt()
            valid_gt = valid_gt * (flow_gt_speed < max_val_flow)
            valid_gt = valid_gt.contiguous()

            if cfe_activate:
                if val_id == 0:
                    print("WARNING TOP and BOTTOM OF IMAGE CROPPED!")
                valid_gt[:100, :] = 0
                valid_gt[412:, :] = 0
                valid_gt[:, :20] = 0
                valid_gt[:, 1004:] = 0

            epe = torch.sum((flow - flow_gt) ** 2, dim=0).sqrt()
            val = valid_gt >= 0.5
            epe_list.append(epe[val].cpu().numpy())

        epe_any.append(epe_list)
        epe_all = np.concatenate(epe_list)
        epe = np.mean(epe_all)
        epe_m.append(epe)

        print("Validation EFT {} EPE : {}".format(dstype, epe))
        results[dstype + '_epe'] = epe

    # epe_final_all = np.concatenate(epe_any)
    epe_final = np.mean(epe_m)
    print("Validation EFT (final) EPE : {}".format(epe_final))
    results['final'] = epe_final

    if cfe_activate:
        print("CFE is activated: Flow is 360")

    return results


@torch.no_grad()
def inference_on_dir(model,
                     inference_dir,
                     output_path='output',
                     padding_factor=8,
                     inference_size=None,
                     paired_data=False,  # dir of paired testdata instead of a sequence
                     save_flo_flow=False,  # save as .flo for quantative evaluation
                     attn_splits_list=None,
                     corr_radius_list=None,
                     prop_radius_list=None,
                     pred_bidir_flow=False,
                     fwd_bwd_consistency_check=False,
                     ):
    """ Inference on a directory """
    model.eval()

    if fwd_bwd_consistency_check:
        assert pred_bidir_flow

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    filenames = sorted(glob(inference_dir + '/*'))
    print('%d images found' % len(filenames))

    stride = 2 if paired_data else 1

    if paired_data:
        assert len(filenames) % 2 == 0

    for test_id in range(0, len(filenames) - 1, stride):

        image1 = frame_utils.read_gen(filenames[test_id])
        image2 = frame_utils.read_gen(filenames[test_id + 1])

        image1 = np.array(image1).astype(np.uint8)
        image2 = np.array(image2).astype(np.uint8)

        if len(image1.shape) == 2:  # gray image, for example, HD1K
            image1 = np.tile(image1[..., None], (1, 1, 3))
            image2 = np.tile(image2[..., None], (1, 1, 3))
        else:
            image1 = image1[..., :3]
            image2 = image2[..., :3]

        image1 = torch.from_numpy(image1).permute(2, 0, 1).float()
        image2 = torch.from_numpy(image2).permute(2, 0, 1).float()

        if inference_size is None:
            padder = InputPadder(image1.shape, padding_factor=padding_factor)
            image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())
        else:
            image1, image2 = image1[None].cuda(), image2[None].cuda()

        # resize before inference
        if inference_size is not None:
            assert isinstance(inference_size, list) or isinstance(inference_size, tuple)
            ori_size = image1.shape[-2:]
            image1 = F.interpolate(image1, size=inference_size, mode='bilinear',
                                   align_corners=True)
            image2 = F.interpolate(image2, size=inference_size, mode='bilinear',
                                   align_corners=True)

        results_dict = model(image1, image2,
                             attn_splits_list=attn_splits_list,
                             corr_radius_list=corr_radius_list,
                             prop_radius_list=prop_radius_list,
                             pred_bidir_flow=pred_bidir_flow,
                             )

        flow_pr = results_dict['flow_preds'][-1]  # [B, 2, H, W]

        # resize back
        if inference_size is not None:
            flow_pr = F.interpolate(flow_pr, size=ori_size, mode='bilinear',
                                    align_corners=True)
            flow_pr[:, 0] = flow_pr[:, 0] * ori_size[-1] / inference_size[-1]
            flow_pr[:, 1] = flow_pr[:, 1] * ori_size[-2] / inference_size[-2]

        if inference_size is None:
            flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()  # [H, W, 2]
        else:
            flow = flow_pr[0].permute(1, 2, 0).cpu().numpy()  # [H, W, 2]

        output_file = os.path.join(output_path, os.path.basename(filenames[test_id])[:-4] + '_flow.png')

        # save vis flow
        save_vis_flow_tofile(flow, output_file)

        # also predict backward flow
        if pred_bidir_flow:
            assert flow_pr.size(0) == 2  # [2, H, W, 2]

            if inference_size is None:
                flow_bwd = padder.unpad(flow_pr[1]).permute(1, 2, 0).cpu().numpy()  # [H, W, 2]
            else:
                flow_bwd = flow_pr[1].permute(1, 2, 0).cpu().numpy()  # [H, W, 2]

            output_file = os.path.join(output_path, os.path.basename(filenames[test_id])[:-4] + '_flow_bwd.png')

            # save vis flow
            save_vis_flow_tofile(flow_bwd, output_file)

            # forward-backward consistency check
            # occlusion is 1
            if fwd_bwd_consistency_check:
                if inference_size is None:
                    fwd_flow = padder.unpad(flow_pr[0]).unsqueeze(0)  # [1, 2, H, W]
                    bwd_flow = padder.unpad(flow_pr[1]).unsqueeze(0)  # [1, 2, H, W]
                else:
                    fwd_flow = flow_pr[0].unsqueeze(0)
                    bwd_flow = flow_pr[1].unsqueeze(0)

                fwd_occ, bwd_occ = forward_backward_consistency_check(fwd_flow, bwd_flow)  # [1, H, W] float

                fwd_occ_file = os.path.join(output_path, os.path.basename(filenames[test_id])[:-4] + '_occ.png')
                bwd_occ_file = os.path.join(output_path, os.path.basename(filenames[test_id])[:-4] + '_occ_bwd.png')

                Image.fromarray((fwd_occ[0].cpu().numpy() * 255.).astype(np.uint8)).save(fwd_occ_file)
                Image.fromarray((bwd_occ[0].cpu().numpy() * 255.).astype(np.uint8)).save(bwd_occ_file)

        if save_flo_flow:
            output_file = os.path.join(output_path, os.path.basename(filenames[test_id])[:-4] + '_pred.flo')
            frame_utils.writeFlow(output_file, flow)
