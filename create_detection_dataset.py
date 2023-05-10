import numpy as np
import torch
from torchvision.ops.boxes import box_iou
from scipy.optimize import linear_sum_assignment
import os
import json


class DetectionDataset():

    def __init__(self, args):
        self.args = args

        label_file = f'CLEVR_{args.split}_scenes.json'

        with open(os.path.join(args.clevr_path, 'scenes', label_file), 'r') as f:
            scenes = json.load(f)

        self.image_idx = []
        pos_data = []
        size_data = []
        depths = []
        ns = []

        if args.task == 'CLEVR3':
            for data in scenes['scenes']:
                if len(data['objects']) == 3:
                    self.image_idx.append(data['image_index'])
                    pos = [d['pixel_coords'][0:2] for d in data['objects']]
                    sizes = [d['3d_coords'][2] for d in data['objects']] # z coordinate is needed for approximate size
                    depth = [d['pixel_coords'][2] for d in data['objects']] #z coordinate is approximately size

                    ns.append(len(pos))
                    pos_data.append(pos)
                    size_data.append(sizes)
                    depths.append(depth)
        elif args.task == 'CLEVR6':
            for data in scenes['scenes']:
                if len(data['objects']) <= 6:
                    self.image_idx.append(data['image_index'])
                    pos = [d['pixel_coords'][0:2] for d in data['objects']]
                    sizes = [d['3d_coords'][2] for d in data['objects']] #z coordinate is approximately size
                    depth = [d['pixel_coords'][2] for d in data['objects']] #z coordinate is approximately size

                    ns.append(len(pos))
                    pos_data.append(np.pad(np.array(pos), ((0, 6 - len(pos)),(0,0)), mode='constant', constant_values=0.0))
                    size_data.append(np.pad(np.array(sizes), (0, 6 - len(sizes)), mode='constant', constant_values=0.15))
                    depths.append(np.pad(np.array(depth), (0, 6 - len(depth)), mode='constant', constant_values=1.0))
        else:
            raise ValueError('Model name not recognized')

        pos_data = np.array(pos_data, 'float')
        size_data = np.array(size_data, 'float')


        #extent: 1 x 320/480
        # NOTE: inverting the y coordinate in pixels, because is from top left down
        sign = np.array([[[1.0, -1.0]]])
        offset = np.array([[[-480., 320.]]]) / 2
        scale = np.array([[[480., 480.]]])
        pos_data = (offset + sign * pos_data) / scale
        sizes = np.array(size_data)
        depths = np.array(depths)
        self.ns = np.array(ns)

        # Approximate size:
        sizes = sizes * 1.0 / np.sqrt(depths + 1e-5)

        pos_data = pos_data.astype('float32')
        sizes = sizes.astype('float32')

        self.img_path = os.path.join(args.clevr_path,'images', args.split, f'CLEVR_{args.split}')

        self.boxes = np.concatenate([pos_data, np.expand_dims(sizes, 2)], 2)

    def __getitem__(self, idx):
        img_path = f'{self.img_path}_{self.image_idx[idx]:06d}.png'
        box = self.boxes[idx, 0:self.ns[idx]]
        return img_path, box


def all_iou(predictions, gt):
    """Calculate the IOUs between two sets of boxes
    Args:
        predictions: (N,3) shaped tensor
        gt: (M,3) shaped tensor
    returns:
        tensor (N,M) of ious
    """
    prediction_boxes = torch.stack(
        [
            predictions[:, 0] - predictions[:, 2] * 0.5,
            predictions[:, 1] - predictions[:, 2] * 0.5,
            predictions[:, 0] + predictions[:, 2] * 0.5,
            predictions[:, 1] + predictions[:, 2] * 0.5,
        ],
        dim=-1,
    )
    gt_boxes = torch.stack(
        [
            gt[:, 0] - gt[:, 2] * 0.5,
            gt[:, 1] - gt[:, 2] * 0.5,
            gt[:, 0] + gt[:, 2] * 0.5,
            gt[:, 1] + gt[:, 2] * 0.5,
        ],
        dim=-1,
    )

    return box_iou(prediction_boxes, gt_boxes)

def match_iou(iou):
    """
    input: ious, Tensor([N,M])
    output:
        Tensor([N])

    Note: in order to penalize over/under prediction, tensors will be padded to square (max(N,M))

    """

    iou = iou.cpu().numpy()
    nm_max = max(*iou.shape)

    rows_to_pad = nm_max - iou.shape[0]
    cols_to_pad = nm_max - iou.shape[1]

    iou = np.pad(iou, ((0, rows_to_pad), (0, cols_to_pad)))

    rows, cols = linear_sum_assignment(iou, maximize=True)

    matched_ious = iou[rows, cols]

    return torch.tensor(matched_ious).float()

def iou(predictions, gt):
    """Calculate the matched IOU between two sets of boxes
    Args:
        predictions: (N,3) shaped tensor
        gt: (M,3) shaped tensor
    returns:
        average of the matched iou's
    """
    ious = all_iou(predictions, gt)
    return match_iou(ious).mean()

def add_arguments(parser):
    parser.add_argument('--clevr_path', type=str, required=True)
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--task', type=str, default='CLEVR6')
