import torch
import math
import numpy as np
import torchvision.transforms as transforms
from .libs import TensorDict
import tools.processing_utils as prutils


def stack_tensors(x):
    if isinstance(x, (list, tuple)) and isinstance(x[0], torch.Tensor):
        return torch.stack(x)
    return x


class BaseProcessing:
    """ Base class for Processing. Processing class is used to process the data returned by a dataset, before passing it
     through the network. For example, it can be used to crop a search region around the object, apply various data
     augmentations, etc."""
    def __init__(self, transform=transforms.ToTensor(), train_transform=None, test_transform=None, joint_transform=None):
        """
        args:
            transform       - The set of transformations to be applied on the images. Used only if train_transform or
                                test_transform is None.
            train_transform - The set of transformations to be applied on the train images. If None, the 'transform'
                                argument is used instead.
            test_transform  - The set of transformations to be applied on the test images. If None, the 'transform'
                                argument is used instead.
            joint_transform - The set of transformations to be applied 'jointly' on the train and test images.  For
                                example, it can be used to convert both test and train images to grayscale.
        """
        self.transform = {'train': transform if train_transform is None else train_transform,
                          'test':  transform if test_transform is None else test_transform,
                          'joint': joint_transform}

    def __call__(self, data: TensorDict):
        raise NotImplementedError


class DSProcessing(BaseProcessing):
    """ The processing class used for training ATOM. The images are processed in the following way.
    First, the target bounding box is jittered by adding some noise. Next, a square region (called search region )
    centered at the jittered target center, and of area search_area_factor^2 times the area of the jittered box is
    cropped from the image. The reason for jittering the target box is to avoid learning the bias that the target is
    always at the center of the search region. The search region is then resized to a fixed size given by the
    argument output_sz. A set of proposals are then generated for the test images by jittering the ground truth box.

    """

    def __init__(self, search_area_factor, output_sz, center_jitter_factor, scale_jitter_factor, proposal_params,
                 mode='pair', *args, **kwargs):
        """
        args:
            search_area_factor - The size of the search region  relative to the target size.
            output_sz - An integer, denoting the size to which the search region is resized. The search region is always
                        square.
            center_jitter_factor - A dict containing the amount of jittering to be applied to the target center before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            scale_jitter_factor - A dict containing the amount of jittering to be applied to the target size before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            proposal_params - Arguments for the proposal generation process. See _generate_proposals for details.
            mode - Either 'pair' or 'sequence'. If mode='sequence', then output has an extra dimension for frames
        """
        super().__init__(*args, **kwargs)
        self.search_area_factor = search_area_factor
        self.output_sz = output_sz
        self.center_jitter_factor = center_jitter_factor
        self.scale_jitter_factor = scale_jitter_factor
        self.proposal_params = proposal_params
        self.mode = mode

    def _get_jittered_box(self, box, mode):
        """ Jitter the input box
        args:
            box - input bounding box
            mode - string 'train' or 'test' indicating train or test data

        returns:
            torch.Tensor - jittered box
        """

        jittered_size = box[2:4] * torch.exp(torch.randn(2) * self.scale_jitter_factor[mode])
        max_offset = (jittered_size.prod().sqrt() * torch.tensor(self.center_jitter_factor[mode]).float())
        jittered_center = box[0:2] + 0.5 * box[2:4] + max_offset * (torch.rand(2) - 0.5)

        return torch.cat((jittered_center - 0.5 * jittered_size, jittered_size), dim=0)

    def _generate_proposals(self, box):
        """ Generates proposals by adding noise to the input box
        args:
            box - input box

        returns:
            torch.Tensor - Array of shape (num_proposals, 4) containing proposals
            torch.Tensor - Array of shape (num_proposals,) containing IoU overlap of each proposal with the input box. The
                        IoU is mapped to [-1, 1]
        """
        # Generate proposals
        num_proposals = self.proposal_params['boxes_per_frame']
        proposal_method = self.proposal_params.get('proposal_method', 'default')

        if proposal_method == 'default':
            proposals = torch.zeros((num_proposals, 4))
            gt_iou = torch.zeros(num_proposals)
            for i in range(num_proposals):
                proposals[i, :], gt_iou[i] = prutils.perturb_box(box, min_iou=self.proposal_params['min_iou'],
                                                                 sigma_factor=self.proposal_params['sigma_factor'])
        elif proposal_method == 'gmm':
            proposals, _, _ = prutils.sample_box_gmm(box, self.proposal_params['proposal_sigma'],
                                                                             num_samples=num_proposals)
            gt_iou = prutils.iou(box.view(1,4), proposals.view(-1,4))

        # Map to [-1, 1]
        gt_iou = gt_iou * 2 - 1
        return proposals, gt_iou

    def __call__(self, data: TensorDict):
        """
        args:
            data - The input data, should contain the following fields:
                'train_images', test_images', 'train_anno', 'test_anno'
        returns:
            TensorDict - output data block with following fields:
                'train_images', 'test_images', 'train_anno', 'test_anno', 'test_proposals', 'proposal_iou'
        """
        # Apply joint transforms
        if self.transform['joint'] is not None:
            data['train_images'], data['train_anno'] = self.transform['joint'](image=data['train_images'], bbox=data['train_anno'])
            data['test_images'], data['test_anno'] = self.transform['joint'](image=data['test_images'], bbox=data['test_anno'], new_roll=False)

        for s in ['train', 'test']:
            assert self.mode == 'sequence' or len(data[s + '_images']) == 1, \
                "In pair mode, num train/test frames must be 1"

            # Add a uniform noise to the center pos
            jittered_anno = [self._get_jittered_box(a, s) for a in data[s + '_anno']]

            # Crop image region centered at jittered_anno box
            crops, boxes, _ = prutils.jittered_center_crop(data[s + '_images'], jittered_anno, data[s + '_anno'],
                                                           self.search_area_factor, self.output_sz)

            # Apply transforms
            data[s + '_images'], data[s + '_anno'] = self.transform[s](image=crops, bbox=boxes, joint=False)

        # Generate proposals
        frame2_proposals, gt_iou = zip(*[self._generate_proposals(a) for a in data['test_anno']])

        data['test_proposals'] = list(frame2_proposals)
        data['proposal_iou'] = list(gt_iou)

        # Prepare output
        if self.mode == 'sequence':
            data = data.apply(stack_tensors)
        else:
            data = data.apply(lambda x: x[0] if isinstance(x, list) else x)

        return data
