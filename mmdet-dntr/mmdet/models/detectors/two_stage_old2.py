# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch
import torch.nn as nn
from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .base import BaseDetector

########################## add for CL part ##########################
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.E = nn.Sequential(
            # nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            # nn.BatchNorm2d(256),
            # nn.LeakyReLU(0.1, True),
            # nn.Conv2d(256, 256, kernel_size=3, padding=1),
            # nn.BatchNorm2d(256),
            # nn.LeakyReLU(0.1, True),
            # nn.AdaptiveAvgPool2d(1),
            ##################################################################
            nn.Conv2d(256, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.mlp = nn.Sequential(
            nn.Linear(256, 256),
            nn.LeakyReLU(0.1, True),
            nn.Linear(256, 256),
        )

    def forward(self, x):
        fea = self.E(x).squeeze(-1).squeeze(-1)
        out = self.mlp(fea)

        return out

class Cosine_similarity(nn.Module):
    def __init__(self,temp=0.1):
        super(Cosine_similarity, self).__init__()
        self.temp = temp

    def forward(self, v1, v2):
        inner = torch.sum(v1[:]*v2[:])
        sim = torch.div(inner, self.temp)
        out = torch.exp(sim)
        return out

class Contrastive_Loss(nn.Module):
    def __init__(self):
        super(Contrastive_Loss, self).__init__()
        device = torch.device("cuda")
        self.cos = Cosine_similarity().to(device)
        self.loc_E = Encoder().to(device)
        self.sem_E = Encoder().to(device)
        base_c = 256
        self.channel_transfer = []
        for cnt in range (4):
            self.channel_transfer.append(nn.Conv2d(base_c*pow(2,cnt), base_c, kernel_size=1).to(device))
        
            

    def forward(self, x_b, x):
        '''     
        X_B SHAPE ####################
            torch.Size([b, 3, 800, 800]) ==> No need
            torch.Size([b, 256, 200, 200]) 
            torch.Size([b, 512, 100, 100])
            torch.Size([b, 1024, 50, 50])
            torch.Size([b, 2048, 25, 25])
        X SHAPE ####################
            torch.Size([b, 256, 200, 200])
            torch.Size([b, 256, 100, 100])
            torch.Size([b, 256, 50, 50])
            torch.Size([b, 256, 25, 25])
            torch.Size([b, 256, 13, 13]) ==> No need
        '''
        loc_loss = sem_loss = 0
        localization_b = localization = semantic = []
        
        self.x_b = x_b[1:]
        self.x = x[:4]
        scale_size = len(self.x)
        batch_size = len(self.x[0])

        # generate representation (len=256)
        temp_b = []
        for i in range(len(self.x_b)):
            temp_b.append(self.channel_transfer[i](self.x_b[i]))
        for i in range(len(self.x)):
            localization_b.append(self.loc_E(temp_b[i]))
            localization.append(self.loc_E(self.x[i]))
            semantic.append(self.sem_E(self.x[i]))

        # generate localization contrastive loss
        for i in range(scale_size):
            for j in range(batch_size):
                numerator, denominator = 0., 0.
                for k in range(scale_size):
                    for l in range(batch_size):
                        if k==i: # positive
                            numerator = numerator + self.cos(localization[i][j], localization_b[k][l])
                            if l==j:continue
                            else:numerator = numerator + self.cos(localization[i][j], localization[k][l])
                        else:  #negative
                            denominator = denominator + self.cos(localization[i][j], localization[k][l])
                            denominator = denominator + self.cos(localization[i][j], localization_b[k][l])
                denominator = denominator + numerator
                loc_loss = loc_loss - torch.log(torch.div(torch.div(numerator, denominator),float(batch_size-1.)))

        for i in range(scale_size):
            for j in range(batch_size):
                numerator, denominator = 0., 0.
                for k in range(scale_size):
                    for l in range(batch_size):
                        if k==i: # positive
                            numerator = numerator + self.cos(localization_b[i][j], localization[k][l])
                            if l==j:continue
                            else:numerator = numerator + self.cos(localization_b[i][j], localization_b[k][l])
                        else:  #negative
                            denominator = denominator + self.cos(localization_b[i][j], localization_b[k][l])
                            denominator = denominator + self.cos(localization_b[i][j], localization[k][l])
                denominator = denominator + numerator
                loc_loss = loc_loss - torch.log(torch.div(torch.div(numerator, denominator),float(batch_size-1.)))
                
        # generate semantic contrastive loss
        for i in range(scale_size):
            for j in range(batch_size):
                numerator, denominator = 0., 0.
                for k in range(scale_size):
                    for l in range(batch_size):
                        if (l==j)and(k!=i) :
                            numerator = numerator + self.cos(semantic[i][j], semantic[k][l])
                        else:
                            denominator = denominator + self.cos(semantic[i][j], semantic[k][l])
                denominator = denominator + numerator
                sem_loss = sem_loss - torch.log(torch.div(torch.div(numerator, denominator),3.))
        return loc_loss, sem_loss
##############################################################################

@DETECTORS.register_module()
class TwoStageDetector(BaseDetector):
    """Base class for two-stage detectors.

    Two-stage detectors typically consisting of a region proposal network and a
    task-specific regression head.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(TwoStageDetector, self).__init__(init_cfg)
        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            backbone.pretrained = pretrained
        self.backbone = build_backbone(backbone)

        if neck is not None:
            self.neck = build_neck(neck)

        if rpn_head is not None:
            rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
            rpn_head_ = rpn_head.copy()
            rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
            self.rpn_head = build_head(rpn_head_)

        if roi_head is not None:
            # update train and test cfg here for now
            # TODO: refactor assigner & sampler
            rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
            roi_head.update(train_cfg=rcnn_train_cfg)
            roi_head.update(test_cfg=test_cfg.rcnn)
            roi_head.pretrained = pretrained
            self.roi_head = build_head(roi_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        ########################## add for CL part ##########################
        device = torch.device("cuda")
        self.contrastive_loss = Contrastive_Loss().to(device)
        #####################################################################

    @property
    def with_rpn(self):
        """bool: whether the detector has RPN"""
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    @property
    def with_roi_head(self):
        """bool: whether the detector has a RoI head"""
        return hasattr(self, 'roi_head') and self.roi_head is not None


    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        name = "test"
        # 可视化resnet产生的特征
        # from tools.feature_visualization import draw_feature_map
        # draw_feature_map(x, name=name)
        if self.with_neck:
            x = self.neck(x)
            # 可视化FPN产生的特征
            # from tools.feature_visualization import draw_feature_map
            # draw_feature_map(x, name=name)
        return x
        # python demo/image.demo.py img_dir config_dir checkpoint_dir --out-file out_image_path_and_name



    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        outs = ()
        # backbone
        x = self.extract_feat(img)
        # rpn
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            outs = outs + (rpn_outs, )
        proposals = torch.randn(1000, 4).to(img.device)
        # roi_head
        roi_outs = self.roi_head.forward_dummy(x, proposals)
        outs = outs + (roi_outs, )
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        x = self.extract_feat(img)

        losses = dict()
        
        ########################## add for CL part ###################################
        device = torch.device("cuda")
        x_b = self.backbone(img)
        x = self.extract_feat(img)
        loc_loss, sem_loss = self.contrastive_loss(x_b, x)
        # print(sem_loss, loc_loss)
        losses["loc_cl_loss"] = (loc_loss*0.05).to(device)
        losses["sem_cl_loss"] = (sem_loss*0.01).to(device)
        # print(losses["loc_cl_loss"], losses["sem_cl_loss"])
        ##############################################################################
        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg,
                **kwargs)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)
        losses.update(roi_losses)

        return losses

    async def async_simple_test(self,
                                img,
                                img_meta,
                                proposals=None,
                                rescale=False):
        """Async test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img)

        if proposals is None:
            proposal_list = await self.rpn_head.async_simple_test_rpn(
                x, img_meta)
        else:
            proposal_list = proposals

        return await self.roi_head.async_simple_test(
            x, proposal_list, img_meta, rescale=rescale)

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""

        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img)
        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals

        return self.roi_head.simple_test(
            x, proposal_list, img_metas, rescale=rescale)

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        x = self.extract_feats(imgs)
        proposal_list = self.rpn_head.aug_test_rpn(x, img_metas)
        return self.roi_head.aug_test(
            x, proposal_list, img_metas, rescale=rescale)

    def onnx_export(self, img, img_metas):

        img_shape = torch._shape_as_tensor(img)[2:]
        img_metas[0]['img_shape_for_onnx'] = img_shape
        x = self.extract_feat(img)
        proposals = self.rpn_head.onnx_export(x, img_metas)
        if hasattr(self.roi_head, 'onnx_export'):
            return self.roi_head.onnx_export(x, proposals, img_metas)
        else:
            raise NotImplementedError(
                f'{self.__class__.__name__} can not '
                f'be exported to ONNX. Please refer to the '
                f'list of supported models,'
                f'https://mmdetection.readthedocs.io/en/latest/tutorials/pytorch2onnx.html#list-of-supported-models-exportable-to-onnx'  # noqa E501
            )
