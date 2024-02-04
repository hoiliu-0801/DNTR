# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch
import torch.nn as nn

from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .base import BaseDetector
torch.autograd.set_detect_anomaly(True)
########################## add for CL part ##########################

class Level_classifier(nn.Module):
    def __init__(self):
        super(Level_classifier, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(256, 32),
            nn.LeakyReLU(0.1, True),
            nn.Linear(32, 5),
            nn.Softmax()
        )
    
    def forward(self, fea):
        return self.mlp(fea)
        
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.E = nn.Sequential(
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
class Spatial_network(nn.Module):
    def __init__(self):
        super(Spatial_network, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(256, 1024),
            nn.LeakyReLU(0.1, True),
            nn.Linear(1024, 40000),
        )
    def forward(self, x, localization):
        w = self.net(localization)
        w = w.view(x.shape[0], 200, 200)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                x.data[i][j] = x.data[i][j] + x.data[i][j] * w.data[i]
        return x

class Channel_network(nn.Module):
    def __init__(self):
        super(Channel_network, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(256, 256),
            nn.LeakyReLU(0.1, True),
            nn.Linear(256, 256),
            nn.Softmax(dim=1),
        )
    def forward(self, x, semantic):
        w = self.net(semantic)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                x.data[i][j] = x.data[i][j] + x.data[i][j] * w[i][j]
        return x

class Reconstruct_Network(nn.Module):
    def __init__(self):
        super(Reconstruct_Network, self).__init__()
        self.spatial_net = Spatial_network()
        self.channel_net = Channel_network()
    def forward(self, x, localization, semantic):
        loc_x = self.spatial_net(x, localization)
        sem_x = self.channel_net(x, semantic)
        return loc_x+sem_x

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
        self.level_classifier = Level_classifier().to(device)
        self.level_loss = nn.CrossEntropyLoss()
        #### channel transfer for backbone features
        base_c = 256
        self.channel_transfer = []
        for cnt in range (4):
            self.channel_transfer.append(nn.Conv2d(base_c*pow(2,cnt), base_c, kernel_size=1).to(device))

    def localization_forward(self, scale_size, batch_size, localization_b, localization):
        loc_loss = 0
        for i in range(scale_size):
            for j in range(batch_size):
                tmp_loss = 0
                numerator = 0
                denominator = 0
                for l in range(scale_size):
                    for m in range(batch_size):
                        if l!=i:
                            denominator = denominator + self.cos(localization[i][j], localization[l][m])
                            denominator = denominator + self.cos(localization[i][j], localization_b[l][m])
                for m in range(batch_size):
                    if m==j:continue
                    else:numerator = numerator + self.cos(localization[i][j], localization[i][m])
                    tmp_loss = tmp_loss - torch.log(torch.div(numerator, denominator + numerator))
                for m in range(batch_size):
                    numerator = self.cos(localization[i][j], localization_b[i][m])
                    tmp_loss = tmp_loss - torch.log(torch.div(numerator, denominator + numerator))
                loc_loss = loc_loss + torch.div(tmp_loss,float(2*batch_size-1.))

        for i in range(scale_size):
            for j in range(batch_size):
                tmp_loss = 0
                numerator = 0
                denominator = 0
                for l in range(scale_size):
                    for m in range(batch_size):
                        if l!=i:
                            denominator = denominator + self.cos(localization_b[i][j], localization_b[l][m])
                            denominator = denominator + self.cos(localization_b[i][j], localization[l][m])
                for m in range(batch_size):
                    if m==j:continue
                    else:numerator = numerator + self.cos(localization_b[i][j], localization_b[i][m])
                    tmp_loss = tmp_loss - torch.log(torch.div(numerator, denominator + numerator))
                for m in range(batch_size):
                    numerator = self.cos(localization_b[i][j], localization[i][m])
                    tmp_loss = tmp_loss - torch.log(torch.div(numerator, denominator + numerator))
                loc_loss = loc_loss + torch.div(tmp_loss,float(2*batch_size-1.))
        return loc_loss

    def semantic_forward(self, scale_size, batch_size, semantic):
        sem_loss = 0
        for i in range(scale_size):
            for j in range(batch_size):
                tmp_loss = 0
                numerator = 0
                denominator = 0
                for l in range(scale_size):
                    for m in range(batch_size):
                        if m!=j:
                            denominator = denominator + self.cos(semantic[i][j], semantic[l][m])
                for l in range(scale_size):
                    if l==i:continue
                    else:numerator = self.cos(semantic[i][j], semantic[l][j])
                    tmp_loss = tmp_loss - torch.log(torch.div(numerator, denominator + numerator))
                sem_loss = sem_loss + torch.div(tmp_loss,float(scale_size-1.))
        return sem_loss

    def level_forward(self, scale_size, batch_size, localization):
        level_loss = 0
        device = torch.device("cuda")
        for i in range(scale_size):
            for j in range(batch_size):
                level_pred = (self.level_classifier(localization[i][j]))
                level_target = torch.tensor(i).to(device)
                tmp_loss = self.level_loss(level_pred, level_target)
                level_loss = level_loss + tmp_loss

        return level_loss

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
        localization_b = []
        localization = []
        semantic = []

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

        loc_loss = self.localization_forward(scale_size, batch_size, localization_b, localization)
        sem_loss = self.semantic_forward(scale_size, batch_size, semantic)
        # level_loss = self.level_forward(scale_size, batch_size, localization)
        level_loss = 0
        

        return loc_loss, sem_loss, level_loss
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
        # self.reconstruct = Reconstruct_Network().to(device)
        #####################################################################

    @property
    def with_rpn(self):
        """bool: whether the detector has RPN"""
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    @property
    def with_roi_head(self):
        """bool: whether the detector has a RoI head"""
        return hasattr(self, 'roi_head') and self.roi_head is not None

    # def extract_feat(self, img):
    #     """Directly extract features from the backbone+neck."""
    #     x = self.backbone(img)
    #     if self.with_neck:
    #         x = self.neck(x)
    #     return x

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        # name = "test"
        # 可视化resnet产生的特征
        # from tools.feature_visualization import draw_feature_map
        # draw_feature_map(x, name=name)
        if self.with_neck:
            x = self.neck(x)
            # 可视化FPN产生的特征
            # from tools.feature_visualization import draw_feature_map
            # draw_feature_map(x, name=name)
        # exit()
        return x

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
        loc_loss, sem_loss , level_loss= self.contrastive_loss(x_b, x)

        # print(loc_loss, sem_loss)
        losses["loc_cl_loss"] = (loc_loss).to(device)
        losses["sem_cl_loss"] = (sem_loss).to(device)
        losses["level_loss"] = (level_loss).to(device)
        print(losses["loc_cl_loss"], losses["sem_cl_loss"], losses["level_loss"])
        ########################## add for Reconstruct part ##########################
        # x[0] = self.reconstruct(x[0], localization[0], semantic[0])
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

        # print("losses : ",losses)
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
