# Copyright (c) OpenMMLab. All rights reserved.
from threading import local
import warnings
from mmcv import Config
import torch
import torch.nn as nn
from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .base import BaseDetector
import matplotlib.pyplot as plt
import cv2
import os

# torch.autograd.set_detect_anomaly(True)
########################## add for CL part ##########################
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.E = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            # nn.LeakyReLU(0.1, True),
            # nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            # nn.BatchNorm2d(512),
            # nn.LeakyReLU(0.1, True),
            # nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),
            # nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.mlp = nn.Sequential(
            # nn.Linear(1024, 512),
            # nn.LeakyReLU(0.1, True),
            # nn.Linear(512, 256),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.1, True),
            nn.Linear(256, 256),
        )
    def forward(self, x):
        # print("x.shape:",x.shape)  # torch.Size([2, 256, 200, 200])
        fea = self.E(x).squeeze(-1).squeeze(-1)   # torch.Size([2, 256])
        # print("self.E(x).shape:",self.E(x).shape)
        out = self.mlp(fea)  # torch.Size([2, 256])
        # print("out.shape:",out.shape) # torch.Size([2, 256])
        return out

class Cosine_similarity(nn.Module):
    def __init__(self,temp=0.5):
        super(Cosine_similarity, self).__init__()
        self.temp = temp
    def forward(self, v1, v2):
        n1=nn.functional.normalize(v1,dim=0)
        n2=nn.functional.normalize(v2,dim=0)
        inner = torch.dot(n1, n2)
        # print("inner:",inner)
        sim = torch.div(inner, self.temp)
        # print("sim:",sim)
        out = torch.exp(sim)
        # print("out:",out)
        return out

class Contrastive_Loss(nn.Module):
    def __init__(self):
        super(Contrastive_Loss, self).__init__()
        device = torch.device("cuda")
        self.cos = Cosine_similarity().to(device)
        self.loc_E = Encoder().to(device)
        self.sem_E = Encoder().to(device)
        #### channel transfer for backbone features
        base_c = 256
        self.channel_transfer = []
        for cnt in range (4):
            self.channel_transfer.append(nn.Conv2d(base_c*pow(2,cnt), base_c, kernel_size=1).to(device))
     # @torch.no_grad()
    # def init_weights(m,n):
    #     if type(m) == nn.Linear:
    #         m.weight.fill_(n)
    #         print(m)
    #         # nn.init.uniform_(tensor, a=0.0, b=1.0)
    #     elif type(m)==nn.Conv2d:
    #         m.weight.fill_(n)
    #         print(m)
# class filter_nan():
#     def __init__(self,x):
#         super(filter_nan, self).__init__()
#         device = torch.device("cuda")
#         print(x)
#         print(~torch.any(x.isnan()))
#         if ~torch.any(x.isnan()):
#             filtered_tensor = x[~torch.any(x.isnan(),dim=0)]
#             print(filtered_tensor)
#         return filtered_tensor.to(device)

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
        # geo_loss = sem_loss = 0.
        geo_loss = 0.
        sem_loss = 0.
        localization_b = []
        localization = []
        semantic_b =[]
        semantic = []

        self.x_b = x_b[1:]
        self.x = x[:4]  # backbone feature
        scale_size = len(self.x)  # 4
        batch_size = len(self.x[0])

        # generate representation (len=256)
        temp_b = []  # temp_b= final backbone feat
        for i in range(len(self.x_b)):
            temp_b.append(self.channel_transfer[i](self.x_b[i]))  # backbone to top-down
        for i in range(len(self.x)):
            localization_b.append(self.loc_E(temp_b[i]))
            localization.append(self.loc_E(self.x[i]))
            semantic_b.append(self.sem_E(temp_b[i]))
            semantic.append(self.sem_E(self.x[i]))

        # generate contrastive loss
        # geo_pos= 0
        # geo_neg= 0
        for i in range(scale_size):  # top-down x
            for j in range(batch_size):
                numerator = 0.
                denominator = 0.
                # per_geo_pos= 0
                # per_geo_neg= 0
                for k in range(scale_size): # backbone x
                    for l in range(batch_size):
                        if k==i:  # same scale
                            if l==j:  # same batch
                                # geo_pos=geo_pos+1
                                # per_geo_pos=per_geo_pos+1
                                q=localization[i][j]  # ([256])
                                k_pos=localization_b[k][l]
                                numerator = numerator + self.cos(q, k_pos) # same scale & same batch
                            else:continue  # same scale & diff batch
                        else:  # diff scale
                            if l==j:continue
                            else:
                                # geo_neg=geo_neg+1
                                # per_geo_neg=per_geo_neg+1
                                q=localization[i][j]
                                k_neg1=localization[k][l]
                                k_neg2=localization_b[k][l]
                                denominator = denominator + self.cos(q, k_neg1)  # top-down - top-down
                                denominator = denominator + self.cos(q, k_neg2) # top-down - backbone
                # print("per_geo_pos:",per_geo_pos,"per_geo_neg:",per_geo_neg) # 1, 6
                denominator = denominator + numerator
                geo_loss = geo_loss - torch.log(torch.div(numerator, denominator))
        # print("geo_pos:",geo_pos,"geo_neg:",geo_neg) # 8, 48
        # sem_pos= 0
        # sem_neg= 0
        for i in range(scale_size-1):  #  x 的 scale , i != 3
            for j in range(batch_size): #  x 的 batch
                numerator = 0.
                denominator = 0.
                # per_sem_pos = 0
                # per_sem_neg = 0
                for k in range(scale_size):  # x_b 的 scale  k=0,1,2,3
                    for l in range(batch_size):   #  x_b 的 batch
                        if l==j:     # same batch
                            if k==i+1:  # pos: at least 3
                                q = semantic[i][j]
                                k_pos = semantic[k][l]
                                numerator = numerator + self.cos(q, k_pos)
                                # sem_pos = sem_pos+1
                                # per_sem_pos = per_sem_pos+1
                            else:   # same scale & diff scale
                                continue
                        else:       # diff scale

                            q = semantic[i][j]
                            k_neg1 = semantic[k][l]
                            k_neg2 = semantic_b[k][l]
                            denominator = denominator + self.cos(q, k_neg1)
                            denominator = denominator + self.cos(q, k_neg2)

                            # sem_neg = sem_neg+1
                            # per_sem_neg = per_sem_neg+1
                denominator = denominator + numerator
                sem_loss = sem_loss - torch.log(torch.div(numerator, denominator))
                # print("per_sem_pos:",per_sem_pos,"per_sem_neg:",per_sem_neg) # 1,4

        # print("sem_pos:",sem_pos,"sem_neg:",sem_neg) #6,24
        # print("geo_loss, sem_loss:",geo_loss, sem_loss)
        return geo_loss, sem_loss
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

    def extract_feat(self, img, filename):
    # def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)

        # 可视化resnet产生的特征
        # from tools.feature_visualization import draw_feature_map
        # draw_feature_map(x)

        # file_=os.listdir("/mnt/data0/Garmin/DNTR/mmdet-dntr/tools/aitod_psnr/test/test_coco_base")
        # print(file_)
        # exit()
        if self.with_neck:
            x = self.neck(x)
            # for id, feat in enumerate(x):
            #     # print(feat.shape) # torch.Size([2, 256, 144, 256])
            #     # heatmap = feat[:,0,:,:]*0
            #     # 可视化FPN产生的特征
            #     if id == 0:
            #         feat = feat.squeeze(0).cpu().detach().numpy()[0]
            #         feat = cv2.resize(feat, (600, 600))
            #         save_path= "/mnt/data0/Garmin/DNTR/mmdet-dntr/tools/vis/"
            #         plt.imsave(save_path+filename, feat) #
            # from tools.feature_visualization import draw_feature_map
            # draw_feature_map(x, filename)
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
        filename = os.path.basename(img_metas[0]['ori_filename'])
        x = self.extract_feat(img, filename)

        losses = dict()

        ########################## add for CL part ###################################
        device = torch.device("cuda")
        x_b = self.backbone(img)
        x = self.extract_feat(img, filename)
        geo_loss, sem_loss = self.contrastive_loss(x_b, x)

        # # print(geo_loss, sem_loss)
        losses["geo_loss"] = (0.01*geo_loss).to(device)
        losses["sem_loss"] = (0.01*sem_loss).to(device)
        # print(losses["loc_cl_loss"], losses["sem_cl_loss"])
        # print(losses)
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
        filename =  os.path.basename(img_metas[0]['filename'])
        # filename = img_metas[0]['ori_filename']
        assert self.with_bbox, 'Bbox head must be implemented.'
        # x = self.extract_feat(img)
        ## If Visualization ##
        x = self.extract_feat(img, filename)
        #######################
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

