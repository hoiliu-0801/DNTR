import cv2
import mmcv
import numpy as np
import os
import torch
import matplotlib.pyplot as plt

def featuremap_2_heatmap(feature_map):
    assert isinstance(feature_map, torch.Tensor)
    feature_map = feature_map.detach()
    heatmap = feature_map[:,0,:,:]*0 # 因為第0維值太大了?
    heatmaps = []
    ## 把維度C=256通通累加起來的意思 ##
    for c in range(feature_map.shape[1]):
        heatmap+=feature_map[:,c,:,:]
    # print("0:",heatmap.shape) # torch.Size([1, 25, 34])
    heatmap = heatmap.cpu().numpy()
    heatmap = np.mean(heatmap, axis=0)
    # print("1:",heatmap.shape) #(13, 17)
    heatmap = np.maximum(heatmap, 0)
    # print("2:",heatmap.shape)　#(13, 17)
    # heatmap /= np.max(heatmap)
    heatmaps.append(heatmap)
    return heatmaps

# def draw_feature_map(features, save_dir = '/mnt/data0/Garmin/DNTR/mmdet-dntr/tools/aitod_psnr/test/',name = 'c'):
def draw_feature_map(features, filename, save_dir = '/mnt/data0/Garmin/DNTR/mmdet-dntr/tools/vis/'):
    h=800
    w=800

    if isinstance(features, torch.Tensor):
        for heat_maps in features:
            heat_maps=heat_maps.unsqueeze(0)
            heatmaps = featuremap_2_heatmap(heat_maps)
            # 这里的h,w指的是你想要把特征图resize成多大的尺寸
            heatmaps = cv2.resize(heatmaps, (h, w))
            for heatmap in heatmaps:
                heatmap = np.uint8(255 * heatmap)
                # 下面这行将热力图转换为RGB格式 ，如果注释掉就是灰度图
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                superimposed_img = heatmap
                # plt.imshow(superimposed_img,cmap='gray')
                # plt.show()
                # file_=os.listdir("/mnt/data0/Garmin/DNTR/mmdet-dntr/tools/aitod_psnr/test/test_coco_base")
                # if filename in file_:
                cv2.imwrite(save_dir + filename, superimposed_img)
    else:
        for featuremap in features:
            heatmaps = featuremap_2_heatmap(featuremap)
            for heatmap in heatmaps:
                heatmap = np.uint8(255 * heatmap)  # 将热力图转换为RGB格式
                # heatmap = cv2.resize(heatmap, (1920,1080))  # 将热力图的大小调整为与原始图像相同
                heatmap = cv2.resize(heatmap, (h, w))  # 将热力图的大小调整为与原始图像相同
                # heatmap = cv2.resize(heatmap, (1024, 540))  # 将热力图的大小调整为与原始图像相同
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                # superimposed_img = heatmap * 0.5 + img*0.3
                superimposed_img = heatmap
                # print(superimposed_img)
                # plt.imshow(superimposed_img,cmap='gray')
                # plt.show()
                # 下面这些是对特征图进行保存，使用时取消注释
                # cv2.imshow("1",superimposed_img)

                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                # file_=os.listdir("/mnt/data0/Garmin/DNTR/mmdet-dntr/tools/aitod_psnr/test/test_coco_base")
                # if filename in file_:
                cv2.imwrite(save_dir+"img_" + filename, superimposed_img)

                    # cv2.imwrite(save_dir + filename+str(i) +'.png', superimposed_img)
                    # i=i+1
