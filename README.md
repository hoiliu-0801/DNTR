# A DeNoising FPN with Transformer R-CNN for Tiny Object Detection

![method](./figures/dnfpn.png)


A PyTorch implementation and pretrained models for DNTR (DeNoising Transformer R-CNN). We present DN-FPN, a plug-in that suppresses noise generated during the fusion of FPNs. In addition, we renalvate the standard R-CNN to consist of a transformer structure, namely Trans R-CNN.(base)
## Installation and Get Started

Required environments:
* Linux
* Python 3.6+
* PyTorch 1.3+
* CUDA 9.2+
* GCC 5+
* [MMCV](https://mmcv.readthedocs.io/en/latest/#installation)
* [cocoapi-aitod](https://github.com/jwwangchn/cocoapi-aitod)


Installation:

This implementation is based on [MMDetection 2.24.1](https://github.com/open-mmlab/mmdetection). Assume that your environment has satisfied the above requirements, please follow the following steps for installation.

```shell script
git clone https://github.com/hoiliu-0801/DNTR.git
cd dntr
pip install -r requirements/build.txt
python setup.py develop
```

Get Started with single GPU

Training DNTR, for example :

```
python tools/train.py configs/dntr/aitod_dntr_r50_1x.py
```

Testing DNTR, for example :
```
python tools/test.py configs/dntr/aitod_dntr_r50_1x.py
```

## Performance
Table 1. **Training Set:** AI-TOD trainval set, **Validation Set:** AI-TOD test set, 36 epochs, where FRCN, DR denotes Faster R-CNN and DetectoRS, respectively.
|Method | Backbone | mAP | AP<sub>50</sub> | AP<sub>75</sub> |AP<sub>vt</sub> | AP<sub>t</sub>  | AP<sub>s</sub>  | AP<sub>m</sub> | AP<sub>m</sub> |
|:---:|:---:|:---:|:---:|:---:|:---:|:---: |:---: |:---: |:---: |
FRCN | R-50 | 11.1 | 26.3 | 7.6 | 0.0 | 7.2 | 23.3 | 33.6 | 22.9 |
ATSS | R-50 | 12.8 | 30.6 | 8.5 | 1.9 | 11.6 | 19.5 | 29.2 | 25.8 |
ATSS w/ DN-FPN | R-50 | 17.9 | 41.0 | 12.9 | 3.7 | 16.4 | 25.3 | 35.0 | 33.2 |
NWD-RKA | R-50 | 23.4 | 53.5 | 16.8 | 8.7 | 23.8 | 28.5 | 36.0 | 6.9 |
DNTR | R-50 | **26.2** | **56.7** | **20.2** | **12.8** | **26.4** | **31.0** | **37.0** | **7.6** |

Table 2.  **Training Set:** Visdrone train set, **Validation Set:** Visdrone val set, 12 epochs,
|Method | Backbone |AP| AP<sub>50</sub> | AP<sub>75</sub> |
|:---:|:---:|:---:|:---:|:---:|
DNTR | R-50 | 34.4 | 57.9 | 35.3 |
UFPMP w/o DN-FPN| R-50 | 36.6 | 62.4 | 36.7 |
UFPMP w/ DN-FPN | R-50 | **37.8** | **62.7** | **38.6** |

## Citation
@ARTICLE{10518058,
  author={Liu, Hou-I and Tseng, Yu-Wen and Chang, Kai-Cheng and Wang, Pin-Jyun and Shuai, Hong-Han and Cheng, Wen-Huang},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={A DeNoising FPN With Transformer R-CNN for Tiny Object Detection}, 
  year={2024},
  volume={62},
  number={},
  pages={1-15},
}
