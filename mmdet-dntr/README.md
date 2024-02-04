## Installation and Get Started

Required environments:
* Linux
* Python 3.6+
* PyTorch 1.3+
* CUDA 9.2+
* GCC 5+
* [MMCV](https://mmcv.readthedocs.io/en/latest/#installation)
* [cocoapi-aitod](https://github.com/jwwangchn/cocoapi-aitod)


Install TODbox:

Note that our TODbox is based on the [MMDetection 2.24.1](https://github.com/open-mmlab/mmdetection). Assume that your environment has satisfied the above requirements, please follow the following steps for installation.

```shell script
git clone https://github.com/Chasel-Tsui/mmdet-aitod.git
cd mmdet-nwdrka
pip install -r requirements/build.txt
python setup.py develop
```

Get Started

Train a network with with single GPU, for example, Faster R-CNN w/ NWD-RKA:

```
python tools/train.py configs_nwdrka/nwdrka/aitod_faster_r50_nwdrka_1x.py
```

## Performance
Table 1. **Training Set:** AI-TOD-v2 trainval set, **Validation Set:** AI-TOD-v2 test set, 12 epochs
Method | Backbone | mAP | AP<sub>50</sub> | AP<sub>75</sub> |AP<sub>vt</sub> | AP<sub>t</sub>  | AP<sub>s</sub>  | AP<sub>m</sub>  
--- |:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:
FR | R-50 | 12.8 | 29.9 | 9.4 | 0.0 | 9.2| 24.6 | 37.0 
DR | R-50 | 16.1 | 35.5 | 12.5 | 0.1 | 12.6 | 28.3 | **40.0**
FR w/ NWD-RKA | R-50 | 21.4 | 53.2 | 12.5 | 7.7 | 20.7 | 26.8 | 35.2 
DR w/ NWD-RKA | R-50 | **24.7** | **57.4** | **17.1** | **9.7** | **24.2** | **29.3** | 39.3

FR denotes Faster R-CNN, DR denotes DetectoRS

For your convenience, we also provide the performance of the model trained on **AI-TOD-v2 train set** and validated on the **AI-TOD-v2 val set**. 
Table 2. **Training Set:** AI-TOD-v2 train set, **Validation Set:** AI-TOD-v2 val set, 12 epochs
Method | Backbone | mAP | AP<sub>50</sub> | AP<sub>75</sub> |AP<sub>vt</sub> | AP<sub>t</sub>  | AP<sub>s</sub>  | AP<sub>m</sub>  
--- |:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:
FR | R-50 | 12.9 | 29.5 | 9.2 | 0.0 | 9.5 | 27.3 | 37.2 
FR w/ NWD-RKA | R-50 | **21.9** | **51.8** | **13.9** | **5.8** | **21.8** | **27.3** | **37.8** 



Table 3.  **Training Set:** DOTA-v2 train set, **Validation Set:** DOTA-v2 val set, 12 epochs, HBB Task
Method | Backbone | mAP | AP<sub>50</sub> | AP<sub>75</sub> |AP<sub>vt</sub> | AP<sub>t</sub>  | AP<sub>s</sub>  | AP<sub>m</sub>  
--- |:---:|:---:|:---:|:---:|:---:|:---:|:---: |:---:
FR | R-50 | 35.6 | 59.5 | 37.2 | 0.0 | 7.1 | 28.9 | 42.1 
DR | R-50 | 40.8 | 62.6 | 44.4 | 0.0 | 7.0 | 29.9 | 47.8 
FR w/ NWD-RKA | R-50 | 36.4 | 61.5 | 37.6 | 1.5 | 10.4 | 29.4 | 43.2 
DR w/ NWD-RKA | R-50 | **41.9** | **66.3** | **44.4** | **1.9** | **10.6** | **30.3** | **48.5** 



Please refer to the paper for detailed performance on the AI-TOD, AI-TOD-v2, DOTA-v2 and VisDrone2019.
