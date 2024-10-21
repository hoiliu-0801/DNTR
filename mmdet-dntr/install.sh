conda create -n DNTR python=3.7 --y
# Install pytorch
conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=11.0 -c pytorch --y
# Required packages
pip install -r requirements/build.txt
pip install yapf==0.40.0
pip install numba
pip install timm
pip install torchprofile
python setup.py develop
# Install cocoapi
pip install pycocotools
# Install aitodcocoapi
pip install cython==0.29.36
pip install "git+https://github.com/jwwangchn/cocoapi-aitod.git#subdirectory=aitodpycocotools"
#Install mmcv
#pip install mmcv==2.0.0rc4 -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7/index.html
pip install -U openmim
mim install mmcv-full==1.6.0