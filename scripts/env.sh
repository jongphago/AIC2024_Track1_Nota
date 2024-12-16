conda create -n nota python=3.8.20 -y
conda activate nota

conda install cudatoolkit=11.1 -c conda-forge -y
conda install cudnn=8 -c conda-forge -y

pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html

pip install cython
pip install xtcocotools

pip install openmim

mim install mmengine "mmcv==2.0.1"

conda clean --all
git clone https://github.com/open-mmlab/mmpose.git mmpose
git checkout main
FORCE_CUDA="1"
pip install -r requirements/build.txt
pip install --no-cache-dir -e .

mim install mmdet==3.0.0

pip3 install matplotlib \
                scipy \
                Pillow \
                numpy \
                prettytable \
                easydict \
                scikit-learn \
                pyyaml \
                yacs \
                termcolor \
                tabulate \
                tensorboard \
                opencv-python \
                pyyaml \
                yacs \
                termcolor \
                scikit-learn \
                tabulate \
                gdown \
                faiss-gpu \
                lap \
                cython-bbox
pip3 install ultralytics