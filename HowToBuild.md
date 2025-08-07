* Setup
```shell
# Install packages
sudo apt-get install librhash0 librhash-dev

# Install cusparselt
wget https://developer.download.nvidia.com/compute/cusparselt/0.7.1/local_installers/cusparselt-local-tegra-repo-ubuntu2204-0.7.1_1.0-1_arm64.deb
sudo dpkg -i cusparselt-local-tegra-repo-ubuntu2204-0.7.1_1.0-1_arm64.deb
sudo cp /var/cusparselt-local-tegra-repo-ubuntu2204-0.7.1/cusparselt-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install libcusparselt0 libcusparselt-dev

# make sure to start with a fresh environment
conda env remove -n tvm21

# create the conda environment with build dependency
conda create -n tvm21 -c conda-forge \
    "llvmdev=15" \
    "cmake>=3.24" \
    git \
    cython psutil matplotlib tqdm opencv cloudpickle \
    "numpy=1.26.4" \
    "jpeg=9e" "libjpeg-turbo" "libtiff" "openexr" "libdeflate" \
    python=3.10

# enter the build environment
conda activate tvm21
# Pytorch
python3 -m pip install https://developer.download.nvidia.com/compute/redist/jp/v60/pytorch/torch-2.4.0a0+3bcc3cddb5.nv24.07.16234504-cp310-cp310-linux_aarch64.whl
# torchvision
git clone  https://github.com/pytorch/vision torchvision
cd torchvision/
git checkout v0.19.0
python3 setup.py install --user
```

* Build
```shell
git submodule update --init --recursive
mkdir build
cd tvm/build
cp ../config.cmake.default config.cmake

cmake .. && cmake --build . --parallel $(nproc)
```

* Export
```
export TVM_HOME=/home/rubis/workspace/tvm-segment-21
export PYTHONPATH=$TVM_HOME/python:$PYTHONPATH
export TVM_LIBRARY_PATH=$TVM_HOME/build
export LD_LIBRARY_PATH=$TVM_LIBRARY_PATH:$LD_LIBRARY_PATH
```