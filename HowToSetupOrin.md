# Install CUDA
* Check whehter CUDA is already installed or not
    ```bash
    nvcc --version
    # If CUDA version is appeared, skip next step
    ```
* If not, install CUDA
    ```bash
    sudo apt install nvidia-jetpack
    ```

* Add CUDA configuration to `~/.bashrc`
    ```bash
    echo '# CUDA' >> ~/.bashrc
    echo 'export PATH="/usr/local/cuda/bin:$PATH"' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH"' >> ~/.bashrc
    echo 'export CMAKE_PREFIX_PATH="/usr/local/cuda:$CMAKE_PREFIX_PATH"' >> ~/.bashrc
    ```

# Setup conda environment
* Install miniconda
    ```bash
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh
    sh Miniconda3-latest-Linux-aarch64.sh
    ```

* Setup conda environment
    ```bash
    conda create -n tvm -c conda-forge \
    "llvmdev=15" \
    "cmake>=3.24" \
    git \
    python=3.10
    ```

# Install pytorch
* Install cusparselt
    ```bash
    sudo apt install python3-pip libopenblas-dev curl
    wget raw.githubusercontent.com/pytorch/pytorch/5c6af2b583709f6176898c017424dc9981023c28/.ci/docker/common/install_cusparselt.sh
    sudo su
    export CUDA_VERSION=12.2
    bash ./install_cusparselt.sh
    exit
    ```

* Install wheel files to Orin [[link](https://drive.google.com/drive/folders/1jkai70O10WKGZQpdnE0KATExZt3gbbUn?usp=sharing)]

    (1) `torch-2.3.0-cp310-cp310-linux_aarch64.whl`

    (2) `torchvision-0.18.0a0+6043bc2-cp310-cp310-linux_aarch64.whl`

    (3) `torchaudio-2.3.0+952ea74-cp310-cp310-linux_aarch64.whl`

* Install pytorch in Orin
    ```bash
    conda activate tvm
    pip install torch-2.3.0-cp310-cp310-linux_aarch64.whl
    pip install torchvision-0.18.0a0+6043bc2-cp310-cp310-linux_aarch64.whl
    pip install torchaudio-2.3.0+952ea74-cp310-cp310-linux_aarch64.whl
    exit
    ```

# Build TVM
* Install packages for tvm
    ```bash
    conda activate tvm
    conda install Cython psutil matplotlib "numpy<2"
    ```
* Build TVM
    ```bash
    # At the workspace direcotry
    conda activate tvm
    git submodule update --init --recursive
    mkdir build
    cd build
    cp ../config.cmake.default config.cmake

    cmake .. && cmake --build . --parallel $(nproc)
    ```

* Set envrionment variables
    ```bash
    echo '# TVM' >> ~/.bashrc
    echo 'export TVM_HOME="$HOME/workspace/tvm-segment"' >> ~/.bashrc
    echo 'export PYTHONPATH="$TVM_HOME/build:$TVM_HOME/python"' >> ~/.bashrc
    echo 'export PATH="$TVM_HOME/build:$PATH"' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH="$TVM_HOME/build:$LD_LIBRARY_PATH"' >> ~/.bashrc
    source ~/.bashrc
    ```