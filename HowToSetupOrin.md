# Install CUDA
1. Check whehter CUDA is already installed or not
    ```bash
    nvcc --version
    # If CUDA version is appeared, skip next step
    ```
2. If not, install CUDA
    **NOTE: Don't select a driver**
    ```bash
    sudo apt-get install -y build-essential
    wget https://developer.download.nvidia.com/compute/cuda/12.2.2/local_installers/cuda_12.2.2_535.104.05_linux_sbsa.run
    chmod +x cuda_12.2.2_*.run
    sudo ./cuda_12.2.2_*.run # Select the CUDA toolkit only (Don't select a driver) 
    ```
3. Install cudnn & cupti
    ```bash
    sudo apt install libcudnn8 libcudnn8-dev cuda-cupti-12-2
    ```

# Setup conda environment
4. Install miniconda
    ```bash
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh
    sh Miniconda3-latest-Linux-aarch64.sh
    ```

5. Setup conda environment
    ```bash
    conda create -n tvm -c conda-forge \
    "llvmdev=15" \
    "cmake>=3.24" \
    git \
    python=3.10
    ```

# Install pytorch
6. Install cusparselt
    ```bash
    sudo apt install python3-pip libopenblas-dev curl
    wget raw.githubusercontent.com/pytorch/pytorch/5c6af2b583709f6176898c017424dc9981023c28/.ci/docker/common/install_cusparselt.sh
    sudo su
    export CUDA_VERSION=12.2
    bash ./install_cusparselt.sh
    exit
    ```

7. Install wheel files to Orin [[link](https://drive.google.com/drive/folders/1jkai70O10WKGZQpdnE0KATExZt3gbbUn?usp=sharing)]

    (1) `torch-2.3.0-cp310-cp310-linux_aarch64.whl`

    (2) `torchvision-0.18.0a0+6043bc2-cp310-cp310-linux_aarch64.whl`

    (3) `torchaudio-2.3.0+952ea74-cp310-cp310-linux_aarch64.whl`

8. Install pytorch in Orin
    ```bash
    conda activate tvm
    pip install torch-2.3.0-cp310-cp310-linux_aarch64.whl
    pip install torchvision-0.18.0a0+6043bc2-cp310-cp310-linux_aarch64.whl
    pip install torchaudio-2.3.0+952ea74-cp310-cp310-linux_aarch64.whl
    ```

# Build TVM
    ```bash
    # At the workspace direcotry
    conda activate tvm
    git submodule update --init --recursive
    mkdir build
    cd tvm/build
    cp ../config.cmake.default config.cmake

    cmake .. && cmake --build . --parallel $(nproc)
    ```
