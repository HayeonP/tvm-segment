
* Setup conda environment
    ```bash
    conda create -n tvm -c conda-forge \
    "llvmdev=15" \
    "cmake>=3.24" \
    git \
    python=3.10
    ```

* Build TVM
    ```bash
    # At the workspace direcotry
    conda activate tvm
    git submodule update --init --recursive
    mkdir build
    cd tvm/build
    cp ../config.cmake.default config.cmake

    cmake .. && cmake --build . --parallel $(nproc)
    ```