* Setup
```shell
# make sure to start with a fresh environment
conda env remove -n tvm20

# create the conda environment with build dependency
conda create -n tvm20 -c conda-forge \
    "llvmdev=15" \
    "cmake>=3.24" \
    git \
    python=3.11

# enter the build environment
conda activate tvm20
```

* Build
```shell
cd tvm
git submodule update --init --recursive
mkdir build
cd tvm/build
cp ../config.cmake.default config.cmake

cmake .. && cmake --build . --parallel $(nproc)
```


