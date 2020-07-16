# torchac: Fast Entropy Coding in PyTorch

# _WIP: Code still lives in [L3C-Pytorch](https://github.com/fab-jul/L3C-PyTorch/tree/master/src/torchac)_

We implemented an entropy coding module as a C++ extension for PyTorch, because no existing fast Python entropy
 coding module was available.

The implementation is based on [this blog post](https://marknelson.us/posts/2014/10/19/data-compression-with-arithmetic-coding.html),
meaning that we implement _arithmetic coding_.
It is **not optimized**, however, it's much faster than doing the equivalent thing in pure-Python (because of all the
 bit-shifts etc.). Encoding an entire `512 x 512` image happens in 0.202s (see Appendix A in the paper).

A good starting point for optimizing the code would probably be the [`range_coder.cc`](https://github.com/tensorflow/compression/blob/master/tensorflow_compression/cc/kernels/range_coder.cc)
implementation of
[TFC](https://tensorflow.github.io/compression/).


#### GPU and CPU support

The module can be built with or without CUDA. The only difference between the CUDA and non-CUDA versions is:
With CUDA, `_get_uint16_cdf` from `torchac.py` is done with a simple/non-optimized CUDA kernel (`torchac_kernel.cu`),
which has one benefit: we can directly write into shared memory! This saves an expensive copying step from GPU to CPU.

However, compiling with CUDA is probably a hassle. We tested with
- GCC 5.5 and NVCC 9.0
- GCC 7.4 and NVCC 10.1 (update 2)
- _Did not work_: GCC 6.0 and NVCC 9
Please comment if you have insights into which other configurations work (or don't.)

The main part (arithmetic coding), is always on CPU.

#### Compiling

_Step 1_: Make sure a **recent `gcc` is available** in `$PATH` by running `gcc --version` (tested with version 5.5).
If you want CUDA support, make sure `nvcc -V` gives the desired version (tested with nvcc version 9.0).

_Step 1b, macOS only_ (tested with 10.14): Set the following
```bash
export CC="clang++ -std=libc++"
export MACOSX_DEPLOYMENT_TARGET=10.14
```

_Step 2_:
 ```bash
 conda activate l3c_env
 cd src/torchac
 COMPILE_CUDA=auto python setup.py install
 ```
- `COMPILE_CUDA=auto`: Use CUDA if a `gcc` between 5 and 6, and `nvcc` 9 is avaiable
- `COMPILE_CUDA=force`: Use CUDA, don't check `gcc` or `nvcc`
- `COMPILE_CUDA=no`: Don't use CUDA

This installs a package called `torchac-backend-cpu` or `torchac-backend-gpu` in your `pip`. 
Both can be installed simultaneously. See also next subsection.

_Step 3_: To test if it works, you can do
  ```
 conda activate l3c_env
 cd src/torchac
 python -c "import torchac"
 ```
It should not print any error messages.
