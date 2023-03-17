"""
This is a buncha stuff that is on the edge of possibility, totally unsafe, and completely unadvised to use in production...sounds fun yeah?
"""
import os
import cv2  # noqa
import pycuda  # noqa
import pycuda.gpuarray as gpuarray
import pycuda.autoinit  # noqa
import torch  # noqa
import numpy as np
import cupy as cp
import pytorch_pfn_extras as ppe
import PIL
from PIL import Image
from torchvision.transforms import ToTensor, Resize  # noqa

ppe.cuda.use_torch_mempool_in_cupy()


################ OPTIONS THAT MAY OR MAY NOT ACTUALLY HELP US ####################
os.environ["OMP_NUM_THREADS"] = '4'  #
os.environ["OMP_SCHEDULE"] = 'STATIC'  #
os.environ["OMP_PROC_BIND"] = 'CLOSE'  #
os.environ["LD_PRELOAD"] = '<jemalloc.so/tcmalloc.so>:$LD_PRELOAD'  #
#
torch.backends.cudnn.benchmark = True  # noqa                                    #
torch.backends.cuda.matmul.allow_tf32 = True  # noqa                             #
torch.backends.cudnn.allow_tf32 = True  # noqa                                   #
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True  # noqa #
torch.jit.enable_onednn_fusion(True)  # noqa                                     #
torch.backends.cuda.cufft_plan_cache.max_size = 8192  # noqa                     #


##################################################################################


def pinned_array(array):
    """
    https://stackoverflow.com/questions/47455294/asynchronous-gpu-memory-transfer-with-cupy
    """
    mem = cp.cuda.alloc_pinned_memory(array.nbytes)
    src = np.frombuffer(
        mem, array.dtype, array.size).reshape(array.shape)
    src[...] = array
    stream = cp.cuda.Stream()
    src = cp.empty_like(src)
    src.set(array, stream=stream)
    return src, stream


def get_pinned_array(data):
    """
    https://github.com/cupy/cupy/issues/3452#issuecomment-903212530
    Return populated pinned memory.
    Parameters
    ----------
    data : cupy.ndarray or numpy.ndarray
        The array to be copied to shared buffer
    """

    mem = cp.cuda.alloc_pinned_memory(data.nbytes)
    ret = np.frombuffer(mem, data.dtype, data.size).reshape(data.shape)
    ret[...] = data

    return ret


def mapped_malloc(size):
    """
    https://github.com/cupy/cupy/issues/3452#issuecomment-903212530

    mapped_pinned_mem_pool = cp.cuda.PinnedMemoryPool(mapped_malloc)
    cp.cuda.set_pinned_memory_allocator(mapped_pinned_mem_pool.malloc)
    """
    mem = cp.cuda.PinnedMemory(size, cp.cuda.runtime.hostAllocMapped)
    return cp.cuda.PinnedMemoryPointer(mem, 0)


class PMemory(cp.cuda.memory.BaseMemory):
    """
    https://stackoverflow.com/questions/57752516/how-to-use-cuda-pinned-zero-copy-memory-for-a-memory-mapped-file

    This fucker is **FAST**.

    cp.cuda.set_allocator(my_pinned_allocator)

    # Create 4 .npy files, ~4GB each
    for i in range(4):
        print(i)
        numpyMemmap = np.memmap( 'reg.memmap'+str(i), dtype='float32', mode='w+', shape=( 10000000 , 100))
        np.save('reg.memmap'+str(i) , numpyMemmap )
        del numpyMemmap
        os.remove('reg.memmap'+str(i) )

    # Check if they load correctly with np.load.
    NPYmemmap = []
    for i in range(4):
        print(i)
        NPYmemmap.append(np.load( 'reg.memmap'+str(i)+'.npy' , mmap_mode = 'r+' )  )
    del NPYmemmap

    # allocate pinned memory storage
    CPYmemmap = []
    for i in range(4):
        print(i)
        CPYmemmap.append(cp.load( 'reg.memmap'+str(i)+'.npy' , mmap_mode = 'r+' )  )
    cp.cuda.set_allocator(None)
    """

    def __init__(self, size):
        cp.cuda.memory.BaseMemory.__init__(self)
        self.size = size
        self.device_id = cp.cuda.device.get_device_id()
        self.ptr = 0
        if size > 0:
            self.ptr = cp.cuda.runtime.hostAlloc(size, 0)

    def __del__(self):
        """
        Clear the memory.
        """
        if self.ptr:
            cp.cuda.runtime.freeHost(self.ptr)
        return self


def from_cupy(arr):
    """
    Cast CuPy array as pycuda GPU array.
    Remember this can only be run after the import pycuda.autoinit statement.
    """
    shape = arr.shape
    dtype = arr.dtype

    def alloc(_):
        """
        Dummy.
        """
        return arr.data.ptr

    if arr.flags.c_contiguous:
        order = 'C'
    elif arr.flags.f_contiguous:
        order = 'F'
    else:
        raise ValueError('arr order cannot be determined')
    return gpuarray.GPUArray(
        shape=shape,
        dtype=dtype,
        allocator=alloc,
        order=order
    )


numpy_opencv_type_map = {
    np.uint8: [cv2.CV_8U, cv2.CV_8UC1, cv2.CV_8UC2, cv2.CV_8UC3, cv2.CV_8UC4],  # Only these are allowed in a GpuMat.
    np.int8: [cv2.CV_8S, cv2.CV_8SC1, cv2.CV_8SC2, cv2.CV_8SC3, cv2.CV_8SC4],
    np.uint16: [cv2.CV_16U, cv2.CV_16UC1, cv2.CV_16UC2, cv2.CV_16UC3, cv2.CV_16UC4],
    np.int16: [cv2.CV_16S, cv2.CV_16SC1, cv2.CV_16SC2, cv2.CV_16SC3, cv2.CV_16SC4],
    np.int32: [cv2.CV_32S, cv2.CV_32SC1, cv2.CV_32SC2, cv2.CV_32SC3, cv2.CV_32SC4],
    np.float32: [cv2.CV_32F, cv2.CV_32FC1, cv2.CV_32FC2, cv2.CV_32FC3, cv2.CV_32FC4],
    np.float64: [cv2.CV_64F, cv2.CV_64FC1, cv2.CV_64FC2, cv2.CV_64FC3, cv2.CV_64FC4]
}

# [0, 0, 8, 16, 24]


class CV2CPArrayInterface:
    """
    https://github.com/rapidsai/cucim/issues/329
    https://gist.github.com/szagoruyko/dccce13465df1542621b728fcc15df53

    http://ninghang.blogspot.com/2012/11/list-of-mat-type-in-opencv.html?m=1
    """

    def __init__(self, gpu_mat):
        w, h = gpu_mat.size()
        type_map = {
            cv2.CV_8U: "|u1",  # noqa
            cv2.CV_8UC1: "|u1",  # noqa
            cv2.CV_8UC2: "|u1",  # noqa
            cv2.CV_8UC3: "|u1",  # noqa
            cv2.CV_8UC4: "|u1",  # noqa
            cv2.CV_8S: "|i1",  # noqa
            cv2.CV_8SC1: "|i1",  # noqa
            cv2.CV_8SC2: "|i1",  # noqa
            cv2.CV_8SC3: "|i1",  # noqa
            cv2.CV_8SC4: "|i1",  # noqa
            cv2.CV_16U: "<u2",  # noqa
            cv2.CV_16UC1: "<u2",  # noqa
            cv2.CV_16UC2: "<u2",  # noqa
            cv2.CV_16UC3: "<u2",  # noqa
            cv2.CV_16UC4: "<u2",  # noqa
            cv2.CV_16S: "<i2",  # noqa
            cv2.CV_16SC1: "<i2",  # noqa
            cv2.CV_16SC2: "<i2",  # noqa
            cv2.CV_16SC3: "<i2",  # noqa
            cv2.CV_16SC4: "<i2",  # noqa
            cv2.CV_32S: "<i4",  # noqa
            cv2.CV_32SC1: "<i4",  # noqa
            cv2.CV_32SC2: "<i4",  # noqa
            cv2.CV_32SC3: "<i4",  # noqa
            cv2.CV_32SC4: "<i4",  # noqa
            cv2.CV_32F: "<f4",  # noqa
            cv2.CV_32FC1: "<f4",  # noqa
            cv2.CV_32FC2: "<f4",  # noqa
            cv2.CV_32FC3: "<f4",  # noqa
            cv2.CV_32FC4: "<f4",  # noqa
            cv2.CV_64F: "<f8",  # noqa
            cv2.CV_64FC1: "<f8",  # noqa
            cv2.CV_64FC2: "<f8",  # noqa
            cv2.CV_64FC3: "<f8",  # noqa
            cv2.CV_64FC4: "<f8",  # noqa
        }
        self.__cuda_array_interface__ = {
            "version": 3,
            "shape": (h, w, gpu_mat.channels()),
            "typestr": type_map[gpu_mat.type()],
            "descr": [("", type_map[gpu_mat.type()])],
            "stream": 1,
            "strides": (gpu_mat.step, gpu_mat.elemSize(), gpu_mat.elemSize1()),
            "data": (gpu_mat.cudaPtr(), False),
        }


def cudaGpuMat2cupy(mat: cv2.cuda_GpuMat) -> cp.ndarray:  # noqa
    """
    This will convert a gpu mat directly into a cupy array.
    """
    i_face = CV2CPArrayInterface
    interface = cp.asarray(i_face(mat))
    return interface


def cupy2Tensor(mat: cp.ndarray) -> torch.Tensor:  # noqa
    """
    This will convert a gpu mat directly into a torch.Tensor object in GPU memory.

    """
    dl = torch.as_tensor(mat, device='cuda')  # https://docs.cupy.dev/en/stable/user_guide/interoperability.html#pytorch
    return dl


def tensor2Cupy(mat: torch.Tensor) -> cp.ndarray:
    """
    https://discuss.pytorch.org/t/convert-torch-tensors-directly-to-cupy-tensors/2752/4
    """
    mat = cp.asarray(mat)
    return mat


def cupy2cudaGpuMat(mat: cp.ndarray) -> cv2.cuda_GpuMat:  # noqa
    """
    Heh, lets try the impossible...

    Test:

    #### experiment ####
    cap = someImageFromAround.png
    frame = cv2.cuda.GpuMat(cap)  # Upload.
    arry = cudaGpuMat2cupy(frame)  # Convert into cupy array.
    test_mat = cupy2cudaGpuMat(arry)  # Convert back into cv2.cuda_GpuMat.
    test_mat = test_mat.download().astype(np.uint8)  # Test conversion output.
    # test_mat = cap.astype(np.uint8)  # Test original  # Alternatively view the original image.
    cv2.imshow('holy fuck it works', test_mat)
    cv2.waitKey(0)
    """
    type_map = numpy_opencv_type_map

    mat = cp.ascontiguousarray(mat)
    dtype = mat.dtype.type
    if dtype != np.uint8:
        dtype = cp.uint8
        mat = mat.astype(dtype)
    dtypes = type_map[dtype]

    if len(mat.shape) == 2:
        height, width = mat.shape
        channel = 1
    elif len(mat.shape) == 3:
        height, width = mat.shape[:2]
        channel = mat.shape[-1]
    else:
        print('Unable to determine array shape', mat.shape)
        raise ValueError
    if channel > 4:
        print(f'unable to convert {channel} number of channels to Cuda_GpuMat')
        raise ValueError
    try:
        cv_dtype = dtypes[channel]
    except IndexError:
        print(f'unable to convert {channel} number of channels to Cuda_GpuMat')
        raise IndexError

    gpuMat = cv2.cuda_GpuMat((width, height), cv_dtype)  # noqa
    pycuda.driver.memcpy_dtod(gpuMat.cudaPtr(), mat.data.ptr, mat.size)  # noqa
    return gpuMat


def tensor2cudaGpuMat(mat: torch.Tensor) -> cv2.cuda_GpuMat:
    """
    Aptly named.
    """
    mat = tensor2Cupy(mat)
    mat = cupy2cudaGpuMat(mat)
    return mat


def cudaGpuMat2Tensor(mat: cv2.cuda_GpuMat) -> torch.Tensor:  # noqa
    """
    Uses the logic above.
    """
    mat = cudaGpuMat2cupy(mat)
    mat = cupy2Tensor(mat)
    return mat


def convert_mat(
        mat: [np.ndarray, cv2.cuda_GpuMat, cp.ndarray, torch.tensor, list, PIL.Image.Image],  # noqa
        output_type: type) -> [np.ndarray, cv2.cuda_GpuMat, cp.ndarray, torch.tensor, list, PIL.Image.Image]:  # noqa
    """
    This will unify type conversions into a single statement, so we can get past all the weird conversion problems.

    """

    def output_err(_output_type):
        """
        This is just an error notifier.
        """
        print('unable to process output:', _output_type)
        raise TypeError

    if not isinstance(mat, output_type):  # skip operations if the mat is already the desired type.
        if isinstance(mat, PIL.Image.Image) or isinstance(mat, np.ndarray) or isinstance(mat, list):
            if not isinstance(mat, np.ndarray):  # PIL/list 2 numpy.
                mat = np.asarray(mat)
            if output_type == list:  # PIL/numpy 2 list.
                mat = mat.tolist()
            elif output_type == PIL.Image:  # List/numpy 2 PIL.
                mat = Image.fromarray(mat)
            elif output_type == cv2.cuda_GpuMat:  # PIL/numpy/list to opencv.
                mat = cv2.cuda_GpuMat(mat)  # noqa
            elif output_type == cp.ndarray:  # PIL/numpy/list 2 cupy.
                mat = cp.asarray(mat)
            elif output_type == torch.Tensor:  # PIL/numpy/list 2 torch.
                mat = torch.from_numpy(mat).to(torch.device(0))
            else:
                output_err(output_type)
        elif isinstance(mat, cp.ndarray):
            if output_type in [np.ndarray, PIL.Image, list]:  # Cupy 2 numpy.
                mat = cp.asnumpy(mat)
                if output_type == list:  # Cupy 2 list.
                    mat = mat.tolist()  # noqa
                elif output_type == PIL.Image:  # Cupy 2 PIL.
                    mat = Image.fromarray(mat)
            elif output_type == cv2.cuda_GpuMat:  # Cupy 2 opencv.
                mat = cupy2cudaGpuMat(mat)
            elif output_type == torch.Tensor:  # Cupy 2 torch.
                mat = cupy2Tensor(mat)
            else:
                output_err(output_type)
        elif isinstance(mat, cv2.cuda_GpuMat):  # noqa
            if output_type in [list, np.ndarray, PIL.Image]:  # Opencv 2 numpy.
                mat = mat.download()  # noqa
                if output_type == list:  # Opencv 2 list.
                    mat = mat.tolist()
                elif output_type == PIL.Image:  # Opencv 2 PIL.
                    mat = Image.fromarray(mat)
            elif output_type == cp.ndarray:  # Opencv 2 cupy.
                mat = cudaGpuMat2cupy(mat)  # noqa
            elif output_type == torch.Tensor:  # Opencv 2 torch.
                mat = cudaGpuMat2Tensor(mat)
            else:
                output_err(output_type)
        elif isinstance(mat, torch.Tensor):
            if output_type in [np.ndarray, PIL.Image, list]:  # Torch 2 numpy.
                mat = mat.detach().cpu().numpy()
                if output_type == PIL.Image:  # Torch 2 PIL.
                    mat = Image.fromarray(mat)
                elif output_type == list:  # Torch 2 list.
                    mat = mat.tolist()
            elif output_type == cv2.cuda_GpuMat:  # Torch 2 opencv.
                mat = tensor2cudaGpuMat(mat)
            elif output_type == cp.ndarray:  # Torch 2 cupy.
                mat = tensor2Cupy(mat)
            else:
                output_err(output_type)
        else:
            print('unable to process source:', type(mat))
            raise TypeError
    return mat


def convert_many(mats: list, output_types: [type, list[type]]):
    """
    Converts arrays of stuff
    """
    results = list()
    if not isinstance(output_types, list):
        output_types = [output_types] * len(mats)
    for mat, output_type in zip(mats, output_types):
        results.append(convert_mat(mat, output_type))
    return results


def convert_mat_test(single: bool = False):
    """
    This will put the logic above through a few tests.

    We will only test the GPU operations here as the remainder are pretty standard.
    """
    if single:
        mat = np.full((3, 3, 1), (150,))
    else:
        mat = np.full((3, 3, 3), (0, 150, 255))
    print('original\n', mat, mat.dtype)
    mat_1 = convert_mat(mat, cv2.cuda_GpuMat)  # noqa
    print('np 2 opencv\n', mat_1.size(), mat_1.type())
    mat_2 = convert_mat(mat_1, np.ndarray)
    print('opencv 2 np\n', mat_2, mat_2.dtype)
    mat_3 = convert_mat(mat_1, cp.ndarray)
    print('opencv 2 cupy\n', mat_3, mat_3.dtype)
    mat_4 = convert_mat(mat_1, torch.Tensor)
    print('opencv 2 torch\n', mat_4)
    mat_5 = convert_mat(mat_4, cp.ndarray)
    print('torch 2 cupy', mat_5, mat_5.dtype)
    mat_6 = convert_mat(mat_5, cv2.cuda_GpuMat)  # noqa
    print('torch 2 opencv\n', mat_6.size(), mat_6.type())
    mat_7 = convert_mat(mat_3, torch.Tensor)
    print('cupy 2 torch\n', mat_7)
    mat_8 = convert_mat(mat_3, cv2.cuda_GpuMat)  # noqa
    print('cupy to opencv\n', mat_8.size(), mat_8.type())
    mat_9 = convert_mat(mat, torch.Tensor)
    print('numpy 2 torch\n', mat_9)
    mat_10 = convert_mat(mat, cp.ndarray)
    print('numpy 2 cupy', mat_10, mat_10.dtype)
    # paydirt.
