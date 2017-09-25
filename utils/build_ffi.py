import glob
import torch
from os import path
from torch.utils.ffi import create_extension

base_dir = path.dirname(path.abspath(__file__))
extra_objects = [path.join(base_dir, 'build/ball_query_gpu.so'), path.join(base_dir, 'build/group_points_gpu.so')]
print(extra_objects)
extra_objects += glob.glob('/usr/local/cuda/lib64/*.a')

ffi = create_extension(
    'pointnet2',
    headers=['cinclude/ball_query_wrapper.h', 'cinclude/group_points_wrapper.h'],
    sources=['csrc/ball_query.c', 'csrc/group_points.c'],
    define_macros=[('WITH_CUDA', None)],
    relative_to=__file__,
    with_cuda=True,
    extra_objects=extra_objects,
    include_dirs=[path.join(base_dir, 'cinclude')])

if __name__ == "__main__":
    assert torch.cuda.is_available(), "Needs CUDA!"
    ffi.build()
