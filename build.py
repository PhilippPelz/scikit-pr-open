import inspect, os
import torch
from torch.utils.ffi import create_extension
import site
this_file = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
print('============this file path:' ,this_file)
#sources = ['lib/my_lib.c']
#headers = ['lib/my_lib.h']
sources = []
headers = []
defines = []
with_cuda = False

site_path = site.getsitepackages()[0]
lib_path = site_path + '/torch/lib/'

if torch.cuda.is_available():
    print('Including CUDA code.')
    sources += ['lib/skpr.c']
    headers += ['lib/skpr.h']
    defines += [('WITH_CUDA', None)]
    with_cuda = True

ffi = create_extension(
    'skpr._ext.skpr_thnn',
    headers=headers,
    sources=sources,
    define_macros=defines,
    relative_to=__file__,
    with_cuda=with_cuda,
    include_dirs= [this_file + '/lib/THZNN'],
    library_dirs= ['lib', lib_path],
    # libraries = [lib_path + 'libTHZNN.so.1',lib_path +'libTHC.so.1',lib_path+'libTH.so.1']
    libraries = ['THZNN','THC','TH']
)


ffi.build()
