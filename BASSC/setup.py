from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
from Cython.Distutils import build_ext

filename = 'gmm_help.pyx'

ext_modules=[
    Extension("gmm_help",
              [filename],
              libraries=["m"],
              extra_compile_args = ["-O3", "-ffast-math", "-march=native", "-fopenmp" ],
              extra_link_args=['-fopenmp']
              )
]

setup(
    name='gmm_help',
    ext_modules=cythonize(ext_modules),
)

filename = 'elki_vanilla_modified.pyx'

ext_modules=[
    Extension("elki_vanilla_modified",
              [filename],
              libraries=["m"],
              extra_compile_args = ["-O3", "-ffast-math", "-march=native", "-fopenmp" ],
              extra_link_args=['-fopenmp']
              )
]

setup(
    name='elki_vanilla_modified',
    ext_modules=cythonize(ext_modules),
)
