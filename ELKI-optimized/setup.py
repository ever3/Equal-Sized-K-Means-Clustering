from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
from Cython.Distutils import build_ext

filename = 'argsort.pyx'

ext_modules=[
    Extension("argsort",
              [filename],
              libraries=["m"],
              extra_compile_args = ["-O3", "-ffast-math", "-march=native", "-fopenmp" ],
              extra_link_args=['-fopenmp']
              ) 
]

setup(
    name='argsort',
    ext_modules=cythonize(ext_modules),
)

filename = 'elki_vanilla.pyx'

ext_modules=[
    Extension("elki_vanilla",
              [filename],
              libraries=["m"],
              extra_compile_args = ["-O3", "-ffast-math", "-march=native", "-fopenmp" ],
              extra_link_args=['-fopenmp']
              ) 
]

setup(
    name='elki_vanilla',
    ext_modules=cythonize(ext_modules),
)
