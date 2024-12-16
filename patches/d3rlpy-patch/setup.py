import os

from setuptools import setup, Extension, find_packages

os.environ['CFLAGS'] = '-std=c++11'

# get __version__ variable
here = os.path.abspath(os.path.dirname(__file__))
exec(open(os.path.join(here, 'src/d3rlpy_patch', '_version.py')).read())

if __name__ == "__main__":
    from numpy import get_include
    from Cython.Build import cythonize

    # setup Cython build
    ext = Extension('d3rlpy_patch.dataset',
                    sources=['src/d3rlpy_patch/dataset.pyx'],
                    include_dirs=[get_include(), './src/d3rlpy_patch/cpp/include'],
                    language='c++',
                    extra_compile_args=["-std=c++11", "-O3", "-ffast-math"],
                    extra_link_args=["-std=c++11"])

    ext_modules = cythonize([ext],
                            compiler_directives={
                                'linetrace': True,
                                'binding': True
                            }, language_level=3, force=True)

    # main setup
    setup(name="d3rlpy_patch",
          version=__version__,
          description="Patch for an offline deep reinforcement learning library",
          # long_description=open("README.md").read(),
          # long_description_content_type="text/markdown",
          # url="https://github.com/takuseno/d3rlpy",
          author="CJY-C",
          author_email="masarity@outlook.com",
          # license="MIT License",
          # packages=find_packages(exclude=["tests*"]),
          python_requires=">=3.9.0",
          zip_safe=False,
          package_data={'': ['*.pyx',
                                   '*.pxd',
                                   '*.h',
                                   '*.pyi',
                                   'py.typed']},
          ext_modules=ext_modules,
          # entry_points={'console_scripts': ['d3rlpy=d3rlpy.cli:cli']}
          )
