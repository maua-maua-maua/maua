from setuptools import setup
from torch.utils import cpp_extension

setup(
    name="efficient_quantile",
    ext_modules=[cpp_extension.CppExtension("efficient_quantile", ["efficient_quantile.cpp"])],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
)
