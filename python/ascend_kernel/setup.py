from setuptools import setup, find_packages
from setuptools.dist import Distribution

class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True

setup(name="ascend-l2-bench", version="0.1.0", packages=find_packages(),
      package_data={"ascend_kernel": ["lib/*.so", "config.ini"]},
      distclass=BinaryDistribution, python_requires=">=3.9",
      description="Ascend C L2 reuse-distance benchmark")
