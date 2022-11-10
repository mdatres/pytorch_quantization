import setuptools
import os

setuptools.setup(
    name='pytorch_quantization',
    version='0.0.1',
    description='A package to perform Quantization based on Pytorch.',
    author='Massimiliano Datres',
    author_email='mdatres@fbk.eu',
    packages=['.'.join(['pytorch_quantization', p]) for p in setuptools.find_packages(os.path.join(os.path.curdir))],
    install_requires=[
        'torch',
        'pandas',
        'tensorboard'
    ],
)