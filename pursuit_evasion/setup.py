from setuptools import setup, find_packages

setup(
    name="pursuit_evasion",
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'gym',
        'pybullet',
        'numpy',
        'matplotlib',
        'control',
        'stable-baselines3',
        'torch',
        'tensorboard'
    ],
    description='A pursuit-evasion environment for training drone agents',
    author='Your Name',
    author_email='your.email@example.com',
)