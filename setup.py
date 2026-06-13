from setuptools import setup, find_packages

setup(
    name="quinnhybrid",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "tqdm>=4.65.0",
    ],
)
