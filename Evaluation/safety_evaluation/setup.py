from setuptools import setup, find_packages

setup(
    name="evaluation_for_safety",
    version="0.1.0",
    description="A Python package for evaluating model safety performance",
    author="SafetyBench Team",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "torch>=1.10.0",
        "transformers>=4.20.0",
        "tqdm>=4.62.0",
        "scikit-learn>=0.24.0"
    ],
    python_requires=">=3.7",
) 