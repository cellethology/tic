from setuptools import setup, find_packages

setup(
    name="tic",
    version="0.0.0",
    author="Jiahao Zhang",
    description=(
        "Temperal Inference of tumer Cells"
    ),
    packages=find_packages(include=['adapters','adapters.*','core', 'core.*', 'tools', 'tool.*','utils', 'utils.*']),
    python_requires=">=3.7",
)