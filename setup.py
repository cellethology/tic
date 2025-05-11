from setuptools import setup, find_packages

def read_requirements():
    with open("requirements.txt", "r") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="tic",
    version="2.0.0",  # 2025/05/08
    author="Jiahao Zhang",
    description="Tumor Inference of Causality in EMT Progression",
    packages=find_packages(include=['tic', 'tic.*', 'utils', 'utils.*']),
    python_requires=">=3.10", 
    install_requires=read_requirements(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    url="https://github.com/cellethology/tic.git",
)