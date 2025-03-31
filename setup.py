from setuptools import setup, find_packages

def read_requirements():
    with open("requirements.txt", "r") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="tic",
    version="1.1.0",  # 2025/03/31
    author="Jiahao Zhang",
    description="Temporal Inference of Tumor Cells",
    packages=find_packages(include=['tic', 'tic.*', 'utils', 'utils.*']),
    python_requires=">=3.9", # torch 2.5.* requires python >= 3.9
    install_requires=read_requirements(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)