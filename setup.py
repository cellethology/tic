from setuptools import setup, find_packages

setup(
    name="tic",
    version="0.1.0",
    author="Jiahao Zhang",
    description=(
        "Temporal Inference of Tumor Cells"
    ),
    packages=find_packages(include=['adapters', 'adapters.*', 'core', 'core.*', 'tools', 'tools.*', 'utils', 'utils.*']),
    python_requires=">=3.7",
    install_requires=[
        "hydra-core>=1.2.0",
        "umap-learn>=0.5.0",
        "statsmodels>=0.12.2",
    ],
    extras_require={
        "submodules": [
            # Instructions for installing submodules as local editable packages
            "spacegm @ file://tools/space-gm",
            "pyslingshot @ file://tools/slingshot",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
