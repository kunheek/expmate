from setuptools import setup, find_packages

setup(
    name="expmate",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pyyaml>=6.0",
        "tqdm>=4.64.0",
    ],
    extras_require={
        "torch": ["torch>=1.12.0"],
        "jax": ["jax>=0.4.0", "flax>=0.6.0"],
    },
    entry_points={
        "console_scripts": [
            "expmate=expmate.cli:main",
        ],
    },
)
