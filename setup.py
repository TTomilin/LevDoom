from setuptools import find_packages, setup

levdoom_requirements = [
    "vizdoom",
    "scipy"
]

rl_requirements = [
    "tianshou",
    "wandb",
]

setup(
    name="LevDoom",
    version='1.0.1',
    description="LevDoom: A Generalization Benchmark for Deep Reinforcement Learning",
    long_description=open("README.md", encoding="utf8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/thu-ml/tianshou",
    author="Tristan Tomilin",
    author_email="tristan.tomilin@hotmail.com",
    license="MIT",
    python_requires=">=3.8",
    keywords=["vizdoom", "reinforcement learning", "benchmarking", "generalization"],
    packages=find_packages(),
    include_package_data=True,
    install_requires=levdoom_requirements,
    extras_require={'rl': rl_requirements},
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
