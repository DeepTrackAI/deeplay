from setuptools import setup, find_packages

with open("requirements.txt", "r") as file:
    required = file.read().splitlines()

setup(
    name="deeplay",
    version="0.0.5",
    license="MIT",
    author="Benjamin Midtvedt",
    author_email="email@example.com",
    packages=find_packages(),
    url="https://github.com/softmatterlab/deeplay",
    keywords="example project",
    install_requires=required,
)
