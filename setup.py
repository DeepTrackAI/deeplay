from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as file:
    required = file.read().splitlines()

setup(
    name="deeplay",
    version="0.0.6",
    license="MIT",
    packages=find_packages(),
    author=(
        "Benjamin Midtvedt, Jesus Pineda, Henrik Klein Moberg, "
        "Harshith Bachimanchi, Carlo Manzo, Giovanni Volpe"
    ),
    description=(
        "An AI-powered platform for advancing deep learning research "
        "and applications, developed by DeepTrackAI."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DeepTrackAI/deeplay",
    keywords=[
        "AI",
        "Deep Learning",
        "Machine Learning",
        "Data Science",
        "Research Platform",
        "Artificial Intelligence",
        "Technology",
    ],
    python_requires=">=3.8",
    install_requires=required,
)
