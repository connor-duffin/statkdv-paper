from setuptools import find_packages, setup

with open("README.md", "r") as f:
    readme = f.read()

setup(
    name="statfenics",
    version="0.0.1",
    author="Connor Duffin",
    author_email="connor.p.duffin@gmail.com",
    description="Routines to use statFEM within FEniCS",
    long_description=readme,
    url="https://github.com/connor-duffin/statfenics",
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    packages=find_packages(exclude=[
        "tests", "*.tests", "*.tests.*", "tests.*"
    ]),
    python_requires=">=3.6"
)
