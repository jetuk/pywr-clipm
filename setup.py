import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pywr-clipm",
    version="0.1.0",
    author="James E. Tomlinson",
    author_email="tomo.bbe@gmail.com",
    description="An OpenCL solver for Pywr.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    package_data={
        "": ["*.cl"]
    },
    install_requires=[
        "pyopencl",
        "numpy",
        "scikit-sparse",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
