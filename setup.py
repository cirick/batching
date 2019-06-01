from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="batching",
    version="0.0.8",
    description="Batching is a set of tools to format data for training sequence models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cirick/batching",
    download_url="https://github.com/cirick/batching/archive/v0.0.8.tar.gz",
    author="Charles Irick",
    author_email="cirick@gmail.com",
    include_package_data=True,
    license="MIT",
    packages=["batching"],
    install_requires=[
        "numpy>=1.16.3",
        "pandas>=0.24.2",
        "scikit-learn>=0.21.2",
        "tensorflow>=1.13.1",
        "boto3>=1.9.156"
    ],
    zip_safe=False,
)
