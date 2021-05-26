from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="batching",
    version="1.1.1",
    description="Batching is a set of tools to format data for training sequence models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cirick/batching",
    download_url="https://github.com/cirick/batching/archive/v1.0.8.tar.gz",
    author="Charles Irick",
    author_email="cirick@gmail.com",
    include_package_data=True,
    license="MIT",
    packages=["batching"],
    install_requires=[
        "numpy>=1.19.2",
        "pandas>=1.2.2",
        "scikit-learn>=0.24.1",
        "tensorflow>=2.4.1",
        "boto3>=1.17.7"
    ],
    zip_safe=False,
)
