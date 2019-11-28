from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="batching",
    version="1.0.6",
    description="Batching is a set of tools to format data for training sequence models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cirick/batching",
    download_url="https://github.com/cirick/batching/archive/v1.0.4.tar.gz",
    author="Charles Irick",
    author_email="cirick@gmail.com",
    include_package_data=True,
    license="MIT",
    packages=["batching"],
    install_requires=[
        "numpy>=1.17.4",
        "pandas>=0.25.3",
        "scikit-learn>=0.21.3",
        "tensorflow>=2.0.0",
        "boto3>=1.10.28"
    ],
    zip_safe=False,
)
