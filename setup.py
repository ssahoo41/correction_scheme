from setuptools import setup, find_packages

setup(
    name="correction_scheme",  # top-level
    version="0.1.0",
    # locate anything with with __init__.py and treat as a package
    packages=find_packages(),
    # use requirements.txt to pull these in
    install_requires=[*open('requirements.txt').read().splitlines(),
                      'NNSubsampling @ git++https://github.com/ray38/NNSubsampling.git']
    python_requires=">=3.9",
    author="Lisette del Pino & Jagriti Sahoo",
    author_email="jsahoo@gatech.edu", "lpino3@gatech.edu",
    description="A package for molecular correction schemes",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
)
