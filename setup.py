from setuptools import find_packages, setup
setup(
    name="methabc",
    version="0.0.1",
    description="",
    python_requires=">=3.11",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)
