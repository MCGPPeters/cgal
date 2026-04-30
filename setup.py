"""Setup script for CGAL package."""

from setuptools import setup, find_packages

setup(
    name="cgal",
    version="0.1.0",
    description="CGAL - Consensus-Gated Associative Learning",
    author="CGAL Contributors",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
        ],
    },
)
