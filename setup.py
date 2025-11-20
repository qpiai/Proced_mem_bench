"""
Setup script for procedural-memory-benchmark package.

For modern installations, use: pip install .
For editable installations, use: pip install -e .
For full installation with baselines: pip install .[all]
"""

from setuptools import setup, find_packages

setup(
    name="procedural-memory-benchmark",
    version="0.1.0",
    packages=find_packages(),
    package_data={
        "procedural_memory_benchmark": [
            "benchmark/data/*.json",
            "data/corpus/*.json",
        ],
    },
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=[],
    extras_require={
        "baselines": [
            "chromadb>=0.4.24",
            "sentence-transformers>=2.3.1",
            "torch>=1.9.0",
            "numpy>=1.21.0",
        ],
        "llm_judge": [
            "openai>=1.12.0",
            "python-dotenv>=1.0.1",
        ],
        "all": [
            "chromadb>=0.4.24",
            "sentence-transformers>=2.3.1",
            "torch>=1.9.0",
            "numpy>=1.21.0",
            "openai>=1.12.0",
            "python-dotenv>=1.0.1",
        ],
    },
    author="Procedural Memory Research Team",
    description="A benchmark for evaluating procedural memory retrieval systems",
    long_description=open("README.md").read() if __name__ == "__main__" else "",
    long_description_content_type="text/markdown",
    keywords="procedural memory benchmark retrieval NLP AI",
    url="https://github.com/yourusername/procedural-memory-benchmark",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
