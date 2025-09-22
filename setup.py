#!/usr/bin/env python3

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="mcp-web-search",
    version="1.0.0",
    author="Fouad Boutaleb",
    author_email="fouad@example.com",
    description="A lightweight Model Context Protocol (MCP) server for intelligent web search with AI-powered ranking",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Fouadbtlb/mcp-web-search",
    project_urls={
        "Bug Tracker": "https://github.com/Fouadbtlb/mcp-web-search/issues",
        "Docker Hub": "https://hub.docker.com/r/boutalebfouad/mcp-web-search",
        "Documentation": "https://github.com/Fouadbtlb/mcp-web-search#readme",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Internet :: WWW/HTTP :: Indexing/Search",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "mcp-web-search=src.server:main",
            "mcp-web-search-stdio=src.mcp_server:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.yml", "*.yaml"],
    },
    keywords="mcp model-context-protocol web-search ai semantic-search ollama docker",
)