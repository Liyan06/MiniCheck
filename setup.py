import os
from setuptools import setup, find_packages

with open(os.path.join(os.path.dirname(__file__), 'README.md'), 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setup(
    name="minicheck",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        line.strip() for line in open('requirements.txt').readlines()
    ],
    extras_require={
        "llm": ["vllm"]
    },
    author="Liyan Tang",
    author_email="lytang@utexas.edu",
    description="MiniCheck: Efficient Fact-Checking of LLMs on Grounding Documents",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/minicheck",
    classifiers=[
        "Programming Language :: Python :: 3",
        'License :: OSI Approved :: Apache Software License',
    ],
    python_requires='>=3.8',
)
