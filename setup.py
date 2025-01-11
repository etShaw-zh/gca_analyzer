from setuptools import setup, find_packages

setup(
    name="gca_analyzer",
    version="0.2.0",
    packages=find_packages(),
    install_requires=[
        'pandas>=1.3.0',
        'numpy>=1.20.0',
        'jieba>=0.42.1',
        'scikit-learn>=1.0.0',
        'matplotlib>=3.4.0',
        'seaborn>=0.11.0',
        'networkx>=2.6.0',
        'plotly>=5.3.0'
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="A package for Group Conversation Analysis with improved text processing and visualization",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/gca_analyzer",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires='>=3.8',
    include_package_data=True,
    package_data={
        'gca_analyzer': ['data/*.txt'],
    }
)
