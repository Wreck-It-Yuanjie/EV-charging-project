from setuptools import setup, find_packages

setup(
    name="your_package_name",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "gluonts",
        "matplotlib",
        "optuna",
        "torch",
        "pathlib"
    ],
    entry_points={
        "console_scripts": [
            "run-analysis=your_package_name:run_analysis",
        ],
    },
    author="Yuanjie Tu",
    author_email="yuanjietu@gmail.com",
    description="A package for EV charging demand prediction with various aggregation methods and estimators.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yuanjietu/EV-charging-project",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)