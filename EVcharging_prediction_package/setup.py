from setuptools import setup, find_packages

setup(
    name='my_gluonts_package',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'gluonts',
        'numpy',
        'pandas',
        'torch',  # or other dependencies if needed
    ],
    include_package_data=True,
    package_data={
        'my_gluonts_package': ['trained_models/*'],
    },
    entry_points={
        'console_scripts': [
            'my_script=my_gluonts_package.cli:main',  # Optional: if you have a CLI entry point
        ],
    },
)