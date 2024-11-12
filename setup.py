from setuptools import setup, find_packages
setup(
name='spottedpy',
version='0.1.7',
author='E. Withnell',
author_email='eloise.withnell.20@ucl.ac.uk',
description='Spatial hotspot analysis',
packages=['spottedpy'],
download_url='https://github.com/secrierlab/SpottedPy/archive/refs/tags/0.1.1.zip',
classifiers=[
'Programming Language :: Python :: 3',
'License :: OSI Approved :: MIT License',
'Operating System :: OS Independent',
],
install_requires=[
        'numpy==1.23.4',
        'pandas==2.2.2',
        'scipy==1.13.0',
        'matplotlib==3.8.4',
        'seaborn==0.13.2',
        'tqdm==4.66.5',
        'anndata==0.10.8',
        'scanpy==1.10.1',
        'session_info==1.0.0',
        'squidpy==1.4.1',
    ],
python_requires='>=3.9',
)