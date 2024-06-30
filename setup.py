# -*- coding: utf-8 -*-
import argparse
import os
import subprocess
import sys
import setuptools.command.build_py
import setuptools.command.develop
from setuptools import find_packages, setup

version = '1.0.2'
cwd = os.path.dirname(os.path.abspath(__file__))


class BuildPy(setuptools.command.build_py.build_py):  # pylint: disable=too-many-ancestors
    def run(self):
        self.create_version_file()
        setuptools.command.build_py.build_py.run(self)

    @staticmethod
    def create_version_file():
        print('-- Building version ' + version)
        version_path = os.path.join(cwd, 'version.py')
        with open(version_path, 'w') as f:
            f.write("__version__ = '{}'\n".format(version))


def pip_install(package_name):
    subprocess.call([sys.executable, '-m', 'pip', 'install', package_name])


requirements_train = open(os.path.join(cwd, 'requirements_train.txt'), 'r').readlines()
requirements_infer = open(os.path.join(cwd, 'requirements_infer.txt'), 'r').readlines()

with open('README.md', "r", encoding="utf-8") as readme_file:
    README = readme_file.read()

setup(
    name='wetts',
    version=version,
    url='',
    author='xiepengyuan',
    author_email='xiepengyuan@corp.netease.com',
    description='wetts',
    long_description=README,
    long_description_content_type="text/markdown",
    # package
    include_package_data=True,
    packages=find_packages(include=['wetts*']),
    package_data={
        "wetts": ["tts_frontend/tokenizer/data/*.txt", "tts_frontend/tokenizer/data/*.json"],
    },
    cmdclass={
        'build_py': BuildPy,
    },
    install_requires=[],
    extras_require={
        'train': [
            requirements_train
        ],
        'infer': [
            requirements_infer
        ]
    },
    find_links=[
        'https://pypi.tuna.tsinghua.edu.cn/simple/'
    ],
    python_requires='>=3.8',
    zip_safe=False
)
