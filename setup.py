from setuptools import setup, find_packages
from typing import List

HYPEN_E_DOT = '-e .'

def get_requirements(filename: str) -> List[str]:
    '''
    This function reads the requirements from a file and returns them as a list.
    '''
    requirements=[]
    with open(filename, 'r') as file:
        requirements = file.readlines()
        requirements = [req.replace('\n', '') for req in requirements]

    if HYPEN_E_DOT in requirements:
        requirements.remove(HYPEN_E_DOT)

setup(
    name="mlproject",
    version="0.1",
    author="Yohan Shanuka",
    author_email="yshanuka123@gmail.com",
    description="A machine learning project",
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)