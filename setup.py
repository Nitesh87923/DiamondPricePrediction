from setuptools import find_packages,setup
from typing import List
e_extra='-e .'
def get_requires(file_path:str)->List[str]:
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]
        if e_extra in requirements:
            requirements.remove(e_extra)
        return requirements
setup(
name='DiamondPricePrediction',
version='0.0.1',
author='Nitesh',
author_email='niteshdeepak2002@gmail.com',
install_requires=get_requires('requirements.txt'),
packages=find_packages()
)