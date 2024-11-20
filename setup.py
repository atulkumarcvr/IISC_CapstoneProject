from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = "-e ."
#fundtion
def get_requirements(file_path:str)->List[str]:
    '''
    This function will return the list of requirements
    '''
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        # to remove the \n- newline character 
        requirements = [req.replace('\n','') for req in requirements]
        if HYPHEN_E_DOT in requirements:
            return requirements


    setup(
        name = 'Capstone_Project',
        version = '0.0.1',
        author='Atul Kumar',
        author_email='atulkumar.cvr@outlook.com',
        packages=find_packages(),
    installl_requires = get_requirements('requirements.txt')
    )