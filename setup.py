from setuptools import find_packages, setup

setup(name='kobart',
      version=0.2,
      url='https://github.com/SKT-AI/KoBART.git',
      license='midified MIT',
      author='Heewon Jeon',
      author_email='madjakarta@gmail.com',
      description='KoBART (Korean BART)',
      packages=find_packages(where=".", exclude=(
          'tests',
          'scripts',
          'examples'
      )),
      long_description=open('README.md', encoding='utf-8').read(),
      zip_safe=False,
      include_package_data=True,
      install_requires=[
          'transformers == 4.1.1',
          'torch == 1.7.1'
      ]
      )
