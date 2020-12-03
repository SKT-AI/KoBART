from kobart import __version__
from setuptools import find_packages, setup

setup(name='kobart',
      version=__version__,
      url='https://github.com/SKT-AI/KoBART',
      license='midified MIT',
      author='Heewon Jeon',
      author_email='madjakarta@gmail.com',
      description='KoBART (Korean BART)',
      packages=find_packages(where=".", exclude=(
          'tests',
          'scripts'
      )),
      long_description=open('README.md', encoding='utf-8').read(),
      zip_safe=False,
      include_package_data=True,
      )
