from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='nelegolizer',
    version='0.1.0',
    description='Python package using neural network to transform 3D object into a LEGO layout',
    long_description=readme,
    author='Damian Lewandowski',
    author_email='lewandd9@gmail.com',
    url='https://github.com/lewandd/nelegolizer',
    license=license,
    install_requires=['numpy','pandas','pyvista', 'torch', 'torchvision', 'torchaudio'],
    packages=find_packages()
)
