from setuptools import setup, find_packages

setup(
    name='ctrl_sandbox',
    version='0.1.0',
    author='tojomcmo',
    author_email='tojomcmo@gmail.com',
    description='Tools and algorithms for controls exploration',
    packages=find_packages(where='src'),  # Tells setuptools to look for packages in src
    package_dir={'': 'src'},  # Tells setuptools that the package directory is src
    install_requires=[
        # List your project's dependencies here
        # e.g., 'numpy', 'pandas'
    ],
)