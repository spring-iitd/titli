from setuptools import setup, find_packages

setup(
    name='titli',
    version='0.0.6',
    description='A library for collection of IDS and tools for evaluating them',
    long_description=open("README.md").read(),
    long_description_content_type = "text/markdown",
    url='https://github.com/spg-iitd/raids',
    packages=find_packages(),
    license='MIT',
    author="Subrat Kumar Swain",
    author_email='mailofswainsubrat@gmail.com',
    keywords='ids adversarial network nids',
    install_requires=[
        # Add your project's dependencies here
        'scapy', 'numpy', 'pandas', 'matplotlib', 'scikit-learn', 'scipy', 'tqdm', 'torch', 'torchvision'
    ],
    classifiers=[
        # Add classifiers that match your project
        # Check https://pypi.org/classifiers/ for the full list
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Security",
    ],
    zip_safe=False,
    python_requires='>=3.10',
)
