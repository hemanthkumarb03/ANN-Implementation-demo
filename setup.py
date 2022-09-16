from setuptools import setup

with open("README.md",'r',encoding='utf-8') as f:
    description = f.read()

setup(
    name = "src",
    version = "0.0.1",
    author = "Hemanth",
    description = "A small package for DL Pipeline",
    long_description=description,
    
)