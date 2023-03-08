from setuptools import setup

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='mfnlc',
    version='0.0.1',
    packages=[''],
    url='https://github.com/ZikangXiong/mf-nlc',
    license='MIT',
    author='Zikang Xiong',
    author_email='zikangxiong@gmail.com',
    description='Model Free Neural Lyapunov Control',
    install_requires=required
)
