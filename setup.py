from setuptools import setup

setup(
    name="Multi-Agent-DRL",
    version="1.0",
    description="Deep Multi-Agent Reinforcement Learning (MAD) in a Common-Pool Resource System",
    author="Ker Zhou",
    packages=["mad", "mad/model"],
    package_dir={"": "src"},
    python_requires=">=3.11, <4",
    entry_points={
        'console_scripts': [
            'mad-experiment = mad.main:main'
        ],
    },
    install_requires=[
        "numpy==1.26.4",
        "tensorflow==2.16.1",
        "matplotlib==3.9.0",
        "pydantic==2.7.4"
    ]
)
