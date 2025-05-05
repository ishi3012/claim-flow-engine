from setuptools import setup, find_packages

setup(
    name="claimflowengine",
    version="0.1.0",
    author="Shilpa Musale",
    author_email="shilpa.musale02@gmail.com",
    description="An intelligent healthcare claim routing system using ML and RL-style policies.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/ishi3012/ClaimFlowEngine",
    packages=find_packages(exclude=["tests", "notebooks", "data"]),
    install_requires=[
        "pandas>=1.5",
        "numpy>=1.21",
        "scikit-learn>=1.0",
        "xgboost>=1.7",
        "lightgbm>=3.3",
        "umap-learn>=0.5",
        "hdbscan>=0.8",
        "sentence-transformers>=2.2",
        "scipy>=1.7",
        "fastapi>=0.95",
        "uvicorn[standard]>=0.22",
        "mlflow>=2.0",
        "joblib>=1.2"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)