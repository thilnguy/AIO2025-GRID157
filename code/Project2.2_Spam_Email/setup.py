from setuptools import setup, find_packages

setup(
    name="spam_ham_email_classifier",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        # Your dependencies here, e.g. 'numpy', 'torch', etc.
    ],
    entry_points={
        "console_scripts": [
            "vector_store=model.vector_store:main",  # TODO: just script template, need to implement
        ],
    },
)
