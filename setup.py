from setuptools import setup, find_packages
import sys

# Platform-specific dependencies
if sys.platform == "darwin":  # macOS
    tensorflow_deps = ["tensorflow-macos>=2.10.0", "tensorflow-metal>=1.0.0"]
else:  # Linux/Windows
    tensorflow_deps = ["tensorflow>=2.10.0"]

setup(
    name='suika_rl',
    version='0.1.0',    # your package version
    packages=find_packages(where='suika_env'),
    package_dir={'': 'suika_env'},
    install_requires=[
        'gymnasium',  # and other necessary packages
        'selenium',
        'ipdb',
        'imageio',
        'numpy>=1.21.0',
        'pillow>=8.0.0',
        'matplotlib>=3.5.0',
        'scikit-learn>=1.0.0',
    ] + tensorflow_deps,
    python_requires='>=3.8',
)