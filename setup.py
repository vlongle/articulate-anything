from setuptools import setup
from setuptools import setup, find_packages


INSTALL_REQUIRES = [
    'google-generativeai',
    'clip @ git+https://github.com/openai/CLIP.git',
    'cotracker @ git+https://github.com/facebookresearch/co-tracker.git@5951295e0ac49068824f75a497ae6749379ec62b', ## cotracker2
    'ipykernel',
    'matplotlib',
    'markdown2',
    'scikit-learn',
    'opencv-python',
    'sapien==2.2.2',
    'pybullet',
    'numpy==1.26.4',  # pybullet requires numpy<2
    'seaborn',
    'trimesh',
    'rich',
    'GPUtil',
    'hydra-core',
    'termcolor',
    'python-dotenv',
    'plotly',
    'nbformat',
    'astor',
    'gradio',
    'openai',
    'anthropic',
    'moviepy',
    # 'gradio @ git+https://github.com/gradio-app/gradio.git@main', ## gradio 5.0-dev has video gallery support
    # see: https://github.com/gradio-app/gradio/pull/9052
]

setup(
    name='articulate_anything',
    version='0.1.0',
    description='articulate_anything',
    author='Long Le',
    install_requires=INSTALL_REQUIRES,
    python_requires='>=3.9',
    packages=find_packages(
        include=['articulate_anything', 'articulate_anything.*']),

)
