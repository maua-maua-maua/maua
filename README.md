# Maua: Creative Machine Learning

## Installation
```bash
conda create -n maua python=3.8 pytorch torchvision torchaudio cudatoolkit=11.3 cudatoolkit-dev=11.3 cudnn mpi4py Cython pip=21.3.1 -c nvidia -c pytorch -c conda-forge
conda activate maua
pip install -r requirements.txt
pip install -r audio/requirements.txt
```
