# MPS Adjoint

This code repository is developed as a part of an existing research project estimating active tension in cardiac micromuscles. Solving an inverse problem, formulated as a PDE-constrained minimization problem, we find a spatial-temoral distribution of active tension and a spatial distribution of the fiber angle for any selected sample.


## Install

### Using docker
The easiest way to run the code is to use the provided docker image. For this you need to have [Docker](https://docs.docker.com/get-docker/). Once docker is installed you can pull the image
```
docker pull ghcr.io/aashildte/mpsadjoint:latest
```
and start a new container (sharing your current directory with the container)
```
docker run --name mpsadjoint -w /home/shared -v $PWD:/home/shared -it ghcr.io/aashildte/mpsadjoint:latest
```
The scripts are located in `/app` inside the container

### Using pip
The code in this repository is pure python and can therefore be installed with `pip` using the command
```
python -m pip install git+https://github.com/aashildte/mpsadjoint
```
However, to run the code you need to have FEniCS. See https://fenicsproject.org/download/archive/ for more info about how to install FEniCS.

## Steps to reproduce results

TBW

## Need help
Open an [issue](https://github.com/aashildte/mpsadjoint/issues/new) or send me en e-mail me at aashild@simula.no if you have questions.
