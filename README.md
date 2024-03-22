# MPS Adjoint

This code repository is developed as a part of an existing research project estimating active tension in cardiac micromuscles. Solving an inverse problem, formulated as a PDE-constrained minimization problem, we find a spatial-temoral distribution of active tension and a spatial distribution of the fiber angle for any selected sample.


## Install

### Using docker
You can run the code using the provided docker image. For this you need to have [Docker](https://docs.docker.com/get-docker/). Once docker is installed you can pull the image
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
However, to run the code you need to have FEniCS and Pyadjoint/Dolfin adjoint. See https://fenicsproject.org/download/archive/ and https://www.dolfin-adjoint.org/en/latest/download/index.html for more info about how to install these (or follow the conda instructions below).

### Using conda

In order to run the code with Conda, install Pyadjont as (following the description [here](https://anaconda.org/conda-forge/dolfin-adjoint));

```
conda install -c conda-forge dolfin-adjoint
```

then the optimization software Cyipopt (as described [here](https://cyipopt.readthedocs.io/en/stable/install.html#using-conda)) as:

```
conda install -c conda-forge cyipopt
```

## Scripts
This code has developed for the paper "Estimation of active tension in cardiac microtissues by solving a PDE-constrained optimization problem". All figures should be reproduceable from the code here along with the data submitted with the paper.

The scripts folder contains script for the synthetic data, and for running inversions in the three phases with brightfield data as input (in order, as described for each figure).

## Documentation
See http://aashildte.github.io/mpsadjoint

## Need help
Open an [issue](https://github.com/aashildte/mpsadjoint/issues/new) or send me en e-mail me at aashild@simula.no if you have questions.
