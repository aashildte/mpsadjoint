FROM ghcr.io/scientificcomputing/fenics-gmsh:2023-08-16



COPY . /app
WORKDIR /app

RUN apt-get update && \
    apt-get install coinor-libipopt-dev -y && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN python3 -m pip install "."
