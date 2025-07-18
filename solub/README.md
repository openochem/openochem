# Building and Running Docker to reproduce solubility models

Installation and scripts are for the "normal" docker. 

Install docker, git and clone image:

```
git clone https://github.com/openochem/openochem.git
cd openochem
```

### Get models and overwrite default ones

```
wget https://files.ochem.eu/solub.tar.gz
tar -zxf solub.tar.gz; rm solub.tar.gz
mv *.sql demo-data 
```

## The Environment Image

First, you need to build the environment image with all dependencies. Just run the following from the root of this repository:

```bash
docker build -t ochem-env -f solub/Dockerfile-env .
```

To compile in China (few packages will be disabled) enable ARG CHINA in "docker/Dockerfile-env" and compile multiple times until it will succeed:

```bash build_ochem_china.sh```

After build this image will contain external software and libraries so make sure that you understand their terms and conditions (see https://github.com/openochem/ochem-external-tools).

## Deployment

You can build and deploy a standalone demo version of ochem with `docker-compose` (on some systems `docker compose`) . To build the images, just run the following from the repository root:

```bash
docker-compose -f solub/ochem-demo.yml build
```

If successful, you should be able to run the whole ochem ecosystem as follows:

```bash
docker-compose -f solub/ochem-demo.yml up
```

The OCHEM will be available at http://localhost:8080 and http://localhost:7080/metaserver

N.B.! If you would like to use GPU, follow installation instructions at https://github.com/NVIDIA/nvidia-docker 
and enable  nvidia docker requirements in docker/ochem-demo.yml  and start GPU-enabled servers (servers/gpu)

      #resources:
        #reservations:
          #devices:
          #  - driver: nvidia
          #    count: all
          #    capabilities: [ gpu ]

N.B.! If you have a local installation of mysql, it will interfere with the docker. Stop the local mysql before running "up" command.


## Citation

Tetko, I. V.; van Deursen, R.; Godin, G. Be Aware of Overfitting by Hyperparameter Optimization! J. Cheminformatics 2024, 16 (1), 139. https://doi.org/10.1186/s13321-024-00934-w.


