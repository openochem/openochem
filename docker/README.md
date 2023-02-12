# Building and Running with Docker

Installation and scripts are for the "normal" docker. 

## The Environment Image

First, you need to build the environment image with all dependencies. Just run the following from the root of this repository:

```bash
docker build -t ochem-env -f docker/Dockerfile-env .
```

After build this image will contain external software and libraries so make sure that you understand their terms and conditions (see https://github.com/openochem/ochem-external-tools).

## Demo Deployment

You can build and deploy a standalone demo version of ochem with `docker-compose`. To build the images, just run the following from the repository root:

```bash
docker-compose -f docker/ochem-demo.yml build
```

If successful, you should be able to run the whole ochem ecosystem as follows:

```bash
docker-compose -f docker/ochem-demo.yml up
```


N.B.! If you would like to use GPU, follow installation instructions at https://github.com/NVIDIA/nvidia-docker 
and enable  nvidia docker requirements in docker/ochem-demo.yml  and start GPU-enabled servers (servers/gpu)

      #resources:
        #reservations:
          #devices:
          #  - driver: nvidia
          #    count: all
          #    capabilities: [ gpu ]

N.B.! If you upgrade from the previous version, delete flyway_schema_history table in ochem_demo and restart the docker:

docker exec ochem-mariadb mariadb ochem_demo -e "drop table flyway_schema_history"
