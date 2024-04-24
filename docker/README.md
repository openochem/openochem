# Building and Running with Docker

Installation and scripts are for the "normal" docker. 

Install docker, git and clone image:

```git clone https://github.com/openochem/openochem.git
cd openochem
```

## The Environment Image

First, you need to build the environment image with all dependencies. Just run the following from the root of this repository:

```bash
docker build -t ochem-env -f docker/Dockerfile-env .
```

To compile in China (few packages will be disabled) use:
```bash
docker build -t ochem-env -f docker/Dockerfile-China .
```

After build this image will contain external software and libraries so make sure that you understand their terms and conditions (see https://github.com/openochem/ochem-external-tools).

## Demo Deployment

You can build and deploy a standalone demo version of ochem with `docker-compose` (on some systems `docker compose`) . To build the images, just run the following from the repository root:

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

N.B.! If you have a local installation of mysql, it will interfere with the docker. Stop the local mysql before running "up" command.