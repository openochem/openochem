# Building and Running with Docker

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