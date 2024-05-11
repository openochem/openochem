while ! docker build -t ochem-env -f docker/Dockerfile-China . ; do sleep 4 ; done ; echo succeed ;
while ! docker-compose -f docker/ochem-demo.yml build ; do sleep 4 ; done ; echo succeed