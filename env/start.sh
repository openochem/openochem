#!/bin/bash

set -u
set -e

DIR=`pwd`
NAME=ochemenv
DIR=$DIR/$NAME

mkdir -p $DIR/tmp
singularity  instance start -e -B $DIR:/ochem -B $DIR/tmp:/etc/ochem/ $DIR/$NAME.sif $NAME
singularity exec instance://ochemenv /etc/source/ochem/bin/startochem.sh >/dev/null 2>/dev/null

echo "OCHEM is starting and will be available in few minutes. Waiting 15 seconds to start servers."

sleep 15
cd ../servers
bash start.sh ochem

echo "OCHEM servers are  starting and will be available in few minutes."
