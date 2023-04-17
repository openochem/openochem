#!/bin/bash

SERVER=$1
SERVERPATH="."
WEB=http://localhost:7080/metaserver/

SERVER=${SERVER//\/}
DIRECTORY=$SERVERPATH/$SERVER/runs

if [ "$EUID" -eq 0 ]
  then echo "Cannot start servers with root privileges"
  exit
fi

if [ -z $SERVER ]; then 
   echo "use as $0 servername"; 
else
   bash stop.sh $SERVER
   mkdir -p $DIRECTORY/tmp; mkdir -p $DIRECTORY/servers
   wget --no-check-certificate -O $SERVERPATH/$SERVER/release.zip $WEB/update
   singularity  instance start -B $SERVERPATH/$SERVER:/etc/ochem -B $DIRECTORY/servers:/etc/cs_servers -B $DIRECTORY/tmp:/tmp --nv $SERVERPATH/$SERVER/$SERVER.sif $SERVER
   singularity exec instance://$SERVER /etc/source/ochem/bin/startservers.sh >/dev/null 2>/dev/null 
   echo "started $SERVER"
fi
