#!/bin/bash

SERVER=$1
SERVERPATH="."

SERVER=${SERVER//\/}
DIRECTORY=$SERVERPATH/$SERVER/runs

if [ -z $SERVER ]; then 
   echo "use as $0 servername"; 
else
   singularity instance stop $SERVER
   rm -r $DIRECTORY
   rm $SERVERPATH/$SERVER/release.zip
fi
