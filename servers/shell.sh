#!/bin/bash

SERVER=$1

SERVER=${SERVER//\/}

if [ -z "$SERVER" ]; then 
   echo "use as $0 servername"; 
   else
   echo "shell  $SERVER ";
   singularity shell instance://$SERVER
fi
