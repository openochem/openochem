#!/bin/bash

# installation script

#make all files  executables

DIR=`pwd`

export ROOT_DIR=DIR

chmod +x build*/*
chmod +x build*/*/*
chmod +x env/*
chmod +x servers/*
cd $DIR/servers/cpu/; ln -sf ../../env/ochemenv/ochemenv.sif cpu.sif
cd $DIR/servers/gpu/; ln -sf ../../env/ochemenv/ochemenv.sif gpu.sif
cd $DIR/env; ln -sfn ochemenv/tmp/ochem-tomcat/logs/catalina.out tomcat.out; ln -sfn ochemenv/tmp/metaserver-tomcat/logs/catalina.out meta.out
cd $DIR

echo "Done!"