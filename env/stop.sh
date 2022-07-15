#!/bin/bash

set -u
set -e

NAME=ochemenv

echo "singularity instance stop $NAME"

singularity instance stop $NAME || true;
rm -rf $NAME/tmp >/dev/null 2>/dev/null
rm -rf ~/.singularity/instances/sing/qsar/bigchem/$NAME >/dev/null 2>/dev/null