#!/bin/bash

set -u
set +e

echo "Starting ochem on ${HOSTNAME}."
export OCHEM_SERVER=${OCHEM_SERVER:-cpu}

# prepare files
echo "Preparing files..."
cd openochem

NAME=${HOSTNAME}
ARCH=`dpkg --print-architecture`

if [ -n "$NAME" ]; then
	if [[ "$NAME" == *@(\/|:)* ]]; then
		echo 'host should be without http:// or port - predefined ports will be used'
		exit 1
	fi
        sed -i -r "s|^(ochem.root_host).*localhost.*|\1 = http://$NAME:8080|" env/ochemenv/cfg/ochem.cfg # visible web site, unless was manualy changed
fi

# Getting name for remote calculations (if any)
OCHEM=$(sed -n 's/ochem.root_host =*\(.*\)/\1/p' env/ochemenv/cfg/ochem.cfg) # actual web site
OCHEM=${OCHEM//[[:blank:]]/}
sed -i -r "s|(ochemURL).*|\1>$OCHEM</\1>|" env/ochemenv/cfg/version-template.xml # synchronize unless previously modified

for i in servers/cpu/*.xml ; do
    [[ -f "$i" ]] || continue
    sed -i -r "s|amd64|$ARCH|" "$i"
done

for i in servers/gpu/*.xml ; do
    [[ -f "$i" ]] || continue
    sed -i -r "s|amd64|$ARCH|" "$i"
done

chmod +x env/*
chmod +x servers/*
cd env
ln -sfn ochemenv/tmp/ochem-tomcat/logs/catalina.out tomcat.out
ln -sfn ochemenv/tmp/metaserver-tomcat/logs/catalina.out meta.out
mkdir -p /ochem/tmp/cs_release/
cp -r ochemenv/. /ochem/
ln -sfn /etc/ochem ochemenv/tmp

echo "Awaiting for mariadb start"

sleep 60

echo "Unpacking and starting tomcats..."
bash /etc/source/ochem/bin/startochem.sh

echo "Waiting for online status..."
sleep 15

echo "Unpacking and starting servers..."
cd ../servers
SERVER=$OCHEM_SERVER
SERVERPATH=`pwd`

SERVER=${SERVER//\/}
DIRECTORY=$SERVERPATH/$SERVER/runs
mkdir -p $DIRECTORY
mkdir -p /etc/cs_servers

META=http://localhost:7080/metaserver/ # default for local servers

rm -rf $DIRECTORY
   rm -f $SERVERPATH/$SERVER/release.zip
   wget -q -O /etc/ochem/release.zip $META/update
   ln -sfn /etc/cs_servers $DIRECTORY/servers
   ln -sfn /tmp $DIRECTORY/tmp
   ln -sfn $SERVERPATH/$SERVER/* /etc/ochem

   bash /etc/source/ochem/bin/startservers.sh
   echo "started $SERVER"

echo "All Done!"
cd ${OCHEM_HOME}

tail -f openochem/env/tomcat.out -f openochem/env/meta.out