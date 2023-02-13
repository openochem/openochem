#!/bin/bash

set -u
set +e

export OCHEM_HOSTNAME=${OCHEM_HOSTNAME:-localhost}
echo "Starting ochem on ${OCHEM_HOSTNAME}."
export OCHEM_SERVER=${OCHEM_SERVER:-ochem}

# prepare files
echo "Preparing files..."
cd openochem

NAME=${OCHEM_HOSTNAME}
ARCH=`dpkg --print-architecture`

if [ -n "$NAME" ]; then
	if [[ "$NAME" == *@(\/|:)* ]]; then
		echo 'host should be without http:// or port - predefined ports will be used'
		exit 1
	fi
	sed -i -r "s|(ochemURL).*|\1>http://$NAME:8080</\1>|" env/ochemenv/cfg/version-template.xml
	sed -i -r "s|(mongoDbURL).*|\1>mongodb://ochem-mongo</\1>|" env/ochemenv/cfg/version-template.xml
	sed -i -r "s|(jdbc:mariadb://).*(/struc.*)|\1ochem-mariadb\2|" env/ochemenv/cfg/version-template.xml

	sed -i -r "s|^(ochem.root_host).*|\1 = http://$NAME:8080|" env/ochemenv/cfg/ochem.cfg
	sed -i -r "s|^(metaserver.default_url).*|\1 = http://$NAME:7080/metaserver|"  env/ochemenv/cfg/ochem.cfg
	sed -i -r "s|^(mongodb.host).*|\1 = mongodb://ochem-mongo|" env/ochemenv/cfg/ochem.cfg

	sed -i -r "s|^(mongodb.host).*|\1 = mongodb://ochem-mongo|" env/ochemenv/cfg/metaserver.cfg
	sed -i -r "s|^(WEB).*|\1 = http://$NAME:7080/metaserver|" env/ochemenv/cfg/metaserver.cfg
	sed -i -r "s|^(WEB).*|\1=http://$NAME:7080/metaserver|" servers/start.sh

	for i in servers/ochem/*.xml ; do
		[[ -f "$i" ]] || continue
		sed -i -r "s|amd64|$ARCH|" "$i"
		sed -i -r "s|(metaserverURL).*|\1>http://$NAME:7080/metaserver</\1>|" "$i"
		sed -i -r "s|(ochemURL).*|\1>http://$NAME:8080</\1>|" "$i"
	done

	for i in servers/gpu/*.xml ; do
		[[ -f "$i" ]] || continue
		sed -i -r "s|amd64|$ARCH|" "$i"
		sed -i -r "s|(metaserverURL).*|\1>http://$NAME:7080/metaserver</\1>|" "$i"
		sed -i -r "s|(ochemURL).*|\1>http://$NAME:8080</\1>|" "$i"
	done

fi

chmod +x env/*
chmod +x servers/*
cd env
ln -sfn ochemenv/tmp/ochem-tomcat/logs/catalina.out tomcat.out
ln -sfn ochemenv/tmp/metaserver-tomcat/logs/catalina.out meta.out
mkdir -p /ochem/tmp/cs_release/
cp -r ochemenv/. /ochem/
ln -sfn /etc/ochem ochemenv/tmp

echo "Done."

echo "Unpacking and starting tomcats..."
sh /etc/source/ochem/bin/startochem.sh METAMEMORY=2048 OCHEMEMORY=4096

echo "Waiting for online status..."
sleep 15

echo "Unpacking and starting servers..."
cd ../servers
SERVER=$OCHEM_SERVER
SERVERPATH=`pwd`
WEB=http://$NAME:7080/metaserver/

SERVER=${SERVER//\/}
DIRECTORY=$SERVERPATH/$SERVER/runs
mkdir -p $DIRECTORY
mkdir -p /etc/cs_servers

rm -rf $DIRECTORY
   rm -f $SERVERPATH/$SERVER/release.zip
   wget -O /etc/ochem/release.zip $WEB/update
   ln -sfn /etc/cs_servers $DIRECTORY/servers
   ln -sfn /tmp $DIRECTORY/tmp
   ln -sfn $SERVERPATH/$SERVER/* /etc/ochem

   sh /etc/source/ochem/bin/startservers.sh
   echo "started $SERVER"

echo "All Done!"
cd ${OCHEM_HOME}

tail -f openochem/env/tomcat.out -f openochem/env/meta.out

