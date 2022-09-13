#!/bin/bash

set -u
set +e

OCHEM_HOSTNAME=${OCHEM_HOSTNAME:-localhost}
echo "Starting ochem on ${OCHEM_HOSTNAME}."

# prepare files
echo "Preparing files..."
cd openochem

NAME=${OCHEM_HOSTNAME}

if [ -n "$NAME" ]; then
	if [[ "$NAME" == *@(\/|:)* ]]; then
		echo 'host should be without http:// or port - predefined ports will be used'
		exit 1
	fi
	sed -i -r "s|(ochemURL).*|\1>http://$NAME:8080</\1>|" env/ochemenv/cfg/version-template.xml
	sed -i -r "s|(mongoDbURL).*|\1>mongodb://ochem-mongo</\1>|" env/ochemenv/cfg/version-template.xml
	sed -i -r "s|(jdbc:mysql://).*(/struc.*)|\1ochem-mariadb\2|" env/ochemenv/cfg/version-template.xml

	sed -i -r "s|^(ochem.root_host).*|\1 = http://$NAME:8080|" env/ochemenv/cfg/ochem.cfg
	sed -i -r "s|^(metaserver.default_url).*|\1 = http://$NAME:7080/metaserver|"  env/ochemenv/cfg/ochem.cfg
	sed -i -r "s|^(mongodb.host).*|\1 = mongodb://ochem-mongo|" env/ochemenv/cfg/ochem.cfg

	sed -i -r "s|^(mongodb.host).*|\1 = mongodb://ochem-mongo|" env/ochemenv/cfg/metaserver.cfg
	sed -i -r "s|^(WEB).*|\1 = http://$NAME:7080/metaserver|" env/ochemenv/cfg/metaserver.cfg
	sed -i -r "s|^(WEB).*|\1=http://$NAME:7080/metaserver|" servers/start.sh

	for i in servers/ochem/*.xml ; do
		[[ -f "$i" ]] || continue
		sed -i -r "s|(metaserverURL).*|\1>http://$NAME:7080/metaserver</\1>|" "$i"
		sed -i -r "s|(ochemURL).*|\1>http://$NAME:8080</\1>|" "$i"
	done

	for i in servers/gpu/*.xml ; do
		[[ -f "$i" ]] || continue
		sed -i -r "s|(metaserverURL).*|\1>http://$NAME:7080/metaserver</\1>|" "$i"
		sed -i -r "s|(ochemURL).*|\1>http://$NAME:8080</\1>|" "$i"
	done

fi

chmod +x openochem/env/*
chmod +x openochem/servers/*
cd openochem/env
ln -sfn ochemenv/tmp/ochem-tomcat/logs/catalina.out tomcat.out
ln -sfn ochemenv/tmp/metaserver-tomcat/logs/catalina.out meta.out
mkdir -p ochem/tmp/cs_release/
cp -r ochemenv/* /ochem/*
ln -sfn /etc/ochem ochemenv/tmp

echo "Done."

echo "Starting tomcats..."
bash /etc/source/ochem/bin/startochem.sh
echo "Done..."
