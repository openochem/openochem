#!/bin/bash
NAME=$1

if [ -n "$NAME" ]; then
	if [[ "$NAME" == *@(\/|:)* ]]; then
		echo 'host should be without http:// or port - predefined ports will be used'
		exit 1
	fi
	sed -i -r "s|(ochemURL).*|\1>http://$NAME:8080</\1>|" env/ochemenv/cfg/version-template.xml
	sed -i -r "s|(mongoDbURL).*|\1>mongodb://$NAME</\1>|" env/ochemenv/cfg/version-template.xml
	sed -i -r "s|(jdbc:mysql://).*(/struc.*)|\1$NAME\2|" env/ochemenv/cfg/version-template.xml
	
	sed -i -r "s|^(ochem.root_host).*|\1 = http://$NAME:8080|" env/ochemenv/cfg/ochem.cfg
	sed -i -r "s|^(metaserver.default_url).*|\1 = http://$NAME:7080/metaserver|"  env/ochemenv/cfg/ochem.cfg
	sed -i -r "s|^(mongodb.host).*|\1 = mongodb://$NAME|" env/ochemenv/cfg/ochem.cfg

	sed -i -r "s|^(mongodb.host).*|\1 = mongodb://$NAME|" env/ochemenv/cfg/metaserver.cfg
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

bash setup.sh

cd env
bash stop.sh
bash start.sh


