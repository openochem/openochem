#!/bin/bash

init_db () {
        sleep 5
        chmod +x insert_databases.sh
	./insert_databases.sh && echo "DB Initialized." && ls /var/lib/mysql
        touch /var/lib/mysql/ochem_initialized
}

[[ ! -f /var/lib/mysql/ochem_initialized ]] && init_db &
/usr/local/bin/docker-entrypoint.sh "$@"
