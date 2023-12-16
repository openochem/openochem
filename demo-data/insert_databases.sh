
#Assumes that insert are done under sudo account, otherwise uncomment below and provide root password instead of xxxx

gunzip *.gz

#ROOT=-uroot -pxxxx
ROOT=

mariadb $ROOT < sqlsetup.sql
mariadb $ROOT metaserver_demo < metaserver_demo.sql 
mariadb $ROOT fragment_demo < fragment_demo.sql 
mariadb $ROOT structures_demo < structures_demo.sql 
mariadb $ROOT --max_allowed_packet=1G ochem_demo < ochem_demo.sql 
mariadb $ROOT ochem_demo -e "drop table flyway_schema_history"

