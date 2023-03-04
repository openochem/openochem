
#Assumes that insert are done under sudo account, otherwise uncomment below and provide root password instead of xxxx

gunzip *.gz

#ROOT=-uroot -pxxxx
ROOT=

mysql $ROOT < sqlsetup.sql
mysql $ROOT metaserver_demo < metaserver_demo.sql 
mysql $ROOT ochem_demo < ochem_demo.sql 
mysql $ROOT ochem_demo -e "drop table flyway_schema_history"
mysql $ROOT fragment_demo < fragment_demo.sql 
mysql $ROOT structures_demo < structures_demo.sql 
