# requires uroot account

create database metaserver_demo;
create database ochem_demo;
create database structures_demo;
create database fragment_demo;

CREATE USER 'ochem'@'%' IDENTIFIED BY 'ochem_db_password';

grant ALL PRIVILEGES on metaserver_demo.* to "ochem"@"%";
grant ALL PRIVILEGES on ochem_demo.* to "ochem"@"%";
grant ALL PRIVILEGES on structures_demo.* to "ochem"@"%";
grant ALL PRIVILEGES on fragment_demo.* to "ochem"@"%";
