-- MariaDB dump 10.19-11.3.2-MariaDB, for debian-linux-gnu (x86_64)
--
-- Host: localhost    Database: metaserver_demo
-- ------------------------------------------------------
-- Server version	11.3.2-MariaDB-1:11.3.2+maria~ubu2204

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Table structure for table `ArchivedTask`
--

DROP TABLE IF EXISTS `ArchivedTask`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `ArchivedTask` (
  `task_id` int(10) unsigned NOT NULL AUTO_INCREMENT,
  `reference_id` char(72) DEFAULT NULL,
  `configuration` longblob DEFAULT NULL,
  `data` longblob DEFAULT NULL,
  `status` varchar(20) DEFAULT NULL,
  `result` longblob DEFAULT NULL,
  `task_type` varchar(50) NOT NULL,
  `calc_server_id` varchar(60) DEFAULT NULL,
  `detailed_status` longtext DEFAULT NULL,
  `priority` double DEFAULT 0,
  `time` timestamp NULL DEFAULT NULL,
  `parent_task_id` int(10) unsigned DEFAULT NULL,
  `time_assigned` timestamp NULL DEFAULT NULL,
  `time_completed` timestamp NULL DEFAULT NULL,
  `datarows` int(11) DEFAULT NULL,
  `cols` int(11) DEFAULT NULL,
  `client` varchar(100) DEFAULT NULL,
  `task_name` varchar(100) DEFAULT NULL,
  `user` varchar(50) DEFAULT NULL,
  `peak_memory_usage` int(11) DEFAULT NULL,
  `scheduled_kill` tinyint(1) NOT NULL DEFAULT 0,
  `last_access` timestamp NULL DEFAULT NULL,
  `task_md5` char(32) DEFAULT NULL,
  `ref_count` int(11) NOT NULL DEFAULT 0,
  `debug` tinyint(1) NOT NULL DEFAULT 0,
  `preferred_server` varchar(255) DEFAULT NULL,
  `min_required_memory` int(10) DEFAULT NULL,
  `resubmitted` int(11) DEFAULT NULL,
  `nonzero` int(11) DEFAULT NULL,
  PRIMARY KEY (`task_id`),
  KEY `TaskTypeIndex` (`task_type`),
  KEY `ParentTaskIdIndex` (`parent_task_id`),
  KEY `ID_TIME_COMPLETED` (`time_completed`),
  KEY `CompletedTaskId` (`time_completed`,`task_id`),
  KEY `ID_Priority` (`priority`),
  KEY `StatTypePri_Index` (`status`,`task_type`,`priority`),
  KEY `ScheduledKillId` (`scheduled_kill`,`last_access`),
  KEY `MD5_Id` (`task_md5`)
) ENGINE=MyISAM DEFAULT CHARSET=latin1 COLLATE=latin1_swedish_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `ArchivedTask`
--

LOCK TABLES `ArchivedTask` WRITE;
/*!40000 ALTER TABLE `ArchivedTask` DISABLE KEYS */;
/*!40000 ALTER TABLE `ArchivedTask` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `StatisticsLog`
--

DROP TABLE IF EXISTS `StatisticsLog`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `StatisticsLog` (
  `log_id` int(11) NOT NULL AUTO_INCREMENT,
  `date` timestamp NOT NULL DEFAULT current_timestamp() ON UPDATE current_timestamp(),
  `assigned_tasks` int(11) DEFAULT NULL,
  `online_servers` int(11) DEFAULT NULL,
  `connections_per_second` int(11) DEFAULT NULL,
  `memory_used` int(11) DEFAULT NULL,
  `lost_connections_per_second` int(11) DEFAULT NULL,
  `new_tasks` int(11) DEFAULT NULL,
  `completed_tasks` int(11) DEFAULT NULL,
  `errors` int(11) DEFAULT NULL,
  `tasks_in_queue` int(11) DEFAULT NULL,
  PRIMARY KEY (`log_id`)
) ENGINE=MyISAM AUTO_INCREMENT=6 DEFAULT CHARSET=latin1 COLLATE=latin1_swedish_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `StatisticsLog`
--

LOCK TABLES `StatisticsLog` WRITE;
/*!40000 ALTER TABLE `StatisticsLog` DISABLE KEYS */;
INSERT INTO `StatisticsLog` VALUES
(1,'2024-04-24 15:21:16',0,NULL,0,NULL,0,0,0,0,NULL),
(2,'2024-04-24 15:22:16',0,NULL,0,NULL,0,0,0,0,NULL),
(3,'2024-04-24 15:23:16',0,NULL,0,NULL,0,0,0,0,NULL),
(4,'2024-04-24 15:24:16',0,NULL,0,NULL,0,0,0,0,NULL),
(5,'2024-04-24 15:25:16',0,NULL,0,NULL,0,0,0,0,NULL);
/*!40000 ALTER TABLE `StatisticsLog` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `Task`
--

DROP TABLE IF EXISTS `Task`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `Task` (
  `task_id` int(10) unsigned NOT NULL AUTO_INCREMENT,
  `reference_id` char(72) DEFAULT NULL,
  `configuration` longblob DEFAULT NULL,
  `data` longblob DEFAULT NULL,
  `status` varchar(20) DEFAULT NULL,
  `result` longblob DEFAULT NULL,
  `task_type` varchar(50) NOT NULL,
  `calc_server_id` varchar(60) DEFAULT NULL,
  `detailed_status` longtext DEFAULT NULL,
  `priority` double DEFAULT 0,
  `time` timestamp NULL DEFAULT NULL,
  `parent_task_id` int(10) unsigned DEFAULT NULL,
  `time_assigned` timestamp NULL DEFAULT NULL,
  `time_completed` timestamp NULL DEFAULT NULL,
  `datarows` int(11) DEFAULT NULL,
  `cols` int(11) DEFAULT NULL,
  `client` varchar(100) DEFAULT NULL,
  `task_name` varchar(100) DEFAULT NULL,
  `user` varchar(50) DEFAULT NULL,
  `peak_memory_usage` int(11) DEFAULT NULL,
  `scheduled_kill` tinyint(1) NOT NULL DEFAULT 0,
  `last_access` timestamp NULL DEFAULT NULL,
  `task_md5` char(32) DEFAULT NULL,
  `ref_count` int(11) NOT NULL DEFAULT 0,
  `debug` tinyint(1) NOT NULL DEFAULT 0,
  `preferred_server` varchar(255) DEFAULT NULL,
  `min_required_memory` int(10) DEFAULT NULL,
  `resubmitted` int(11) DEFAULT NULL,
  `nonzero` int(11) DEFAULT NULL,
  PRIMARY KEY (`task_id`),
  KEY `TaskTypeIndex` (`task_type`),
  KEY `ParentTaskIdIndex` (`parent_task_id`),
  KEY `ID_TIME_COMPLETED` (`time_completed`),
  KEY `CompletedTaskId` (`time_completed`,`task_id`),
  KEY `ID_Priority` (`priority`),
  KEY `StatTypePri_Index` (`status`,`task_type`,`priority`),
  KEY `ScheduledKillId` (`scheduled_kill`,`last_access`),
  KEY `MD5_Id` (`task_md5`)
) ENGINE=MyISAM DEFAULT CHARSET=latin1 COLLATE=latin1_swedish_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `Task`
--

LOCK TABLES `Task` WRITE;
/*!40000 ALTER TABLE `Task` DISABLE KEYS */;
/*!40000 ALTER TABLE `Task` ENABLE KEYS */;
UNLOCK TABLES;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2024-04-24 15:25:50
