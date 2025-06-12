Tutorial: Installation and Running of Programs
(the following are the required steps for installation and running of the python programs on the Hadoop cluster. 
 In step 7 specify the file-name of the program you wish to run.)
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------


STEPS:
-------------------
* Login to ECS server and SSH into a cluster node: "ssh co246a-1" 
* Place .jar files for Spark and Hadoop libraries (spark 3.5.1, hadoop 3.3.6) in local folder
* Set up environment variables in a ‘SetupSparkClasspath.csh’ file:
------------------------------------------------------------------------------
export HADOOP_VERSION=3.3.6
export HADOOP_HOME=/local/Hadoop/hadoop-$HADOOP_VERSION
export SPARK_HOME=/local/spark
export HADOOP_CONF_DIR=$HADOOP_HOME/etc/hadoop
export YARN_CONF_DIR=$HADOOP_HOME/etc/hadoop
export JAVA_HOME="/usr/pkg/java/sun-8"
export PATH=${PATH}:$JAVA_HOME:$HADOOP_HOME/bin:$SPARK_HOME/bin
export LD_LIBRARY_PATH=$HADOOP_HOME/lib/native:$JAVA_HOME/jre/lib/amd64/server
-------------------------------------------------------------------------------

* Run the following commands: “source SetupSparkClasspath.csh”, “need java8”
* Create your directory in the hadoop cluster: “hdfs dfs -mkdir /user/{username}/input”
* Move input file (data file) into ‘input’ directory within Hadoop: “hdfs dfs -put…”   
* Finally, submit your python program (eg. DT.py) to be run on the Hadoop cluster: “spark-submit --master yarn --py-files DT.py DT.py”
* Program results will be outputted
