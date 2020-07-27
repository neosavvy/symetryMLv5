# To Run PySpark Example in Notebook:

That is the [amazon-s3-example-pyspark.ipynb](amazon-s3-example-pyspark.ipynb) / notebook:

```bash
export PYSPARK_DRIVER_PYTHON="jupyter"
export PYSPARK_DRIVER_PYTHON_OPTS="notebook"

export SPARKHOME=YOUR_SPARK_HOME
export PATH_TO_JARS=${SPARKHOME}/jars

${SPARKHOME}/bin/pyspark --master local[2] --jars /opt/symetry/lib/sym-spark-assembly.jar,${PATH_TO_JARS}/aws-java-sdk-1.7.4.jar,${PATH_TO_JARS}/hadoop-aws-2.7.3.jar,${PATH_TO_JARS}/jets3t-0.9.4.jar --driver-java-options -Dsym.lic.loc=/opt/symetry/sym.lic
```


# To Run The Other Examples in Notebook

```bash
jupyter notebook
```
