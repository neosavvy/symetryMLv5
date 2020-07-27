package com.sml.examples.scala

/**
 * To run via spark-shell localy for example:
 *
 * $SPARK_HOME/bin/spark-shell --master local[1] --jars /opt/symetry/lib/sym-spark-assembly.jar
 *
 * <$SPARK_HOME is where the Apache Spark is located, usually in /opt/spark>
 */
/**
 * Created by neil on 15-11-19.
 */

import com.rtlm.constants.CoreConstants._
import com.rtlm.json.DataFrame
import com.rtlm.util.AttributeTypes
import com.sml.shell.SparkShellSymetryProject
import com.sml.shell.SymShellConfig
import com.sml.shell.Util._

object IrisS3Example {

  def run(sc:org.apache.spark.SparkContext)={

    println("amazonExample.scala start")

    val awsAccessKeyId           = System.getenv("AWS_ACCESS_KEY")
    val awsSecretAccessKey       = System.getenv("AWS_SECRET_KEY")

    //println("awsAccessKeyId:" + awsAccessKeyId)

    val hadoopConf=sc.hadoopConfiguration
    hadoopConf.set("fs.s3n.awsAccessKeyId", awsAccessKeyId)
    hadoopConf.set("fs.s3n.awsSecretAccessKey", awsSecretAccessKey)

    // 1: Create RDD (Resilient Distributed Data) from your data, Specify Attribute Names, and their Types.
    // the dataset can be on Amazon s3: For example s3n://sml-oregon
    val myrdd  = sc.textFile("s3n://sml-oregon/datasets/susy/SUSYmini.csv")
    val attributeNames = myrdd.first.split(",")         // The first line of CSV file are the name of the attributes
    val attributeTypes = Array(AttributeTypes.TYPE_BINARY)++Array.fill[Char](attributeNames.length-1)(AttributeTypes.TYPE_CONTINOUS)
    /* Specify attribute types:
        AttributeTypes.TYPE_CONTINOUS
        AttributeTypes.TYPE_BINARY
        AttributeTypes.TYPE_STRING
        AttributeTypes.TYPE_LIST
        AttributeTypes.TYPE_IGNORE
        AttributeTypes.TYPE_CATEGORY
     */

    // 2: Create a distributed SymetryProject
    val projectName = "susyExample-TEST"
    val userName    = "c1"
    val projectType = 0 // 0: Using CPU, 11:using GPU
    val p = new SparkShellSymetryProject(
        userName,
        projectName,
        projectType)
    /*
     */
    p.learn(sc, myrdd, attributeNames, attributeTypes) // learn data
    // 3: Explore univariate
    println("Univaraite Exploration of Attribute number 4")
    val stats1 = p.univariate(4) // exploration , see whether the project has been built correctly
    println(stats1)
    p.univariate("lepton-2-pT") // You may pass the name of the attribute as well

    println("Univaraite Exploration of the lepton-2-pT Attribute")
    val stats2 = p.univariate("lepton-2-pT") // You may pass the name of the attribute as well
    println(stats2)
    println("Skewness:")
    println(stats2(UNI_SKEWNESS))

    // 4: PCA
    val eigens1 = p.pca((1 to 18).toArray) // returns a Map containing eigenvalues,eigenvectors

    // or pass the Attributes name
    val eigens2 = p.pca(Array(
      "lepton-1-pT",
      "lepton-1-eta", "lepton-1-phi",
      "lepton-2-pT", "lepton-2-eta"))

    // 5: Build model
    val tar   = Array(0)  // The first Attribute is used as Target
    val input = (1 to 18).toArray // Input attributes

    p.buildModel(input, tar, "lsvm", "mySvmModel")       // The Model is now built

    // 6: make predictions (There are different ways to make prediction

    // One row of data (Attributes are comma separated);
    // Here in this model, the first Attribute is the Target,
    // we put an arbitrary value as it will be ignored in the prediction

    val test1 = Array("-1","1.667973", "0.0641906","-1.2251714",
      "0.506102", "-0.3389389", "1.6725428",
      "3.475464", "-1.2191363", "0.0129545",
      "3.775173", "1.0459771",  "0.5680512",
      "0.481928", "0.0000000", "0.4484102",
      "0.205355", "1.3218934", "0.3775840")
    val df1 = new DataFrame
    df1.setAttributeNames(attributeNames)
    df1.setAttributeTypes(attributeTypes)
    df1.addTuple(test1)
    val res1     = p.predict(df1,"mySvmModel")
    println("Predicted response:" + res1(0))  // should be 1

    // The response for this example should be 0
    val test2 = Array("-1","1.001869","-0.471788","0.555614",
      "1.233368","1.255548","-1.052491",
      "0.437615","-1.333052","0.326858",
      "-0.111678","1.435708","0.755201",
      "0.466779","0.454541","1.446331",
      "0.592259","1.325197","0.083014")

    df1.clear
    df1.addTuple(test2)
    val res2     = p.predict(df1,"mySvmModel")
    println("Predicted response:" + res2(0)) // should be 0

    // :7 You can delete the Model to release the used memory
    p.deleteModel("mySvmModel")

    // 8: Tou can delete the Project to release the used memory
    p.deleteProject
    
    println("amazonExample.scala end")
  }
}

// **************************************
// ******** PLEASE RUN: *****************
// IrisS3Example.run(sc)
