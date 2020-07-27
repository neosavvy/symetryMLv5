

/**
 *
 * $SPARK_HOME/bin/spark-shell --master local[1] --jars /opt/symetry/lib/sym-spark-assembly.jar 
 *
 *  <$SPARK_HOME is where the Apache Spark is located, usually in /opt/spark>
 */

import com.rtlm.constants.CoreConstants
import com.rtlm.json.DataFrame
import com.rtlm.util.AttributeTypes
import com.sml.shell.SparkShellSymetryProject
import com.sml.shell.Util._

object IrisExample {

  /**
   * Create a local Project, which does not need a SPARK cluster. The Project learns the iris dataset
   */
  def run(path:String="../../data/Iris_data.csv") = {
    println("Run Iris Example Start...")

    // 1) create a dataframe
    val irisDataFrame = readCSV(path)
    println(irisDataFrame)
    irisDataFrame.get(1)   // show the first row

    // 2) Create a local SymetryProject and learn DataFrame
    val projectName = "irisExample"
    val userName    = "c1"
    val projectType = 0 // 0: using CPU, 11: using GPU
    val p = new SparkShellSymetryProject(userName,projectName, projectType)
    p.learn(irisDataFrame)

    // 3) a) Exploring the data, b) PCA, c) Building model, d) prediction

    // Univariate statistics
    val m1 = p.univariate(7) // exploration , to check whether the project has been built correctly
    val m2 = p.univariate("sepal_width_b2") // You may pass the name of the attribute as well
    println("skew:" + m1(CoreConstants.UNI_SKEWNESS))  // available

    // Bivariate statistics
    val b1 = p.bivariate("sepal_length","petal_length") //calculates two bivariate statistics
    println("covar:" +b1(CoreConstants.BI_COVARIANCE))
    println("linnCor:" +b1(CoreConstants.BI_LINCORR))

    // Hypothesis Testing
    /* performing a z test  between 1) sepal_length | sepal_width_b1 = 1 
            and  2) sepal_length | sepal_width_b2 = 1 */
    val zres = p.ztest("sepal_length","sepal_width_b1","sepal_width_b2")
    println("ztest z:" +zres(CoreConstants.ZTEST_z))
    println("ztest p:" +zres(CoreConstants.ZTEST_p))

    // PCA
    val eigens1 = p.pca(Array(0,1,3)) // returns a Map containing eigenvalues,eigenvectors

    // 5) Build model
    val tar   = Array(6)  // The seventh Attribute is used as target
    val input = (0 to 3).toArray   // The first four attributes are used as input

    p.buildModel(input,tar,"lda","irisLDAModel")  // Build model

    // 6) Make a prediction
    val col = Array("sepal_length",	"sepal_width",
      "petal_length",	"petal_width")

    val types = Array(
        AttributeTypes.TYPE_CONTINOUS,
        AttributeTypes.TYPE_CONTINOUS,
        AttributeTypes.TYPE_CONTINOUS,
        AttributeTypes.TYPE_CONTINOUS)

    val df = new DataFrame

    df.setAttributeNames(col)
    df.setAttributeTypes(types)

    val d = Array("4.8", "3", "1.4", "0.1")
    df.addTuple(d)

    val res = p.predict(df,"irisLDAModel")
    println("Predicted Response:" + res(0)). // should be 0

    // another prediction test should return 1
    df.clear()
    val d2   = Array("1.2", "0", "1.4", "0.1")
    df.addTuple(d2)
    val res2 = p.predict(df,"irisLDAModel")
    println("Predicted Response:" + res2(0)) // should be 1

    // STEP 6) Delete the Model
    p.deleteModel("irisLDAModel")

    // STEP 7) Delete the Project
    p.deleteProject

    println("RunIrisExample done")
  }
}
// *******************************
// ******** PLEASE RUN: *****************
/* irisExample.run() //or irisExample.run( << an string containing 
    the the path of the iris data >> ) */

