from pyspark import SparkContext
from pyspark import SparkConf
import numpy as np
from pandas import read_csv
from SMLProjectUtil import update_sml_project

print("pyspark-irisExample.py start")

# Comment out these lines if this script is being run interactively through pyspark shell. The shell provides the SparkContext.
conf = SparkConf()
conf.setAppName('pysparkIrisExample')
sc = SparkContext(conf=conf)

# Get the gateway to the JVM through the Spark Context.
gateway = sc._gateway

# Create short name.
sml = gateway.jvm.com.sml.shell

# Create a local, unpersisted SML project that is wrapped by a Scala shell.
projectType = 0 # cpu project
p = sml.PySparkShellSymetryProject("c1", "pysparkIrisExamplePrj", projectType)

# Learn the Iris dataset file incrementally by chunks of lines.
IRIS_FILE = "../../data/irisFiles/Iris_data.csv"
LINE_CHUNK = 50
lineCount = 0
attrNames = None
attrTypes = []
attrTotal = None
dfpIter = read_csv(IRIS_FILE, iterator=True, chunksize=LINE_CHUNK)
for dfp in dfpIter:
    if lineCount == 0:
        attrNames = dfp.columns.values.tolist()
        attrTotal = len(attrNames)

    update_sml_project(p, dfp, sml, attrTypes)

    lineCount += dfp.size / attrTotal
    print("p.learn() %d" % lineCount)

    # Verification
    print(attrTotal)
    m0 = p.univariate(0)
    mLast = p.univariate(attrTotal - 1)
    print(m0)
    print(mLast)


n = attrNames
print(", ".join(n))


t = attrTypes
print(", ".join(t))


# Exploring the data
m1 = p.univariate(7)                      # exploration , to check whether the project has been built correctly
m2 = p.univariate("sepal_width_b2")       # You may pass the name of the attribute as well
print(m1)
print(m2)
print(m1["skewness"])
print(m2["skewness"])


print(len(attrNames))


# Measure some univariate statistics
x = range(1, len(attrNames))
attrVariance = [p.univariate(i-1)["variance"] for i in x]
attrMean = [p.univariate(i-1)["mean"] for i in x]


b1 = p.bivariate("sepal_length", "petal_length")  # calculates two bivariate statistics
print(b1["linCorr"])
print(b1["covar"])


# calculate the pairwise correlation coefficients
linCorr = []
for attr1 in attrNames:
    temp = []
    for attr2 in attrNames:
        b1 = p.bivariate(attr1, attr2)  # calculates two bivariate statistics
        temp.append(b1["linCorr"])
    linCorr.append(temp)

zres = p.ztest("sepal_length", "sepal_width_b1", "sepal_width_b2")
print(zres["z"])
print(zres["zp"])


e = p.pca([0, 1, 3])  # returns a tuple[eigenvalues,eigenvectors]
print(e["EigenValues"])                        # eigen-values
print(e["EigenVectors"])                       # eigen-vectors


# Build model
tar = [13]                              # The forth Attribute is used as Target
input = range(0, 4)                    # The first four attribute used as as Input

p.buildModel(input, tar, "lda", "irisLDAModel")      # The Model is know built


col = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

types = ["C", "C", "C", "C"]
df = sml.PyDataFrame()

df.setAttributeNames(col)
df.setAttributeTypes(types)
d = ["4.9", "2.4", "3.3", "1"]
df.addTuple(d)

results1 = p.predict(df, "irisLDAModel")
# should be one, that is true
print(results1["res"])


df.clear()
d2 = ["4.3", "3", "1.0", "0.1"]
df.addTuple(d2)
results2 = p.predict(df, "irisLDAModel")
# should be 0, false
print(results2["res"])


# Delete the Model
p.deleteModel("irisLDAModel")

# Delete the Project
p.deleteProject()

print("pyspark-irisExample.py end")

