## Spark Application - execute with spark-submit：
#spark-submit app.py
## Imports
from pyspark import SparkConf, SparkContext
#from pyspark import mllib.linalg.Vector
#from pyspark import mllib.stat.{MultivariateStatisticalSummary, Statistics}
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.linalg import Vector
from pyspark.mllib.stat import MultivariateStatisticalSummary
from pyspark.mllib.stat import Statistics
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import LogisticRegressionWithLBFGS
from pyspark.mllib.classification import LogisticRegressionModel
from pyspark.mllib.tree                import DecisionTree
from pyspark.mllib.evaluation      import MulticlassMetrics

## Module Constants
APP_NAME = "PregnantApplication"

def tokenize(item):
    label_text = item.split(",")[9]
    label = float(label_text) -1
    wife_age_text = item.split(",")[0]
    wife_edu_text = item.split(",")[1]
    if wife_edu_text == "1":
        feature_wife_edu = 0
    else:
        feature_wife_edu = 1
        hus_edu_text = item.split(",")[2]
        if hus_edu_text == "1":
            feature_hus_edu = 0
        else:
            feature_hus_edu = 1
            feature_kid= item.split(",")[3]
            feature_text5= item.split(",")[4]
            if feature_text5 == "1":
                islam = 0
            else:
                islam = 1
                work = item.split(",")[5]
                feature_occ= item.split(",")[6]
                if feature_occ == "1":
                    occ = 0
                else:
                    occ = 1
                    feature_index= item.split(",")[7]
                    if feature_index == "1":
                        index = 0
                        else:
                            index = 1
                            feature_social= item.split(",")[8]
                            vector = Vectors.dense(float(wife_age_text),
                                                   float(feature_wife_edu),
                                                   float(feature_hus_edu),
                                                   float(feature_kid),
                                                   float(islam),
                                                   float(work),
                                                   float(occ),
                                                   float(index),
                                                   float(feature_social))
                            item = LabeledPoint(label,vector)
                            return item
## Main functionality

def main(sc):
    biyun_lines = sc.textFile("cmc.data")
    #print(biyun_lines.collect())
    biyun_points = biyun_lines.map(lambda item:tokenize(item))
    print(biyun_points)
    splits = biyun_points.randomSplit([0.922, 0.078],11)
    training = splits[0]
    testing = splits[1]
    print(training.count())
    print(testing.count())
    model = DecisionTree.trainClassifier(
        training,
        numClasses=3,
        categoricalFeaturesInfo={},
        impurity='gini', maxDepth=10, maxBins=20)
    print(model)
    print('Learned classification tree model:')
    print(model.toDebugString())
# Evaluate model on test instances and compute test error
predictions = model.predict(testing.map(lambda x: x.features))
labelsAndPredictions = testing.map(lambda lp: lp.label).zip(predictions)
testErr = labelsAndPredictions.filter(lambda lp: lp[0] != lp[1]).count() / float(testing.count())
print('Test Error = ' + str(testErr))
predictionList = []
for item in testing.collect():
    predictionList.append([model.predict(item.features),item.label])
    predictionAndLabels = sc.parallelize(predictionList)
    for pitem in predictionAndLabels.collect():
        print(pitem)
        metrics = MulticlassMetrics(predictionAndLabels)
        print("Accurancy:"+str(metrics.accuracy))
        print("ConfusionMatrix")
        print(metrics.confusionMatrix())
"""
Print out the Accurancy
"""
sc.stop()
if __name__ == "__main__":
    # Configure Spark
    conf = SparkConf().setAppName(APP_NAME)
    conf = conf.setMaster("local[*]")
    sc   = SparkContext(conf=conf)
    # Execute Main functionality
    main(sc)
