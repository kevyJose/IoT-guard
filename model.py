### Multi-class classification using Random Forest Classifier on IoT data
### Evaluation technique: F1 Score (with Cross-Validation)

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, monotonically_increasing_id
from pyspark.ml.feature import StringIndexer, VectorAssembler, MinMaxScaler, PCA
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
import numpy as np
import time


# Initialize Spark session
spark = SparkSession.builder.appName("IoT Intrusion Program (cross-validation)").getOrCreate()


# Load data
df = spark.read.csv('individual/data/IoT-data.csv', header=True, inferSchema=True)
# print(df.show(20))

# PREPROCESSING
#----------------------------------------------------
print("\nPREPROCESSING...")

# random sampling (10% sample)
df = df.sample(fraction=0.1, seed=42)

# X and y
X = df.select(df.columns[:-1])
y = df.select(df.columns[-1])

# label encoding
indexer = StringIndexer(inputCol="label", outputCol="labelIndex")
y = indexer.fit(y).transform(y)
y = y.select("labelIndex")
# print('num of unique vals:  ', y.select("labelIndex").distinct().count())

# convert all columns to double-type
X = X.select([col(column).cast("double").alias(column) for column in X.columns])

# feature assembly
assembler = VectorAssembler(inputCols=X.columns, outputCol="features")
X = assembler.transform(X)

# PCA feature construction
pca = PCA(k=30, inputCol="features", outputCol="pca_features")
X = pca.fit(X).transform(X)
X = X.select("pca_features")
# X.show(10)

# normalise the features
scaler = MinMaxScaler(inputCol="pca_features", outputCol="scaledFeatures")
X = scaler.fit(X).transform(X)
X = X.select("scaledFeatures")

# Add monotonically increasing id to both X and y
X = X.withColumn("id", monotonically_increasing_id())
y = y.withColumn("id", monotonically_increasing_id())

# append y to X
data = X.join(y, "id").select(col("scaledFeatures").alias("features"), col("labelIndex").alias("label"))
# data.show(10)
#----------------------------------------------------



# RUNNING
print("\nRUNNING...")

# Initialise evaluator
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")

# Create RandomForest model
rf = RandomForestClassifier(featuresCol="features", labelCol="label")

# Create a ParamGridBuilder for hyperparameter tuning
paramGrid = ParamGridBuilder().addGrid(rf.numTrees, [10, 20]).build()

# Create CrossValidator
crossval = CrossValidator(estimator=rf,
                          estimatorParamMaps=paramGrid,
                          evaluator=evaluator,
                          numFolds=3)

# Track total running time
start_time = time.time()

# Split data
train_df, test_df = data.randomSplit([0.8, 0.2], seed=42)

# Run cross-validation and choose the best set of parameters
cv_model = crossval.fit(train_df)

# Make predictions
train_preds = cv_model.transform(train_df)
test_preds = cv_model.transform(test_df)

# Evaluate the model on training data
train_accuracy = evaluator.evaluate(train_preds)

# Evaluate the model on test data
test_accuracy = evaluator.evaluate(test_preds)

# Calculate total running time
end_time = time.time()
total_runtime = end_time - start_time

print(f"\n\n----------------------------------------------------")
print(f"RESULTS from Random-Forest Classifier (with cross-validation):")
print(f"\nTrain Accuracy: {train_accuracy}")
print(f"Test Accuracy: {test_accuracy}")
print(f"\nTotal running time of the whole program: {total_runtime:.2f} seconds")

# Stop the Spark session
spark.stop()