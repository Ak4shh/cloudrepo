from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression, DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler
import sys
import os

def prepare_data(input_data):
    # Standardize column names
    input_data = input_data.toDF(*[col.strip('"') for col in input_data.columns])

    label_column = 'quality'

    # Index categorical labels
    indexer = StringIndexer(inputCol=label_column, outputCol="label")
    input_data = indexer.fit(input_data).transform(input_data)

    # Select feature columns
    feature_columns = [col for col in input_data.columns if col not in [label_column, "label"]]

    # Assemble feature columns
    assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
    return assembler.transform(input_data)

def train_model(train_path, val_path, model_output):
    # Initialize Spark session
    spark = SparkSession.builder.appName("WineQualityTrainer").getOrCreate()

    # Read data
    train_raw = spark.read.csv(train_path, header=True, inferSchema=True, sep=";")
    val_raw = spark.read.csv(val_path, header=True, inferSchema=True, sep=";")

    train_data = prepare_data(train_raw)
    val_data = prepare_data(val_raw)

    # Define models
    classifiers = [
        RandomForestClassifier(labelCol="label", featuresCol="features"),
        LogisticRegression(labelCol="label", featuresCol="features"),
        DecisionTreeClassifier(labelCol="label", featuresCol="features"),
    ]

    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")

    best_model = None
    best_f1 = 0

    for model in classifiers:
        pipeline = Pipeline(stages=[model])
        param_grid = ParamGridBuilder().addGrid(model.maxDepth if hasattr(model, "maxDepth") else model.numTrees, [5, 10]).build()
        cv = CrossValidator(estimator=pipeline, estimatorParamMaps=param_grid, evaluator=evaluator, numFolds=3)
        cv_model = cv.fit(train_data)
        
        f1 = evaluator.evaluate(cv_model.transform(val_data))
        if f1 > best_f1:
            best_model = cv_model.bestModel
            best_f1 = f1

    best_model.save(os.path.join(os.getcwd(), model_output))
    spark.stop()

if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(1)

    train_model(sys.argv[1], sys.argv[2], sys.argv[3])
