from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import sys
import os

def prepare_data(input_data):
    input_data = input_data.toDF(*[col.strip('"') for col in input_data.columns])
    label_column = 'quality'

    indexer = StringIndexer(inputCol=label_column, outputCol="label")
    input_data = indexer.fit(input_data).transform(input_data)

    feature_columns = [col for col in input_data.columns if col not in [label_column, "label"]]
    assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
    return assembler.transform(input_data)

def predict_using_model(test_path, model_path):
    spark = SparkSession.builder.appName("WineQualityPredictor").getOrCreate()

    test_raw = spark.read.csv(test_path, header=True, inferSchema=True, sep=";")
    test_data = prepare_data(test_raw)

    model = PipelineModel.load(os.path.join(os.getcwd(), model_path))
    predictions = model.transform(test_data)

    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")

    print(f"Test Accuracy: {evaluator.evaluate(predictions, {evaluator.metricName: 'accuracy'})}")
    print(f"Test F1 Score: {evaluator.evaluate(predictions)}")

    spark.stop()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(1)

    predict_using_model(sys.argv[1], sys.argv[2])
