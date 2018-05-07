# spark-shell --packages com.amazonaws:sagemaker-spark_2.11:spark_2.2.0-1.0

# Train, deploy and invoke a standalone SageMaker model from Spark: MNIST (784) --> KMeans (10)

import org.apache.spark.sql.SparkSession
import com.amazonaws.services.sagemaker.sparksdk.{IAMRole,SageMakerResourceCleanup}
import com.amazonaws.services.sagemaker.sparksdk.algorithms.KMeansSageMakerEstimator
import com.amazonaws.services.sagemaker.AmazonSageMakerClientBuilder

val spark = SparkSession.builder.getOrCreate

val region = "us-east-1"
val trainingData = spark.read.format("libsvm").option("numFeatures", "784").load(s"s3://sagemaker-sample-data-$region/spark/mnist/train/")
trainingData.count
trainingData.take(1)

val testData = spark.read.format("libsvm").option("numFeatures", "784").load(s"s3://sagemaker-sample-data-$region/spark/mnist/test/")

val roleArn = "arn:aws:iam::ACCOUNT_NUMBER:role/ROLE_NAME"

val estimator = new KMeansSageMakerEstimator(sagemakerRole = IAMRole(roleArn), trainingInstanceType = "ml.p2.xlarge", trainingInstanceCount = 1, endpointInstanceType = "ml.c4.xlarge", endpointInitialInstanceCount = 1).setK(10).setFeatureDim(784)

val model = estimator.fit(trainingData)

val transformedData = model.transform(testData)

val sagemakerClient = AmazonSageMakerClientBuilder.defaultClient
val cleanup = new SageMakerResourceCleanup(sagemakerClient)
cleanup.deleteResources(model.getCreatedResources)
