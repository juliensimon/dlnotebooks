# spark-shell --packages com.amazonaws:sagemaker-spark_2.11:spark_2.2.0-1.0

# Train, deploy and invoke a SageMaker model from a Spark pipeline : MNIST (784) --> PCA (50) --> KMeans (10)

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.PCA
import org.apache.spark.sql.SparkSession
import com.amazonaws.services.sagemaker.{AmazonSageMaker, AmazonSageMakerClientBuilder}
import com.amazonaws.services.sagemaker.sparksdk.{IAMRole, SageMakerResourceCleanup
import com.amazonaws.services.sagemaker.sparksdk.algorithms.KMeansSageMakerEstimator
import com.amazonaws.services.sagemaker.sparksdk.transformation.serializers.ProtobufRequestRowSerializer

val spark = SparkSession.builder.getOrCreate

val region = "us-east-1"
val trainingData = spark.read.format("libsvm").option("numFeatures", "784").load(s"s3://sagemaker-sample-data-$region/spark/mnist/train/")
val testData = spark.read.format("libsvm").option("numFeatures", "784").load(s"s3://sagemaker-sample-data-$region/spark/mnist/test/")

val roleArn = "arn:aws:iam::ACCOUNT_NUMBER:role/ROLE_NAME"

val pcaEstimator = new PCA().setInputCol("features").setOutputCol("projectedFeatures").setK(50)

val kMeansSageMakerEstimator = new KMeansSageMakerEstimator(sagemakerRole = IAMRole(roleArn), requestRowSerializer = new ProtobufRequestRowSerializer(featuresColumnName = "projectedFeatures"), trainingSparkDataFormatOptions = Map("featuresColumnName" -> "projectedFeatures"), trainingInstanceType = "ml.p2.xlarge", trainingInstanceCount = 1, endpointInstanceType = "ml.c4.xlarge", endpointInitialInstanceCount = 1).setK(10).setFeatureDim(50)

val pipeline = new Pipeline().setStages(Array(pcaEstimator, kMeansSageMakerEstimator))

val pipelineModel = pipeline.fit(trainingData)

val transformedData = pipelineModel.transform(testData)
transformedData.show()
