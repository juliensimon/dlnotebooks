# pyspark --packages com.amazonaws:sagemaker-spark_2.11:spark_2.1.1-1.0

# Train, deploy and invoke a standalone SageMaker model from Spark: MNIST (784) --> KMeans (10)

from sagemaker_pyspark import IAMRole
from sagemaker_pyspark.algorithms import KMeansSageMakerEstimator

iam_role = "arn:aws:iam::ACCOUNT_NUMBER:role/ROLE_NAME"
region = "us-east-1"
training_data = spark.read.format("libsvm").option("numFeatures", "784").load("s3a://sagemaker-sample-data-{}/spark/mnist/train/".format(region))

test_data = spark.read.format("libsvm").option("numFeatures", "784").load("s3a://sagemaker-sample-data-{}/spark/mnist/train/".format(region))

kmeans_estimator = KMeansSageMakerEstimator(trainingInstanceType="ml.m4.xlarge", trainingInstanceCount=1, endpointInstanceType="ml.m4.xlarge", endpointInitialInstanceCount=1,sagemakerRole=IAMRole(iam_role))

kmeans_estimator.setK(10)
kmeans_estimator.setFeatureDim(784)

kmeans_model = kmeans_estimator.fit(training_data)

transformed_data = kmeans_model.transform(test_data)
transformed_data.show()
