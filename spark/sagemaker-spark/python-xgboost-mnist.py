# pyspark --packages com.amazonaws:sagemaker-spark_2.11:spark_2.1.1-1.0

# Train, deploy and invoke a standalone SageMaker model from Spark: XGBoost

from sagemaker_pyspark import IAMRole
from sagemaker_pyspark.algorithms import XGBoostSageMakerEstimator

iam_role = "arn:aws:iam::ACCOUNT_NUMBER:role/ROLE_NAME"
region = "us-east-1"

training_data=spark.read.format("libsvm").option("numFeatures","784").load("s3a://sagemaker-sample-data-{}/spark/mnist/train/".format(region))
test_data = spark.read.format("libsvm").option("numFeatures","784").load("s3a://sagemaker-sample-data-{}/spark/mnist/train/".format(region))

xgboost_estimator = XGBoostSageMakerEstimator(trainingInstanceType="ml.m4.xlarge", trainingInstanceCount=1, endpointInstanceType="ml.m4.xlarge", endpointInitialInstanceCount=1, sagemakerRole=IAMRole(iam_role))

xgboost_estimator.setObjective('multi:softmax')
xgboost_estimator.setNumRound(25)
xgboost_estimator.setNumClasses(10)

xgboost_model = xgboost_estimator.fit(training_data)

transformed_data = xgboost_model.transform(test_data)
transformed_data.show()
