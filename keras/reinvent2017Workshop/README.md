This is a notebook tutorial for TensorFlow (mainly thorugh Keras) on MNIST data

You will go through building a simple fully connected (dense - DNN) network, then improve it using convolution (CNN), and then you will explore RNN (LSTM) for the same problem

## Launching your AMI 

http://bit.ly/dlami-blog

Windows users can use this bootcamp at: https://github.com/awslabs/aws-ai-bootcamp-labs

Note that there are a few new AMI, choose the one with Conda:

"Deep Learning AMI (Amazon Linux) Version 1.0 - ami-77eb3a0f

Deep Learning AMI with **Conda-based** virtual environments for Apache MXNet, TensorFlow, Caffe2, PyTorch, Theano, CNTK and Keras"

Make sure that you have the **keypair** you are using or download the new one that you created

Connecting to the instance and opening an SSH tunnel for Jupyter on port 8888 (Ubuntu or Amazon Linux):

ssh -i user.pem -L localhost:8888:localhost:8888 **ubuntu**@ec2-ip-ip-ip-ip.region.compute.amazonaws.com

ssh -i user.pem -L localhost:8888:localhost:8888 **ec2-user**@ec2-ip-ip-ip-ip.region.compute.amazonaws.com

### Clone this Notebook

> git clone https://github.com/guyernest/TensorFlowTutorials.git

### Launch Jupyter

> jupyter notebook

### TensorBoard 

In the jupyter terminal start TensorBoard and point it to the log directory used in the notebook

> tensorboard --logdir=~/TensorFlowTutorials/logs/

#### Using DeepLearning AMI on EC2

Opening SSH tunnel for TensorBoard default port 6006 (Ubuntu or Amazon Linux):

ssh -i user.pem -L localhost:6006:localhost:6006 **ubuntu**@ec2-ip-ip-ip-ip.region.compute.amazonaws.com

ssh -i user.pem -L localhost:6006:localhost:6006 **ec2-user**@ec2-ip-ip-ip-ip.region.compute.amazonaws.com

#### Using Amazon SageMaker

Append the port number after the /proxy/ URL, for example:

https://<NB-NAME>.notebook.<REGION>.sagemaker.aws/proxy/6006/
