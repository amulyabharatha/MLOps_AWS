export SG_ROLE= arn:aws:iam::657605447075:role/service-role/AmazonSageMaker-ExecutionRole-20230911T145444 #An AWS IAM Role with Full SG Access (I give full S3 Access too fwiw )
export WORK_DIRECTORY= #the name of your local data_dir, i.e., data
export PREFIX= #the prefix you want for you S3 bucket e.g., DEMO-scikit-iris-gus
export FRAMEWORK_VERSION=1.0-1
export SCRIPT_PATH=train.py
export INSTANCE_TYPE= #the instance size you want to use, i.e., ml.m5.large
#added roles