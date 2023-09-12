import os
import numpy as np
import pandas as pd
import boto3
import sagemaker
from sagemaker.sklearn.estimator import SKLearn


# S3 prefix
#role = os.environ["SG_ROLE"]
#PREFIX = os.environ.get("PREFIX", "DEMO-scikit-iris")
#WORK_DIRECTORY = os.environ.get("WORK_DIRECTORY", "data")
#FRAMEWORK_VERSION = os.environ.get("FRAMEWORK_VERSION", "1.0-1")
#SCRIPT_PATH = os.environ.get("SCRIPT_PATH", "train.py")
#INSTANCE_TYPE = os.environ.get("INSTANCE_TYPE", "ml.m5.large")
#INITIAL_INSTANCE_COUNT = int(os.environ.get("INITIAL_INSTANCE_COUNT", 1))
#USE_SPOT_INSTANCES = True
#MAX_RUN = 3600
#MAX_WAIT = 7200

# S3 prefix
role = os.environ["SG_ROLE"]
#PREFIX = os.environ["PREFIX"]
PREFIX = os.environ.get("PREFIX", "DEMO-scikit-iris")
#WORK_DIRECTORY = os.environ["WORK_DIRECTORY"]
WORK_DIRECTORY = os.environ.get("WORK_DIRECTORY", "data")
#FRAMEWORK_VERSION = os.environ["FRAMEWORK_VERSION"]
FRAMEWORK_VERSION = os.environ.get("FRAMEWORK_VERSION", "1.0-1")
#SCRIPT_PATH = os.environ["SCRIPT_PATH"]
SCRIPT_PATH = os.environ.get("SCRIPT_PATH", "train.py")
#INSTANCE_TYPE = os.environ["INSTANCE_TYPE"]
INSTANCE_TYPE = os.environ.get("INSTANCE_TYPE", "ml.m5.large")

def get_sg_session():
    sagemaker_session = sagemaker.Session()
    print ("sagemaker_session is:",sagemaker_session)
    return sagemaker_session


def load_data(sg_session):
    os.makedirs("./data", exist_ok=True)
    s3_client = boto3.client("s3")
    s3_client.download_file(
        f"sagemaker-sample-files", "datasets/tabular/iris/iris.data", "./data/iris.csv")
        
    df_iris = pd.read_csv("./data/iris.csv", header=None)
    df_iris[4] = df_iris[4].map({"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2})
    iris = df_iris[[4, 0, 1, 2, 3]].to_numpy()
    np.savetxt("./data/iris.csv", iris, delimiter=",", fmt="%1.1f, %1.3f, %1.3f, %1.3f, %1.3f")
    print ("df_iris head is at:",df_iris.head())
    train_input = sg_session.upload_data(
        WORK_DIRECTORY, key_prefix="{}/{}".format(PREFIX, WORK_DIRECTORY)
        )
    print ("train input data is :",train_input)
    return train_input

def train_model(sg_session, train_input):
    sklearn = SKLearn(
        entry_point=SCRIPT_PATH,
        framework_version=FRAMEWORK_VERSION,
        instance_type=INSTANCE_TYPE,
        role=role,
        sagemaker_session=sg_session,
        hyperparameters={"max_leaf_nodes": 30},
        )
    z = sklearn.fit({"train": train_input})
    print (z)

if __name__ == "__main__":
    sg_session = get_sg_session()
    train_input = load_data(sg_session)
    train_model(sg_session, train_input)