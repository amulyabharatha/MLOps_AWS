import os
import numpy as np
import pandas as pd
import boto3
import sagemaker
from sagemaker.sklearn.estimator import SKLearn

# S3 prefix
role = os.environ["SG_ROLE"]
PREFIX = os.environ["PREFIX"]
WORK_DIRECTORY = os.environ["WORK_DIRECTORY"]
FRAMEWORK_VERSION = os.environ["FRAMEWORK_VERSION"]
SCRIPT_PATH = os.environ["SCRIPT_PATH"]
INSTANCE_TYPE = os.environ["INSTANCE_TYPE"]

def get_sg_session():
    sagemaker_session = sagemaker.Session()
    print (sagemaker_session)
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
    print (df_iris.head())
    train_input = sg_session.upload_data(
        WORK_DIRECTORY, key_prefix="{}/{}".format(PREFIX, WORK_DIRECTORY)
        )
    print (train_input)
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
