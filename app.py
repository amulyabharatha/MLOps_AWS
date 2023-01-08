import os
import time
import itertools
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
INITIAL_INSTANCE_COUNT = int(os.environ["INITIAL_INSTANCE_COUNT"])
USE_SPOT_INSTANCES = True if os.environ["USE_SPOT_INSTANCES"]=="yes" else False
MAX_RUN = int(os.environ["MAX_RUN"])
MAX_WAIT = 7200 if USE_SPOT_INSTANCES else None


def get_sg_session():
    sagemaker_session = sagemaker.Session()
    bucket = sagemaker.Session().default_bucket()
    print(bucket)
    print(sagemaker_session)
    session_details = {"sg_session": sagemaker_session, "bucket_name": bucket}
    return session_details


def load_data(sg_session):
    os.makedirs("./data", exist_ok=True)
    s3_client = boto3.client("s3")
    s3_client.download_file(
        f"sagemaker-sample-files", "datasets/tabular/iris/iris.data", "./data/iris.csv"
    )

    df_iris = pd.read_csv("./data/iris.csv", header=None)
    df_iris[4] = df_iris[4].map(
        {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}
    )
    iris = df_iris[[4, 0, 1, 2, 3]].to_numpy()
    np.savetxt(
        "./data/iris.csv", iris, delimiter=",", fmt="%1.1f, %1.3f, %1.3f, %1.3f, %1.3f"
    )
    print(df_iris.head())
    train_input = sg_session.upload_data(
        WORK_DIRECTORY, key_prefix="{}/{}".format(PREFIX, WORK_DIRECTORY)
    )
    print(train_input)
    return train_input


def train_model(session_details, train_input):
    job_name = "DEMO-xgboost-spot-1-" + time.strftime(
        "%Y-%m-%d-%H-%M-%S", time.gmtime()
    )
    print("Training job", job_name)
    bucket = session_details["bucket_name"]
    sg_session = session_details["sg_session"]
    checkpoint_s3_uri = (
        "s3://{}/{}/checkpoints/{}".format(bucket, PREFIX, job_name)
        if USE_SPOT_INSTANCES
        else None
    )
    print("Checkpoint path:", checkpoint_s3_uri)

    sklearn = SKLearn(
        entry_point=SCRIPT_PATH,
        framework_version=FRAMEWORK_VERSION,
        instance_type=INSTANCE_TYPE,
        role=role,
        sagemaker_session=sg_session,
        hyperparameters={"max_leaf_nodes": 30},
        use_spot_instances=USE_SPOT_INSTANCES,
        max_run=MAX_RUN,
        max_wait=MAX_WAIT,
        checkpoint_s3_uri=checkpoint_s3_uri,
    )
    z = sklearn.fit({"train": train_input})
    print(z)
    return sklearn


def gen_test_data():
    shape = pd.read_csv("data/iris.csv", header=None)
    a = [50 * i for i in range(3)]
    b = [40 + i for i in range(10)]
    indices = [i + j for i, j in itertools.product(a, b)]

    test_data = shape.iloc[indices[:-1]]
    test_X = test_data.iloc[:, 1:]
    test_y = test_data.iloc[:, 0]
    test_data_dict = {"test_X": test_X, "test_y": test_y}
    return test_data_dict


def serve_model(trained_model, test_data):
    predictor = trained_model.deploy(
        initial_instance_count=INITIAL_INSTANCE_COUNT, instance_type=INSTANCE_TYPE
    )
    print(predictor.predict(test_data["test_X"].values))
    print(test_data["test_y"].values)
    predictor.delete_endpoint()


if __name__ == "__main__":
    session_details = get_sg_session()
    train_input = load_data(session_details["sg_session"])
    trained_model = train_model(session_details, train_input)
    test_data = gen_test_data()
    serve_model(trained_model, test_data)
