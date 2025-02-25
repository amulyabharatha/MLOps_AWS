import os
import time
import itertools
import numpy as np
import pandas as pd
import boto3
import sagemaker
from sagemaker.sklearn.estimator import SKLearn, SKLearnModel

# S3 prefix
role = os.environ["SG_ROLE"]
PREFIX = os.environ.get("PREFIX", "DEMO-scikit-iris")
WORK_DIRECTORY = os.environ.get("WORK_DIRECTORY", "data")
FRAMEWORK_VERSION = os.environ.get("FRAMEWORK_VERSION", "1.0-1")
SCRIPT_PATH = os.environ.get("SCRIPT_PATH", "train.py")
INSTANCE_TYPE = os.environ.get("INSTANCE_TYPE", "ml.m5.large")
INITIAL_INSTANCE_COUNT = int(os.environ.get("INITIAL_INSTANCE_COUNT", 1))
USE_SPOT_INSTANCES = True
MAX_RUN = 3600
MAX_WAIT = 7200
#getting session details
print("gettting sg session details")
def get_sg_session():
    sagemaker_session = sagemaker.Session()
    bucket = sagemaker.Session().default_bucket()
    print("bucket name is",bucket)
    print("sagemaker_session name is",sagemaker_session)
    session_details = {"sg_session": sagemaker_session, "bucket_name": bucket}
    print("sg session details are:", session_details)
    return session_details

print("loading data sg_session data.csv")
def load_data(sg_session):
    os.makedirs("./data", exist_ok=True)
    s3_client = boto3.client("s3")
    s3_client.download_file(
        f"sagemaker-sample-files", "datasets/tabular/iris/iris.data", "./data/iris.csv"
    )

    print("reading iris csv file")
    df_iris = pd.read_csv("./data/iris.csv", header=None)
    df_iris[4] = df_iris[4].map(
        {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}
    )
    print("iris to numpy")
    iris = df_iris[[4, 0, 1, 2, 3]].to_numpy()
    np.savetxt(
        "./data/iris.csv", iris, delimiter=",", fmt="%1.1f, %1.3f, %1.3f, %1.3f, %1.3f"
    )
    print("df_iris.head() is:",df_iris.head())
    train_input = sg_session.upload_data(
        WORK_DIRECTORY, key_prefix="{}/{}".format(PREFIX, WORK_DIRECTORY)
    )
    print("training input is:",train_input)
    return train_input

print("training model started")
def train_model(session_details, train_input):
    job_name = "DEMO-xgboost-spot-1-" + time.strftime(
        "%Y-%m-%d-%H-%M-%S", time.gmtime()
    )
    print("job_name is:", job_name)
    bucket = session_details["bucket_name"]
    sg_session = session_details["sg_session"]
    print("bucket and sg_session name from training job is:" ,bucket,sg_session)
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
    sklearn.fit({"train": train_input})
    image_uri = sklearn.image_uri
    model_path = sklearn.model_data
    model_data = {"image_uri": sklearn.image_uri,
                  "model_path": sklearn.model_data,
                  "estimator": sklearn}
    print(image_uri, model_path)
    return model_data


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


def serve_model(model_data, test_data):
    estimator = model_data["estimator"]
    predictor = estimator.deploy(
        initial_instance_count=INITIAL_INSTANCE_COUNT, instance_type=INSTANCE_TYPE
    )
    print(predictor.predict(test_data["test_X"].values))
    print(test_data["test_y"].values)
    predictor.delete_endpoint()

def serve_saved_model(model_data, test_data):
    sklearn_model = SKLearnModel(model_data=model_data['model_path'],
                                 role=role,
                                 entry_point="train.py",
                                 framework_version=FRAMEWORK_VERSION)
    predictor = sklearn_model.deploy(instance_type=INSTANCE_TYPE, initial_instance_count=1)
    print(predictor.predict(test_data["test_X"].values))
    print(test_data["test_y"].values)
    #predictor.delete_endpoint()


if __name__ == "__main__":
    session_details = get_sg_session()
    train_input = load_data(session_details["sg_session"])
    train_model(session_details, train_input)
    #model_data = train_model(session_details, train_input)
    test_data = gen_test_data()
    #serve_model(model_data, test_data)
    model_data = {"model_path": "s3://sagemaker-us-east-1-657605447075/sagemaker-scikit-learn-2023-11-03-04-17-11-091/output/model.tar.gz"}
    #serve_model(model_data, test_data)
    serve_saved_model(model_data, test_data)
