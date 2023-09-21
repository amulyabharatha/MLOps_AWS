FROM python:3.8

RUN pip3 install --no-cache scikit-learn pandas joblib flask requests boto3 tabulate

COPY train.py /usr/bin/train

RUN chmod 755 /usr/bin/train 

EXPOSE 8080
 