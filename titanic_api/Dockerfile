FROM python:3.10.8

# Create a user that will run the app
RUN adduser --disabled-password --gecos '' ml-api-user

WORKDIR /opt/titanic_api

ARG PIP_EXTRA_INDEX_URL

# Install requirements

ADD ./titanic_api /opt/titanic_api/
RUN pip install --upgrade pip
RUN pip install -r /opt/titanic_api/requirements.txt

RUN chmod +X /opt/titanic_api/run.sh
RUN chown -R ml-api-user:ml-api-user ./

USER ml-api-user

EXPOSE 8001

CMD ["bash", "./run.sh"]