 #!/bin/bash
export experiment_tracker_name="titanic_experiment_tracker"
export stack_name="titanic_stack"
export mlflow_secret_name="mlflow_secret"

zenml connect --url=$ZENML_SERVER_URL \
              --username=$ZENML_USERNAME \
              --password=$ZENML_PASSWORD
zenml stack set default
zenml stack delete $stack_name -y
zenml experiment-tracker delete $experiment_tracker_name
zenml secret delete $mlflow_secret_name -y
