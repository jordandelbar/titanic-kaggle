 #!/bin/bash
export experiment_tracker_name="titanic_experiment_tracker"
export stack_name="titanic_stack"
export mlflow_secret_name="mlflow_secret"

if [ -d "./.zen" ]; then
    echo ".zen folder already exists"
else
    zenml init
fi
# Connect to ZenML server
zenml connect --url=$ZENML_SERVER_URL \
              --username=$ZENML_USERNAME \
              --password=$ZENML_PASSWORD
# Create secret for our experiment tracker
zenml secret create $mlflow_secret_name \
    --username=$MLFLOW_TRACKING_USERNAME \
    --password=$MLFLOW_TRACKING_PASSWORD
# Register our experiment tracker
zenml experiment-tracker register $experiment_tracker_name \
 --flavor=mlflow --tracking_uri=$MLFLOW_TRACKING_URI \
 --tracking_username={{mlflow_secret.username}} \
 --tracking_password={{mlflow_secret.password}}
# Register our stack
zenml stack register $stack_name \
-o default \
-a default \
-e $experiment_tracker_name \
# Set our stack
zenml stack set $stack_name
