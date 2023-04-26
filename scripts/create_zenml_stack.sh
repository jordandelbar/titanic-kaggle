 #!/bin/bash
export experiment_tracker_name="titanic_experiment_tracker"
export stack_name="titanic_stack"
export mlflow_secret_name="mlflow_secret"

zenml secret create $mlflow_secret_name \
    --username=$MLFLOW_TRACKING_USERNAME \
    --password=$MLFLOW_TRACKING_PASSWORD

zenml experiment-tracker register $experiment_tracker_name \
 --flavor=mlflow --tracking_uri=$MLFLOW_TRACKING_URI \
 --tracking_username={{mlflow_secret.username}} \
 --tracking_password={{mlflow_secret.password}}

zenml stack register $stack_name \
-o default \
-a default \
-e $experiment_tracker_name \

zenml stack set $stack_name
