 #!/bin/bash
export secrets_manager_name="titanic_secret_manager"
export experiment_tracker_name="titanic_experiment_tracker"
export stack_name="titanic_stack"
export mlflow_secret_name="mlflow_secret"

zenml secrets-manager register $secrets_manager_name --flavor=local

zenml experiment-tracker register $experiment_tracker_name \
 --flavor=mlflow --tracking_uri=$MLFLOW_TRACKING_URI \
 --tracking_username={{$mlflow_secret_name.username}} \
 --tracking_password={{$mlflow_secret_name.password}}

zenml stack register $stack_name \
-o default \
-a default \
-e $experiment_tracker_name \
-x $secrets_manager_name

zenml stack set $stack_name

zenml secrets-manager secret register $mlflow_secret_name \
--username=$MLFLOW_TRACKING_USERNAME \
--password=$MLFLOW_TRACKING_PASSWORD
