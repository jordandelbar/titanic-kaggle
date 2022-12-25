from typing import Dict, List, Tuple

import numpy
from mlflow import MlflowClient


def _get_registered_models(model_name: str) -> Dict:
    """Return all the registered models under a certain name in Mlflow

    Args:
        model_name (str): name of the models registered

    Raises:
        NameError: raise if there are no models registered under that name

    Returns:
        Dict: dictionnary with the model version as key and model information as value
    """
    client = MlflowClient()
    registered_models = dict()
    # We record every registered model under a certain name
    for mv in client.search_model_versions(f"name='{model_name}'"):
        registered_models[dict(mv)["version"]] = dict(mv)

    # If there is no model registered under that name we raise an error
    if len(registered_models) == 0:
        raise NameError("no model registered with that name yet")

    return registered_models


def _get_metric_list(model_dictionnary: Dict, metric: str) -> List[float]:
    """Return a metric list for all registered model in a dictionnary

    Args:
        model_dictionnary (Dict): dict of models registered
        metric (str): name of the metric to be returned

    Returns:
        List[float]: list of model metrics
    """
    client = MlflowClient()

    # We retrieve all the specified metrics of a registered model dictionnary
    metrics_list = list()
    for key in model_dictionnary.keys():
        metrics_list.append(
            client.get_run(model_dictionnary[key]["run_id"]).data.to_dictionary()[
                "metrics"
            ][metric]
        )
    return metrics_list


def _get_best_models_in_mlflow(
    model_name: str, metric_to_check: str
) -> List[Tuple[str, float]]:
    """Return the two best models registered in the Mlflow server

    Args:
        model_name (str): name of the model to be checked
        metric_to_check (str): metric to compare to determine best models

    Returns:
        List[Tuple[str, float]]: list of tuple with best and second best model
                                 and the metric value associated
    """
    client = MlflowClient()

    registered_models = _get_registered_models(model_name=model_name)

    version_metrics = dict()
    for key in registered_models.keys():
        version_metrics[key] = client.get_run(
            registered_models[key]["run_id"]
        ).data.to_dictionary()["metrics"][metric_to_check]

    best_model = sorted(version_metrics, key=version_metrics.get)[-1]
    best_model_value = sorted(version_metrics.values())[-1]

    # Check if there is already a second model else we set it up to 0
    try:
        second_best_model = sorted(version_metrics, key=version_metrics.get)[-2]
        second_best_model_value = sorted(version_metrics.values())[-2]
    except IndexError:
        second_best_model = "0"
        second_best_model_value = 0

    return [
        (best_model, best_model_value),
        (second_best_model, second_best_model_value),
    ]


def _get_second_best_model_metric(model_name: str, metric_to_check: str) -> float:
    """Return the second best model metric

    Args:
        model_name (str): name of the model
        metric_to_check (str): metric to be returned

    Returns:
        float: metric of the second best model
    """
    return _get_best_models_in_mlflow(
        model_name=model_name, metric_to_check=metric_to_check
    )[1][1]


def registering_model_decision(
    model_name: str, model_accuracy: float, model_f1_score: float
) -> bool:
    """Decision engine to decide if a model should be registered or not

    Args:
        model_name (str): name of the model on the Mlflow server
        model_accuracy (float): new model accuracy
        model_f1_score (float): new model f1 score

    Returns:
        bool: True if the model should be registered, False if not
    """
    registering_decision = False
    f1_score_mean = numpy.mean(
        _get_metric_list(
            model_dictionnary=_get_registered_models(model_name=model_name),
            metric="f1 score",
        )
    )
    # We register a model only if its accurary is better than the second best
    # model and if its f1 score is better or equal to the mean of f1 score of the
    # different registered models. This is completely arbitrary but gives an
    # example and inspiration of the rules engine you can design
    if (
        model_accuracy
        > _get_second_best_model_metric(
            model_name=model_name, metric_to_check="accuracy"
        )
        and model_f1_score >= f1_score_mean
    ):
        registering_decision = True
    return registering_decision


def promote_models(model_name: str, metric_to_check: str) -> None:
    """Promoting engine to clean all the model and put them in the
       correct stage

    Args:
        model_name (str): name of the model on the Mlflow server
        metric_to_check (str): metric to determine best models
    """
    best_models = _get_best_models_in_mlflow(
        model_name=model_name, metric_to_check=metric_to_check
    )
    best_models_list = [best_models[0][0], best_models[1][0]]
    client = MlflowClient()

    # The best model is put in production stage
    client.transition_model_version_stage(
        name=model_name,
        version=best_models_list[0],
        stage="Production",
    )

    if best_models_list[1] != "0":
        # If it exists, the second best model is put in staging
        client.transition_model_version_stage(
            name=model_name,
            version=best_models_list[1],
            stage="Staging",
        )
    # This is completely arbitraty but in this little use case
    # I don't need to perform integration tests, reason why I use
    # Staging and Production stages as best model and back-up model
    registered_models = dict()
    for mv in client.search_model_versions(f"name='{model_name}'"):
        registered_models[dict(mv)["version"]] = dict(mv)

    # Every other model is archived
    for model_version in list(
        set(list(registered_models.keys())) - set(best_models_list)
    ):
        client.transition_model_version_stage(
            name=model_name,
            version=model_version,
            stage="Archived",
        )
