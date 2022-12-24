from typing import Dict


def model_register_decision(
    current_model_metrics: Dict, new_model_metrics: Dict
) -> bool:
    """
    Compare metrics of a model currently in production
    versus a new model to decide if the new model should
    be set to a different stage

    Args:
        current_model_metrics (Dict): Dict with the current model metrics
        new_model_metrics (Dict): Dict with the new model metrics

    Returns:
        bool: True if the new model has better metrics, False otherwise
    """
    current_model_score = 0
    new_model_score = 0
    for key in current_model_metrics:
        if current_model_metrics[key] < new_model_metrics[key]:
            new_model_score += 1
        elif current_model_metrics[key] > new_model_metrics[key]:
            current_model_score += 1
    return new_model_score > current_model_score
