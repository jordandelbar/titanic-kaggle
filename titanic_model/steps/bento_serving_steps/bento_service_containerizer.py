import bentoml
from zenml.steps import step


@step
def bento_containerizer(bentoml_service_name: str) -> None:
    """Containerizes a bentoml service to a docker image

    Args:
        bentoml_service_name (str): name of the bento service
            to be containerized
    """
    bentoml.container.build(bento_tag=bentoml_service_name)
