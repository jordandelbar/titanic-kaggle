import bentoml
from zenml.steps import step


@step
def bento_containerizer(bentoml_service_name: str) -> None:

    bentoml.container.build(bento_tag=bentoml_service_name)
