from pytorch_lightning.loggers import MLFlowLogger


def build_logger(
    tracking_uri: str,
    experiment_name: str,
    run_name: str | None,
) -> MLFlowLogger:

    return MLFlowLogger(
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
        run_name=run_name,
    )
