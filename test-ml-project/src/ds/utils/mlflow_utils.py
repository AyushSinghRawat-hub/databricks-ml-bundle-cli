import mlflow
import os
from typing import Dict, Optional

def start_run(name: str, tags: dict = None):
    """Start MLflow run with Databricks configuration"""
    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")
    return mlflow.start_run(run_name=name, tags=tags)

def register_model(model_uri: str, registered_model_name: str):
    """Register model to Unity Catalog Model Registry"""
    return mlflow.register_model(model_uri=model_uri, name=registered_model_name)


def log_segmentation_metrics(metrics: Dict[str, float]):
    """Log segmentation specific metrics"""
    for metric_name, value in metrics.items():
        mlflow.log_metric(metric_name, value)

def log_segmentation_artifacts(model_path: str, sample_images: Optional[str] = None):
    """Log segmentation model artifacts and sample results"""
    mlflow.log_artifact(model_path)
    if sample_images:
        mlflow.log_artifact(sample_images, "sample_segmentations")
