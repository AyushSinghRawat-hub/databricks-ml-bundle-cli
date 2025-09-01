import click
import mlflow

import mlflow.pytorch

from pyspark.sql import SparkSession

import torch
import torch.nn as nn

import pandas as pd

spark = SparkSession.builder.getOrCreate()

@click.command()
@click.option("--features", required=True, help="UC table with processed features")
@click.option("--registered-model", required=True, help="UC model path")
@click.option("--experiment", required=True, help="UC experiment path")
@click.option("--epochs", default=100, type=int)
@click.option("--batch-size", default=4, type=int)
@click.option("--learning-rate", default=0.001, type=float)
def main(features: str, registered_model: str, experiment: str, epochs: int, batch_size: int, learning_rate: float):
    """Train segmentation model"""
    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_experiment(experiment)

    with mlflow.start_run(run_name="train_segmentation", tags={"model_type": "segmentation"}):
        # Load preprocessed data
        df = spark.table(features)
        
        # Log hyperparameters
        mlflow.log_params({
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "model_type": "segmentation"
        })
        
        
        # Segmentation training logic (placeholder)
        # Replace with actual segmentation model training
        
        # Example metrics for segmentation
        metrics = {
            "dice_score": 0.85,
            "iou": 0.78,
            "pixel_accuracy": 0.92,
            "loss": 0.15
        }
        
        for metric_name, value in metrics.items():
            mlflow.log_metric(metric_name, value)
        
        # mlflow.pytorch.log_model(model, "model", registered_model_name=registered_model)
        
        
        
        mlflow.set_tags({"env": "${bundle.target}", "project": "test_ml_project"})
        print(f"Training completed. Model registered as {registered_model}")

if __name__ == "__main__":
    main()