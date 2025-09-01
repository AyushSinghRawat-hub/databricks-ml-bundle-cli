import click
import mlflow
from mlflow.tracking import MlflowClient

@click.command()
@click.option("--registered-model", required=True)

@click.option("--min-dice-score", default=0.80, type=float)

@click.option("--promote-stage", default="Staging")
@click.option("--experiment", required=True)


def main(registered_model: str, min_dice_score: float, promote_stage: str, experiment: str):
    """Register and promote segmentation model based on quality gates"""
    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")
    client = MlflowClient()

    # Find best run by dice score metric
    runs = client.search_runs(
        experiment_ids=[client.get_experiment_by_name(experiment).experiment_id],
        filter_string=f"metrics.dice_score >= {min_dice_score}",
        run_view_type=1,
        max_results=1,
        order_by=["metrics.dice_score DESC"]
    )
    if not runs:
        raise SystemExit(f"No run met the quality bar (dice_score >= {min_dice_score})")


    run = runs[0]
    # Find the newest version created by log_model (same run)
    mv = [v for v in client.search_model_versions(f"name='{registered_model}'") if v.run_id == run.info.run_id]
    if not mv:
        raise SystemExit("No model version found for winning run")

    version = sorted(mv, key=lambda v: int(v.version))[-1]
    client.transition_model_version_stage(name=registered_model, version=version.version, stage=promote_stage)
    print(f"Promoted {registered_model} v{version.version} to {promote_stage}")

if __name__ == "__main__":
    main()