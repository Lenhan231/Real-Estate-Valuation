import wandb


def log_to_wandb(
    project_name: str,
    run_name: str,
    config: dict,
    metrics: dict,
    pred_vs_actual_path: str = None,
    feature_importance_path: str = None,
):
    """Log training results to Weights & Biases."""
    run = wandb.init(
        project=project_name,
        name=run_name,
        config=config,
    )

    wandb.log(metrics)

    if pred_vs_actual_path:
        wandb.log({"pred_vs_actual": wandb.Image(str(pred_vs_actual_path))})

    if feature_importance_path:
        wandb.log({"feature_importance": wandb.Image(str(feature_importance_path))})

    wandb.finish()
