import os


def split_wandb_project(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("W&B project must be a nonempty string")
    value = value.strip()
    if "/" not in value:
        return None, value
    parts = value.split("/")
    if len(parts) != 2 or not all(parts):
        raise ValueError("W&B destination must be project or entity/project")
    return parts[0], parts[1]


def configure_environment():
    """Accept short credential aliases without storing or printing credentials."""
    for alias, canonical in (("OPENAI_KEY", "OPENAI_API_KEY"), ("WANDB_KEY", "WANDB_API_KEY")):
        if os.environ.get(alias) and not os.environ.get(canonical):
            os.environ[canonical] = os.environ[alias]
    if os.environ.get("WANDB_PROJECT"):
        entity, project = split_wandb_project(os.environ["WANDB_PROJECT"])
        # W&B validates environment settings before applying wandb.init kwargs.
        os.environ["WANDB_PROJECT"] = project
        if entity and not os.environ.get("WANDB_ENTITY"):
            os.environ["WANDB_ENTITY"] = entity
