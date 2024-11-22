import dotenv

dotenv.load_dotenv(
    override=True,
)

import os
import warnings

os.environ["HYDRA_FULL_ERROR"] = "1"
warnings.filterwarnings("ignore")

import pandas as pd

import hydra
from omegaconf import DictConfig


@hydra.main(
    config_path="configs/",
    config_name="hierarchical_inference.yaml",
)
def hierarchical_inference(
    config: DictConfig,
) -> None:
    submission_path = config.submission_path
    submission_names = config.submission_names
    if len(submission_names) - 1 != config.num_labels:
        raise ValueError(
            f"Count of the submission files - 1 and number of labels must be the same. priority_order: {submission_names}, num_labels: {config.num_labels}"
        )

    target_column_name = config.target_column_name
    priority_order = config.priority_order
    if len(priority_order) != config.num_labels:
        raise ValueError(
            f"Lengths of the priority order and number of labels must be the same. priority_order: {priority_order}, num_labels: {config.num_labels}"
        )

    save_name = config.save_name

    submission_dfs = []
    for submission_name in submission_names:
        submission_df = pd.read_csv(f"{submission_path}/{submission_name}.csv")
        submission_dfs.append(submission_df)

    for i, submission_df in enumerate(submission_dfs):
        submission_df = submission_df[[target_column_name]].astype(int)
        submission_df.columns = [i]

    df = pd.concat(
        submission_dfs,
        axis=1,
    )
    df[target_column_name] = None

    for priority in priority_order:
        df.loc[
            df[target_column_name].isnull() & (df.iloc[:, priority].squeeze() == 1),
            target_column_name,
        ] = priority

    df.loc[df[target_column_name].isnull(), target_column_name] = df.iloc[:, -1]
    df[target_column_name] = df[target_column_name].astype(int)

    df.to_csv(
        f"{submission_path}/{save_name}.csv",
        index=False,
    )


if __name__ == "__main__":
    hierarchical_inference()
