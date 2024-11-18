import dotenv

dotenv.load_dotenv(
    override=True,
)

import pandas as pd
from sklearn.model_selection import train_test_split

import hydra
from omegaconf import DictConfig


@hydra.main(
    config_path="../../configs/",
    config_name="huggingface.yaml",
)
def split_metadata(
    config: DictConfig,
) -> None:
    dataset = pd.read_csv(f"{config.connected_dir}/metadata/metadata.csv")
    train_dataset, test_dataset = train_test_split(
        dataset,
        test_size=config.split_ratio,
        random_state=config.seed,
        shuffle=True,
        stratify=dataset[config.target_column_name],
    )
    train_dataset = train_dataset.reset_index(drop=True)
    test_dataset = test_dataset.reset_index(drop=True)

    train_dataset.to_csv(
        f"{config.connected_dir}/metadata/train.csv",
        index=False,
    )
    test_dataset.to_csv(
        f"{config.connected_dir}/metadata/test.csv",
        index=False,
    )


if __name__ == "__main__":
    split_metadata()
