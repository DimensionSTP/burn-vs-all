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
    metadata = pd.read_csv(f"{config.connected_dir}/metadata/metadata.csv")
    train_metadata, test_metadata = train_test_split(
        metadata,
        test_size=config.split_ratio,
        random_state=config.seed,
        shuffle=True,
        stratify=metadata[config.target_column_name],
    )
    train_metadata = train_metadata.reset_index(drop=True)
    test_metadata = test_metadata.reset_index(drop=True)

    train_metadata.to_csv(
        f"{config.connected_dir}/metadata/train.csv",
        index=False,
    )
    test_metadata.to_csv(
        f"{config.connected_dir}/metadata/test.csv",
        index=False,
    )


if __name__ == "__main__":
    split_metadata()
