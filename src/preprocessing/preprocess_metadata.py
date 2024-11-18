import dotenv

dotenv.load_dotenv(
    override=True,
)

import pandas as pd
import cv2
from tqdm import tqdm

import hydra
from omegaconf import DictConfig


@hydra.main(
    config_path="../../configs/",
    config_name="huggingface.yaml",
)
def preprocess_metadata(
    config: DictConfig,
) -> None:
    metadata = pd.read_csv(f"{config.connected_dir}/metadata/metadata.csv")
    metadata[config.target_column_name] = metadata[config.target_column_name].astype(
        int
    )
    metadata[config.coordinates_column_name.x1] = metadata[
        config.coordinates_column_name.x1
    ].astype(int)
    metadata[config.coordinates_column_name.y1] = metadata[
        config.coordinates_column_name.y1
    ].astype(int)
    metadata[config.coordinates_column_name.x2] = metadata[
        config.coordinates_column_name.x2
    ].astype(int)
    metadata[config.coordinates_column_name.y2] = metadata[
        config.coordinates_column_name.y2
    ].astype(int)

    exceptions = []
    for _, row in tqdm(metadata.iterrows(), total=len(metadata)):
        image_dir = row[config.image_dir_column_name]
        image_file = row[config.image_file_column_name]
        x1 = row[config.coordinates_column_name.x1]
        y1 = row[config.coordinates_column_name.y1]
        x2 = row[config.coordinates_column_name.x2]
        y2 = row[config.coordinates_column_name.y2]
        image_path = f"{config.connected_dir}/data/{image_dir}/images/{image_file}"
        image = cv2.imread(image_path)
        if image is None:
            exceptions.append(
                (
                    image_dir,
                    image_file,
                )
            )

        try:
            image = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB,
            )
        except:
            exceptions.append(
                (
                    image_dir,
                    image_file,
                )
            )

        image = image[y1:y2, x1:x2]
        if image is None:
            exceptions.append(
                (
                    image_dir,
                    image_file,
                )
            )

        try:
            image = cv2.resize(
                image,
                (
                    config.image_size,
                    config.image_size,
                ),
                interpolation=cv2.INTER_CUBIC,
            )[
                0 : config.image_size,
                0 : config.image_size,
                :,
            ]
        except:
            exceptions.append(
                (
                    image_dir,
                    image_file,
                )
            )

    exceptions_df = pd.DataFrame(
        exceptions,
        columns=[
            config.image_dir_column_name,
            config.image_file_column_name,
        ],
    )

    filtered_metadata = metadata.merge(
        exceptions_df,
        on=[
            config.image_dir_column_name,
            config.image_file_column_name,
        ],
        how="left",
        indicator=True,
    )
    filtered_metadata = filtered_metadata[filtered_metadata["_merge"] == "left_only"]
    filtered_metadata = filtered_metadata.drop(columns=["_merge"])

    filtered_metadata.to_csv(
        f"{config.connected_dir}/metadata/metadata.csv",
        index=False,
    )


if __name__ == "__main__":
    preprocess_metadata()
