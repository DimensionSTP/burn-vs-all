from typing import Dict, Any, List

import pandas as pd
import cv2
from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2


class BurnSkinDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        metadata_path: str,
        split: str,
        split_ratio: float,
        seed: int,
        classification_type: int,
        is_crop: bool,
        image_dir_column_name: str,
        image_file_column_name: str,
        target_column_name: str,
        coordinates_column_name: Dict[str, str],
        num_devices: int,
        batch_size: int,
        image_size: int,
        augmentation_probability: float,
        augmentations: List[str],
    ) -> None:
        self.data_path = data_path
        self.metadata_path = metadata_path
        self.split = split
        self.split_ratio = split_ratio
        self.seed = seed
        self.classification_type = classification_type
        if self.classification_type not in range(5):
            raise ValueError(
                f"Invalid data type: {self.classification_type}. Choose in [0, 1, 2, 3, 4]."
            )
        self.is_crop = is_crop
        self.image_dir_column_name = image_dir_column_name
        self.image_file_column_name = image_file_column_name
        self.target_column_name = target_column_name
        self.coordinates_column_name = coordinates_column_name
        self.num_devices = num_devices
        self.batch_size = batch_size
        self.image_size = image_size
        self.augmentation_probability = augmentation_probability
        self.augmentations = augmentations
        metadata = self.get_metadata()
        self.image_paths = metadata["image_paths"]
        self.labels = metadata["labels"]
        self.x1 = metadata["x1"]
        self.y1 = metadata["y1"]
        self.x2 = metadata["x2"]
        self.y2 = metadata["y2"]
        self.transform = self.get_transform()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(
        self,
        idx: int,
    ) -> Dict[str, Any]:
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )
        if self.is_crop:
            x1, y1, x2, y2 = self.x1[idx], self.y1[idx], self.x2[idx], self.y2[idx]
            image = image[y1:y2, x1:x2]
        image = cv2.resize(
            image,
            (
                self.image_size,
                self.image_size,
            ),
            interpolation=cv2.INTER_CUBIC,
        )[
            0 : self.image_size,
            0 : self.image_size,
            :,
        ]
        image = self.transform(image=image)["image"]
        label = self.labels[idx]
        return {
            "image": image,
            "label": label,
            "index": idx,
        }

    def get_metadata(self) -> Dict[str, List[Any]]:
        if self.split in ["train", "val"]:
            csv_path = f"{self.metadata_path}/train.csv"
            data = pd.read_csv(csv_path)
            train_data, val_data = train_test_split(
                data,
                test_size=self.split_ratio,
                random_state=self.seed,
                shuffle=True,
                stratify=data[self.target_column_name],
            )
            if self.split == "train":
                data = train_data
            else:
                data = val_data
        elif self.split == "test":
            csv_path = f"{self.metadata_path}/{self.split}.csv"
            data = pd.read_csv(csv_path)
        elif self.split == "predict":
            csv_path = f"{self.metadata_path}/test.csv"
            data = pd.read_csv(csv_path)
            if self.num_devices > 1:
                last_row = data.iloc[-1]
                total_batch_size = self.num_devices * self.batch_size
                remainder = (len(data) % total_batch_size) % self.num_devices
                if remainder != 0:
                    num_dummies = self.num_devices - remainder
                    repeated_rows = pd.DataFrame([last_row] * num_dummies)
                    repeated_rows.reset_index(
                        drop=True,
                        inplace=True,
                    )
                    data = pd.concat(
                        [
                            data,
                            repeated_rows,
                        ],
                        ignore_index=True,
                    )
        else:
            raise ValueError(f"Inavalid split: {self.split}")

        image_paths = data.apply(
            lambda row: f"{self.data_path}/{row[self.image_dir_column_name]}/images/{row[self.image_file_column_name]}",
            axis=1,
        ).tolist()
        labels = data[self.target_column_name].astype(int).tolist()
        if self.classification_type in [0, 1, 2, 3]:
            labels = [1 if label == self.classification_type else 0 for label in labels]
        x1 = data[self.coordinates_column_name.x1].tolist()
        y1 = data[self.coordinates_column_name.y1].tolist()
        x2 = data[self.coordinates_column_name.x2].tolist()
        y2 = data[self.coordinates_column_name.y2].tolist()
        return {
            "image_paths": image_paths,
            "labels": labels,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
        }

    def get_transform(self) -> A.Compose:
        transforms = []
        if self.split in ["train", "val"]:
            for aug in self.augmentations:
                if aug == "rotate30":
                    transforms.append(
                        A.Rotate(
                            limit=[30, 30],
                            p=self.augmentation_probability,
                        )
                    )
                elif aug == "rotate45":
                    transforms.append(
                        A.Rotate(
                            limit=[45, 45],
                            p=self.augmentation_probability,
                        )
                    )
                elif aug == "rotate90":
                    transforms.append(
                        A.Rotate(
                            limit=[90, 90],
                            p=self.augmentation_probability,
                        )
                    )
                elif aug == "hflip":
                    transforms.append(
                        A.HorizontalFlip(
                            p=self.augmentation_probability,
                        )
                    )
                elif aug == "vflip":
                    transforms.append(
                        A.VerticalFlip(
                            p=self.augmentation_probability,
                        )
                    )
                elif aug == "noise":
                    transforms.append(
                        A.GaussNoise(
                            p=self.augmentation_probability,
                        )
                    )
                elif aug == "blur":
                    transforms.append(
                        A.Blur(
                            blur_limit=7,
                            p=self.augmentation_probability,
                        )
                    )
            transforms.append(
                A.Normalize(
                    mean=[
                        0.485,
                        0.456,
                        0.406,
                    ],
                    std=[
                        0.229,
                        0.224,
                        0.225,
                    ],
                )
            )
            transforms.append(ToTensorV2())
            return A.Compose(transforms)
        else:
            transforms.append(
                A.Normalize(
                    mean=[
                        0.485,
                        0.456,
                        0.406,
                    ],
                    std=[
                        0.229,
                        0.224,
                        0.225,
                    ],
                )
            )
            transforms.append(ToTensorV2())
            return A.Compose(transforms)
