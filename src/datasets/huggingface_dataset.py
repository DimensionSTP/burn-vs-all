from typing import Dict, Any, List

import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset

from transformers import AutoFeatureExtractor

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
        image_path_column_name: str,
        target_column_name: str,
        coordinates_column_name: Dict[str, str],
        num_devices: int,
        batch_size: int,
        image_size: int,
        pretrained_model_name: str,
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
        self.image_path_column_name = image_path_column_name
        self.target_column_name = target_column_name
        self.coordinates_column_name = coordinates_column_name
        self.num_devices = num_devices
        self.batch_size = batch_size
        self.image_size = image_size
        self.data_encoder = AutoFeatureExtractor.from_pretrained(
            pretrained_model_name,
        )
        self.augmentation_probability = augmentation_probability
        self.augmentations = augmentations
        self.transform = self.get_transform()
        self.text_max_length = text_max_length

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(
        self,
        idx: int,
    ) -> Dict[str, Any]:
        if self.modality in ["image", "multi-modality"]:
            data = np.array(Image.open(self.datas[idx]).convert("RGB")) / 255.0
            data = self.transform(image=data)["image"]
            if self.modality == "image":
                encoded = self.encode_image(data)
            else:
                encoded = self.encode_text(data)
        else:
            data = self.normalize_string(self.datas[idx])
            encoded = self.encode_text(data)
        encoded["labels"] = torch.tensor(
            [self.labels[idx]],
            dtype=torch.long,
        ).squeeze(0)
        return {
            "encoded": encoded,
            "index": idx,
        }

    def get_dataset(self) -> Dict[str, List[Any]]:
        if self.split in ["train", "val"]:
            csv_path = f"{self.data_path}/train.csv"
            data = pd.read_csv(csv_path)
            data = data.fillna("_")
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
            csv_path = f"{self.data_path}/{self.split}.csv"
            data = pd.read_csv(csv_path)
            data = data.fillna("_")
        elif self.split == "predict":
            csv_path = f"{self.data_path}/test.csv"
            data = pd.read_csv(csv_path)
            data = data.fillna("_")
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

        if self.modality in ["image", "multi-modality"]:
            if self.split in ["train", "test"]:
                datas = [
                    f"{self.data_path}/{self.split}/{file_name}"
                    for file_name in data["ID"]
                ]
            elif self.split == "val":
                datas = [
                    f"{self.data_path}/train/{file_name}" for file_name in data["ID"]
                ]
            else:
                datas = [
                    f"{self.data_path}/test/{file_name}" for file_name in data["ID"]
                ]
        else:
            datas = data["text"].tolist()
        labels = data[self.target_column_name].tolist()
        return {
            "datas": datas,
            "labels": labels,
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
            transforms.append(ToTensorV2())
            return A.Compose(transforms)
        else:
            transforms.append(ToTensorV2())
            return A.Compose(transforms)

    def encode_image(
        self,
        data: np.ndarray,
    ) -> Dict[str, torch.Tensor]:
        encoded = self.data_encoder(
            data,
            return_tensors="pt",
        )
        encoded = {k: v.squeeze(0) for k, v in encoded.items()}
        return encoded

    @staticmethod
    def normalize_string(
        data: str,
    ) -> str:
        data = re.sub(
            r"[\s]",
            r" ",
            str(data),
        )
        data = re.sub(
            r"[^a-zA-Z가-힣ㄱ-ㅎ0-9.!?]+",
            r" ",
            str(data),
        )
        return data

    def encode_text(
        self,
        data: str,
    ) -> Dict[str, torch.Tensor]:
        encoded = self.data_encoder(
            data,
            padding="max_length",
            max_length=self.text_max_length,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=True,
        )
        encoded = {k: v.squeeze(0) for k, v in encoded.items()}
        return encoded
