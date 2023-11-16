import tqdm
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import pandas as pd
from PIL import Image


IMAGE_FOLDER = "/root/ILSVRC/Data/CLS-LOC/val"
CSV_TARGETS = "/root/LOC_val_solution.csv"
CLASS_MAPPING_PATH = "/root/LOC_synset_mapping.txt"


class ImageNetDataset(Dataset):
    @staticmethod
    def get_class_mapping(class_mapping_path):
        class_mapping = dict()
        with open(class_mapping_path) as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                class_name = line.split()[0]
                class_mapping[class_name] = i
        return class_mapping

    def __init__(self, image_folder, csv_targets, class_mapping_path, transforms, pre_load=False, dataset_size=None):
        class_mapping = ImageNetDataset.get_class_mapping(class_mapping_path)
        tmp = pd.read_csv(csv_targets, sep=',')
        names = tmp['ImageId'].tolist()
        class_names = tmp['PredictionString'].apply(lambda x: x.split()[0]).tolist()
        class_id = list(map(class_mapping.get, class_names))

        _targets_mapping = dict(zip(names, class_id))

        _image_paths = sorted(list(Path(image_folder).glob("**/*.JPEG")))
        if dataset_size is not None:
            _image_paths = _image_paths[:dataset_size]

        self.pre_load = pre_load
        if pre_load:
            print("preloading images")
            self.images_and_classes = []
            for _image_path in tqdm.tqdm(_image_paths):
                image = Image.open(_image_path).convert('RGB')
                tensor = transforms(image)

                image_name = str(_image_path).split('/')[-1].split('.')[0]
                class_id = _targets_mapping[image_name]

                self.images_and_classes.append((tensor, class_id))
        else:
            self._image_paths = _image_paths
            self.transforms = transforms
            self._targets_mapping = _targets_mapping

    def __getitem__(self, item):
        if self.pre_load:
            return self.images_and_classes[item]
        else:
            _image_path = self._image_paths[item]
            image = Image.open(_image_path).convert('RGB')
            tensor = self.transforms(image)

            image_name = str(_image_path).split('/')[-1].split('.')[0]
            class_id = self._targets_mapping[image_name]
            return tensor, class_id


    def __len__(self):
        return len(self.images_and_classes)
