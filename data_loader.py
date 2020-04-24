from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import torch


class DataGenerator(DataLoader):
    def __init__(self, is_infinite=False, device=torch.device('cpu'), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = device
        self.is_infinite = is_infinite
        self.reload_iterator()

    def reload_iterator(self):
        self.dataset_iterator = super().__iter__()

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.dataset_iterator)
        except StopIteration:
            if self.is_infinite:
                self.reload_iterator()
            batch = next(self.dataset_iterator)
        batch = [elem.to(self.device) for elem in batch]
        return batch

    def get_classes_to_idx(self):
        return self.dataset.dataset.class_to_idx

    def get_classes(self):
        return self.dataset.dataset.classes


def create_data_generators(dataset_name, domain, data_path="data", batch_size=16,
                           transformations=None, num_workers=1, split_ratios=[0.8, 0.1, 0.1],
                           image_size=500, infinite_train=False, device=torch.device('cpu')):
    """
    Args:
        dataset_name (string)
        domain (string)
            - valid domain of the dataset dataset_name
        data_path (string)
            - valid path, which contains dataset_name folder
        batch_size (int)
        transformations (callable)
            - optional transform applied on image sample
        num_workers (int)
            - multi-process data loading
        split_ratios (list of ints, len(split_ratios) = 3)
            - ratios of train, validation and test parts

    Return:
        3 data generators  - for train, validation and test data
    """
    if transformations is None:
        transformations = transforms.Compose([
                                                transforms.Resize(image_size),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225]),
                                            ])

    dataset = create_dataset(dataset_name, domain, data_path, transformations, device)

    len_dataset = len(dataset)
    train_size = int(len_dataset * split_ratios[0])
    val_size = int(len_dataset * split_ratios[1])
    test_size = len_dataset - train_size - val_size

    torch.manual_seed(42)
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_dataloader = DataGenerator(is_infinite=infinite_train, device=device, dataset=train_dataset,
                                     batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    val_dataloader = DataGenerator(is_infinite=False, device=device, dataset=val_dataset,
                                   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_dataloader = DataGenerator(is_infinite=False, device=device, dataset=test_dataset,
                                    batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_dataloader, val_dataloader, test_dataloader


def create_dataset(dataset_name, domain, data_path, transformations, device):
    """
    Args:
        dataset_name (string)
        domain (string)
            - valid domain of the dataset dataset_name
        data_path (string)
            - valid path, which contains dataset_name folder
        transformations (callable)
            - optional transform to be applied on an image sample

    Return:
        torchvision.dataset object
    """

    assert dataset_name in ["office-31"], f"Dataset {dataset_name} is not implemented"

    if dataset_name == "office-31":

        dataset_domains = ["amazon", "dslr", "webcam"]

        assert domain in dataset_domains, f"Incorrect domain {domain}: " + \
            f"dataset {dataset_name} domains: {dataset_domains}"

        dataset = ImageFolder(f"{data_path}/{dataset_name}/{domain}/images", transform=transformations)

    return dataset
