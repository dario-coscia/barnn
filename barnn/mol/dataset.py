import torch
import lightning.pytorch as pl
from .utils import download_data, load_smiles_dataset, PreprocessSmiles
from torch.utils.data import DataLoader


class SmilesDataset(torch.utils.data.Dataset):
    """
    Custom PyTorch Dataset that takes a file containing tokenized SMILES.
    """

    def __init__(self, smiles_list):
        self._smiles_list = smiles_list

    def __getitem__(self, i):
        mol     = self._smiles_list[i]
        X = torch.tensor(mol[:-1], dtype=torch.long)
        y = torch.tensor(mol[1:], dtype=torch.long)
        return X, y

    def __len__(self):
        return len(self._smiles_list)
    

class TDCDataModule(pl.LightningDataModule):
    def __init__(self, batch_size = 128, dataset_folder=None):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_folder = dataset_folder
        self.preprocesser = PreprocessSmiles()
        self.tokenizer = None               # overwritten in prepare_data
        self.vocabulary = None              # overwritten in prepare_data
        self.smiles_data = None             # overwritten in prepare_data

    def initialize(self):
        # download
        if self.dataset_folder is None:
            print('Downloading ChemBL dataset from TDC and preprocess it.')
            raw_smiles = download_data(self.dataset_folder)
            do_preprocessing = True
        else:
            raw_smiles = load_smiles_dataset(self.dataset_folder)
            do_preprocessing = False

        self.smiles_data = self.preprocesser.preprocess_smiles(raw_smiles, do_preprocessing)
        # set tokenizer and vocabulary
        self.tokenizer = self.preprocesser.tokenizer
        self.vocabulary = self.preprocesser.vocabulary

    def setup(self, stage):
        # Assign train/val datasets for use in dataloaders
        self.smiles_train = SmilesDataset(
            smiles_list=self.smiles_data["train"]
        )
        self.smiles_val = SmilesDataset(
            smiles_list=self.smiles_data["val"]
        )
        self.smiles_test= SmilesDataset(
            smiles_list=self.smiles_data["val"]
        )

    def train_dataloader(self):
        return DataLoader(self.smiles_train,
                          batch_size=self.batch_size,
                          shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.smiles_val,
                          batch_size=self.batch_size,
                          shuffle=True)

    def predict_dataloader(self):
        return DataLoader(self.smiles_test,
                          batch_size=self.batch_size,
                          shuffle=True)