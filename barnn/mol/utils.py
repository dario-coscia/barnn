import re
import rdkit
import rdkit.Chem
import rdkit.Chem.Draw
from tdc.generation import MolGen
import rdkit
from rdkit.Chem import SaltRemover
from rdkit.Chem.rdmolops import RemoveStereochemistry
from rdkit.Chem import SaltRemover
from rdkit import rdBase
rdBase.DisableLog("rdApp.error")
import os
import argparse


class Vocabulary:
    """
    Stores the tokens and their conversion to vocabulary indexes.
    """
    def __init__(self, tokens):
        token2label = {"[PAD]": 0} # padding index is zero
        token2label.update({token: i for i, token in enumerate(sorted(set(token for inp in tokens for token in inp)), start=1)})
        self.vocabulary_map = token2label
        self.inverted_vocabulary_map = {v: k for k, v in token2label.items()}

    def __len__(self):
        return len(self.vocabulary_map.keys())
    
    def token2labels(self, sequences):
        """
        Decodes a vocabulary index array into a list of tokens.
        """
        return [[self.vocabulary_map[s] for s in seq] for seq in sequences]
    
    def labels2tokens(self, tokens):
        """
        Decodes a vocabulary index array into a list of tokens.
        """
        return [[self.inverted_vocabulary_map[t] for t in tok] for tok in tokens]


class SMILESTokenizer:
    """
    Handles the tokenization and untokenization of SMILES.

    REGEX from: https://github.com/molML/s4-for-de-novo-drug-design
    """
    _ELEMENTS_STR = r"(?<=\[)Cs(?=\])|Si|Xe|Ba|Rb|Ra|Sr|Dy|Li|Kr|Bi|Mn|He|Am|Pu|Cm|Pm|Ne|Th|Ni|Pr|Fe|Lu|Pa|Fm|Tm|Tb|Er|Be|Al|Gd|Eu|te|As|Pt|Lr|Sm|Ca|La|Ti|Te|Ac|Cf|Rf|Na|Cu|Au|Nd|Ag|Se|se|Zn|Mg|Br|Cl|Pb|U|V|K|C|B|H|N|O|S|P|F|I|b|c|n|o|s|p"
    __REGEXES = rf"(\[[^\]]+]|{_ELEMENTS_STR}|" + r"\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%\d{2}|\d)"
    _RE_PATTERN = {'regex': re.compile(__REGEXES)}

    def tokenize(self, smiles, with_beg_and_end=True):
        """
        Tokenize a list of SMILES into a list of list of tokens.
        """
        return [["[BEG]"] + self.single_tokenize(smile) + ["[END]"] if with_beg_and_end else self.single_tokenize(smile) for smile in smiles]
    
    def single_tokenize(self, smile):
        """
        Tokenize a SMILES string into a list of tokens.
        """
        regex = self._RE_PATTERN["regex"]
        tokens = regex.findall(smile)
        return tokens

    def untokenize(self, tokens):
        """
        Untokenizes SMILES tokens.
        """
        return "".join(token for token in tokens if token not in {"[BEG]", "[END]", "[PAD]"})

class PreprocessSmiles:
    allowed_atoms = {'C', 'H', 'O', 'N', 'S', 'P', 'F', 'Cl', 'Br', 'I'}

    def __init__(self, max_length=100):
        self._tokenizer = SMILESTokenizer()
        self._vocabulary = None
        self.max_length = max_length

    def preprocess_smiles(self, smiles, do_preprocessing=False):
        """
        Preprocess smiles (if do_preprocessing=True) by:
            1. Check if the molecule is valid
            2. Retain only allowed atoms
            3. Remove salts and disconnected structures
            4. Get canonical SMILES and select SMILES smaller than max_length
            5. Tokenize the smiles
            6. Create the vocabulary
            7. Pad the molecules to a fixed max_length
            8. Create a list of label encoded and padded molecules
        Returning the valid smiles after preprocessing. Each molecule is a
        list of integers representing the SMILES tokens.
        """
        # 1-4. Get valid smiles
        valid_smiles = {}
        for key, smile_list in smiles.items():
            valid_smiles[key] = [self._preprocess_smiles(mol, do_preprocessing) for mol in smile_list]
        # 5. Tokenize smiles
        data = [item for sublist in valid_smiles.values() for item in sublist]
        tokens = self._tokenizer.tokenize(data)
        # 6. Built the vocabulary
        self._vocabulary = Vocabulary(tokens)
        # 7. Pad the molecules
        final_smiles = {}
        for key, smile_list in valid_smiles.items():
            tokens = self._tokenizer.tokenize(smile_list)
            padded_tokes = self._pad_sequences(tokens)
            final_smiles[key] = self.vocabulary.token2labels(padded_tokes)
        return final_smiles
    
    @property
    def vocabulary(self):
        """
        Returns the vocabulary of the SMILES strings.
        """
        return self._vocabulary
    
    @property
    def tokenizer(self):
        return self._tokenizer
    
    def _pad_sequences(self, sequences, padding_value="[PAD]"):
        """
        Pad sequences to a max_length where padding is done at the end of
        the sequences.
        """
        return [(seq + [padding_value] * max(self.max_length - len(seq), 0))[:self.max_length] for seq in sequences]

    def _contains_only_allowed_atoms(self, mol):
        """
        Only selecte atoms allowed atoms, see PreprocessSmiles.allowed_atoms.
        """
        for atom in mol.GetAtoms():
            if atom.GetSymbol() not in self.allowed_atoms:
                return False
        return True

    def _remove_salts(self, mol):
        """
        Remove salts and disconnected structures.
        """
        remover = SaltRemover.SaltRemover()
        return remover.StripMol(mol)

    def _remove_stereo_and_charge(self, mol):
        """
        Remove stereochemistry annotations and charge.
        """
        RemoveStereochemistry(mol)
        rdkit.Chem.SanitizeMol(mol)
        return mol

    def _smiles_token_count(self, smiles):
        """
        Token count for SMILES strings.
        """
        return len(smiles)
    
    def _preprocess_smiles(self, smiles, do_preprocessing=False):
        """
        Preprocess smiles by:
            1. Check if the molecule is valid
            2. Retain only allowed atoms
            3. Remove salts and disconnected structures
            4. Get canonical SMILES and select SMILES smaller than max_length
        Notice, this is the prepocessing done by https://www.nature.com/articles/s41467-024-50469-9
        if the same dataset is used preprocessing is not needed. The folder
        data/ contains the same datasets.
        """
        if not do_preprocessing:
            return smiles
        mol = rdkit.Chem.MolFromSmiles(smiles)
        # Check if the molecule is valid
        if mol is None:
            return None
        # Retain only allowed atoms
        if not self._contains_only_allowed_atoms(mol):
            return None
        # Remove salts and disconnected structures
        mol = self._remove_salts(mol)
        # Remove stereochemistry and neutralize charge
        mol = self._remove_stereo_and_charge(mol)
        # Get canonical SMILES and check token count
        canon_smiles = rdkit.Chem.MolToSmiles(mol, canonical=True)
        if self._smiles_token_count(canon_smiles) > self.max_length:
            return None
        return canon_smiles

def download_data(dataset_name, seed=111, frac=[0.8, 0.2, 0.0]):
    # get the data
    data = MolGen(name = dataset_name)
    split = data.get_split(seed=seed, frac=frac)
    train_smiles = split['train'].smiles.to_list()
    val_smiles = split['valid'].smiles.to_list()
    # return dataset of smiles
    dataset = {'train' : train_smiles, 
               'val' : val_smiles}
    return dataset

def load_smiles_dataset(folder_path):
    train_path = os.path.join(folder_path, 'train.txt')
    val_path = os.path.join(folder_path, 'val.txt')
    with open(train_path, 'r') as train_file:
        train_smiles = [line.strip() for line in train_file]
    with open(val_path, 'r') as val_file:
        val_smiles = [line.strip() for line in val_file]
    return {'train' : train_smiles, 
            'val' : val_smiles}

def get_valid_smiles(smiles_list):
    valid_smiles = []
    for smile in smiles_list:
        mol = rdkit.Chem.MolFromSmiles(smile)
        if mol:  # If mol is not None, the SMILES is valid
            valid_smiles.append(smile)
    return valid_smiles

def plot_smiles(smiles, filename='molecules', smileinrows=3):
    total_smiles = smileinrows ** 2  # Total number of molecules to display
    valid_smiles = get_valid_smiles(smiles)  # Get valid SMILES from the input list
    valid_smiles = valid_smiles[:total_smiles]  # Limit the number of valid SMILES to the grid size
    
    # Convert valid SMILES to RDKit molecule objects
    mols = [rdkit.Chem.MolFromSmiles(smile) for smile in valid_smiles]
    
    # Create the grid image of molecules
    img = rdkit.Chem.Draw.MolsToGridImage(mols=mols,
                               molsPerRow=smileinrows)

    # Save the image to a file
    img.save(f'{filename}.png')