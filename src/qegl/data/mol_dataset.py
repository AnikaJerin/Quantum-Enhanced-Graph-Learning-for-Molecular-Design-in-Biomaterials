# qegl/data/mol_dataset.py

import os
import pandas as pd
from torch_geometric.data import Data, InMemoryDataset
from rdkit import Chem
from rdkit.Chem import AllChem
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch.serialization import safe_globals


def mol_to_graph(mol):
    """
    Convert an RDKit Mol object into a PyTorch Geometric Data object.
    """
    # Node features (atoms)
    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append([
            atom.GetAtomicNum(),
            atom.GetTotalDegree(),
            atom.GetFormalCharge(),
            atom.GetHybridization().real,
            atom.GetTotalNumHs(),
            atom.GetIsAromatic()
        ])
    x = torch.tensor(atom_features, dtype=torch.float)

    # Edges (bonds)
    edge_index = []
    edge_attr = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index.append((i, j))
        edge_index.append((j, i))

        edge_type = [
            bond.GetBondTypeAsDouble(),
            bond.GetIsConjugated(),
            bond.IsInRing()
        ]
        edge_attr.append(edge_type)
        edge_attr.append(edge_type)

    if len(edge_index) > 0:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    else:
        # Handle molecules with no bonds
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 3), dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data


class MoleculeDataset(InMemoryDataset):
    def __init__(self, root, csv_file, task="regression", transform=None, pre_transform=None):
        self.csv_file = csv_file
        self.task = task
        super().__init__(root, transform, pre_transform)

        if os.path.exists(self.processed_paths[0]):
            # Only load if processed file exists
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            # Trigger processing if not yet processed
            self.process()
            self.data, self.slices = torch.load(self.processed_paths[0])
    # def __init__(self, root, csv_file, task="regression", transform=None, pre_transform=None):
    #     """
    #     Args:
    #         root (str): dataset root folder
    #         csv_file (str): path to CSV containing SMILES and target(s)
    #         task (str): 'regression' or 'classification' or 'multitask'
    #     """
    #     self.csv_file = csv_file
    #     self.task = task
    #     super().__init__(root, transform, pre_transform)

    #     self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [os.path.basename(self.csv_file)]

    @property
    def processed_file_names(self):
        return ["molecule_dataset.pt"]

    def download(self):
        # Nothing to download, CSV is already provided
        pass

    def process(self):
        df = pd.read_csv(self.csv_file)

        data_list = []
        for _, row in df.iterrows():
            smiles = row["smiles"]
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue

            data = mol_to_graph(mol)

            # Handle regression, classification, multitask
            if self.task == "regression":
                y = torch.tensor([float(row["gap"])], dtype=torch.float)
            elif self.task == "classification":
                y = torch.tensor([int(row["toxicity"])], dtype=torch.long)
            elif self.task == "multitask":
                y = torch.tensor([float(row["gap"]), float(row["toxicity"])], dtype=torch.float)
            else:
                raise ValueError(f"Unknown task {self.task}")

            data.y = y
            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
