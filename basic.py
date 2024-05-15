from rdkit import Chem


# 1. read
def get_mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    Chem.Kekulize(mol)
    return mol


# 2. output
def get_smiles(mol):
    return Chem.MolToSmiles(mol, kekuleSmiles=True)


# 3. standard
def sanitize(mol):
    try:
        smiles = get_smiles(mol)
        mol = get_mol(smiles)
    except Exception as e:
        return None
    return mol

