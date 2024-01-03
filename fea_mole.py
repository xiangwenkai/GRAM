# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import torch
import rdkit
from rdkit import Chem


from rdkit.Chem import Lipinski, rdMolDescriptors
from rdkit.Chem.AtomPairs.Utils import NumPiElectrons
RD_PT = Chem.rdchem.GetPeriodicTable()
from rdkit.Chem import Crippen

import dgl
import glob
from rdkit import Chem
import pickle
import os
from dgl import backend 
import sys
from torch import nn
table=Chem.GetPeriodicTable()

def atom_hybridization(atom):
    """One hot encoding for the hybridization of an atom.
    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.
    allowable_set : list of rdkit.Chem.rdchem.HybridizationType
        Atom hybridizations to consider. Default: ``Chem.rdchem.HybridizationType.SP``,
        ``Chem.rdchem.HybridizationType.SP2``, ``Chem.rdchem.HybridizationType.SP3``,
        ``Chem.rdchem.HybridizationType.SP3D``, ``Chem.rdchem.HybridizationType.SP3D2``.
    encode_unknown : bool
        If True, map inputs not in the allowable set to the
        additional last element. (Default: False)
    Returns
    -------
    list
        List of boolean values where at most one value is True.
    See Also
    --------
    one_hot_encoding
    """
    
    return int(atom.GetHybridization())

def atom_partial_charge(atom):
    """Get Gasteiger partial charge for an atom.
    For using this function, you must have called ``AllChem.ComputeGasteigerCharges(mol)``
    to compute Gasteiger charges.
    Occasionally, we can get nan or infinity Gasteiger charges, in which case we will set
    the result to be 0.
    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.
    Returns
    -------
    list
        List containing one float only.
    """
    gasteiger_charge = atom.GetProp('_GasteigerCharge')
    if gasteiger_charge in ['-nan', 'nan', '-inf', 'inf']:
        gasteiger_charge = 0
    return float(gasteiger_charge)
def is_h_acceptor(atom):
    """ Is an H acceptor? """

    m = atom.GetOwningMol()
    idx = atom.GetIdx()
    return idx in [i[0] for i in Lipinski._HAcceptors(m)]   
def is_h_donor(a):
    """ Is an H donor? """

    m = a.GetOwningMol()
    idx = a.GetIdx()
    return idx in [i[0] for i in Lipinski._HDonors(m)]


def is_hetero(a):
    """ Is a heteroatom? """

    m = a.GetOwningMol()
    idx = a.GetIdx()
    return idx in [i[0] for i in Lipinski._Heteroatoms(m)]
def explicit_valence(a):
    """ Explicit valence of atom """
    return a.GetExplicitValence()
def implicit_valence(a):
    """ Implicit valence of atom """

    return a.GetImplicitValence()
def n_valence_electrons(a):
    """ return the number of valance electrons an atom has """

    return RD_PT.GetNOuterElecs(a.GetAtomicNum())
def n_pi_electrons(a):
    """ returns number of pi electrons """

    return NumPiElectrons(a)

def degree(a):
    """ returns the degree of the atom """

    return a.GetDegree()
def formal_charge(a):
    """ Formal charge of atom """

    return a.GetFormalCharge()
def num_implicit_hydrogens(a):
    """ Number of implicit hydrogens """

    return a.GetNumImplicitHs()


def num_explicit_hydrogens(a):
    """ Number of explicit hydrodgens """

    return a.GetNumExplicitHs()


def n_hydrogens(a):
    """ Number of hydrogens """

    return num_implicit_hydrogens(a) + num_explicit_hydrogens(a)
def n_lone_pairs(a):
    """ returns the number of lone pairs assicitaed with the atom """

    return int(0.5 * (n_valence_electrons(a) - degree(a) - n_hydrogens(a) -
                      formal_charge(a) - n_pi_electrons(a)))
def crippen_log_p_contrib(a):
    """ Hacky way of getting logP contribution. """

    idx = a.GetIdx()
    m = a.GetOwningMol()
    return Crippen._GetAtomContribs(m)[idx][0]
def crippen_molar_refractivity_contrib(a):
    """ Hacky way of getting molar refractivity contribution. """

    idx = a.GetIdx()
    m = a.GetOwningMol()
    return Crippen._GetAtomContribs(m)[idx][1]
def tpsa_contrib(a):
    """ Hacky way of getting total polar surface area contribution. """

    idx = a.GetIdx()
    m = a.GetOwningMol()
    return rdMolDescriptors._CalcTPSAContribs(m)[idx]
def atomic_mass(a):
    """ Atomic mass of atom """

    return a.GetMass()
def labute_asa_contrib(a):
    """ Hacky way of getting accessible surface area contribution. """

    idx = a.GetIdx()
    m = a.GetOwningMol()
    return rdMolDescriptors._CalcLabuteASAContribs(m)[0][idx]
def van_der_waals_radius(a):
    """ returns van der waals radius of the atom """
    return PeriodicTable.GetRvdw(rdkit.Chem.GetPeriodicTable(),+a.GetAtomicNum())
def get_min_ring(a):
    mol_=a.GetOwningMol()  
    r=mol_.GetRingInfo()
    r_lst=r.AtomRings()
    min_ring=min([len(i) for i in r_lst])
    return min_ring
def get_min_ring(b):
    mol_=b.GetOwningMol()  
    r=mol_.GetRingInfo()
    r_lst=r.AtomRings()
    
    count_ring=[len(i) for i in r_lst]
    if count_ring==[]:
        min_ring=0
    else:
        min_ring=min(count_ring)
    return min_ring
def one_hot_encoding(x, allowable_set, encode_unknown=False):
    """One-hot encoding.
    Parameters
    ----------
    x
        Value to encode.
    allowable_set : list
        The elements of the allowable_set should be of the
        same type as x.
    encode_unknown : bool
        If True, map inputs not in the allowable set to the
        additional last element.
    Returns
    -------
    list
        List of boolean values where at most one value is True.
        The list is of length ``len(allowable_set)`` if ``encode_unknown=False``
        and ``len(allowable_set) + 1`` otherwise.
    Examples
    --------
    >>> from dgllife.utils import one_hot_encoding
    >>> one_hot_encoding('C', ['C', 'O'])
    [True, False]
    >>> one_hot_encoding('S', ['C', 'O'])
    [False, False]
    >>> one_hot_encoding('S', ['C', 'O'], encode_unknown=True)
    [False, False, True]
    """
    if encode_unknown and (allowable_set[-1] is not None):
        allowable_set.append(None)

    if encode_unknown and (x not in allowable_set):
        x = None

    return list(map(lambda s: x == s, allowable_set))
def atom_type_one_hot(atom, allowable_set=None, encode_unknown=False):
    """One hot encoding for the type of an atom.
    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.
    allowable_set : list of str
        Atom types to consider. Default: ``C``, ``N``, ``O``, ``S``, ``F``, ``Si``, ``P``,
        ``Cl``, ``Br``, ``Mg``, ``Na``, ``Ca``, ``Fe``, ``As``, ``Al``, ``I``, ``B``, ``V``,
        ``K``, ``Tl``, ``Yb``, ``Sb``, ``Sn``, ``Ag``, ``Pd``, ``Co``, ``Se``, ``Ti``, ``Zn``,
        ``H``, ``Li``, ``Ge``, ``Cu``, ``Au``, ``Ni``, ``Cd``, ``In``, ``Mn``, ``Zr``, ``Cr``,
        ``Pt``, ``Hg``, ``Pb``.
    encode_unknown : bool
        If True, map inputs not in the allowable set to the
        additional last element. (Default: False)
    Returns
    -------
    list
        List of boolean values where at most one value is True.
    See Also
    --------
    one_hot_encoding
    atomic_number
    atomic_number_one_hot
    """
    if allowable_set is None:
        allowable_set = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'I', 'B', 'H']
    return one_hot_encoding(atom.GetSymbol(), allowable_set, encode_unknown)
def atom_degree_one_hot(atom, allowable_set=None, encode_unknown=False):
    """One hot encoding for the degree of an atom.
    Note that the result will be different depending on whether the Hs are
    explicitly modeled in the graph.
    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.
    allowable_set : list of int
        Atom degrees to consider. Default: ``0`` - ``10``.
    encode_unknown : bool
        If True, map inputs not in the allowable set to the
        additional last element. (Default: False)
    Returns
    -------
    list
        List of boolean values where at most one value is True.
    See Also
    --------
    one_hot_encoding
    atom_degree
    atom_total_degree
    atom_total_degree_one_hot
    """
    if allowable_set is None:
        allowable_set = list(range(7))
    return one_hot_encoding(atom.GetDegree(), allowable_set, encode_unknown)
def atom_hybridization_one_hot(atom, allowable_set=None, encode_unknown=False):
    """One hot encoding for the hybridization of an atom.
    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.
    allowable_set : list of rdkit.Chem.rdchem.HybridizationType
        Atom hybridizations to consider. Default: ``Chem.rdchem.HybridizationType.SP``,
        ``Chem.rdchem.HybridizationType.SP2``, ``Chem.rdchem.HybridizationType.SP3``,
        ``Chem.rdchem.HybridizationType.SP3D``, ``Chem.rdchem.HybridizationType.SP3D2``.
    encode_unknown : bool
        If True, map inputs not in the allowable set to the
        additional last element. (Default: False)
    Returns
    -------
    list
        List of boolean values where at most one value is True.
    See Also
    --------
    one_hot_encoding
    """
    if allowable_set is None:
        allowable_set = [Chem.rdchem.HybridizationType.SP,
                         Chem.rdchem.HybridizationType.SP2,
                         Chem.rdchem.HybridizationType.SP3,
                         Chem.rdchem.HybridizationType.SP3D,
                         Chem.rdchem.HybridizationType.SP3D2]
    return one_hot_encoding(atom.GetHybridization(), allowable_set, encode_unknown)
def atom_chirality_type_one_hot(atom, allowable_set=None, encode_unknown=False):
    """One hot encoding for the chirality type of an atom.
    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.
    allowable_set : list of str
        Chirality types to consider. Default: ``R``, ``S``.
    encode_unknown : bool
        If True, map inputs not in the allowable set to the
        additional last element. (Default: False)
    Returns
    -------
    list
        List containing one bool only.
    See Also
    --------
    one_hot_encoding
    atom_chiral_tag_one_hot
    """
    if not atom.HasProp('_CIPCode'):
        return [False, False]

    if allowable_set is None:
        allowable_set = ['R', 'S']
    return one_hot_encoding(atom.GetProp('_CIPCode'), allowable_set, encode_unknown)
def bond_type_one_hot(bond, allowable_set=None, encode_unknown=False):
    """One hot encoding for the type of a bond.
    Parameters
    ----------
    bond : rdkit.Chem.rdchem.Bond
        RDKit bond instance.
    allowable_set : list of Chem.rdchem.BondType
        Bond types to consider. Default: ``Chem.rdchem.BondType.SINGLE``,
        ``Chem.rdchem.BondType.DOUBLE``, ``Chem.rdchem.BondType.TRIPLE``,
        ``Chem.rdchem.BondType.AROMATIC``.
    encode_unknown : bool
        If True, map inputs not in the allowable set to the
        additional last element. (Default: False)
    Returns
    -------
    list
        List of boolean values where at most one value is True.
    See Also
    --------
    one_hot_encoding
    """
    if allowable_set is None:
        allowable_set = [Chem.rdchem.BondType.SINGLE,
                         Chem.rdchem.BondType.DOUBLE,
                         Chem.rdchem.BondType.TRIPLE,
                         Chem.rdchem.BondType.AROMATIC]
    return one_hot_encoding(bond.GetBondType(), allowable_set, encode_unknown)
def featurize_atoms(mol):
    feats = []
#    AllChem.ComputeGasteigerCharges(mol)
    num_atom=len(mol.GetAtoms())

#    print(Chem.MolToSmiles(mol))
    for atom in mol.GetAtoms():
#        dict_a=atom.GetPropsAsDict()
#        acsf=list(envascf(d,atom.GetIdx()))
#        print(acsf)
#        print(dict_a['molFileAlias'])
        atom_num=atom.GetAtomicNum()
        feats.append(  [atom_num,table.GetRvdw(atom_num)]+atom_hybridization_one_hot(atom)+[
                      
                      
                     
                      formal_charge(atom),
                      
                      atom.GetNumRadicalElectrons(),
                      int(atom.GetIsAromatic()),
                      
                    num_implicit_hydrogens(atom),
                    num_explicit_hydrogens(atom),
                    explicit_valence(atom),
                    implicit_valence(atom),
                      atom.GetTotalDegree(),
                     ]+atom_chirality_type_one_hot(atom))
    
    return {'hv': torch.Tensor(feats).reshape(num_atom, -1).float(),}



def construct_bigraph_from_mol_int(mol, node_featurize):
    g = dgl.graph(([], []), idtype=torch.int32)

    # Add nodes
   
    
    num_atoms = mol.GetNumAtoms()
    g.add_nodes(num_atoms)
    g.ndata.update(node_featurize(mol))
    src_list = []
    dst_list = []
    bond_fea=[]
    for i in range(num_atoms):
        bf=[0]*7        # [0, 0, 0, 0, ...]
        bond_fea.append(bf) # [[0,0,0,0,...],[0,0,0,0,...],[0,0,0,0,...],...]
        src_list.append(i)  # [0,1,2,3,4,...]
        dst_list.append(i)
        
    for b in mol.GetBonds():
        bf=[int(b.GetIsConjugated()), int(b.GetIsAromatic()),int(b.IsInRing()),]+bond_type_one_hot(b)
        
        bond_fea.append(bf)
        bond_fea.append(bf)    # 因为是有向图，所以需要append两次
        begin_idx=b.GetBeginAtomIdx()
        end_idx=b.GetEndAtomIdx()
        src_list.extend([begin_idx,end_idx])
        dst_list.extend([end_idx,begin_idx])
    

    g.add_edges(torch.IntTensor(src_list), torch.IntTensor(dst_list))
    
    g.edata.update({'he':torch.Tensor(bond_fea)})
    return g

def nan2zero(a):
    a = torch.where(torch.isnan(a), torch.full_like(a, 0), a)
    return a
def inf2zero(a):
    a = torch.where(torch.isinf(a), torch.full_like(a, 0), a)
    return a

        

    
if __name__=="__main__":
    import glob
    from rdkit import Chem
    import pickle
    import os
    from dgl import backend 
    import numpy as np
    import pandas as pd
    from tqdm import tqdm
    from rdkit.Chem import AllChem
    path = '/home/nilin/3Dconformer_final/data/qm9_G.csv'
    with open('/home/nilin/3Dconformer_final/data/qm9_G.csv','rb') as p:
        df = pickle.load(p)
    for i in tqdm(range(df.shape[0])):
        mu = df.iloc[i].mu
        alpha = df.iloc[i].alpha
        HOMO = df.iloc[i].homo
        LUMO = df.iloc[i].lumo
        gap = df.iloc[i].gap
        R2 = df.iloc[i].r2
        ZPVE = df.iloc[i].zpve
        U0 = df.iloc[i].u0
        U = df.iloc[i].u298
        H = df.iloc[i].h298
        G_qm9 = df.iloc[i].g298
        Cv = df.iloc[i].cv
        # smi = df.at[i,'smiles']
        mol_3d = list(df.iloc[i].mol_3d)[0]
        G = list(df.iloc[i].G_matrix)[0]
        d = AllChem.Get3DDistanceMatrix(mol_3d)
        g = construct_bigraph_from_mol_int(mol_3d,featurize_atoms)
        # noise = torch.from_numpy(np.random.normal(mu, sigma, (g.ndata['hv'].shape[0],int(g.ndata['hv'].shape[1]/5))))
        # g.ndata['hv'] = torch.cat((g.ndata['hv'] , noise),1)
        g_G_dir = "/home/nilin/3Dconformer_final/data/mol_g_G_12_qm9"
        filename = "g_G_" + str(i)
        g_G_path = os.path.join(g_G_dir,filename)
        with open(g_G_path,"wb") as g_G_file:
            pickle.dump((mol_3d,g,G,mu,alpha,HOMO,LUMO,gap,R2,ZPVE,U0,U,H,G_qm9,Cv), g_G_file)
        # print('hv',g.ndata['hv'].shape)
        # print('he',g.edata['he'].shape)
  