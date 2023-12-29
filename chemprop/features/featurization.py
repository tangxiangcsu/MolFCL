from argparse import Namespace
from typing import List, Tuple, Union
import pandas as pd
from rdkit import Chem
import torch
import numpy as np
from torch.utils.data import Dataset
import pickle
import pdb
from rdkit.Chem import BRICS
from rdkit.Chem import Descriptors
from rdkit.Chem.BRICS import FindBRICSBonds,BRICSDecompose,BreakBRICSBonds,BRICSBuild
import os
from rdkit import RDConfig
from rdkit import RDLogger 
from rdkit.Chem import ChemicalFeatures 
import pickle
from rdkit.Chem import MACCSkeys
from dgl.dataloading import GraphDataLoader
import dgl
import networkx as nx
import matplotlib.pyplot as plt
# from chemprop.descriptors.rdNormalizedDescriptors import RDKit2DNormalized
# from multiprocessing import Pool

RDLogger.DisableLog('rdApp.*')  

# Atom feature sizes
MAX_ATOMIC_NUM = 110
PHARM_FEATURE_SIZE= 182
ATOM_FEATURES = {
    'atomic_num': list(range(MAX_ATOMIC_NUM)),
    'degree': [0, 1, 2, 3, 4, 5],
    'formal_charge': [-1, -2, 1, 2, 0],
    'chiral_tag': [0, 1, 2, 3],
    'num_Hs': [0, 1, 2, 3, 4],
    'hybridization': [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2
    ],
}

# rdkit
fdefName = os.path.join(RDConfig.RDDataDir,'BaseFeatures.fdef')
FACTORY = ChemicalFeatures.BuildFeatureFactory(fdefName)
declist = Descriptors.descList
calc = {}
for (i,j) in declist:
    calc[i] = j
# Distance feature sizes
PATH_DISTANCE_BINS = list(range(10))
THREE_D_DISTANCE_MAX = 20
THREE_D_DISTANCE_STEP = 1
THREE_D_DISTANCE_BINS = list(range(0, THREE_D_DISTANCE_MAX + 1, THREE_D_DISTANCE_STEP))

# len(choices) + 1 to include room for uncommon values; + 2 at end for IsAromatic and mass
ATOM_FDIM = sum(len(choices) + 1 for choices in ATOM_FEATURES.values()) + 2
BOND_FDIM = 14
REACT_FDIM = 34
# Memoization
SMILES_TO_GRAPH = {}

def clear_cache():
    """Clears featurization cache."""
    global SMILES_TO_GRAPH
    SMILES_TO_GRAPH = {}

def get_pharm_fdim() -> int:
    return PHARM_FEATURE_SIZE

def get_atom_fdim() -> int:
    """
    Gets the dimensionality of atom features.

    :param: Arguments.
    """
    return ATOM_FDIM

def get_bond_fdim() -> int:
    """
    Gets the dimensionality of bond features.

    :param: Arguments.
    """
    return BOND_FDIM

def get_react_fdim() -> int:
    return REACT_FDIM

def onek_encoding_unk(value: int, choices: List[int]) -> List[int]:
    """
    Creates a one-hot encoding.

    :param value: The value for which the encoding should be one.
    :param choices: A list of possible values.
    :return: A one-hot encoding of the value in a list of length len(choices) + 1.
    If value is not in the list of choices, then the final element in the encoding is 1.
    """
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1

    return encoding

def atom_features(atom: Chem.rdchem.Atom, functional_groups: List[int] = None) -> List[Union[bool, int, float]]:
    """
    Builds a feature vector for an atom.

    :param atom: An RDKit atom.
    :param functional_groups: A k-hot vector indicating the functional groups the atom belongs to.
    :return: A list containing the atom features.
    """
    
    features = onek_encoding_unk(atom.GetAtomicNum() - 1, ATOM_FEATURES['atomic_num']) + \
           onek_encoding_unk(atom.GetTotalDegree(), ATOM_FEATURES['degree']) + \
           onek_encoding_unk(atom.GetFormalCharge(), ATOM_FEATURES['formal_charge']) + \
           onek_encoding_unk(int(atom.GetChiralTag()), ATOM_FEATURES['chiral_tag']) + \
           onek_encoding_unk(int(atom.GetTotalNumHs()), ATOM_FEATURES['num_Hs']) + \
           onek_encoding_unk(int(atom.GetHybridization()), ATOM_FEATURES['hybridization']) + \
           [1 if atom.GetIsAromatic() else 0] + \
           [atom.GetMass() * 0.01]  # scaled to about the same range as other features
    if functional_groups is not None:
        features += functional_groups
    return features

def bond_features(bond: Chem.rdchem.Bond) -> List[Union[bool, int, float]]:
    """
    Builds a feature vector for a bond.

    :param bond: A RDKit bond.
    :return: A list containing the bond features.
    """
    if bond is None:
        fbond = [1] + [0] * (BOND_FDIM - 1)
    else:
        bt = bond.GetBondType()
        fbond = [
            0,  # bond is not None
            bt == Chem.rdchem.BondType.SINGLE,
            bt == Chem.rdchem.BondType.DOUBLE,
            bt == Chem.rdchem.BondType.TRIPLE,
            bt == Chem.rdchem.BondType.AROMATIC,
            (bond.GetIsConjugated() if bt is not None else 0),
            (bond.IsInRing() if bt is not None else 0)
        ]
        fbond += onek_encoding_unk(int(bond.GetStereo()), list(range(6)))
    return fbond

eletype_list = [i for i in range(118)]
hrc2emb,react2emb = {},{}
for eletype in eletype_list:
    hrc_emb = np.random.rand(14)
    hrc2emb[eletype] = hrc_emb
for i in range(17):
    for j in range(17):
        react2emb[(i,j)] = np.random.rand(34)
def hrc_features(ele):
    fhrc = hrc2emb[ele]
    return fhrc.tolist()

def react_features(a,b):
    return react2emb[(a,b)]
# Word2Vec得到的子结构特征，以及反应信息
# frag2emb = pickle.load(open('./cache/fg2emb.pkl', 'rb'))# 官能团嵌入特征
# react2emb = pickle.load(open('./cache/react2emb.pkl', 'rb'))

with open('./chemprop/data/funcgroup.txt', "r") as f:
    funcgroups = f.read().strip().split('\n')
    name = [i.split()[0] for i in funcgroups]
    smart = [Chem.MolFromSmarts(i.split()[1]) for i in funcgroups]
    smart2name = dict(zip(smart, name))
    func2index = {smart2name[sm]:i for i,sm in enumerate(smart)}

# df = pd.read_csv("./chemprop/data/funcgroup.csv")
# smiles_lst = df['smiles'].to_list()  # 示例的分子结构
# name = df['name'].to_list()
# smart = [Chem.MolFromSmarts(s) for s in df['smart'].to_list()]
# smart2name = dict(zip(smart, name))
# func2index = {smart2name[sm]:i for i,sm in enumerate(smart)}


def maccskeys_emb(mol):
    return list(MACCSkeys.GenMACCSKeys(mol))

def pharm_property_types_feats(mol,factory=FACTORY): 
    types = [i.split('.')[1] for i in factory.GetFeatureDefs().keys()]
    feats = [i.GetType() for i in factory.GetFeaturesForMol(mol)]
    result = [0] * len(types)
    for i in range(len(types)):
        if types[i] in list(set(feats)):
            result[i] = 1
    return result

def get_PharmElement(mol_pharm):
    atom_symbol={'C':0,'H':0,'O':0,'N':0,'P':0,
                 'S':0,'F':0,'CL':0,'Br':0,'other':0,}
    feat_pharm_element =[]
    # [C,H,O,N,P,S,F,CL,Br,other]
    for atom in mol_pharm.GetAtoms():
        if atom.GetSymbol() in atom_symbol.keys():
            atom_symbol[atom.GetSymbol()]+=1
        else:
            atom_symbol['other']+=1
    for key,value in atom_symbol.items():
        feat_pharm_element+=[value]
    return feat_pharm_element

def brics_features(mol,pretrain=False):
    fragsmiles = [Chem.MolToSmiles(x,True) for x in Chem.GetMolFrags(BreakBRICSBonds(mol),asMols=True)]
    # fragsmiles1 = Chem.MolToSmiles(BreakBRICSBonds(mol),True).split('.')
    break_bonds = [mol.GetBondBetweenAtoms(i[0][0],i[0][1]).GetIdx() for i in FindBRICSBonds(mol)]
    if break_bonds == []:
        tmp = mol
    else:
        tmp = Chem.FragmentOnBonds(mol,break_bonds,addDummies=False)
    frags_idx_lst = Chem.GetMolFrags(tmp)
    pharm_feats = {}
    atom2pharmid = {}
    for idx,frag_idx in enumerate(frags_idx_lst):
        for atom_idx in frag_idx:
            atom2pharmid[atom_idx] = idx
        try:
            frag_pharm = fragsmiles[idx]
            mol_pharm = Chem.MolFromSmiles(frag_pharm)
            pharm_feat= GetBRICSFeature(mol_pharm)
        except:
            print(f'generate Pharm feature make a error in {Chem.MolToSmiles(mol)}')
            pharm_feat=[0]*PHARM_FEATURE_SIZE
        pharm_feats[idx] = pharm_feat
    
    '''for sm in smart:
        if mol.HasSubstructMatch(sm):
            atom_indices = mol.GetSubstructMatches(sm)
            for atom_lst in atom_indices:
                if atom_lst in frags_idx_lst:
                    smart_exist[smart2name[sm]]=True'''
    return pharm_feats,atom2pharmid,frags_idx_lst

def GetBRICSBondFeature(react_1,react_2,pretrain=False):
    result = []
    start_action_bond = int(react_1) if (react_1 !='7a' and react_1 !='7b') else 7
    end_action_bond = int(react_2) if (react_2 !='7a' and react_2 !='7b') else 7
    try:
        result = list(react2emb[f'L{start_action_bond}>>L{end_action_bond}'])
    except:
        result = np.random.rand(14)
        print('cant find reactive info in react2emb')
    return result
def GetBRICSFeature(mol_pharm):
    try:
        pharm_feat= [calc['TPSA'](mol_pharm)*0.01]+[calc['MolLogP'](mol_pharm)]+\
                                [calc['HeavyAtomMolWt'](mol_pharm)*0.01]+[1 if mol_pharm.GetRingInfo().NumRings()>0 else 0]+\
                                [mol_pharm.GetRingInfo().NumRings()]+\
                                get_PharmElement(mol_pharm)+maccskeys_emb(mol_pharm)
    except:
        pharm_feat = [0]*get_pharm_fdim()
    return pharm_feat

def GetBRICSBondFeature_Hetero(react_1,react_2):
    result = []
    start_action_bond = int(react_1) if (react_1 !='7a' and react_1 !='7b') else 7
    end_action_bond = int(react_2) if (react_2 !='7a' and react_2 !='7b') else 7
    # return react_features(start_action_bond,end_action_bond)
    emb_0 = [0 for i in range(17)]
    emb_1 = [0 for i in range(17)]
    emb_0[start_action_bond] = 0
    emb_1[end_action_bond] = 0
    result = emb_0 + emb_1
    return result

def GetBricsBonds(mol):
    bonds_tmp = FindBRICSBonds(mol)
    bonds = [b for b in bonds_tmp]
    result = {}
    for item in bonds:# item[0] is atom, item[1] is brics type
        result.update({(int(item[0][0]), int(item[0][1])):[int(item[1][0]), int(item[1][1])]})
        result.update({(int(item[0][1]), int(item[0][0])):[int(item[1][1]), int(item[1][0])]})
    return result

def hyper_features(mol,smiles):
    try:
        f = maccskeys_emb(mol)+pharm_property_types_feats(mol)
    except:
        emb_0 = [0 for i in range(167)]
        emb_1 = [0 for i in range(27)]
        f = emb_0+emb_1
    return f

def HyperGraphBRICSFeature(mol):
    # fragsmiles = [Chem.MolToSmiles(x,True) for x in Chem.GetMolFrags(BreakBRICSBonds(mol),asMols=True)]
    raw_idxs = Chem.GetMolFrags(mol)
    # 以下的设置假设fragsmiles 和 frags_idx 对应上
    # break_bonds and frags_idx 不能对应上
    atom2pharmid = {}
    Hyper_edge_feature = []
    FG_feature = []
    Hyper_edge_type = []
    sour,targ = [],[]
    global_hyper_num = 0
    raw_mol = mol
    break_bonds = [mol.GetBondBetweenAtoms(i[0][0],i[0][1]).GetIdx() for i in FindBRICSBonds(mol)]
    raw_break_bonds = list(FindBRICSBonds(mol))
    if break_bonds == []:
        tmp = mol
    else:
        tmp = Chem.FragmentOnBonds(mol,break_bonds,addDummies=False)
    frags_idx_lst = Chem.GetMolFrags(tmp)
    for f_x,fg_idx in enumerate(frags_idx_lst):
        fg_mol = Chem.MolFromSmiles(Chem.MolFragmentToSmiles(raw_mol, fg_idx))
        fg_mol_smiles = Chem.MolFragmentToSmiles(raw_mol, fg_idx)
        FG_feature.append(hyper_features(fg_mol,fg_mol_smiles))
        for atom_idx in fg_idx:
            atom2pharmid[atom_idx] = f_x

    for rid,raw_idx in enumerate(raw_idxs):
        mol =  Chem.MolFromSmiles(Chem.MolFragmentToSmiles(raw_mol, raw_idx))# 原子信息重新编号
        mol_smiles = Chem.MolFragmentToSmiles(raw_mol, raw_idx)
        FIND_break = [b for b in raw_break_bonds if (b[0][0] in raw_idx) and (b[0][1] in raw_idx)]
        # tmp = Chem.FragmentOnBonds(mol,FIND_break,addDummies=False)
        # frags_idx = Chem.GetMolFrags(tmp)# 原子信息重新编号
        hyper2pharm_fg=[(i,fg_idx) for i,fg_idx in enumerate(frags_idx_lst) if set(fg_idx).issubset(set(raw_idx))] # 形成的碎片编号，和原来对应
        if FIND_break ==[]:
            assert len(hyper2pharm_fg)==1
            Hyper_edge_feature.append(hyper_features(mol,mol_smiles))
            sour+=[global_hyper_num]
            targ+=[hyper2pharm_fg[0][0]]
            global_hyper_num+=1
            continue
        for id,b in enumerate(FIND_break):
            #temp_smiles = [Chem.MolToSmiles(x,True) for x in Chem.GetMolFrags(BreakBRICSBonds(mol,[b]),asMols=True)]
            tmp = Chem.FragmentOnBonds(raw_mol,[raw_mol.GetBondBetweenAtoms(b[0][0],b[0][1]).GetIdx()],addDummies=False)
            hyperfg_idx = Chem.GetMolFrags(tmp)
            new_hyperfg_idx = []
            for h_fg_idx in hyperfg_idx:
                if set(h_fg_idx).issubset(set(raw_idx)):
                    new_hyperfg_idx.append(h_fg_idx)
            assert len(new_hyperfg_idx)==2
            hyper12pharm,hyper22pharm = 0,0
            hyper1_id = global_hyper_num
            hyper2_id = global_hyper_num+1
            for fg in hyper2pharm_fg:
                if set(fg[1]).issubset(set(new_hyperfg_idx[0])):
                    hyper12pharm+=1
                    sour.extend([hyper1_id])
                    targ.extend([fg[0]])
                if set(fg[1]).issubset(set(new_hyperfg_idx[1])):
                    hyper22pharm+=1
                    sour.extend([hyper2_id])
                    targ.extend([fg[0]])
            assert hyper12pharm+hyper22pharm==len(hyper2pharm_fg)
            temp_smiles = [Chem.MolFragmentToSmiles(raw_mol, n_fg) for n_fg in new_hyperfg_idx]
            hyper1 = Chem.MolFromSmiles(temp_smiles[0]) # 形成两个片段，用来匹配的
            hyper2 = Chem.MolFromSmiles(temp_smiles[1])
            h1_feat= hyper_features(hyper1,temp_smiles[0])
            h2_feat= hyper_features(hyper2,temp_smiles[1])
            Hyper_edge_feature.append(h1_feat)
            Hyper_edge_feature.append(h2_feat)
            global_hyper_num+=2
            Hyper_edge_type.append([hyper1_id,hyper2_id,(b[1][0],b[1][1])])
    return Hyper_edge_feature,FG_feature,sour,targ,atom2pharmid,Hyper_edge_type

def match_BRICS(smiles,brics2emb):
    fg_emb = [[1] * 300]
    pad_fg = [[0] * 300]
    mapping = [] # 一个碎片最多映射15个原子
    mol = Chem.MolFromSmiles(smiles)
    f_brics,atom2pharmid,frags_idx_lst = brics_features(mol,False)
    if brics2emb.get(smiles) is not None:
        pharm_feat = brics2emb[smiles].tolist()
        fg_emb.extend(pharm_feat)
    for atom_lst in frags_idx_lst:
        atom_l = [x+1 for x in list(atom_lst)]
        if len(atom_l)>15:
            atom_l = atom_l[:15]
        else:
            atom_l = atom_l+[0]*(15-len(atom_l))
        mapping.extend([atom_l])
    if len(fg_emb) > 13:
        fg_emb = fg_emb[:13]
    return fg_emb,mapping

def match_BRICS_group(mol,pretrain):
    fg_emb = []
    mathch_smart2name = []
    mapping = [] # 一个碎片最多映射15个原子
    f_brics,atom2pharmid,frags_idx_lst = brics_features(mol,pretrain)
    fg_emb.extend(list(f_brics.values()))# fragment特征
    for sm in smart:
        if mol.HasSubstructMatch(sm):
            atom_indices = mol.GetSubstructMatches(sm)
            for atom_lst in atom_indices:
                atom_l = [x+1 for x in list(atom_lst)]
                if len(atom_l)>15:
                    atom_l = atom_l[:15]
                else:
                    atom_l = atom_l+[0]*(15-len(atom_l))
                mapping.extend([atom_l])
                mathch_smart2name.append(smart2name[sm])
    '''for atom_lst in frags_idx_lst:
        atom_l = [x+1 for x in list(atom_lst)]
        if len(atom_l)>15:
            atom_l = atom_l[:15]
        else:
            atom_l = atom_l+[0]*(15-len(atom_l))
        mapping.extend([atom_l])'''
    if len(mapping)==0:
        atom_l = [x+1 for x in range(mol.GetNumAtoms())]
        if len(atom_l)>15:
            atom_l = atom_l[:15]
        else:
            atom_l = atom_l+[0]*(15-len(atom_l))
        mapping.extend([atom_l])

    return fg_emb,mapping,mathch_smart2name            

def match_group(mol):
    mapping = []
    func2atom,mathch_smart2name = [],[]
    for sm in smart:
        if mol.HasSubstructMatch(sm):
            atom_indices = mol.GetSubstructMatch(sm)
            mapping.extend([func2index[smart2name[sm]]])
            for atom_lst in [atom_indices]:
                atom_l = [x+1 for x in list(atom_lst)]
                if len(atom_l)>15:
                    atom_l = atom_l[:15]
                else:
                    atom_l = atom_l+[0]*(15-len(atom_l))
                func2atom.extend([atom_l])
                mathch_smart2name.append(smart2name[sm])
    return mapping,func2atom,mathch_smart2name

class MolGraph:
    """
    A MolGraph represents the graph structure and featurization of a single molecule.

    A MolGraph computes the following attributes:
    - smiles: Smiles string.
    - n_atoms: The number of atoms in the molecule.
    - n_bonds: The number of bonds in the molecule.
    - f_atoms: A mapping from an atom index to a list atom features.
    - f_bonds: A mapping from a bond index to a list of bond features.
    - a2b: A mapping from an atom index to a list of incoming bond indices.
    - b2a: A mapping from a bond index to the index of the atom the bond originates from.
    - b2revb: A mapping from a bond index to the index of the reverse bond.
    """

    def __init__(self, smiles: str, args: Namespace, pretrain: bool, brics2emb = None):
        """
        Computes the graph structure and featurization of a molecule.

        :param smiles: A smiles string.
        :param args: Arguments.
        """
        self.brics2emb = brics2emb
        self.smiles = smiles
        self.n_atoms = 0  # number of atoms
        self.n_bonds = 0  # number of bonds
        self.f_atoms = []  # mapping from atom index to atom features
        self.f_bonds = []  # mapping from bond index to concat(in_atom, bond) features
        
        self.n_real_atoms = 0

        self.a2b = []  # mapping from atom index to incoming bond indices
        self.b2a = []  # mapping from bond index to the index of the atom the bond is coming from
        self.b2revb = []  # mapping from bond index to the index of the reverse bond
        self.bonds = []
        # Convert smiles to molecule
        mol = Chem.MolFromSmiles(smiles)
        self.f_brics,self.atom2pharmid,self.frags_idx_lst = brics_features(mol,pretrain)
        self.f_brics = list(self.f_brics.values())
        self.n_brics = len(self.f_brics)
        self.n_reacts = 0
        self.f_reacts = []
        self.p2r = []
        self.r2p = []
        self.r2revb = []
        self.reacts = []
        self.mapping = []
        self.func2atom = []
        self.n_mapping = 0
        self.n_func2atom = 0

        self.smiles_descriptor = []
        if args.step=='BRICS_prompt' and 'frag_attention' in args.add_step:
            self.mapping,self.func2atom,_ = match_group(mol)
            self.n_mapping = len(self.mapping)
            self.n_func2atom = len(self.func2atom)
        self.pretrain = pretrain
        result = GetBricsBonds(mol)
        
        if not self.pretrain:
            # fake the number of "atoms" if we are collapsing substructures
            self.n_atoms = mol.GetNumAtoms()
            # Get atom features
            for i, atom in enumerate(mol.GetAtoms()):
                self.f_atoms.append(atom_features(atom))
            self.f_atoms = [self.f_atoms[i] for i in range(self.n_atoms)]
            for _ in range(self.n_atoms):
                self.a2b.append([])

            # Get bond features
            for a1 in range(self.n_atoms):
                for a2 in range(a1 + 1, self.n_atoms):
                    bond = mol.GetBondBetweenAtoms(a1, a2)
                    if bond is None:
                        continue
                    f_bond = bond_features(bond)

                    if args.atom_messages:
                        self.f_bonds.append(f_bond)
                        self.f_bonds.append(f_bond)
                    else:
                        self.f_bonds.append(self.f_atoms[a1] + f_bond)
                        self.f_bonds.append(self.f_atoms[a2] + f_bond)
                    # Update index mappings
                    b1 = self.n_bonds
                    b2 = b1 + 1
                    self.a2b[a2].append(b1)  # b1 = a1 --> a2
                    self.b2a.append(a1)
                    self.a2b[a1].append(b2)  # b2 = a2 --> a1
                    self.b2a.append(a2)
                    self.b2revb.append(b2)
                    self.b2revb.append(b1)
                    self.n_bonds += 2
                    self.bonds.append(np.array([a1, a2]))   

            # Get pharm features
            for _ in range(self.n_brics):
                self.p2r.append([])
            # Get react features
            for p1 in range(self.n_brics):
                for p2 in range(p1+1,self.n_brics):
                    find = False
                    for a1 in self.frags_idx_lst[p1]:
                        for a2 in self.frags_idx_lst[p2]:
                            if (a1,a2)in result:
                                f_bond1 = GetBRICSBondFeature_Hetero(result[(a1,a2)][0],result[(a1,a2)][1])
                                f_bond2 = GetBRICSBondFeature_Hetero(result[(a2,a1)][0],result[(a2,a1)][1])
                                find=True
                            if find:break
                        if find:break
                    if not find:
                        continue
                    if args.atom_messages:
                        self.f_reacts.append(f_bond1)
                        self.f_reacts.append(f_bond2)
                    else:
                        self.f_reacts.append(self.f_brics[p1]+f_bond1)
                        self.f_reacts.append(self.f_brics[p2]+f_bond2)
                    # Update index mappings
                    b1 = self.n_reacts
                    b2 = b1 + 1
                    self.p2r[p2].append(b1)  # b1 = p1 --> p2
                    self.r2p.append(p1)
                    self.p2r[p1].append(b2)  # b2 = p2 --> p1
                    self.r2p.append(p2)
                    self.r2revb.append(b2)
                    self.r2revb.append(b1)
                    self.n_reacts += 2
                    self.reacts.append(np.array([p1, p2]))
        else:
            # fake the number of "atoms" if we are collapsing substructures
            self.n_real_atoms = mol.GetNumAtoms()
            # Get atom features
            for i, atom in enumerate(mol.GetAtoms()):
                self.f_atoms.append(atom_features(atom))
            self.n_atoms += self.n_brics+self.n_real_atoms
            self.f_atoms += self.f_brics
                        
            self.f_atoms = [self.f_atoms[i] for i in range(self.n_atoms)]
            for _ in range(self.n_atoms):
                self.a2b.append([])

            # Get bond features
            for a1 in range(self.n_atoms):
                for a2 in range(a1 + 1, self.n_atoms):
                    find=False
                    if a2 < self.n_real_atoms:
                        bond = mol.GetBondBetweenAtoms(a1, a2)
                        if bond is None:
                            continue
                        # f_bond = self.f_atoms[a1] + bond_features(bond)
                        f_bond = bond_features(bond)
                    elif a1 < self.n_real_atoms and a2 >= self.n_real_atoms:# a1为原子节点，a2为官能团节点
                        if self.atom2pharmid[a1] == a2-self.n_real_atoms:
                            f_bond = hrc_features(mol.GetAtomWithIdx(a1).GetAtomicNum())
                        else:
                            continue
                    elif a1 >= self.n_real_atoms:# 官能团连接级
                        p1 = self.frags_idx_lst[a1-self.n_real_atoms]
                        p2 = self.frags_idx_lst[a2-self.n_real_atoms]
                        for a1 in p1:
                            for a2 in p2:
                                if (a1,a2)in result:
                                    f_bond1 = GetBRICSBondFeature(result[(a1,a2)][0],result[(a1,a2)][1])
                                    f_bond2 = GetBRICSBondFeature(result[(a2,a1)][0],result[(a2,a1)][1])
                                    find=True
                                if find:break
                            if find:break
                        if not find:
                            continue
                    if args.atom_messages:
                        if find:
                            self.f_bonds.append(f_bond1)
                            self.f_bonds.append(f_bond2)
                        else:
                            self.f_bonds.append(f_bond)
                            self.f_bonds.append(f_bond)
                    else:
                        if find:
                            self.f_bonds.append(self.f_atoms[a1] + f_bond1)
                            self.f_bonds.append(self.f_atoms[a2] + f_bond2)
                        else:
                            self.f_bonds.append(self.f_atoms[a1] + f_bond)
                            self.f_bonds.append(self.f_atoms[a2] + f_bond)
                        
                    # Update index mappings
                    b1 = self.n_bonds
                    b2 = b1 + 1
                    self.a2b[a2].append(b1)  # b1 = a1 --> a2
                    self.b2a.append(a1)
                    self.a2b[a1].append(b2)  # b2 = a2 --> a1
                    self.b2a.append(a2)
                    self.b2revb.append(b2)
                    self.b2revb.append(b1)
                    self.n_bonds += 2
                    self.bonds.append(np.array([a1, a2]))

class BatchMolGraph:
    """
    A BatchMolGraph represents the graph structure and featurization of a batch of molecules.

    A BatchMolGraph contains the attributes of a MolGraph plus:
    - smiles_batch: A list of smiles strings.
    - n_mols: The number of molecules in the batch.
    - atom_fdim: The dimensionality of the atom features.
    - bond_fdim: The dimensionality of the bond features (technically the combined atom/bond features).
    - a_scope: A list of tuples indicating the start and end atom indices for each molecule.
    - b_scope: A list of tuples indicating the start and end bond indices for each molecule.
    - max_num_bonds: The maximum number of bonds neighboring an atom in this batch.
    - b2b: (Optional) A mapping from a bond index to incoming bond indices.
    - a2a: (Optional): A mapping from an atom index to neighboring atom indices.
    """

    def __init__(self, mol_graphs,descriptors, args: Namespace):
        self.smiles_batch = [mol_graph.smiles for mol_graph in mol_graphs]
        self.n_mols = len(self.smiles_batch)

        self.atom_fdim = get_atom_fdim()
        self.bond_fdim = get_bond_fdim() + (not args.atom_messages) * self.atom_fdim # * 2
        self.pharm_fdim = get_pharm_fdim()
        self.react_fdim = get_react_fdim() + (not args.atom_messages) * self.pharm_fdim # * 2
        # Start n_atoms and n_bonds at 1 b/c zero padding
        self.n_atoms = 1  # number of atoms (start at 1 b/c need index 0 as padding)
        self.n_bonds = 1  # number of bonds (start at 1 b/c need index 0 as padding)
        self.n_brics = 1
        self.n_reacts = 1
        self.n_group = 1
        self.n_mapping = 0
        self.n_func2atom = 0

        self.atom_num = []
        self.brics_num = []
        self.group_num = []
        self.a_scope = []  # list of tuples indicating (start_atom_index, num_atoms) for each molecule
        self.b_scope = []  # list of tuples indicating (start_bond_index, num_bonds) for each molecule
        self.brics_scope = []
        self.react_scope = []
        self.group_scope = []
        self.mapping_scope = []
        self.func2atom_scope = []

        # All start with zero padding so that indexing with zero padding returns zeros
        f_atoms = [[0] * self.atom_fdim]  # atom features
        f_bonds = [[0] * self.bond_fdim]  # combined atom/bond features
        f_brics = [[0] * self.pharm_fdim] # pharm features
        f_reacts = [[0] * self.react_fdim] # combined pharm/react features
        if args.step=='BRICS_prompt' and 'frag_attention' in args.add_step:
            f_group = [[0] * 13]
        else:
            f_group = [[0] * self.pharm_fdim]
        mapping = []
        func2atom = []

        a2b = [[]]  # mapping from atom index to incoming bond indices
        b2a = [0]  # mapping from bond index to the index of the atom the bond is coming from
        b2revb = [0]  # mapping from bond index to the index of the reverse bond
        bonds = [[0,0]]
        
        p2r = [[]]
        r2p = [0]
        r2revb = [0]
        reacts = [[0,0]]
        for mol_graph in mol_graphs:
            f_atoms.extend(mol_graph.f_atoms)
            f_bonds.extend(mol_graph.f_bonds)
            f_brics.extend(mol_graph.f_brics)
            f_reacts.extend(mol_graph.f_reacts)
            # f_group.extend(mol_graph.f_group)
            mapping.extend(mol_graph.mapping)
            func2atom.extend(mol_graph.func2atom)

            for a in range(mol_graph.n_atoms):
                a2b.append([b + self.n_bonds for b in mol_graph.a2b[a]]) #  if b!=-1 else 0
            
            for p in range(mol_graph.n_brics):
                p2r.append([r + self.n_reacts for r in mol_graph.p2r[p]])
    
            for b in range(mol_graph.n_bonds):
                b2a.append(self.n_atoms + mol_graph.b2a[b])
                b2revb.append(self.n_bonds + mol_graph.b2revb[b])
                bonds.append([b2a[-1], 
                              self.n_atoms + mol_graph.b2a[mol_graph.b2revb[b]]])
            
            for r in range(mol_graph.n_reacts):
                r2p.append(self.n_brics + mol_graph.r2p[r])
                r2revb.append(self.n_reacts + mol_graph.r2revb[r])
                reacts.append([r2p[-1],
                              self.n_brics + mol_graph.r2p[mol_graph.r2revb[r]]])
            
            self.a_scope.append((self.n_atoms, mol_graph.n_atoms))
            self.b_scope.append((self.n_bonds, mol_graph.n_bonds))
            self.atom_num.append(mol_graph.n_atoms)
            self.n_atoms += mol_graph.n_atoms
            self.n_bonds += mol_graph.n_bonds
            
            self.brics_scope.append((self.n_brics, mol_graph.n_brics))
            self.react_scope.append((self.n_reacts, mol_graph.n_reacts))
            self.brics_num.append(mol_graph.n_brics)
            self.n_brics += mol_graph.n_brics
            self.n_reacts += mol_graph.n_reacts
            
            # self.group_scope.append((self.n_group,mol_graph.n_group))
            # self.group_num.append(mol_graph.n_group)
            # self.n_group += mol_graph.n_group

            self.mapping_scope.append((self.n_mapping,mol_graph.n_mapping))
            self.n_mapping += mol_graph.n_mapping
            
            self.func2atom_scope.append((self.n_func2atom,mol_graph.n_func2atom))
            self.n_func2atom+=mol_graph.n_func2atom

        bonds = np.array(bonds).transpose(1,0)
        reacts = np.array(reacts).transpose(1,0)
        
        self.max_num_bonds = max(1, max(len(in_bonds) for in_bonds in a2b)) # max with 1 to fix a crash in rare case of all single-heavy-atom mols
        self.max_num_reacts = max(1, max(len(in_bonds) for in_bonds in p2r))

        self.f_atoms = torch.FloatTensor(f_atoms)
        self.f_bonds = torch.FloatTensor(f_bonds)
        self.a2b = torch.LongTensor([a2b[a][:self.max_num_bonds] + [0] * (self.max_num_bonds - len(a2b[a])) for a in range(self.n_atoms)])
        self.b2a = torch.LongTensor(b2a)
        self.bonds = torch.LongTensor(bonds)
        self.b2revb = torch.LongTensor(b2revb)
        self.b2b = None  # try to avoid computing b2b b/c O(n_atoms^3)
        self.a2a = None  # only needed if using atom messages

        self.f_brics = torch.FloatTensor(f_brics)
        self.f_reacts = torch.FloatTensor(f_reacts)
        self.p2r = torch.LongTensor([p2r[p][:self.max_num_reacts] + [0] * (self.max_num_reacts - len(p2r[p])) for p in range(self.n_brics)])
        self.r2p = torch.LongTensor(r2p)
        self.reacts = torch.LongTensor(reacts)
        self.r2revb = torch.LongTensor(r2revb)
        self.r2r = None
        self.p2p = None

        self.descriptors = torch.FloatTensor(descriptors)
        # self.f_group = torch.FloatTensor(f_group)
        self.mapping = torch.LongTensor(mapping)
        self.func2atom = torch.LongTensor(func2atom)
    def get_components(self):
        """
        Returns the components of the BatchMolGraph.

        :return: A tuple containing PyTorch tensors with the atom features, bond features, and graph structure
        and two lists indicating the scope of the atoms and bonds (i.e. which molecules they belong to).
        """
        return self.f_atoms, self.f_bonds, self.a2b, self.b2a, self.b2revb, self.a_scope, self.atom_num, \
            self.f_brics, self.f_reacts, self.p2r, self.r2p, self.r2revb, self.brics_scope, self.brics_num,\
            self.mapping, self.mapping_scope,self.func2atom,self.func2atom_scope,self.descriptors

    def get_b2b(self) -> torch.LongTensor:
        """
        Computes (if necessary) and returns a mapping from each bond index to all the incoming bond indices.

        :return: A PyTorch tensor containing the mapping from each bond index to all the incoming bond indices.
        """

        if self.b2b is None:
            b2b = self.a2b[self.b2a]  # num_bonds x max_num_bonds
            # b2b includes reverse edge for each bond so need to mask out
            revmask = (b2b != self.b2revb.unsqueeze(1).repeat(1, b2b.size(1))).long()  # num_bonds x max_num_bonds
            self.b2b = b2b * revmask

        return self.b2b

    def get_a2a(self) -> torch.LongTensor:
        """
        Computes (if necessary) and returns a mapping from each atom index to all neighboring atom indices.

        :return: A PyTorch tensor containing the mapping from each bond index to all the incodming bond indices.
        """
        if self.a2a is None:
            # b = a1 --> a2
            # a2b maps a2 to all incoming bonds b
            # b2a maps each bond b to the atom it comes from a1
            # thus b2a[a2b] maps atom a2 to neighboring atoms a1
            self.a2a = self.b2a[self.a2b]  # num_atoms x max_num_bonds

        return self.a2a

def mol2graph(smiles_batch: List[str],
              args: Namespace, pretrain: bool) -> BatchMolGraph:
    """
    Converts a list of SMILES strings to a BatchMolGraph containing the batch of molecular graphs.

    :param smiles_batch: A list of SMILES strings.
    :param args: Arguments.
    :return: A BatchMolGraph containing the combined molecular graph for the molecules
    """
    '''if args.step == 'BRICS_prompt' and 'frag_attention' in args.add_step :
        brics2emb = pickle.load(open(f'./embedding/{args.exp_id}_frag.pkl','rb'))
    else:'''
    brics2emb = None
    mol_graphs = []
    descriptors = []
    # des = np.load(f"./chemprop/data/{args.exp_id}_descriptor.npy",allow_pickle=True).tolist()
    for smiles in smiles_batch:
        mol_graph = MolGraph(smiles, args, pretrain, brics2emb)
        mol_graphs.append(mol_graph)
        # descriptors.append(list(Chem.RDKFingerprint(Chem.MolFromSmiles(smiles), minPath=1, maxPath=7, fpSize=900)))
    return BatchMolGraph(mol_graphs, descriptors, args)
    
def Mol2HeteroGraph(mol,args:Namespace):
    #build graphs
    edge_types = [('a','b','a'),('p','r','p'),('a','j','p'), ('p','j','a')]
    pharm_feats,atom2pharmid,frags_idx_lst = brics_features(mol,pretrain=False)
    result = GetBricsBonds(mol)
    edges = {k:[] for k in edge_types}
    for bond in mol.GetBonds():
        edges[('a','b','a')].append([bond.GetBeginAtomIdx(),bond.GetEndAtomIdx()])
        edges[('a','b','a')].append([bond.GetEndAtomIdx(),bond.GetBeginAtomIdx()])
    for (a1,a2) in result.keys():
        edges[('p','r','p')].append([atom2pharmid[a1],atom2pharmid[a2]])
    
    for k,v in atom2pharmid.items():
        edges[('a','j','p')].append([k,v])
        edges[('p','j','a')].append([v,k])
    g = dgl.heterograph(edges)
    # atom view
    f_atom = []
    for idx in g.nodes('a'):
        atom = mol.GetAtomWithIdx(idx.item())
        f_atom.append(atom_features(atom))

    f_bond = []
    src,dst = g.edges(etype=('a','b','a'))
    for i in range(g.num_edges(etype=('a','b','a'))):
        if not args.atom_messages:
            f_bond.append(f_atom[src[i].item()]+bond_features(mol.GetBondBetweenAtoms(src[i].item(),dst[i].item())))
        else:
            f_bond.append(bond_features(mol.GetBondBetweenAtoms(src[i].item(),dst[i].item())))
    g.edges[('a','b','a')].data['x'] = torch.FloatTensor(f_bond)
    f_atom = torch.FloatTensor(f_atom)
    g.nodes['a'].data['f'] = f_atom
    dim_atom = len(f_atom[0])

    # pharm view
    f_pharm = []
    for k,v in pharm_feats.items():
        f_pharm.append(v)
    

    f_reac = []
    src, dst = g.edges(etype=('p','r','p'))
    for idx in range(g.num_edges(etype=('p','r','p'))):
        p1 = src[idx].item()
        p2 = dst[idx].item()
        for k,v in result.items():
            if  p1 ==atom2pharmid[k[0]] and p2==atom2pharmid[k[1]]:
                if not args.atom_messages:
                    f_reac.append(f_pharm[p1]+GetBRICSBondFeature_Hetero(v[0],v[1]))
                else:
                    f_reac.append(GetBRICSBondFeature_Hetero(v[0],v[1]))
    
    g.edges[('p','r','p')].data['x'] = torch.FloatTensor(f_reac)
    g.nodes['p'].data['f'] = torch.FloatTensor(f_pharm)
    dim_pharm = len(f_pharm[0])

    dim_atom_padding = g.nodes['a'].data['f'].size()[0] # 原子个数
    dim_pharm_padding = g.nodes['p'].data['f'].size()[0] # 药效团个数
    # junction view
    g.nodes['a'].data['f_junc'] = torch.cat([g.nodes['a'].data['f'], torch.zeros(dim_atom_padding, dim_pharm)], 1)
    g.nodes['p'].data['f_junc'] = torch.cat([torch.zeros(dim_pharm_padding, dim_atom), g.nodes['p'].data['f']], 1)
    
    return g

def Mol2HyperGraph(mol):
    edge_types = [('hyper_edge','con','pharm'),('hyper_edge','react','hyper_edge'),
                 ('pharm','in','hyper_edge'),('pharm','junc','atom'),
                  ('atom','bond','atom'),('atom','junc','pharm')]
    Hyper_edge_feature,FG_feature,sour,targ,atom2pharmid,Hyper_edge_type = HyperGraphBRICSFeature(mol)
    edges = {k:[] for k in edge_types}
    f_reac = []
    for i in range(len(sour)):
        edges[('hyper_edge','con','pharm')].append([sour[i],targ[i]])
        edges[('pharm','in','hyper_edge')].append([targ[i],sour[i]])
    for bond in mol.GetBonds():
        edges[('atom','bond','atom')].append([bond.GetBeginAtomIdx(),bond.GetEndAtomIdx()])
        edges[('atom','bond','atom')].append([bond.GetEndAtomIdx(),bond.GetBeginAtomIdx()])
    for k,v in atom2pharmid.items():
        edges[('atom','junc','pharm')].append([k,v])
        edges[('pharm','junc','atom')].append([v,k])
    for h1,h2,v in Hyper_edge_type:
        edges[('hyper_edge','react','hyper_edge')].append([h1,h2])
        edges[('hyper_edge','react','hyper_edge')].append([h2,h1])
        f_reac.append(GetBRICSBondFeature_Hetero(v[0],v[1]))
        f_reac.append(GetBRICSBondFeature_Hetero(v[1],v[0]))
    g = dgl.heterograph(edges)
    # atom view
    f_atom = []
    for idx in g.nodes('atom'):
        atom = mol.GetAtomWithIdx(idx.item())
        f_atom.append(atom_features(atom))
    f_atom = torch.FloatTensor(f_atom)
    g.nodes['atom'].data['n'] = f_atom

    f_bond = []
    src,dst = g.edges(etype=('atom','bond','atom'))
    for i in range(g.num_edges(etype=('atom','bond','atom'))):
        f_bond.append(bond_features(mol.GetBondBetweenAtoms(src[i].item(),dst[i].item())))
    g.edges[('atom','bond','atom')].data['e'] = torch.FloatTensor(f_bond)


    # pharm view
    f_pharm = torch.FloatTensor(FG_feature)
    g.nodes['pharm'].data['n'] = torch.FloatTensor(f_pharm)
    
    # HyperGraph View
    g.edges[('hyper_edge','react','hyper_edge')].data['e'] = torch.FloatTensor(f_reac)
    g.nodes['hyper_edge'].data['n'] = torch.FloatTensor(Hyper_edge_feature)
    return g

def create_dgl_batch(smiles_batch: List[str], args: Namespace = None, pretrain: bool = None):
    graph,smiles = [],[]
    for smile in smiles_batch:
        mol = Chem.MolFromSmiles(smile)
        g = Mol2HeteroGraph(mol,args)
        graph.append(g)
        smiles.append(smile)
    return dgl.batch(graph),smiles

def create_dglHyper_batch(smiles_batch: List[str], args: Namespace = None, pretrain: bool = None):
    graph = []
    for smile in smiles_batch:
        mol = Chem.MolFromSmiles(smile)
        g = Mol2HyperGraph(mol)
        graph.append(g)
    return dgl.batch(graph)

class Sub_graph(object):
    def __init__(self,smiles_batch,BRICS=True,fg=True) -> None:
        self.smiles_batch = smiles_batch
        self.BRICS = BRICS
        self.fg = fg
    def get_subGrapg(self):
        sub_smiles = []
        for smile in self.smiles_batch:
            try:
                mol = Chem.MolFromSmiles(smile)
            except:
                mol = None
            if mol is None:continue
            delete_atom = []
            delete_bond = []
def brics_decomp(mol):
    n_atoms = mol.GetNumAtoms()
    if n_atoms == 1:
        return [[0]], []

    cliques = []
    breaks = []
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        cliques.append([a1, a2])

    res = list(BRICS.FindBRICSBonds(mol))
    if len(res) == 0:
        return [list(range(n_atoms))], []
    else:
        for bond in res:
            if [bond[0][0], bond[0][1]] in cliques:
                cliques.remove([bond[0][0], bond[0][1]])
            else:
                cliques.remove([bond[0][1], bond[0][0]])
            cliques.append([bond[0][0]])
            cliques.append([bond[0][1]])

    # break bonds between rings and non-ring atoms
    for c in cliques:
        if len(c) > 1:
            if mol.GetAtomWithIdx(c[0]).IsInRing() and not mol.GetAtomWithIdx(c[1]).IsInRing():
                cliques.remove(c)
                cliques.append([c[1]])
                breaks.append(c)
            if mol.GetAtomWithIdx(c[1]).IsInRing() and not mol.GetAtomWithIdx(c[0]).IsInRing():
                cliques.remove(c)
                cliques.append([c[0]])
                breaks.append(c)

    # select atoms at intersections as motif
    for atom in mol.GetAtoms():
        if len(atom.GetNeighbors()) > 2 and not atom.IsInRing():
            cliques.append([atom.GetIdx()])
            for nei in atom.GetNeighbors():
                if [nei.GetIdx(), atom.GetIdx()] in cliques:
                    cliques.remove([nei.GetIdx(), atom.GetIdx()])
                    breaks.append([nei.GetIdx(), atom.GetIdx()])
                elif [atom.GetIdx(), nei.GetIdx()] in cliques:
                    cliques.remove([atom.GetIdx(), nei.GetIdx()])
                    breaks.append([atom.GetIdx(), nei.GetIdx()])
                cliques.append([nei.GetIdx()])

    # merge cliques
    for c in range(len(cliques) - 1):
        if c >= len(cliques):
            break
        for k in range(c + 1, len(cliques)):
            if k >= len(cliques):
                break
            if len(set(cliques[c]) & set(cliques[k])) > 0:
                cliques[c] = list(set(cliques[c]) | set(cliques[k]))
                cliques[k] = []
        cliques = [c for c in cliques if len(c) > 0]
    cliques = [c for c in cliques if len(c) > 0]

    # edges
    edges = []
    for bond in res:
        for c in range(len(cliques)):
            if bond[0][0] in cliques[c]:
                c1 = c
            if bond[0][1] in cliques[c]:
                c2 = c
        edges.append((c1, c2))
    for bond in breaks:
        for c in range(len(cliques)):
            if bond[0] in cliques[c]:
                c1 = c
            if bond[1] in cliques[c]:
                c2 = c
        edges.append((c1, c2))

    return cliques, edges

def get_clique_mol(mol, atoms):
    # get the fragment of clique
    smiles = Chem.MolFragmentToSmiles(mol, atoms, kekuleSmiles=True)
    return smiles

def load_pretrain_motif():
    data_path = './embedding/selected_motifs.txt'
    with open(data_path) as f:
        reader = csv.reader(f)
        motif_lst = []
        for line in reader:
            motif = line[0]
            motif_lst.append(motif)
    return {motif:False for motif in motif_lst}
if __name__=="__main__":
    import csv
    from tqdm import tqdm
    exip = 'zinc15_250K'
    data_path = f'./data/{exip}.csv'
    save_motif_path = f'./embedding/selected_motifs_{exip}.txt'
    with open(data_path) as f:
        reader = csv.reader(f)
        next(reader)  # skip header

        smiles_lst = []
        for line in reader:
            smiles = line[0]
            smiles_lst.append(smiles)
    motif_smart = {}
    smart_name_exit = {smart2name[sm]:0 for sm in smart}
    for smiles in tqdm(smiles_lst):
        mol = Chem.MolFromSmiles(smiles)
        mapping,func2atom,match_smart2name = match_group(mol)
        for ms in match_smart2name:
            smart_name_exit[ms]+=1
    exit_s = 0
    for key,value in smart_name_exit.items():
        if value>0:exit_s+=1
    print(smart_name_exit)
    print(exit_s/len(smart_name_exit.keys()))

    '''fragsmiles = [Chem.MolToSmiles(x,True) for x in Chem.GetMolFrags(BreakBRICSBonds(mol),asMols=True)]
        if len(fragsmiles)==1:# 无法分割
            continue
        for frag in fragsmiles:
            if frag not in motif_smart:
                motif_smart[frag] = 1
            else:
                motif_smart[frag] += 1
    sorted_motif_smart = dict(sorted(motif_smart.items(), key=lambda item: item[1], reverse=True))'''
    '''# 筛选满足阈值的键并保存到 txt 文件
    alpha = 10
    exist_len = 0
    pretrain_motif = load_pretrain_motif()
    for key,value in sorted_motif_smart.items():
        if key in pretrain_motif:
            exist_len+=1
    print(exist_len/len(list(sorted_motif_smart.keys())))
    with open(save_motif_path, "w") as f:
        for key, value in sorted_motif_smart.items():
            if value >= alpha:
                f.write(key + "\n")'''