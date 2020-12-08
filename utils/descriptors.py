import rdkit
from rdkit import Chem
import numpy as np
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from rdkit.DataStructs.cDataStructs import ConvertToNumpyArray

fprint_params = {'bits': 4096, 'radius_multiplier': 1.5, 'rdkit_invariants': True}
confgen_params = {'max_energy_diff': 20.0, 'first': 10}


def ecfp(mol, r=3, nBits=4096, errors_as_zeros=True):
    """
    Дает битовый вектор фингерпринтов для молекулы или строки smiles
    """
    mol = Chem.MolFromSmiles(mol) if not isinstance(mol, rdkit.Chem.rdchem.Mol) else mol
    try:
        arr = np.zeros((1,))
        ConvertToNumpyArray(GetMorganFingerprintAsBitVect(mol, r, nBits), arr)
        return arr.astype(np.float32)
    except:
        print("Error in ecfp")
        return np.NaN if not errors_as_zeros else np.zeros((nBits,), dtype=np.float32)


def ecfp_for_builder(x):
    """
    Добавляет вертикальную размерность 1 для ecfp(x)
    """
    return np.array([ecfp(x)]).astype(np.float32)
