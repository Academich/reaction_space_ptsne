from rdkit.Chem import rdChemReactions
import numpy as np
from rdkit.DataStructs.cDataStructs import ConvertToNumpyArray

FP_TYPES = {"AtomPairFP": rdChemReactions.FingerprintType.AtomPairFP,
            "MorganFP": rdChemReactions.FingerprintType.MorganFP,
            "PatternFP": rdChemReactions.FingerprintType.PatternFP,
            "RDKitFP": rdChemReactions.FingerprintType.RDKitFP,
            "TopologicalTorsion": rdChemReactions.FingerprintType.TopologicalTorsion
            }


def reaction_fps(rx_smi: str,
                 fp_method: str,
                 n_bits: int,
                 fp_type: str,
                 include_agents: bool,
                 agent_weight: int,
                 non_agent_weight: int,
                 bit_ratio_agents: float
                 ) -> np.array:
    # === Parameters section
    params = rdChemReactions.ReactionFingerprintParams()

    # number of bits of the fingerprint
    params.fpSize = n_bits

    # include the agents of a reaction for fingerprint generation
    params.includeAgents = include_agents

    # kind of fingerprint used, e.g AtompairFP.Be aware that only AtompairFP, TopologicalTorsion and MorganFP
    # were supported in the difference fingerprint.
    params.fpType = FP_TYPES[fp_type]
    # ===

    rxn = rdChemReactions.ReactionFromSmarts(
        rx_smi,
        useSmiles=True)

    arr = np.zeros((1,))
    if fp_method == "difference":
        # if agents are included, agents could be weighted compared to reactants and products in difference fingerprints
        params.agentWeight = agent_weight

        # in difference fingerprints weight factor for reactants and products compared to agents
        params.nonAgentWeight = non_agent_weight

        # NOTE: difference fingerprints are not binary
        fps = rdChemReactions.CreateDifferenceFingerprintForReaction(rxn, params)
    elif fp_method == "structural":
        # in structural fingerprints it determines the ratio of bits of the agents in the fingerprint
        params.bitRatioAgents = bit_ratio_agents

        # NOTE: structural fingerprints are binary
        fps = rdChemReactions.CreateStructuralFingerprintForReaction(rxn, params)
    else:
        raise ValueError("Invalid fp_method. Allowed are 'difference' and 'structural'")

    ConvertToNumpyArray(fps, arr)
    return arr
