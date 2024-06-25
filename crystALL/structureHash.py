from typing import Dict
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import hashlib
import base62

def structureHashes(
        structure: Structure
) -> Dict[str, str]:
    """ Generate a list of hashes for a structure. The hash of the structure filed (configuration occupying a structure) 
    itself is not generated here but rather calculated internally within the MongoDB server. Here we abstract it in order 
    to establish the structures themselves as the primary entities shared between entries. We do that (a) through 
    symmetry-reduction and anonymization of the structure, (b) further abstracted by coarse position symmetrization, 
    and lastly by (c) volume normalization.
    """
    
    # Anonymize the structure
    for el in set(structure.species):
        structure.replace_species({el: 'H'})
    structure.remove_oxidation_states()
    for site in structure.sites:
        site.properties = {}
        site.label = {}

    # a. Symmetrized structure with fine precision
    spgAf = SpacegroupAnalyzer(structure, symprec=1e-4)
    symmetrizedStruct = spgAf.get_symmetrized_structure()
    symStructHash = hashlib.md5(
        symmetrizedStruct.to_json().encode())

    # b. Position coarsening and symmetrization / idealization
    # Coarsen the atomic positions to 6 significant digits
    structureDict = structure.as_dict(verbosity=0)
    for i in range(structureDict['sites'].__len__()):
        # Delete the property fields, all should be empty
        del structureDict['sites'][i]['properties']
        for j in range(3):
            # Round atomic positions in structures to 0 plus 5 significant digits
            structureDict['sites'][i]['abc'][j] = round(structureDict['sites'][i]['abc'][j], 5)
    # Coarsen the lattice parameters to 4 decimal digits
    for i in range(3):
        structureDict['lattice']['matrix'][i] = [round(x, 4) for x in structureDict['lattice']['matrix'][i]]
    structure_coarse = Structure.from_dict(structureDict)
    spgAc = SpacegroupAnalyzer(structure_coarse, symprec=0.1)
    coarseSymmetrizedStruct = spgAc.get_symmetrized_structure()
    coarseSymmetrizedHash = hashlib.md5(
        coarseSymmetrizedStruct.to_json().encode())

    # c. Volume normalization
    volumeNormalizedStruct = coarseSymmetrizedStruct.copy().scale_lattice(1)
    volumeNormalizedHash = hashlib.md5(
        volumeNormalizedStruct.to_json().encode())

    return {
        'material.structID': base62.encodebytes(
            symStructHash.digest()),
        'material.coarseStructID': base62.encodebytes(
            coarseSymmetrizedHash.digest()),
        'material.volNormStructID': base62.encodebytes(
            volumeNormalizedHash.digest())
    }