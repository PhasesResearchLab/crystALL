# Base Python imports
import sys
import os
import io
import math
import re
import json
import csv
from itertools import groupby, permutations, product, combinations
from operator import itemgetter
from importlib import resources
from statistics import mean
from typing import List, Dict, Union, Literal

# Common tool libraries
import numpy as np
import pandas as pd
from tqdm import tqdm
from natsort import natsorted
from colorama import Fore, Style
from distinctipy import get_colors
from plotly.offline import plot, iplot
import plotly.graph_objs as go

# Disable undesired warnings from torch and pymatgen
import warnings
warnings.filterwarnings("ignore", message="(?:.|\n)*datapipes.*dataloader2.*deprecated(?:.|\n)*")
warnings.filterwarnings("ignore", message="(?:.|\n)*electronegativity(?:.|\n)*")

# ML / Data
from pymongo.database import Database
from scipy.spatial import ConvexHull
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score, pairwise_distances
from sklearn.cluster import KMeans, DBSCAN

# Atomistic / Atomistic ML
from pysipfenn import Calculator
from pysipfenn.descriptorDefinitions.KS2022 import onlyStructural
from alignn import pretrained
from pymatgen.core import Structure, Composition, Element
from pymatgen.io.vasp.inputs import Poscar
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram

# Top-Level Functions *************************************************************************

def run_local(
        targetFormula: Union[str, Composition],
        prototypePath: str
    ) -> str:
    """
    Run prototype identification and elemental substitution for permutations within the anonymous formula.

    Args:
        targetFormula: The target formula as a string or pymatgen Composition object.
        prototypePath: Path to the directory containing prototype files.

    Returns:
        Path to the directory containing the generated prototypes.

    Raises:
        ValueError: If the target formula is not a string or pymatgen Composition object.
        AssertionError: If the prototype path is not a valid directory.
    """

    assert os.path.isdir(prototypePath), 'Prototype path must be a valid directory'
    
    print(f'\nSearching for structure prototypes for: {targetFormula}')
    print('Structure prototype dataset:\n' + prototypePath)

    if isinstance(targetFormula, str):
        # Formula to pymatgen Composition
        composition = Composition(targetFormula)
    elif isinstance(targetFormula, Composition):
        composition = targetFormula
    else:
        raise ValueError('Target formula must be a string or a pymatgen Composition object')
    assert os.path.isdir(prototypePath), 'Prototype path must be a valid directory'

    aFormula = composition.anonymized_formula
    nElements = composition.to_data_dict['nelements']

    # Initial prototype dataset scan
    fileList = natsorted(os.listdir(prototypePath))
    total = fileList.__len__()
    print(str(total)+' files detected\n')
    # Intermediate directory for identified prototypes
    try:
        os.mkdir(os.path.split(prototypePath)[1] + '-' + aFormula)
        print('Intermediate directory for '+aFormula+' created.')
    except FileExistsError:
        print('Intermediate directory for '+aFormula+' already exist.\nPrototypes selection will be overwritten/updated.\n')

    # Counters
    allS = 0
    nEl = 0
    aF = 0
    print('{0:*^80}'.format('Scanning database for '+aFormula+' compounds.'))
    # Iterate over dataset of all structures
    for file in fileList:
        allS = allS + 1
        if os.path.splitext(file)[1] == '.POSCAR':
            s = Structure.from_file(prototypePath +'/'+ file)
            if s.n_elems == nElements:
                nEl = nEl + 1
                if Composition(s.formula).anonymized_formula == aFormula:
                    sP = Poscar(s)
                    sP.write_file(os.path.split(prototypePath)[1] + '-' + aFormula + '/' + str(aF) + '-' + os.path.splitext(file)[0] + '-' + str.replace(s.formula, ' ', '') + '.POSCAR')
                    aF = aF + 1
                    if aF % 100 == 0:
                        print(str(aF) + ' prototype structures identified.')
        if allS % 5000 == 0:
            print(str(round(allS / total * 100, 1)) + '% database scan completed.')

    print('\nTotal structures detected: ' + str(allS))
    print('\n' + str(nElements) + '-element structures: ' + str(nEl))
    print(str(nElements) + '-element structures fraction: ' + str(round(nEl / total * 100)) + '%')
    print('\n' + aFormula + ': ' + str(aF))
    print(aFormula + ' Fraction from ' + str(nElements) + '-element: ' + str(round(aF / nEl * 100)) + '%')

    # List selected prototypes
    prototypeList = natsorted(os.listdir(os.path.split(prototypePath)[1] + '-' + aFormula))

    # Substitution elements combination.
    print('{0:*^80}'.format(' Substituting elements in the prototypes '))
    sEL = sortedElementsList(composition)
    sELtarget = [item for sublist in sEL for item in sublist]
    print('\nTarget Element Combination: ' + str(sELtarget))

    # Verify need for permutations
    permN = 1
    for multiples in sEL:
        permN = permN * math.factorial(multiples.__len__())

    if permN > 1:
        print('\nTarget composition has degenerate anonymous formula thus requires permutations.')
        print('Up to ' + str(permN) + ' per structure.')
        print('Total number of prototypes increased up to: ' + str(aF * permN) + '\n')
    else:
        print('\nTarget composition indicates discrete element substitution.')

    intPath = f"{os.path.split(prototypePath)[1]}-{aFormula}"
    print('Intermediate Directory: ' + intPath)
    finPath = f"{os.path.split(prototypePath)[1]}-{str(targetFormula).replace(' ', '')}"
    print('Final Directory: ' + finPath)
    try:
        os.mkdir(finPath)
        print('Final directory created.')
    except FileExistsError:
        print('Final directory already exist.\nPrototypes will be overwritten/updated.')

    # Substitute over all condidates over all of their permutations
    allS = 0
    for prototype in prototypeList:
        s = Structure.from_file(intPath + '/' + prototype)

        permuts = elementPermutations(sortedElementsList(Composition(s.formula)))
        spGroupA = SpacegroupAnalyzer(s)
        fileSplit = prototype.split('.')
        newNameRoot = fileSplit[0] + '-' + str(
            spGroupA.get_crystal_system()) + '-'

        p = 0
        for permut in permuts:
            d = {}
            p = p + 1
            st = s.copy()
            for e in range(permut.__len__()):
                d.update({Element(permut[e]): Element(sELtarget[e])})
            st.replace_species(d)
            if permuts.__len__() > 1:
                newName = newNameRoot + '-p' + str(p) + '.POSCAR'
            else:
                newName = newNameRoot + '.POSCAR'
            sP = Poscar(st)
            sP.write_file(finPath + '/' + re.sub(r'[?:*|><\\/]', '-', newName))
        allS = allS + p
        if allS % 100 == 0:
            print(str(allS) + ' substitutions complete.')
    print(f'\nGenerated {Fore.GREEN}{Style.BRIGHT}{str(allS)}{Style.RESET_ALL} {targetFormula} prototypes.')
    return finPath

def run_MPDD(
        targetFormula: Union[str, Composition],
        MPDD: Database,
        collections: List[str],
        targetPrototypes: Literal['native', 'non-native'],
        filterPrototypes: Literal['material.structID', 'material.volNormStructID'],
        limit: int
    ) -> str:
    """
    Run prototype identification and elemental substitution for permutations within the anonymous formula using the MPDD database.

    Args:
        targetFormula: The target formula as a string or pymatgen Composition object.
        MPDD: A pymongo Database object corresponding to the MPDD database.
        collections: List of collection names to search in.
        targetPrototypes: Specifies whether to use 'native' or 'non-native' prototypes.
        filterPrototypes: Specifies the prototype filter to use.
        limit: The maximum number of prototypes to generate.

    Returns:
        Path to the directory containing the generated prototypes.

    Raises:
        ValueError: If the target formula is not a string or pymatgen Composition object.
        AssertionError: If no collections are specified or if MPDD is not a pymongo Database object.
    """

    assert len(collections) > 0, 'At least one collection must be specified'
    assert isinstance(MPDD, Database), 'MPDD must be a pymongo Database object'
    
    print(f'\nSearching for structure prototypes for: {targetFormula}')
    print('Using MPDD Database with collections: '+', '.join(collections))

    if isinstance(targetFormula, str):
        # Formula to pymatgen Composition
        composition = Composition(targetFormula)
    elif isinstance(targetFormula, Composition):
        composition = targetFormula
    else:
        raise ValueError('Target formula must be a string or a pymatgen Composition object')
    assert len(collections) > 0, 'At least one collection must be specified'
    assert isinstance(MPDD, Database), 'MPDD must be a pymongo Database object corresponding to the MPDD database or another database following (at least partially) the same schema'

    aFormula = composition.anonymized_formula
    rFormula = composition.reduced_formula

    print(f'\nLooking for {targetFormula} in MPDD database collections.')

    if targetPrototypes == 'native':

        natPath = f"{str(targetFormula).replace(' ', '')}-{'_'.join(collections)}-native-prototypes"

        try:
            os.mkdir(natPath)
            print('Directory created for MPDD prototypes: '+natPath )
        except FileExistsError:
            print('Directory for MPDD prototypes already exist.\nPrototypes will be overwritten/updated.')

        for col in collections:
            print('Looking through: ' + col)

            pipeline = [
                {"$match": {"material.reducedFormula": rFormula}},
                {"$group": {
                    "_id": "$" + filterPrototypes,
                    "document": {"$first": "$$ROOT"}
                }},
                {"$sort": {"document.properties.EF_SIPFENN_NN30": 1}},
                {"$limit": limit}
            ]

            aggregated_data = list(MPDD[col].aggregate(pipeline))

            for entry in aggregated_data:

                e = entry['document']

                s = Structure.from_dict(json.loads(e['material']['mgJSON']))
                spGroupA = SpacegroupAnalyzer(s)

                name = e['material']['formula'].replace(' ', '') + '-' + str(
                    spGroupA.get_crystal_system())

                if 'parentDatabaseID' in e['metadata']:
                    name = name + '-' + e['metadata']['parentDatabaseID']

                else:
                    name = name + '-from-' + e['metadata']['name'] + '-at-' + e['metadata']['affiliation']

                p = Poscar(s)
                p.write_file(natPath + '/' + re.sub(r'[?:*|><\\/]', '-', name) + '.POSCAR')
        print('Done.\n')

        return natPath

    elif targetPrototypes == 'non-native':

        print('\nDetecting Input Structures:')
        foundTotal = 0
        for col in collections:
            foundN = MPDD[col].count_documents({'material.anonymizedFormula' : aFormula})
            foundTotal += foundN
            print('Found ' + str(foundN) + ' in ' + col)

        print('\nTotal of '+str(foundTotal) + ' structures available.\n')

        # Substitution elements combination.
        print('{0:*^80}'.format(' Substituting elements in the prototypes '))
        sEL = sortedElementsList(composition)
        sELtarget = [item for sublist in sEL for item in sublist]
        print('\nTarget Element Combination: ' + str(sELtarget))

        # Verify need for permutations
        permN = 1
        for multiples in sEL:
            permN = permN * math.factorial(multiples.__len__())

        if permN > 1:
            print('\nTarget composition has degenerate anonymous formula thus requires permutations.')
            print('Up to ' + str(permN) + ' per structure.')
            print('Total number of prototypes increased up to: ' + str(foundTotal * permN) + '\n')
        else:
            print('\nTarget composition indicates discrete element substitution.')

        finPath = f"{str(targetFormula).replace(' ', '')}-{'_'.join(collections)}-prototypes"
        print('Final Prototype Directory: ' + finPath)

        try:
            os.mkdir(finPath)
            print('Final directory created.')
        except FileExistsError:
            print('Final directory already exist.\nPrototypes will be overwritten/updated.')

        # Substitute over all prototypes over all of their permutations
        allS = 0
        for col in collections:
            print('Looking through: ' + col)

            pipeline = [
                {"$match": {"material.anonymizedFormula": aFormula}},
                {"$group": {
                    "_id": "$" + filterPrototypes,
                    "document": {"$first": "$$ROOT"}
                }},
                {"$sort": {"document.properties.EF_SIPFENN_NN30": 1}},
                {"$limit": limit}
            ]

            aggregated_data = list(MPDD[col].aggregate(pipeline, hint={'material.anonymizedFormula': 1}))

            for entry in aggregated_data:

                prototype = entry['document']

                s = Structure.from_dict(json.loads(prototype['material']['mgJSON']))

                permuts = elementPermutations(sortedElementsList(Composition(s.formula)))
                spGroupA = SpacegroupAnalyzer(s)

                nameRoot = prototype['material']['reducedFormula'] + '-' + str(
                    spGroupA.get_crystal_system())

                if 'parentDatabaseID' in prototype['metadata']:
                    nameRoot = nameRoot + '-' + prototype['metadata']['parentDatabaseID'].replace('/','-').replace(':','-').replace('-','')

                p = 0
                for permut in permuts:
                    d = {}
                    p += 1
                    st = s.copy()
                    for e in range(permut.__len__()):
                        d.update({Element(permut[e]): Element(sELtarget[e])})
                    st.replace_species(d)
                    if permuts.__len__() > 1:
                        name = nameRoot + '-p' + str(p) + '.POSCAR'
                    else:
                        name = nameRoot + '.POSCAR'
                    sP = Poscar(st)
                    sP.write_file(finPath + '/' + re.sub(r'[?:*|><\\/]', '-', name))
                allS = allS + p
                if allS % 1000 == 0:
                    print(str(allS) + ' substitutions complete.')

        print(f'\nGenerated {Fore.GREEN}{Style.BRIGHT}{str(allS)}{Style.RESET_ALL} {targetFormula} prototypes.')
        
        return finPath

def run_compositional_MPDD(
        targetCompositionDict: Dict[str, int],
        devs: Dict[str, int],
        MPDD: Database,
        collections: List[str],
        targetPrototypes: Literal['native', 'non-native'],
        filterPrototypes: Literal['material.structID', 'material.volNormStructID'],
        limit: int
    ) -> str:
    """
    Run prototype identification and elemental substitution for permutations within the anonymous composition fractions.

    Args:
        targetCompositionDict: Dictionary with element symbols as keys and their amounts as values.
        devs: Dictionary with element symbols as keys and their allowed deviations as values.
        MPDD: A pymongo Database object corresponding to the MPDD database.
        collections: List of collection names to search in.
        targetPrototypes: Specifies whether to use 'native' or 'non-native' prototypes.
        filterPrototypes: Specifies the prototype filter to use.
        limit: The maximum number of prototypes to generate.

    Returns:
        Path to the directory containing the generated prototypes.
    """
    
    assert len(collections) > 0, 'At least one collection must be specified'
    assert isinstance(MPDD, Database), 'MPDD must be a pymongo Database object'
    
    # Target formula
    meanFormula = ''.join(f'{key}{val}' for key, val in targetCompositionDict.items())
    rangesStr = f"+-({','.join(str(r) for r in devs.values())})"

    print('\nSearching for structure prototypes for: ' + meanFormula + ' ' + rangesStr)
    print('Using MPDD Database with collections: ' + ', '.join(collections))

    if targetPrototypes == 'native':

        # ************************* Try to find materials already in the MPDD database **********************************

        print('\nLooking for ' + meanFormula + ' in MPDD database collections.')

        natPath = '_'.join(collections) + '-native-' + meanFormula + rangesStr

        try:
            os.mkdir(natPath)
            print('Directory created for MPDD prototypes: ' + natPath)
        except FileExistsError:
            print('Directory for MPDD prototypes already exist.\nPrototypes will be overwritten/updated.')

        # Query for matching MPDD entries
        nativeQuery = comp2compQ(targetCompositionDict, devs)

        targetPresent = 0
        for col in collections:
            print('Looking through: ' + col)

            pipeline = [
                {"$match": nativeQuery},
                {"$group": {
                    "_id": "$" + filterPrototypes,
                    "document": {"$first": "$$ROOT"}
                }},
                {"$sort": {"document.properties.EF_SIPFENN_NN30": 1}},
                {"$limit": limit}
            ]

            aggregated_data = list(MPDD[col].aggregate(pipeline))

            for entry in aggregated_data:

                e = entry['document']

                s = Structure.from_dict(json.loads(e['material']['mgJSON']))
                spGroupA = SpacegroupAnalyzer(s)

                name = e['material']['formula'].replace(' ', '') + '-' + str(
                    spGroupA.get_crystal_system())

                if 'parentDatabaseID' in e['metadata']:
                    name = name + '-' + e['metadata']['parentDatabaseID']

                else:
                    name = name + '-from-' + e['metadata']['name'] + '-at-' + e['metadata']['affiliation']

                p = Poscar(s)
                p.write_file(natPath + '/' + re.sub(r'[?:*|><\\/]', '-', name) + '.POSCAR')
                targetPresent += 1

        print('Done.')
        print(f'{targetPresent} target materials found in MPDD.\n')

        return natPath

    elif targetPrototypes == 'non-native':
        # ******** Try to find materials from the MPDD database that can be adjusted to the target formula  **************

        # Sorted fractions dictionary query
        sortedFractionsDictQuery = comp2acompQ(targetCompositionDict, devs)

        # ************************************** Count prototypes ********************************************************

        print('\nDetecting Input Structures:')
        foundTotal = 0
        for col in collections:
            foundN = MPDD[col].count_documents(sortedFractionsDictQuery)
            foundTotal += foundN
            print('Found ' + str(foundN) + ' in ' + col)

        print('\nTotal of ' + str(foundTotal) + ' structures available.\n')

        # ********************************* Substitution elements combination ********************************************

        print('{0:*^80}'.format(' Substituting elements in the prototypes '))

        finPath = '_'.join(collections) + '-' + meanFormula + rangesStr
        print('Final Prototype Directory: ' + finPath)

        try:
            os.mkdir(finPath)
            print('Final directory created.')
        except FileExistsError:
            print('Final directory already exist.\nPrototypes will be overwritten/updated.')

        # All generated prototypes
        allS = 0
        # List of permutation numbers
        permutNList = []

        # Search all collections using the compound index for the first 8 components
        for col in collections:
            print('Looking through: ' + col)

            pipeline = [
                {"$match": sortedFractionsDictQuery},
                {"$group": {
                    "_id": "$" + filterPrototypes,
                    "document": {"$first": "$$ROOT"}
                }},
                {"$sort": {"document.properties.EF_SIPFENN_NN30": 1}},
                {"$limit": limit}
            ]

            aggregated_data = list(MPDD[col].aggregate(pipeline, hint={
                'material.sortedFractionsDict.c1': 1,
                'material.sortedFractionsDict.c2': 1,
                'material.sortedFractionsDict.c3': 1,
                'material.sortedFractionsDict.c4': 1,
                'material.sortedFractionsDict.c5': 1,
                'material.sortedFractionsDict.c6': 1,
                'material.sortedFractionsDict.c7': 1,
                'material.sortedFractionsDict.c8': 1
            }))

            for entry in aggregated_data:

                prototype = entry['document']

                s = Structure.from_dict(json.loads(prototype['material']['mgJSON']))
                c = Composition(s.formula)

                cDict = dict(c.fractional_composition.as_dict())
                for e in cDict:
                    cDict[e] = int(cDict[e] * 1000)

                elements = [element for element in cDict]

                oAS = orderedAllowedSubstitutions(targetCompositionDict, devs, cDict)

                permutNList += [len(oAS)]

                spGroupA = SpacegroupAnalyzer(s)

                nameRoot = prototype['material']['reducedFormula'] + '-' + \
                        str(spGroupA.get_crystal_system())

                if 'parentDatabaseID' in prototype['metadata']:
                    nameRoot = nameRoot + '-' + prototype['metadata']['parentDatabaseID'].replace('/', '-').replace(':','-').replace('-', '')

                p = 1
                for substitutions in oAS:
                    d = {}
                    st = s.copy()
                    for o, r in zip(elements, substitutions):
                        d.update({Element(o): Element(r)})
                    st.replace_species(d)

                    if len(oAS) > 1:
                        name = nameRoot + '-p' + str(p) + '.POSCAR'
                        p += 1
                    else:
                        name = nameRoot + '.POSCAR'
                    sP = Poscar(st)
                    sP.write_file(finPath + '/' + re.sub(r'[?:*|><\\/]', '-', name))
                allS = allS + p
                if allS % 50 == 0:
                    print(str(allS) + ' substitutions complete.')

        if allS>0:
            print(f'\nGenerated {Fore.GREEN}{Style.BRIGHT}{str(allS)}{Style.RESET_ALL} {meanFormula} prototypes.')
            print(f'Mean number of permutations: {mean(permutNList)}')
            print(f'Min / Max number of permutations: {min(permutNList)} / {max(permutNList)}')
        else:
            print('No prototypes were generated.')

        return finPath

def estimate_compositional_MPDD(
        targetCompositionDict: Dict[str, int],
        devs: Dict[str, int],
        MPDD: Database,
        collections: List[str]
    ) -> str:
    """
    Estimate the number of input structures available for substitution into the target composition.

    Args:
        targetCompositionDict: Dictionary with element symbols as keys and their amounts as values.
        devs: Dictionary with element symbols as keys and their allowed deviations as values.
        MPDD: A pymongo Database object corresponding to the MPDD database.
        collections: List of collection names to search in.

    Returns:
        Total number of structures available.
    """

    assert len(collections) > 0, 'At least one collection must be specified'
    assert isinstance(MPDD, Database), 'MPDD must be a pymongo Database object'

    # Target formula
    meanFormula = ''.join(f'{key}{val}' for key, val in targetCompositionDict.items())
    rangesStr = f"+-({','.join(str(r) for r in devs.values())})"

    # Sorted fractions dictionary query
    sortedFractionsDictQuery = comp2acompQ(targetCompositionDict, devs)

    print(f'\nDetecting Input Structures for substitution into {meanFormula} under allowed deviations {rangesStr}')
    foundTotal = 0
    for col in collections:
        foundN = MPDD[col].count_documents(sortedFractionsDictQuery)
        foundTotal += foundN
        print('Found ' + str(foundN) + ' in ' + col)

    print('\nTotal of ' + str(foundTotal) + ' structures available.\n')

    return foundTotal

def sortedElementsList(
        comp: Composition
    ) -> List[List[str]]:
    """
    Return elements grouped by their occurrence in the formula, starting from the least common.

    Args:
        comp: A pymatgen Composition object.

    Returns:
        List of lists of element symbols grouped by their occurrence in the formula.
    """

    fractions = re.sub(r'[A-z]+', '', comp.formula).split()
    elements = re.sub(r'[0-9]+', '', comp.formula).split()
    compl = []
    for i in range(elements.__len__()):
        compl.append([elements[i],int(fractions[i])])
    compl.sort(key=itemgetter(1))
    sortedEL = [[x for x, _ in g]
           for _, g in groupby(compl,key=itemgetter(1))]
    return sortedEL

def elementPermutations(
        sEL: List[List[str]]
    ) -> List[List[str]]:
    """
    Return all permutations of elements within the groups that conserve the anonymous formula of the compound.

    Args:
        sEL: List of lists of element symbols grouped by their occurrence in the formula.

    Returns:
        List of lists of element symbols representing all permutations.
    """

    generatingList = [[list(i) for i in permutations(lev)] for lev in sEL]
    outList = [[item for sublist in list(permutation) for item in sublist] for permutation in product(*generatingList)]
    return outList

def comp2acompQ(
        compDict: Dict[str, int],
        devs: Dict[str, int]
    ) -> Dict[str, Union[Dict[str, int], int]]:
    """
    Return a MPDD query for anonymous composition space with N components.

    Args:
        compDict: Dictionary with element symbols as keys and their amounts as values.
        devs: Dictionary with element symbols as keys and their allowed deviations as values.

    Returns:
        Dictionary representing the MPDD query.
    """

    eVals = []
    for e in compDict:
        eVals.append([compDict[e], devs[e]])
    eVals.sort(key = lambda x: x[0], reverse=True)
    result = dict()
    for eVal, c in zip(eVals, range(eVals.__len__())):
        result.update({f'material.sortedFractionsDict.c{c+1}': {'$gte': (eVal[0]-eVal[1]), '$lte': (eVal[0]+eVal[1])}})
    result.update({f'material.nComponents': len(eVals)})
    return result

def comp2compQ(
        compDict: Dict[str, int],
        devs: Dict[str, int]
    ) -> Dict[str, Union[Dict[str, int], int]]:
    """
    Return a MPDD query for the given composition dictionary with permil fractions.

    Args:
        compDict: Dictionary with element symbols as keys and their amounts as values.
        devs: Dictionary with element symbols as keys and their allowed deviations as values.

    Returns:
        Dictionary representing the MPDD query.
    """

    meanComposition = Composition(' '.join(''.join((key, str(val))) for (key, val) in compDict.items()))
    mpddQuery = dict()
    mpddQuery.update({'material.system': meanComposition.chemical_system})
    for e in compDict:
        key = 'material.compositionDictionary.' + e
        mpddQuery.update({key: {'$gte': (compDict[e] - devs[e]) / 1000, '$lte': (compDict[e] - devs[e]) / 1000}})
    return mpddQuery

def orderedAllowedSubstitutions(
        targetCompDict: Dict[str, int],
        devs: Dict[str, int],
        prototypeCompDict: Dict[str, int]
    ) -> List[List[str]]:
    """
    Return elements grouped by their occurrence in the formula, starting from the least common.

    Args:
        targetCompDict: Dictionary with element symbols as keys and their amounts as values.
        devs: Dictionary with element symbols as keys and their allowed deviations as values.
        prototypeCompDict: Dictionary with element symbols as keys and their amounts as values for the prototype.

    Returns:
        List of lists of element symbols representing allowed substitutions.
    """

    # Finds what elements can be used for substitution
    allInRangeSubstitutions = []
    for c in prototypeCompDict:
        inRangeSubstitutions = []
        val = prototypeCompDict[c]
        for e in targetCompDict:
            if targetCompDict[e] - devs[e] <= val and targetCompDict[e] + devs[e] >= val:
                inRangeSubstitutions += [str(e)]
        allInRangeSubstitutions.append(inRangeSubstitutions)

    # E.g. [['Al', 'Fe', 'N'], ['Al', 'Fe', 'N'], ['N']]
    allInRangeSubstitutionPermutations = list(product(*allInRangeSubstitutions))
    allAllowedSubstitutions = []
    for irsp in allInRangeSubstitutionPermutations:
        if len(irsp)==len(set(irsp)):
            allAllowedSubstitutions += [irsp]
    # E.g. [('Al', 'Fe', 'N'), ('Fe', 'Al', 'N')]

    return allAllowedSubstitutions

# Convenience High-Level API Class ************************************************************

class CrystALLSearch:
    """
    A self-contained environment for (1) obtaining POSCAR files of ``crystALL`` prototypes from either MPDD or a local 
    dataset of POSCAR files, (2) featurizing the prototypes with KS2022 or future ``pySIPFENN`` descriptor calculators,
    (3) getting property predictions using ``pySIPFENN`` or ``ALIGNN``, (4) clustering the prototypes in ``KS2022`` 
    feature space, and (5) selecting prototypes based on desired property and cluster affiliation.
    """
    
    def __init__(
            self,
            targetFormula: Union[str, Composition],
            calculator: Calculator = None
    ) -> None:
        """
        Initialize the CrystALLSearch class.

        Args:
            targetFormula: The target formula as a string or pymatgen Composition object.
            calculator: A pySIPFENN Calculator object. If not provided, a default Calculator is used.

        Raises:
            TypeError: If the target formula is not a string or pymatgen Composition object.
        """

        assert isinstance(targetFormula, (str, Composition)), 'Target formula must be a string or a pymatgen Composition object'

        # Defined at initialization
        if isinstance(targetFormula, str):
            self.targetFormula = Composition(targetFormula)
        elif isinstance(targetFormula, Composition):
            self.targetFormula = targetFormula
        else:
            raise TypeError('Target formula must be a string or a pymatgen Composition object')
        self.targetCompositionDict = {el: round(amt / self.targetFormula.num_atoms * 1000, 0)
                                      for el, amt in self.targetFormula.as_dict().items()}

        stdout, sys.stdout = sys.stdout, io.StringIO()
        self.Calculator = calculator if calculator is not None else Calculator(autoLoad=False, verbose=False)
        sys.stdout = stdout

        # Internal parameters to use during the processing steps
        self.availableObjectives: List[str] = []
        self.predictedObjectives: List[str] = []
        self.ks2022: np.ndarray = None
        self.clusterAffiliation: Dict[int, Dict[str, float]] = {}
        self.prototypes: List[Dict[str, Union[str, float]]] = []
        self.topCandidates: List[Dict[str, Union[str, float]]] = []

        print(f"Initialized CrystALLSearch for target formula: {Fore.MAGENTA}{Style.BRIGHT}{self.targetFormula.formula}{Style.RESET_ALL}")

        print(f"\nChecking for available models on the computer..")

        # Dictionary to record model availability
        self.model_status = {
            'pySIPFENN': False,
            'MPDD-ALIGNN': False
        }

        # Check for pySIPFENN models
        existingCompatibleModels = set(self.Calculator.findCompatibleModels('KS2022')).intersection(set(self.Calculator.network_list_available))
        if existingCompatibleModels:
            self.model_status['pySIPFENN'] = True
        else:
            user_input = input("pySIPFENN models not found. Would you like to automatically run `download_models`? ([y]/n):")
            if user_input.lower() in ('y', ''):
                self.download_models()

        # Check for MPDD-ALIGNN models
        with resources.files('alignn') as alignn_dir:
            for model_dict in pretrained.get_default_models():
                if os.path.exists(os.path.join(alignn_dir, model_dict['model'])):
                    self.model_status['MPDD-ALIGNN'] = True
                    break
            if not self.model_status['MPDD-ALIGNN']:
                user_input = input("MPDD-ALIGNN models not found. Would you like to automatically run `download_models`? ([y]/n):")
                if user_input.lower() in ('y', ''):
                    self.download_models()

        # Print the status of models
        for model, status in self.model_status.items():
            if status:
                try:
                    print(f"{Fore.GREEN}✔ {model} models found.{Style.RESET_ALL}")
                except UnicodeEncodeError:
                    print(f"{Fore.GREEN}√ {model} models found.{Style.RESET_ALL}")
            else:
                try:
                    print(f"{Style.DIM}✘ {model} models not found.{Style.RESET_ALL}")
                except UnicodeEncodeError:
                    print(f"{Style.DIM}x {model} models not found.{Style.RESET_ALL}")

        # Update objectives with models found
        self.update_objectives(self.Calculator)

    def update_objectives(
            self,
            calculator: Calculator,
            verbose: bool = True
    ) -> None:
        """
        Update the available objectives with the models found.

        Args:
            calculator: A pySIPFENN Calculator object.
            verbose: If True, prints the available objectives. Defaults to True.

        Raises:
            AssertionError: If the calculator is not a pySIPFENN Calculator object.
        """

        assert isinstance(calculator, Calculator), "Calculator must be a Calculator object"
        
        if self.model_status['pySIPFENN']:
        
            loadedCompatibleModels = set(self.Calculator.findCompatibleModels('KS2022')).intersection(set(self.Calculator.loadedModels.keys()))

            if not len(loadedCompatibleModels) > 0:
                alternative_model = list(set(self.Calculator.findCompatibleModels('KS2022')).intersection(set(self.Calculator.network_list_available)))[0]
                stdout, sys.stdout = sys.stdout, io.StringIO()
                self.Calculator.loadModels(network = alternative_model)
                sys.stdout = stdout
            
            self.availableObjectives = list(set(calculator.findCompatibleModels('KS2022')).intersection(set(self.Calculator.loadedModels.keys())))

        else:

            print("\npySIPFENN objective required for calculation. Run `download_models` first.")

        self.availableObjectives += ['ALIGNN-MP Formation Energy [eV/atom]', 'ALIGNN-JARVIS Formation Energy [eV/atom]'] if self.model_status['MPDD-ALIGNN'] else []

        if len(self.availableObjectives) >= 1 and verbose:
        
            print(f"\nObjectives available for calculation: {Fore.GREEN}{Style.BRIGHT}{', '.join(self.availableObjectives)}{Style.RESET_ALL}.")

    def __check_objectives(
            self
    ) -> None:
        """
        Check available objectives with predictions.

        Raises:
            AssertionError: If no objectives are available.

        Returns:
            List of common objectives between available objectives and predicted objectives.
        """

        assert isinstance(self.availableObjectives, list) and len(self.availableObjectives) >= 1, "At least one objective is required for classification. Run `download_models` first."
        
        if any(value not in self.predictedObjectives for value in self.availableObjectives):
            print("\nSome objectives do not have predictions calculated and will not be considered for classification.")
            for obj in self.availableObjectives:
                if obj not in self.predictedObjectives:
                    try:
                        print(f"{Style.DIM}✘ {Style.BRIGHT}{obj}{Style.RESET_ALL}")
                    except UnicodeEncodeError:
                        print(f"{Style.DIM}x {Style.BRIGHT}{obj}{Style.RESET_ALL}")
                else:
                    try:
                        print(f"{Fore.GREEN}✔ {Style.BRIGHT}{obj}{Style.RESET_ALL}")
                    except UnicodeEncodeError:
                        print(f"{Fore.GREEN}√ {Style.BRIGHT}{obj}{Style.RESET_ALL}")

    def download_models(
            self
    ) -> None:
        """
        Download the default models for both pySIPFENN and MPDD-ALIGNN.
        """
        
        print(f"Downloading SIPFENN_Krajewski2022_NN30 model for pysipfenn..")
        self.Calculator.downloadModels("SIPFENN_Krajewski2022_NN30")
        self.model_status['pySIPFENN'] = True
        print(f"Downloading models for MPDD-alignn..")
        pretrained.download_default_models()
        self.model_status['MPDD-ALIGNN'] = True
        print(f"Done! Models are ready to use.")

        self.update_objectives(self.Calculator, False)

    def connect_MPDD(
            self,
            MPDD: Database = None,
            collections: List[str] = ['curated']
    ) -> None:
        """
        Establish a connection to the MPDD database and set the collections to search in.

        Args:
            MPDD: A pymongo Database object to connect to. If not provided, uses the existing connection.
            collections: List of collection names to search in. Defaults to ['curated'].

        Raises:
            AssertionError: If MPDD is not a pymongo Database object, if no collections are found in the database, if any specified collection is not in the database, or if no collections are specified.
        """

        self.MPDD = MPDD if MPDD is not None else self.MPDD
        assert isinstance(self.MPDD, Database), 'MPDD must be a pymongo Database object'
        availableCollections = self.MPDD.list_collection_names()
        assert len(availableCollections) > 0, 'No collections found in the database or the database connection has not been established properly'
        assert set(collections).issubset(availableCollections), 'At least one of the specified collections is not available in the database'
        filtered_collections = [col for col in availableCollections if 'test' not in col.lower()]
        print(f"\nConnected to database {Fore.MAGENTA}{Style.BRIGHT}{self.MPDD.name}{Style.RESET_ALL} with collections: {Fore.MAGENTA}{Style.BRIGHT}{', '.join(filtered_collections)}{Style.RESET_ALL}")
        assert len(collections) > 0, 'At least one collection must be specified'
        self.collections = collections
        print(f"Using collections: {Fore.GREEN}{Style.BRIGHT}{', '.join(collections)}{Style.RESET_ALL}")
        
    def generate_MPDD(
            self,
            targetPrototypes: Literal['native', 'non-native'] = 'non-native',
            filterPrototypes: Literal['material.structID', 'material.volNormStructID'] = 'material.structID',
            limit: int = 1000,
            devs: Dict[str, int] = None
    ) -> None:
        """
        Generate prototype materials from the MPDD database based on the target formula.

        Args:
            targetPrototypes: Specifies whether to use 'native' or 'non-native' prototypes. Defaults to 'non-native'.
            filterPrototypes: Specifies the prototype filter to use. Defaults to 'material.structID'.
            limit: The maximum number of prototypes to generate. Defaults to 1000.
            devs: Dictionary with element symbols as keys and their allowed deviations as values. Defaults to None.

        Raises:
            AssertionError: If no connection to the MPDD has been established, or if no collections have been specified.
        """

        assert self.MPDD is not None, 'No database connection identified. Please run `connect_MPDD` first.'
        assert self.collections, 'No collections identified. Please provide a list of collections to search in.'
        assert devs is None or isinstance(devs, dict), 'Deviations must be a dictionary with element symbols as keys and integer values as deviations'
        
        if devs is None:
            self.prototypePath = run_MPDD(self.targetFormula, self.MPDD, self.collections, targetPrototypes, filterPrototypes, limit)
        else:
            self.prototypePath = run_compositional_MPDD(self.targetCompositionDict, devs, self.MPDD, self.collections, targetPrototypes, filterPrototypes, limit)
        inputFiles = natsorted(os.listdir(self.prototypePath))
        self.prototypes = [{"index": i + 1, "name": file} for i, file in enumerate(inputFiles)]

    def generate_local(
            self,
            prototypePath: str = None
    ) -> None:
        """
        Generate prototypes from a local directory containing POSCAR or CIF files.

        Args:
            prototypePath: Path to the directory with prototype files. If not provided, uses the existing path.

        Raises:
            AssertionError: If no prototype directory is specified.
        """

        self.prototypePath = prototypePath if prototypePath is not None else self.prototypePath
        assert self.prototypePath, 'No prototypes folder identified. Please provide a path to the prototypes folder.'
        self.prototypePath = run_local(self.targetFormula, self.prototypePath)
        inputFiles = natsorted(os.listdir(self.prototypePath))
        self.prototypes = [{"index": i, "name": file} for i, file in enumerate(inputFiles)]

    def run_pysipfenn(
            self,
            mode: str = 'serial'
    ) -> None:
        """
        Predict properties for the prototype materials using pySIPFENN models and update the prototypes with the predictions.

        Args:
            mode: Mode to run pySIPFENN. Defaults to 'serial'.

        Raises:
            AssertionError: If no prototype directory is specified, if the prototype directory path is not a string, if no pySIPFENN models are found, or if the number of prototypes does not match the number of property predictions.
        """

        assert self.prototypePath is not None, 'No prototypes folder identified. Please run `generate_MPDD` or `generate_local` first.'
        assert isinstance(self.prototypePath, str), 'Prototypes folder path must be a string'
        assert self.model_status['pySIPFENN'], 'pySIPFENN models not found. Please run `download_models` first.'

        for model in set(self.Calculator.findCompatibleModels('KS2022')).intersection(set(self.availableObjectives)):
            if model not in self.predictedObjectives:
                self.predictedObjectives.append(model)
        
        self.Calculator.runFromDirectory(self.prototypePath, 'KS2022', mode=mode, max_workers=8)
        resultDicts, self.ks2022 = self.Calculator.get_resultDictsWithNames(), np.array(self.Calculator.descriptorData)
        assert len(self.prototypes) == len(resultDicts), 'Number of prototypes does not match number of property predictions'
        
        for result in resultDicts:
            name = result['name']
            for prototype in self.prototypes:
                if prototype['name'] == name:
                    prototype.update(result)
                    for key in result.keys():
                        if isinstance(prototype[key], float):
                            prototype[key] = float(f'{prototype[key]:.4f}')
                    break

        # Update average of the predictions
        self.calculate_average()
        
        print(f"Predicted properties for {Fore.GREEN}{Style.BRIGHT}{len(self.ks2022)}{Style.RESET_ALL} prototypes using pySIPFENN")
    
    def get_alignn_properties(
            self,
            mode: str = 'serial'
    ) -> None:
        """
        Predict properties for the prototype materials using MPDD-ALIGNN models and update the prototypes with the predictions.

        Args:
            mode: Mode to run ALIGNN. Defaults to 'serial'.

        Raises:
            AssertionError: If no prototype directory is specified, if the prototype directory path is not a string, if no MPDD-ALIGNN models are found, or if the number of prototypes does not match the number of property predictions.
        """

        assert self.prototypePath is not None, 'No prototypes folder identified. Please run `generate_MPDD` or `generate_local` first.'
        assert isinstance(self.prototypePath, str), 'Prototypes folder path must be a string'
        assert self.model_status['MPDD-ALIGNN'], 'MPDD-ALIGNN models not found. Please run `download_models` first.'
        
        for model_dict in pretrained.get_default_models():
            if model_dict['name'] in self.availableObjectives and model_dict['name'] not in self.predictedObjectives:
                self.predictedObjectives.append(model_dict['name'])
        
        print("\nModels that will be run: ['ALIGNN-MP Formation Energy [eV/atom]', 'ALIGNN-JARVIS Formation Energy [eV/atom]']")
        print('Calculating descriptors...')
        resultDicts = pretrained.run_models_from_directory(self.prototypePath, mode=mode)
        assert len(self.prototypes) == len(resultDicts), 'Number of prototypes does not match number of property predictions'

        for result in resultDicts:
            name = result['name']
            for prototype in self.prototypes:
                if prototype['name'] == name:
                    prototype.update(result)
                    for key in result.keys():
                        if isinstance(prototype[key], float):
                            prototype[key] = float(f'{prototype[key]:.4f}')
                    break

        # Update average of the predictions
        self.calculate_average()
        
        print(f"Predicted properties for {Fore.GREEN}{Style.BRIGHT}{len(resultDicts)}{Style.RESET_ALL} prototypes using ALIGNN")

    def get_mpdd_convexhull(
            self
    ) -> None:
        """
        Get the MPDD convex hull stability for the prototypes.

        Raises:
            AssertionError: If no connection to the MPDD has been established, or if no prototypes have been identified.
        """

        assert self.MPDD is not None, 'No database connection identified. Please run `connect_MPDD` first.'
        assert len(self.prototypes) > 0, 'No prototypes identified. Please run `generate_MPDD` or `generate_local` first.'

        MPDDcurated = self.MPDD['curated']
        
        eList = self.targetFormula.chemical_system.split('-')
        sysList = []
        for nComponents in range(eList.__len__()):
            for c in combinations(eList, 1+nComponents):
                t = list(c)
                t.sort()
                sysList.append('-'.join(t))
        
        entriesForPD = []
        formulaSet = set()
        i = 0
        for e in MPDDcurated.find(
                {'material.system': {'$in': sysList},
                'properties.STAB_SIPFENN_NN30': 0},
                {'material.reducedFormula': 1, 'metadata.parentDatabaseID': 1,
                'properties.EF_SIPFENN_NN30': 1}):
            i += 1
            formula = e['material']['reducedFormula']
            comp = Composition(formula)
            dH = e['properties']['EF_SIPFENN_NN30']

            name = f'{formula}-{e["_id"]}'
            if 'parentDatabaseID' in e['metadata']:
                name = name + '-' + e['metadata']['parentDatabaseID']

            if dH is not None:
                pdEntry = PDEntry(comp, dH, name)
                if pdEntry.composition not in formulaSet:
                    entriesForPD.append(pdEntry)
                    formulaSet.add(pdEntry.composition)

        for prototype in self.prototypes:
            pdEntry = PDEntry(self.targetFormula, prototype['SIPFENN_Krajewski2022_NN30'], prototype['name'])
            entriesForPD.append(pdEntry)
            pd = PhaseDiagram(entriesForPD)
            stability = float(f'{pd.get_e_above_hull(pdEntry):.4f}')
            prototype.update({'MPDD_Stability': stability})
            entriesForPD.pop()

    def calculate_average(
            self
    ) -> None:
        """
        Calculate the average of the predicted properties for each prototype material.

        Raises:
            AssertionError: If no features are identified or if less than two models are available for averaging.
        """

        assert len(self.prototypes) > 0, 'No prototypes identified. Please run `generate_MPDD` or `generate_local` first.'
        assert self.predictedObjectives, 'No objective predictions identified. Please run `run_pysipfenn` or `get_alignn_properties` first.'

        # Calculate the average of the predictions
        for i, prototype in enumerate(self.prototypes):
            average_prediction = round(sum(prototype[pred] for pred in self.predictedObjectives) / len(self.predictedObjectives), 4)
            self.prototypes[i].update({f'Avg_Formation_Energy': average_prediction})

    def filter_prototypes(
            self,
            max_prototypes: int,
            cutoff: float = 0.5
    ) -> None:
        """
        Filter the prototypes based on the Formation Energy cutoff value provided.

        Args:
            max_prototypes: The maximum number of prototypes to keep.
            cutoff: The Formation Energy cutoff value. Defaults to 0.5.

        Raises:
            AssertionError: If no prototypes are identified or if no features are identified.
        """

        assert len(self.prototypes) > 0, 'No prototypes identified. Please run `generate_MPDD` or `generate_local` first.'
        assert len(self.ks2022) > 0, 'No features identified. Please run `run_pysipfenn` first.'

        if max_prototypes:
            self.prototypes.sort(key=lambda prototype: (prototype['Avg_Formation_Energy'], prototype['index']))
            self.prototypes = self.prototypes[:max_prototypes]
        else:
            min_energy = min(prototype['Avg_Formation_Energy'] for prototype in self.prototypes)
            self.prototypes = [prototype for prototype in self.prototypes if prototype['Avg_Formation_Energy'] < cutoff + min_energy]

        # Delete files not in filtered self.prototypes
        for filename in os.listdir(self.prototypePath):
            if filename not in {prototype['name'] for prototype in self.prototypes}:
                os.remove(os.path.join(self.prototypePath, filename))

        # Update the 'index' key of each dictionary in self.prototypes to its position in the list
        for i, prototype in enumerate(self.prototypes):
            prototype['index'] = i + 1

        # Update self.ks2022 to only include the values at the positions specified by the "index" key in the filtered self.prototypes
        indices = [prototype['index'] - 1 for prototype in self.prototypes]
        self.ks2022 = self.ks2022[indices]

        if max_prototypes:
            print(f"Filtered prototypes to {Fore.GREEN}{Style.BRIGHT}{len(self.prototypes)}{Style.RESET_ALL} based on the maximum number of prototypes of {max_prototypes}")
        
        else:
            print(f"Filtered prototypes to {Fore.GREEN}{Style.BRIGHT}{len(self.prototypes)}{Style.RESET_ALL} based on the cutoff value of {cutoff}")

    def __embed_features(
            self,
            sameComposition: bool
    ) -> None:
        """
        Embed the features in 2D t-SNE space.

        Args:
            sameComposition: If True, only structural features are used. Defaults to False.

        Raises:
            AssertionError: If no features are identified.
        """

        assert len(self.prototypes) >= 5, 'At least 5 prototypes are required to run `cluster_prototypes`.'
        assert self.ks2022 is not None, 'No features identified. Please run `run_pysipfenn` first.'

        if sameComposition:
            features2use = np.array([onlyStructural(v) for v in self.ks2022])
        else:
            features2use = self.ks2022

        # Select, Scale and Embed features in 2D and 3D t-SNE space
        scaler = MinMaxScaler()
        self.scaledks2022 = scaler.fit_transform(features2use)
        tsne = TSNE(n_components = 2, perplexity = 4, init = 'pca', learning_rate=10, max_iter = 2000, metric = 'correlation', angle = 0.2, n_jobs = 1, random_state = 580)
        self.embeddedks2022 = tsne.fit_transform(self.scaledks2022)
        tsne_3d = TSNE(n_components = 3, perplexity = 4, init = 'pca', learning_rate=10, max_iter = 2000, metric = 'correlation', angle = 0.2, n_jobs = 1, random_state = 580)
        self.embeddedks2022_3d = tsne_3d.fit_transform(self.scaledks2022)
        
    def cluster_prototypes_KMeans(
            self,
            n_clusters: int = None,
            sameComposition: bool = False
    ) -> None:
        """
        Cluster the prototypes using KMeans, assign each data point to its cluster, and select the best prototypes.

        Args:
            n_clusters: Number of clusters. If not provided, the optimal number is determined automatically using silhouette score.
            sameComposition: If True, only prototypes with the same composition are considered. Defaults to False.

        Raises:
            AssertionError: If no features are identified, if the number of clusters is not an integer, or if any objective is not allowed.
        """

        assert self.ks2022 is not None, 'No features identified. Please run `run_pysipfenn` first.'
        assert isinstance(n_clusters, int) or n_clusters is None, 'Number of clusters must be an integer'
        assert n_clusters is None or n_clusters <= len(self.prototypes), 'Number of clusters must be less or equal than the number of prototypes'

        # Check if the number of prototypes is sufficient for clustering
        if len(self.prototypes) < 5:
            print(f"At least 5 prototypes are required to run `cluster_prototypes`. {len(self.prototypes)} found.")
            return

        self.__check_objectives()

        # Select, Scale and Embed features in 2D and 3D t-SNE space
        self.__embed_features(sameComposition)

        print("\nNo number of clusters has been provided. Determining the optimal number through `silhouette score`. This may take a few minutes..\n") if n_clusters is None else None
        
        # Determine the optimal number of clusters using silhouette score
        distances = pairwise_distances(self.embeddedks2022)
        distance_threshold = 0.1

        mask = np.ones(len(self.embeddedks2022), dtype=bool)
        for i in range(len(self.embeddedks2022)):
            for j in range(i + 1, len(self.embeddedks2022)):
                if distances[i, j] < distance_threshold:
                    mask[j] = False

        filtered_data = self.embeddedks2022[mask] # Filter out overlapping embeddings
        
        sil_dict = {}
        k_range = [n_clusters] if n_clusters is not None else np.unique(np.geomspace(2, len(filtered_data) - 1, 40).astype(int))
        for k in tqdm(k_range, desc="Processing", ncols=100, bar_format="{l_bar}{bar}{r_bar}"):
            kmeans = KMeans(n_clusters=k, n_init=500, max_iter=500, random_state=580)
            clusterData = kmeans.fit(self.embeddedks2022)
            sil_score = silhouette_score(self.embeddedks2022, kmeans.labels_) - k / len(self.embeddedks2022)
            sil_dict[sil_score] = clusterData

        clusterData = sil_dict[max(sil_dict.keys())]

        print(f"\nClustered prototypes into {clusterData.n_clusters} clusters using KMeans in 2D t-SNE embedded space.\n")

        # Assign each data point to its cluster
        clusters = {}
        for i, label in enumerate(clusterData.labels_):
            self.prototypes[i].update({'cluster': label + 1})
            clusters.setdefault(label + 1, []).append(self.prototypes[i])

        # Select the best prototype per cluster and Create a list of top candidates
        self.__select_prototypes(clusters)

    def cluster_prototypes_DBSCAN(
            self,
            eps: float = 0.3,
            min_samples: int = 1,
            p: int = 1,
            sameComposition: bool = False
    ) -> None:
        """
        Cluster the prototypes using DBSCAN, assign each data point to its cluster, and select the best prototypes.

        Args:
            eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other. Defaults to 0.3.
            min_samples: The number of samples in a neighborhood for a point to be considered as a core point. Defaults to 1.
            p: The power parameter for the Minkowski metric. Defaults to 1.
            sameComposition: If True, only prototypes with the same composition are considered. Defaults to False.

        Raises:
            AssertionError: If no features are identified, if eps is not a float, if min_samples is not an integer, or if any objective is not allowed.
        """

        assert self.ks2022 is not None, 'No features identified. Please run `run_pysipfenn` first.'
        assert isinstance(eps, float), 'eps must be a float'
        assert isinstance(min_samples, int), 'min_samples must be an integer'

        # Check if the number of prototypes is sufficient for clustering
        if len(self.prototypes) < 5:
            print(f"At least 5 prototypes are required to run `cluster_prototypes`. {len(self.prototypes)} found.")
            return

        self.__check_objectives()

        # Select, Scale and Embed features in 2D and 3D t-SNE space
        self.__embed_features(sameComposition)

        # Cluster the prototypes using DBSCAN
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, p=p)
        dbscan.fit(self.scaledks2022)

        # Determine the number of clusters identified by DBSCAN
        n_clusters = len(set(dbscan.labels_)) - (1 if -1 in dbscan.labels_ else 0)
        print(f"\nClustered prototypes into {n_clusters} clusters using DBSCAN.\n")
        
        # Assign each data point to its cluster
        clusters = {}
        for i, label in enumerate(dbscan.labels_):
            self.prototypes[i].update({'cluster': label + 1})
            clusters.setdefault(label + 1, []).append(self.prototypes[i])
      
        # Select the best prototype per cluster and Create a list of top candidates
        self.__select_prototypes(clusters)

    def __select_prototypes(
            self,
            clusters: dict
    ) -> None:
        """
        Select the best prototype per cluster for each objective and store them in self.topCandidates.

        Args:
            clusters: A dictionary mapping cluster labels to lists of prototype dictionaries.

        Returns:
            List of dictionaries representing the selected prototypes.
        """

        print(f"List of prototypes selected per energy of each cluster:")
        
        # Select the best prototype per cluster based on the minimum predicted property
        self.clusterAffiliation = {}
        for objective in self.predictedObjectives:
            print(f"\nObjective: {objective}")
            self.clusterAffiliation[objective] = {}
            for label, cluster in sorted(clusters.items()):
                prototype = min(cluster, key=lambda x: x[objective])
                self.clusterAffiliation[objective].update({prototype['cluster']: prototype['index']})
                print(f"           Cluster {label:02}: {prototype['name']}")
        
        # Create a list of top candidates
        self.topCandidates, i = [], 0
        for objective in self.predictedObjectives:
            for index in self.clusterAffiliation[objective].values():
                self.topCandidates.append(self.prototypes[index - 1].copy())
                self.topCandidates[i].update({'objective': objective})
                i += 1

        return self.topCandidates

    def export_table(
            self,
            content: List[str] = ['allPrototypes', 'topCandidates'],
            file_prefix: str = ''
    ) -> None:
        """
        Export a CSV file with the specified content.

        Args:
            content: Specifies the type of data to export. Can be 'allPrototypes' and/or 'topCandidates'. Defaults to ['allPrototypes', 'topCandidates'].
            file_prefix: A prefix that will be added to the file name. Defaults to ''.

        Raises:
            AssertionError: If no prototypes are identified or if an invalid content type is provided.

        Outputs:
            CSV files: Writes CSV files for 'allPrototypes' and/or 'topCandidates' depending on the content argument. The files are named based on the content type and the file_prefix (if provided).
        """

        assert len(self.prototypes) > 0, 'No prototypes identified. Please run `generate_MPDD` or `generate_local` first.'
        assert not any(value not in ['allPrototypes', 'topCandidates'] for value in content), 'Content should contain only `allPrototypes` and/or `topCandidates`'

        if 'topCandidates' in content and not self.topCandidates:
            print(f"No clustered prototypes identified. Please run `cluster_prototypes` first. \nChanging content to 'allPrototypes' only.")
            content = ['allPrototypes']

        if 'allPrototypes' in content:
            file_name = 'allPrototypes.csv' if file_prefix == '' else file_prefix + '_allPrototypes.csv'
            with open(file_name, 'w', newline='') as output_file:
                dict_writer = csv.DictWriter(output_file, self.prototypes[0].keys())
                dict_writer.writeheader()
                dict_writer.writerows(self.prototypes)
            
            print(f"\nExported {Fore.GREEN}{Style.BRIGHT}{len(self.prototypes)}{Style.RESET_ALL} prototypes to {file_name}")
        
        if 'topCandidates' in content:
            file_name = 'topCandidates.csv' if file_prefix == '' else file_prefix + '_topCandidates.csv'
            topCandidates_exported = [d for d in self.topCandidates if d['objective'] in self.availableObjectives]
            
            with open(file_name, 'w', newline='') as output_file:
                dict_writer = csv.DictWriter(output_file, topCandidates_exported[0].keys())
                dict_writer.writeheader()
                dict_writer.writerows(topCandidates_exported)
            
            print(f"Exported {Fore.GREEN}{Style.BRIGHT}top{Style.RESET_ALL} candidates for {len(self.predictedObjectives)} Objective(s) to {file_name}")

    def plot_barchart(
            self,
            objective: Union[str, List[str]] = None,
            save: bool = False,
            file_name: str = 'barchart_clusterPrototypes',
            colorBlind: bool = False
    ) -> None:
        """
        Plot bar charts for the prototypes.

        Args:
            objective: The objective to plot. Defaults to the first available objective.
            save: If True, saves the plot as an HTML file. Defaults to False.
            file_name: The name of the HTML file to save the plot. Defaults to 'barchart_clusterPrototypes'.
            colorBlind: If True, uses a colorblind-friendly color palette. Defaults to False.

        Raises:
            AssertionError: If the objective is not one of the calculated objectives or their average.
        """

        objective = objective if objective is not None else self.availableObjectives[0]

        assert objective in self.predictedObjectives, 'Objective should be one of the calculated objectives and/or their average: ' + ', '.join(self.predictedObjectives)
        assert isinstance(objective, str) or isinstance(objective, list), 'Objective must be a single value specified as a string or a list with a single string'
        assert not (isinstance(objective, list) and len(objective) != 1), 'If objective is given as a list, it must contain exactly one string'

        if not self.topCandidates:
            print(f"No clustered prototypes identified. Please run `cluster_prototypes` first. Aborting 2D plot generation.")
            return
        
        # Create a DataFrame from the data
        df = pd.DataFrame({
            'cluster': [c['cluster'] for c in self.prototypes],
            'name': [c['name'].split('-')[0] + '-' + c['name'].split('-')[-1].replace('.POSCAR', '') for c in self.prototypes],
            'objective': [c[objective] for c in self.prototypes],
            'index': [c['index'] for c in self.prototypes]
        })

        df = df.sort_values('cluster')

        # Define color set based on clusters
        colorblind_type = "Protanopia" if colorBlind else None
        colors = get_colors(df['cluster'].nunique(), [(1.0, 1.0, 1.0), (1.0, 1.0, 0.0)], colorblind_type=colorblind_type)
        cluster_colors = {cluster: '#{:02x}{:02x}{:02x}'.format(int(colors[i][0]*255), int(colors[i][1]*255), int(colors[i][2]*255)) for i, cluster in enumerate(df['cluster'].unique())}

        # Sort the dataframe by 'objective' in ascending order
        df_sorted = df.sort_values(by=['objective', 'index'])

        # Segment the DataFrame into df_prototypes and df_top_candidates
        df_prototypes = df_sorted[~df_sorted['index'].isin(self.clusterAffiliation[objective].values())]
        df_top_candidates = df_sorted[df_sorted['index'].isin(self.clusterAffiliation[objective].values())]

        # Get the minimum and maximum values for the y-axis range
        y_min = df_sorted['objective'].min() * 1.01
        y_max = df_sorted['objective'].max() * 0.95

        # Create bar plots for prototypes and top candidates
        trace_prototypes = go.Bar(
            x=df_prototypes['index'],
            y=df_prototypes['objective'],
            hovertext=df_prototypes['objective'],
            hoverinfo='text',
            marker=dict(color='#a6a6a6'),
            width=0.4,
            name='Prototypes'
        )

        # Create bar plots for each top candidate cluster
        traces_top_candidates = []
        for cluster in df_top_candidates['cluster'].unique():
            df_cluster = df_top_candidates[df_top_candidates['cluster'] == cluster]
            trace = go.Bar(
                x=df_cluster['index'],
                y=df_cluster['objective'],
                hovertext=df_cluster['objective'],
                hoverinfo='text',
                marker=dict(color=cluster_colors[cluster]),
                width=0.4,
                name=f'Top Candidate (Cluster {cluster})'
            )
            traces_top_candidates.append(trace)

        data = [trace_prototypes] + traces_top_candidates

        layout = go.Layout(
            xaxis=dict(
                side='top',
                tickvals=df['index'],
                ticktext=df['name'],
                tickangle=300,
                showline=True,
                linewidth=1,
                linecolor='black',
                mirror=True
            ),
            yaxis=dict(
                title='Enthalpy of Formation (eV/atom)',
                range=[y_min, y_max],
                showline=True,
                linewidth=1,
                linecolor='black',
                mirror=True
            ),
            template='simple_white'
        )

        figure = go.Figure(data=data, layout=layout)

        if save:
            plot(figure, filename=f'{file_name}.html')
            print(f"Plot saved as {file_name}.html")

        else:
            iplot(figure, filename=file_name)

    def plot_2D(
            self,
            objective: Union[str, List[str]] = None,
            save: bool = False,
            file_name: str = '2D_clusterPrototypes',
            markerSymbol: bool = False,
            colorBlind: bool = False
    ) -> None:
        """
        Plot 2D scatter plots for the prototypes.

        Args:
            objective: The objective to plot. Defaults to the first available objective.
            save: If True, saves the plot as an HTML file. Defaults to False.
            file_name: The name of the HTML file to save the plot. Defaults to '2D_clusterPrototypes'.
            markerSymbol: If True, uses different marker symbols for each cluster. Defaults to False.
            colorBlind: If True, uses a colorblind-friendly color palette. Defaults to False.

        Raises:
            AssertionError: If the objective is not one of the calculated objectives or their average.
        """

        objective = objective if objective is not None else self.availableObjectives[0]

        assert objective in self.predictedObjectives, 'Objective should be one of the calculated objectives and/or their average: ' + ', '.join(self.predictedObjectives)
        assert isinstance(objective, str) or isinstance(objective, list), 'Objective must be a single value specified as a string or a list with a single string'
        assert not (isinstance(objective, list) and len(objective) != 1), 'If objective is given as a list, it must contain exactly one string'

        if not self.topCandidates:
            print(f"No clustered prototypes identified. Please run `cluster_prototypes` first. Aborting 2D plot generation.")
            return
        
        # Create a DataFrame from the data
        df = pd.DataFrame({
            'x': np.transpose(self.embeddedks2022)[0],
            'y': np.transpose(self.embeddedks2022)[1],
            'cluster': [c['cluster'] for c in self.prototypes],
            'name': [c['name'].replace('.POSCAR', '') for c in self.prototypes],
            'objective': [c[objective] for c in self.prototypes],
            'index': [c['index'] for c in self.prototypes]
        })
        
        df = df.sort_values('cluster')

        # Define a color for each cluster
        colorblind_type = "Protanopia" if colorBlind else None
        colors = get_colors(df['cluster'].nunique(), [(1.0, 1.0, 1.0),(1.0, 1.0, 0.0)], colorblind_type=colorblind_type)

        # Define a symbol for each cluster (up to 8)
        symbols = ['circle', 'circle-open', 'cross', 'diamond', 'diamond-open', 'square', 'square-open', 'x']

        # Create a scatter plot for each cluster
        data = []
        for i, cluster in enumerate(df['cluster'].unique()):
            df_cluster = df[df['cluster'] == cluster]
            df_cluster = df_cluster.sort_values(by=['objective', 'index'])

            # Separate the data into top candidates and other cluster members
            df_prototypes = df_cluster[~df_cluster['index'].isin(self.clusterAffiliation[objective].values())]
            df_top_candidates = df_cluster[df_cluster['index'].isin(self.clusterAffiliation[objective].values())]

            hovertext_prototypes = ['<br>'.join(['<b>Prototype: </b>' + '-'.join(c['name'].split('-')[0:2] + [c['name'].split('-')[-1]])] + 
                                    ['<b>Affiliation: </b>' + c['name'].split('-')[2]] + 
                                    ['<b>Formation Energy: </b> ' + str(round(c['objective'], 4))])  
                        for _, c in df_cluster.iterrows()]
            
            hovertext_top_candidates = hovertext_prototypes.pop(0)

            color_discrete_map = {'Top Candidate': '#ff7f0e', 'Other Cluster Entries': '#{:02x}{:02x}{:02x}'.format(int(colors[i][0]*255),int(colors[i][1]*255),int(colors[i][2]*255))}
            
            if markerSymbol and len(df['cluster'].unique()) <= 8:
                symbol = symbols[i]
            else: 
                symbol = symbols[0]

            # Create scatter plot for other cluster members
            scatter_prototypes = dict(
                mode = "markers",
                name = f"Cluster {cluster}",
                type = "scatter",
                x = df_prototypes['x'], 
                y = df_prototypes['y'],
                hovertext = hovertext_prototypes,  # set the hover name
                hoverinfo = 'text+name',  # only show the hovertext
                marker = dict(
                    size=5, 
                    color=color_discrete_map['Other Cluster Entries'],  # set the color for other cluster members
                    symbol=symbol
                )
            )
            data.append(scatter_prototypes)

            # Create scatter plot for top candidates
            scatter_top_candidates = dict(
                mode = "markers",
                name = f"Cluster {cluster}",
                showlegend = False,  # don't show this in the legend
                type = "scatter",
                x = df_top_candidates['x'], 
                y = df_top_candidates['y'],
                hovertext = hovertext_top_candidates,  # set the hover name
                hoverinfo = 'text+name',  # only show the hovertext
                marker = dict(
                    size=5, 
                    color=color_discrete_map['Top Candidate'],  # set the color for top candidates
                    symbol=symbol
                )
            )
            data.append(scatter_top_candidates)

        # Create a shape for each cluster
        shapes = []
        for i, cluster in enumerate(df['cluster'].unique()):
            df_cluster = df[df['cluster'] == cluster]
            points = df_cluster[['x', 'y']].values
            if not np.linalg.matrix_rank(points - points[0], tol=1e-12) < 2 and len(df_cluster) >= 3: # Ensures at least 3 points that are not collinear
                hull = ConvexHull(df_cluster[['x', 'y']])
                polygon = {
                    'type': 'path',
                    'path': 'M ' + ' L '.join(f"{p[0]} {p[1]}" for p in df_cluster.iloc[hull.vertices][['x', 'y']].values) + ' Z',
                    'line': {
                        'color': 'rgb'+str(colors[i]).replace(' ', ''),  # use the color defined for this cluster
                        'width': 1
                    },
                    'fillcolor': 'rgb'+str(colors[i]).replace(' ', ''),  # use the color defined for this cluster
                    'opacity': 0.2
                }
                shapes.append(polygon)

        layout = dict(
            shapes=shapes,
            template='plotly_white',  # use the plotly_white template
            height=500,  # set the height of the plot
            width=500,  # set the width of the plot
            title=f'Embedded Prototypes ({len(self.scaledks2022[0])}D ➡ 2D t-SNE space)<br><sub>🟠 Top Candidates ({objective})</sub>'  # add the title here
        )

        figure = dict(data=data, layout=layout)
        
        if save:
            plot(figure, filename=f'{file_name}.html')
            print(f"Plot saved as {file_name}.html")

        else:
            iplot(figure, filename=file_name)

    def plot_3D(
            self,
            objective: Union[str, List[str]] = None,
            save: bool = False,
            file_name: str = '3D_clusterPrototypes',
            markerSymbol: bool = False,
            colorBlind: bool = False
    ) -> None:
        """
        Plot 3D scatter plots for the prototypes.

        Args:
            objective: The objective to plot. Defaults to the first available objective.
            save: If True, saves the plot as an HTML file. Defaults to False.
            file_name: The name of the HTML file to save the plot. Defaults to '3D_clusterPrototypes'.
            markerSymbol: If True, uses different marker symbols for each cluster. Defaults to False.
            colorBlind: If True, uses a colorblind-friendly color palette. Defaults to False.

        Raises:
            AssertionError: If the objective is not one of the calculated objectives or their average.
        """

        objective = objective if objective is not None else self.availableObjectives[0]

        assert objective in self.predictedObjectives, 'Objective should be one of the calculated objectives and/or their average: ' + ', '.join(self.predictedObjectives)
        assert isinstance(objective, str) or isinstance(objective, list), 'Objective must be a single value specified as a string or a list with a single string'
        assert not (isinstance(objective, list) and len(objective) != 1), 'If objective is given as a list, it must contain exactly one string'

        if not self.topCandidates:
            print(f"No clustered prototypes identified. Please run `cluster_prototypes` first. Aborting 3D plot generation.")
            return

        # Create a DataFrame from the data
        df = pd.DataFrame({
            'x': np.transpose(self.embeddedks2022_3d)[0],
            'y': np.transpose(self.embeddedks2022_3d)[1],
            'z': np.transpose(self.embeddedks2022_3d)[2],
            'cluster': [c['cluster'] for c in self.prototypes],
            'name': [c['name'].replace('.POSCAR', '') for c in self.prototypes],
            'objective': [c[objective] for c in self.prototypes],
            'index': [c['index'] for c in self.prototypes]
        })

        df = df.sort_values('cluster')

        # Define a color for each cluster
        colorblind_type = "Protanopia" if colorBlind else None
        colors = get_colors(df['cluster'].nunique(), [(1.0, 1.0, 1.0),(1.0, 1.0, 0.0)], colorblind_type=colorblind_type)

        # Define a symbol for each cluster (up to 8)
        symbols = ['circle', 'circle-open', 'cross', 'diamond', 'diamond-open', 'square', 'square-open', 'x']

        # Create a scatter plot and a mesh for each cluster
        data = []
        for i, cluster in enumerate(df['cluster'].unique()):
            df_cluster = df[df['cluster'] == cluster]
            df_cluster = df_cluster.sort_values(by=['objective', 'index'])

            # Separate the data into top candidates and other cluster members
            df_prototypes = df_cluster[~df_cluster['index'].isin(self.clusterAffiliation[objective].values())]
            df_top_candidates = df_cluster[df_cluster['index'].isin(self.clusterAffiliation[objective].values())]

            hovertext_prototypes = ['<br>'.join(['<b>Prototype: </b>' + '-'.join(c['name'].split('-')[0:2] + [c['name'].split('-')[-1]])] +
                                    ['<b>Affiliation: </b>' + c['name'].split('-')[2]] + 
                                    ['<b>Formation Energy: </b> ' + str(round(c['objective'], 4))])  
                        for _, c in df_cluster.iterrows()]
            
            hovertext_top_candidates = hovertext_prototypes.pop(0)

            color_discrete_map = {'Top Candidate': '#ff7f0e', 'Other Cluster Entries': '#{:02x}{:02x}{:02x}'.format(int(colors[i][0]*255),int(colors[i][1]*255),int(colors[i][2]*255))}

            if markerSymbol and len(df['cluster'].unique()) <= 8:
                symbol = symbols[i]
            else: 
                symbol = symbols[0]

            # Create scatter plot for other cluster members
            scatter_prototypes = dict(
                mode = "markers",
                name = f"Cluster {cluster}",
                type = "scatter3d",
                x = df_prototypes['x'], 
                y = df_prototypes['y'],
                z = df_prototypes['z'],
                hovertext = hovertext_prototypes,  # set the hover name
                hoverinfo = 'text+name',  # only show the hovertext
                marker = dict(
                    size=3, 
                    color=color_discrete_map['Other Cluster Entries'],  # set the color for other cluster members
                    symbol=symbol
                )
            )
            data.append(scatter_prototypes)

            # Create scatter plot for top candidates
            scatter_top_candidates = dict(
                mode = "markers",
                name = f"Cluster {cluster}",
                showlegend = False,  # don't show this in the legend
                type = "scatter3d",
                x = df_top_candidates['x'], 
                y = df_top_candidates['y'],
                z = df_top_candidates['z'],
                hovertext = hovertext_top_candidates,  # set the hover name
                hoverinfo = 'text+name',  # only show the hovertext
                marker = dict(
                    size=3, 
                    color=color_discrete_map['Top Candidate'],  # set the color for top candidates
                    symbol=symbol
                )
            )
            data.append(scatter_top_candidates)

            # Create mesh for the cluster
            mesh = dict(
                alphahull = 0,
                name = f"Cluster {cluster}",
                opacity = 0.1,
                type = "mesh3d", 
                x = df_cluster['x'], 
                y = df_cluster['y'],
                z = df_cluster['z'],
                color=color_discrete_map['Other Cluster Entries'],  # set the color of the mesh
                hoverinfo = 'text+name'  # don't show any hover information for the mesh
            )
            data.append(mesh)

        layout = dict(
            template='plotly_white',  # use the plotly_white template
            scene=dict(
                xaxis_title='',
                yaxis_title='',
                zaxis_title=''
            ),
            height=500,  # set the height of the plot
            width=500,  # set the width of the plot
            title=f'Embedded Prototypes ({len(self.scaledks2022[0])}D ➡ 3D t-SNE space)<br><sub>🟠 Top Candidates ({objective})</sub>'  # add the title here
        )

        figure = dict(data=data, layout=layout)

        if save:
            plot(figure, filename=f'{file_name}.html')
            print(f"Plot saved as {file_name}.html")

        else:
            iplot(figure, filename=file_name)
