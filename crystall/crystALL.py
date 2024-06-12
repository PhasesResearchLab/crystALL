#Import libraries
import pymatgen.core as mg
import os
from natsort import natsorted
from pymatgen.io.vasp.inputs import Poscar
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import re
from itertools import groupby, permutations, product
from operator import itemgetter
import math
import json
from statistics import mean
import csv
from typing import List, Dict, Union, Tuple
from pymongo.database import Database
from pymongo import MongoClient
from colorama import Fore, Style
from pysipfenn import Calculator
from pysipfenn.descriptorDefinitions.KS2022 import onlyStructural
from alignn import pretrained
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans, DBSCAN
import numpy as np
import plotly.express as px
from plotly.graph_objs import Figure
from importlib import resources

# *********** Top-Level Functions ***********
# Run candidate identification and elemental substitution for permutations within the anonymous formula.
def run_local(
        targetFormula: Union[str, mg.Composition],
        candidatePath: str
    ) -> str:
    
    print(f'\nSearching for structure candidates for: {targetFormula}')
    print('Structure candidate dataset:\n' + candidatePath)

    if isinstance(targetFormula, str):
        #Formula to pymatgen Composition
        composition = mg.Composition(targetFormula)
    elif isinstance(targetFormula, mg.Composition):
        composition = targetFormula
    else:
        raise ValueError('Target formula must be a string or a pymatgen Composition object')
    assert os.path.isdir(candidatePath), 'Candidate path must be a valid directory'

    aFormula = composition.anonymized_formula
    nElements = composition.to_data_dict['nelements']

    #Initial candidate dataset scan
    fileList = natsorted(os.listdir(candidatePath))
    total = fileList.__len__()
    print(str(total)+' files detected\n')
    #Intermediate directory for identified candidates
    try:
        os.mkdir(os.path.split(candidatePath)[1] + '-' + aFormula)
        print('Intermediate directory for '+aFormula+' created.')
    except FileExistsError:
        print('Intermediate directory for '+aFormula+' already exist.\nCandidates selection will be overwritten/updated.\n')

    #Counters
    allS = 0
    nEl = 0
    aF = 0
    print('{0:*^80}'.format('Scanning database for '+aFormula+' compounds.'))
    #Iterate over dataset of all structures
    for file in fileList:
        allS = allS + 1
        if os.path.splitext(file)[1] == '.POSCAR':
            s = mg.Structure.from_file(candidatePath +'/'+ file)
            if s.ntypesp == nElements:
                nEl = nEl + 1
                if mg.Composition(s.formula).anonymized_formula == aFormula:
                    sP = Poscar(s)
                    sP.write_file(os.path.split(candidatePath)[1] + '-' + aFormula + '/' + str(aF) + '-' + os.path.splitext(file)[0] + '-' + str.replace(s.formula, ' ', '') + '.POSCAR')
                    aF = aF + 1
                    if aF % 100 == 0:
                        print(str(aF) + ' candidate structures identified.')
        if allS % 5000 == 0:
            print(str(round(allS / total * 100, 1)) + '% database scan completed.')

    print('\nTotal structures detected: ' + str(allS))
    print('\n' + str(nElements) + '-element structures: ' + str(nEl))
    print(str(nElements) + '-element structures fraction: ' + str(round(nEl / total * 100)) + '%')
    print('\n' + aFormula + ': ' + str(aF))
    print(aFormula + ' Fraction from ' + str(nElements) + '-element: ' + str(round(aF / nEl * 100)) + '%')

    #List selected candidates
    candidateList = natsorted(os.listdir(os.path.split(candidatePath)[1] + '-' + aFormula))

    #Substitution elements combination.
    print('{0:*^80}'.format('Substituting elements in the candidates.'))
    sEL = sortedElementsList(composition)
    sELtarget = [item for sublist in sEL for item in sublist]
    print('\nTarget Element Combination: ' + str(sELtarget))

    #Verify need for permutations
    permN = 1
    for multiples in sEL:
        permN = permN * math.factorial(multiples.__len__())

    if permN > 1:
        print('\nTarget composition has degenerate anonymous formula thus requires permutations.')
        print('Up to ' + str(permN) + ' per structure.')
        print('Total number of candidates increased up to: ' + str(aF * permN) + '\n')
    else:
        print('\nTarget composition indicates discrete element substitution.')

    intPath = f"{os.path.split(candidatePath)[1]}-{aFormula}"
    print('Intermediate Directory: ' + intPath)
    finPath = f"{os.path.split(candidatePath)[1]}-{targetFormula}"
    print('Final Directory: ' + finPath)
    try:
        os.mkdir(finPath)
        print('Final directory created.')
    except FileExistsError:
        print('Final directory already exist.\nCandidates will be overwritten/updated.')

    #Substitute over all condidates over all of their permutations
    allS = 0
    for candidate in candidateList:
        s = mg.Structure.from_file(intPath + '/' + candidate)

        permuts = elementPermutations(sortedElementsList(mg.Composition(s.formula)))
        spGroupA = SpacegroupAnalyzer(s)
        fileSplit = candidate.split('.')
        newNameRoot = fileSplit[0] + '-' + spGroupA.get_crystal_system() + '-' + str(
            spGroupA.get_space_group_number()) + '-'

        p = 0
        for permut in permuts:
            d = {}
            p = p + 1
            st = s.copy()
            for e in range(permut.__len__()):
                d.update({mg.Element(permut[e]): mg.Element(sELtarget[e])})
            st.replace_species(d)
            if permuts.__len__() > 1:
                newName = newNameRoot + str.replace(st.formula, ' ', '') + '-permutation' + str(p) + '.POSCAR'
            else:
                newName = newNameRoot + str.replace(st.formula, ' ', '') + '.POSCAR'
            sP = Poscar(st)
            sP.write_file(finPath + '/' + re.sub(r'[?:*|><\\/]', '-', newName))
        allS = allS + p
        if allS % 100 == 0:
            print(str(allS) + ' substitutions complete.')
    print(f'\nGenerated {str(allS)} {targetFormula} candidates.')
    return finPath

# Run candidate identification and elemental substitution for permutations within the anonymous formula.
def run_MPDD(
        targetFormula: Union[str, mg.Composition],
        MPDD: Database,
        collections: List[str],
        limit: int = 0 #No limit
    ) -> str:

    print(f'\nSearching for structure candidates for: {targetFormula}')
    print('Using MPDD Database with collections: '+', '.join(collections))

    if isinstance(targetFormula, str):
        #Formula to pymatgen Composition
        composition = mg.Composition(targetFormula)
    elif isinstance(targetFormula, mg.Composition):
        composition = targetFormula
    else:
        raise ValueError('Target formula must be a string or a pymatgen Composition object')
    assert len(collections) > 0, 'At least one collection must be specified'
    assert isinstance(MPDD, Database), 'MPDD must be a pymongo Database object corresponding to the MPDD database or another database following (at least partially) the same schema'

    aFormula = composition.anonymized_formula
    rFormula = composition.reduced_formula

    print(f'\nLooking for {targetFormula} in MPDD database collections.')

    natPath = f"{'_'.join(collections)}-native-{targetFormula}"

    try:
        os.mkdir(natPath)
        print('Directory created for MPDD candidates: '+natPath )
    except FileExistsError:
        print('Directory for MPDD candidates already exist.\nCandidates will be overwritten/updated.')

    for col in collections:
        print('Looking through: ' + col)
        for e in MPDD[col].find({'material.reducedFormula' : rFormula}):

            s = mg.Structure.from_dict(json.loads(e['material']['mgJSON']))
            spGroupA = SpacegroupAnalyzer(s)

            name = e['material']['formula'].replace(' ', '') + '-' + spGroupA.get_crystal_system() + '-' + str(
                spGroupA.get_space_group_number())

            if 'parentDatabaseID' in e['metadata']:
                name = name + '-' + e['metadata']['parentDatabaseID']
                if 'parentDatabaseURL' in e['metadata']:
                    namePrint = name + '-' + e['metadata']['parentDatabaseURL']
                else:
                    namePrint = name
            else:
                name = name + '-from-' + e['metadata']['name'] + '-at-' + e['metadata']['affiliation']
                namePrint = name

            print(namePrint)
            p = Poscar(s)
            p.write_file(natPath + '/' + re.sub(r'[?:*|><\\/]', '-', name) + '.POSCAR')
    print('Done.\n')

    print('\nDetecting Candidates:')
    foundTotal = 0
    for col in collections:
        foundN = MPDD[col].count_documents({'material.anonymizedFormula' : aFormula})
        foundTotal += foundN
        print('Found ' + str(foundN) + ' in ' + col)

    print('\nTotal of '+str(foundTotal) + ' candidates found.\n')

    #Substitution elements combination.
    print('{0:*^80}'.format('Substituting elements in the candidates.'))
    sEL = sortedElementsList(composition)
    sELtarget = [item for sublist in sEL for item in sublist]
    print('\nTarget Element Combination: ' + str(sELtarget))

    #Verify need for permutations
    permN = 1
    for multiples in sEL:
        permN = permN * math.factorial(multiples.__len__())

    if permN > 1:
        print('\nTarget composition has degenerate anonymous formula thus requires permutations.')
        print('Up to ' + str(permN) + ' per structure.')
        print('Total number of candidates increased up to: ' + str(foundTotal * permN) + '\n')
    else:
        print('\nTarget composition indicates discrete element substitution.')

    finPath = f"{'_'.join(collections)}-{targetFormula}"
    print('Final Candidate Directory: ' + finPath)

    try:
        os.mkdir(finPath)
        print('Final directory created.')
    except FileExistsError:
        print('Final directory already exist.\nCandidates will be overwritten/updated.')

    #Substitute over all candidates over all of their permutations
    allS = 0
    for col in collections:
        for candidate in MPDD[col].find(
                {'material.anonymizedFormula' : aFormula}
                ).hint([('material.anonymizedFormula', 1)]).limit(limit):

            s = mg.Structure.from_dict(json.loads(candidate['material']['mgJSON']))

            permuts = elementPermutations(sortedElementsList(mg.Composition(s.formula)))
            spGroupA = SpacegroupAnalyzer(s)

            nameRoot = candidate['material']['reducedFormula'] + '-' + spGroupA.get_crystal_system() + '-' + str(
                spGroupA.get_space_group_number())

            if 'parentDatabaseID' in candidate['metadata']:
                nameRoot = nameRoot + '-' + candidate['metadata']['parentDatabaseID'].replace('/','-').replace(':','-')

            p = 0
            for permut in permuts:
                d = {}
                p += 1
                st = s.copy()
                for e in range(permut.__len__()):
                    d.update({mg.Element(permut[e]): mg.Element(sELtarget[e])})
                st.replace_species(d)
                if permuts.__len__() > 1:
                    name = nameRoot + '-' + str.replace(st.formula, ' ', '') + '-permutation_' + str(p) + '.POSCAR'
                else:
                    name = nameRoot + '-' + str.replace(st.formula, ' ', '') + '.POSCAR'
                sP = Poscar(st)
                sP.write_file(finPath + '/' + re.sub(r'[?:*|><\\/]', '-', name))
            allS = allS + p
            if allS % 1000 == 0:
                print(str(allS) + ' substitutions complete.')

    print(f'\nGenerated {str(allS)} {targetFormula} candidates.')
    return finPath

#Run candidate identification and elemental substitution for permutations within the anonymous composition fractions
# E.g.
#targetCompositionDict = {'Nd':338, 'Sn':215, 'Bi':447}
#devs =  {'Nd':15, 'Sn':15, 'Bi':15}
#MPDD = client['MPDD']
#collections = ['community', 'curated']
def run_compositional_MPDD(
        targetCompositionDict: Dict[str, int],
        devs: Dict[str, int],
        MPDD: Database,
        collections: List[str],
        limit: int = 0 #No limit
    ) -> str:

    # Target formula
    meanFormula = ''.join(f'{key}{val}' for key, val in targetCompositionDict.items())
    rangesStr = f"+-({','.join(str(r) for r in devs.values())})"

    print('\nSearching for structure candidates for: ' + meanFormula + ' ' + rangesStr)
    print('Using MPDD Database with collections: ' + ', '.join(collections))

    # ************************* Try to find materials already in the MPDD database **********************************

    print('\nLooking for ' + meanFormula + ' in MPDD database collections.')

    natPath = '_'.join(collections) + '-native-' + meanFormula + rangesStr

    try:
        os.mkdir(natPath)
        print('Directory created for MPDD candidates: ' + natPath)
    except FileExistsError:
        print('Directory for MPDD candidates already exist.\nCandidates will be overwritten/updated.')

    # Query for matching MPDD entries
    nativeQuery = comp2compQ(targetCompositionDict, devs)

    targetPresent = 0
    for col in collections:
        print('Looking through: ' + col)
        for e in MPDD[col].find(nativeQuery):

            s = mg.Structure.from_dict(json.loads(e['material']['mgJSON']))
            spGroupA = SpacegroupAnalyzer(s)

            name = e['material']['formula'].replace(' ', '') + '-' + spGroupA.get_crystal_system() + '-' + str(
                spGroupA.get_space_group_number())

            if 'parentDatabaseID' in e['metadata']:
                name = name + '-' + e['metadata']['parentDatabaseID']
                if 'parentDatabaseURL' in e['metadata']:
                    namePrint = name + '-' + e['metadata']['parentDatabaseURL']
                else:
                    namePrint = name
            else:
                name = name + '-from-' + e['metadata']['name'] + '-at-' + e['metadata']['affiliation']
                namePrint = name

            print(namePrint)
            p = Poscar(s)
            p.write_file(natPath + '/' + re.sub(r'[?:*|><\\/]', '-', name) + '.POSCAR')
            targetPresent += 1

    print('Done.')
    print(f'{targetPresent} target materials found in MPDD.\n')

    # ******** Try to find materials from the MPDD database that can be adjusted to the target formula  **************

    # Sorted fractions dictionary query
    sortedFractionsDictQuery = comp2acompQ(targetCompositionDict, devs)

    # ************************************** Count candidates ********************************************************

    print('\nDetecting Candidates:')
    foundTotal = 0
    for col in collections:
        foundN = MPDD[col].count_documents(sortedFractionsDictQuery)
        foundTotal += foundN
        print('Found ' + str(foundN) + ' in ' + col)

    print('\nTotal of ' + str(foundTotal) + ' candidates available.\n')

    # ********************************* Substitution elements combination ********************************************

    print('{0:*^80}'.format('Substituting elements in the candidates.'))

    finPath = '_'.join(collections) + '-' + meanFormula + rangesStr
    print('Final Candidate Directory: ' + finPath)

    try:
        os.mkdir(finPath)
        print('Final directory created.')
    except FileExistsError:
        print('Final directory already exist.\nCandidates will be overwritten/updated.')

    # All generated candidates
    allS = 0
    # List of permutation numbers
    permutNList = []

    # Search all collections using the compound index for the first 8 components
    for col in collections:
        for candidate in MPDD[col].find(
                sortedFractionsDictQuery
        ).hint([
            ('material.sortedFractionsDict.c1', 1),
            ('material.sortedFractionsDict.c2', 1),
            ('material.sortedFractionsDict.c3', 1),
            ('material.sortedFractionsDict.c4', 1),
            ('material.sortedFractionsDict.c5', 1),
            ('material.sortedFractionsDict.c6', 1),
            ('material.sortedFractionsDict.c7', 1),
            ('material.sortedFractionsDict.c8', 1)
        ]).limit(limit):

            s = mg.Structure.from_dict(json.loads(candidate['material']['mgJSON']))
            c = mg.Composition(s.formula)

            cDict = dict(c.fractional_composition.as_dict())
            for e in cDict:
                cDict[e] = int(cDict[e] * 1000)

            elements = [element for element in cDict]

            oAS = orderedAllowedSubstitutions(targetCompositionDict, devs, cDict)

            permutNList += [len(oAS)]

            spGroupA = SpacegroupAnalyzer(s)

            nameRoot = candidate['material']['reducedFormula'] + '-' + spGroupA.get_crystal_system() + '-' + \
                       str(spGroupA.get_space_group_number())

            if 'parentDatabaseID' in candidate['metadata']:
                nameRoot = nameRoot + '-' + candidate['metadata']['parentDatabaseID'].replace('/', '-').replace(':',
                                                                                                                '-')

            p = 1
            for substitutions in oAS:
                d = {}
                st = s.copy()
                for o, r in zip(elements, substitutions):
                    d.update({mg.Element(o): mg.Element(r)})
                st.replace_species(d)

                if len(oAS) > 1:
                    name = nameRoot + '-' + str.replace(st.formula, ' ', '') + '-permutation_' + str(p) + '.POSCAR'
                    p += 1
                else:
                    name = nameRoot + '-' + str.replace(st.formula, ' ', '') + '.POSCAR'
                sP = Poscar(st)
                sP.write_file(finPath + '/' + re.sub(r'[?:*|><\\/]', '-', name))
            allS = allS + p
            if allS % 50 == 0:
                print(str(allS) + ' substitutions complete.')

    if allS>0:
        print('\nGenerated ' + str(allS) + ' ' + meanFormula + ' candidates.')
        print(f'Mean number of permutations: {mean(permutNList)}')
        print(f'Min / Max number of permutations: {min(permutNList)} / {max(permutNList)}')
    else:
        print('No candidates were generated.')

    return finPath

def estimate_compositional_MPDD(
        targetCompositionDict: Dict[str, int],
        devs: Dict[str, int],
        MPDD: Database,
        collections: List[str]
    ) -> str:

    # Target formula
    meanFormula = ''.join(f'{key}{val}' for key, val in targetCompositionDict.items())
    rangesStr = f"+-({','.join(str(r) for r in devs.values())})"

    # Sorted fractions dictionary query
    sortedFractionsDictQuery = comp2acompQ(targetCompositionDict, devs)

    print(f'\nDetecting Candidates for substitution into {meanFormula} under allowed deviations {rangesStr}')
    foundTotal = 0
    for col in collections:
        foundN = MPDD[col].count_documents(sortedFractionsDictQuery)
        foundTotal += foundN
        print('Found ' + str(foundN) + ' in ' + col)

    print('\nTotal of ' + str(foundTotal) + ' candidates available.\n')

    return foundTotal

# Calcualte distance between structures in POSCAR files and the composition given by the target composition dictionary
# CSV file needs to be in the structure files folder
def compositionalManhattanDistanceAdd(
        structureFiles: str,
        csvfile: str, 
        targetCompositionDict: Dict[str, int]
    ) -> None:

    oldCSVpath = structureFiles + '/' + csvfile
    reader = csv.reader(open(oldCSVpath, 'r'))
    newCSV = open(oldCSVpath.replace('.csv', '') + '_distances.csv', 'w', newline='')
    writer = csv.writer(newCSV)
    headers = next(reader)
    headers.append("Distance")
    writer.writerow(headers)

    for row in reader:
        s = mg.Structure.from_file(structureFiles + '/' + row[0])
        c = mg.Composition(s.formula)
        cDict = dict(c.fractional_composition.as_dict())
        d = 0
        for e in targetCompositionDict:
            d += abs(targetCompositionDict[e] - int(cDict[e] * 1000))
        row.append(str(d))
        writer.writerow(row)
    newCSV.close()

# For a given composition, returns elements grouped by their occurrence in the formula,
# starting from the least common.
# E.g. 'Al2Ni3O2U4As3I4Bi2' returns [['Al', 'Bi', 'O'], ['Ni', 'As'], ['U', 'I']]
def sortedElementsList(
        comp: mg.Composition
    ) -> List[List[str]]:

    fractions = re.sub(r'[A-z]+', '', comp.formula).split()
    elements = re.sub(r'[0-9]+', '', comp.formula).split()
    compl = []
    for i in range(elements.__len__()):
        compl.append([elements[i],int(fractions[i])])
    compl.sort(key=itemgetter(1))
    sortedEL = [[x for x, _ in g]
           for _, g in groupby(compl,key=itemgetter(1))]
    return sortedEL

# For a given list of elements grouped by occurrence in the formula (see sortedElementsList())
# it returns all permutations of elements within the groups that conserve the anonymous
# formula of the compound.
# E.g. 'Al2Ni3O2U4As3I4Bi2Gd1' returns
# [['Gd', 'Al', 'Bi', 'O', 'Ni', 'As', 'U', 'I'],
#  ['Gd', 'Al', 'Bi', 'O', 'Ni', 'As', 'I', 'U'],
#  ['Gd', 'Al', 'Bi', 'O', 'As', 'Ni', 'U', 'I'],
#  ['Gd', 'Al', 'Bi', 'O', 'As', 'Ni', 'I', 'U'],
#  ['Gd', 'Al', 'O', 'Bi', 'Ni', 'As', 'U', 'I'],
#  ['Gd', 'Al', 'O', 'Bi', 'Ni', 'As', 'I', 'U'],
#  ['Gd', 'Al', 'O', 'Bi', 'As', 'Ni', 'U', 'I'],
#  ['Gd', 'Al', 'O', 'Bi', 'As', 'Ni', 'I', 'U'],
#  ['Gd', 'Bi', 'Al', 'O', 'Ni', 'As', 'U', 'I'],
#  ['Gd', 'Bi', 'Al', 'O', 'Ni', 'As', 'I', 'U'],
#  ['Gd', 'Bi', 'Al', 'O', 'As', 'Ni', 'U', 'I'],
#  ['Gd', 'Bi', 'Al', 'O', 'As', 'Ni', 'I', 'U'],
#  ['Gd', 'Bi', 'O', 'Al', 'Ni', 'As', 'U', 'I'],
#  ['Gd', 'Bi', 'O', 'Al', 'Ni', 'As', 'I', 'U'],
#  ['Gd', 'Bi', 'O', 'Al', 'As', 'Ni', 'U', 'I'],
#  ['Gd', 'Bi', 'O', 'Al', 'As', 'Ni', 'I', 'U'],
#  ['Gd', 'O', 'Al', 'Bi', 'Ni', 'As', 'U', 'I'],
#  ['Gd', 'O', 'Al', 'Bi', 'Ni', 'As', 'I', 'U'],
#  ['Gd', 'O', 'Al', 'Bi', 'As', 'Ni', 'U', 'I'],
#  ['Gd', 'O', 'Al', 'Bi', 'As', 'Ni', 'I', 'U'],
#  ['Gd', 'O', 'Bi', 'Al', 'Ni', 'As', 'U', 'I'],
#  ['Gd', 'O', 'Bi', 'Al', 'Ni', 'As', 'I', 'U'],
#  ['Gd', 'O', 'Bi', 'Al', 'As', 'Ni', 'U', 'I'],
#  ['Gd', 'O', 'Bi', 'Al', 'As', 'Ni', 'I', 'U']]
def elementPermutations(
        sEL: List[List[str]]
    ) -> List[List[str]]:
    generatingList = [[list(i) for i in permutations(lev)] for lev in sEL]
    outList = [[item for sublist in list(permutation) for item in sublist] for permutation in product(*generatingList)]
    return outList

# For a given (1) composition dictionary with permil fractions (1/1000)
# E.g. {'Nd':338, 'Sn':215, 'Bi':447}
# and a list of allowed composition deviations
# e.g. {'Nd':50, 'Sn':50, 'Bi':20}
# returns a MPDD query for ananymous composition space with N components and
def comp2acompQ(
        compDict: Dict[str, int],
        devs: Dict[str, int]
    ) -> Dict[str, Union[Dict[str, int], int]]:

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

    meanComposition = mg.Composition(' '.join(''.join((key, str(val))) for (key, val) in compDict.items()))
    mpddQuery = dict()
    mpddQuery.update({'material.system': meanComposition.chemical_system})
    for e in compDict:
        key = 'material.compositionDictionary.' + e
        mpddQuery.update({key: {'$gte': (compDict[e] - devs[e]) / 1000, '$lte': (compDict[e] - devs[e]) / 1000}})
    return mpddQuery

# For a given composition, returns elements grouped by their occurrence in the formula,
# starting from the least common.
# E.g. 'Al2Ni3O2U4As3I4Bi2' returns [['Al', 'Bi', 'O'], ['Ni', 'As'], ['U', 'I']]
def orderedAllowedSubstitutions(
        targetCompDict: Dict[str, int],
        devs: Dict[str, int],
        candidateCompDict: Dict[str, int]
    ) -> List[List[str]]:

    # Finds what eleemnts can be used for substitution
    allInRangeSubstitutions = []
    for c in candidateCompDict:
        inRangeSubstitutions = []
        val = candidateCompDict[c]
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


# *********** Convenience High-Level API Class ***********

class CrystALLSearch:
    """
    A self-contained environment for (1) obtaining POSCAR files of ``crystALL`` candidates from either MPDD or a local 
    dataset of POSCAR files, (2) featurizing the candidates with KS2022 or future ``pySIPFENN`` descriptor calculators,
    (3) getting property predictions using ``pySIPFENN`` or ``ALIGNN``, (4) clustering the candidates in ``KS2022`` 
    feature space, and (5) selecting candidates based on desired property and cluster affiliation.
    """
    
    def __init__(
            self,
            targetFormula: Union[str, mg.Composition],
            MPDD: Database = None,
            collections: List[str] = None,
            candidatePath: str = None,
            calculator: Calculator = None
    ) -> None:
        
        # Defined at initialization
        self.MPDD = MPDD
        self.collections = collections
        self.candidatePath = candidatePath
        if isinstance(targetFormula, str):
            self.targetFormula = mg.Composition(targetFormula)
        elif isinstance(targetFormula, mg.Composition):
            self.targetFormula = targetFormula
        else:
            raise TypeError('Target formula must be a string or a pymatgen Composition object')
        self.targetCompositionDict = {el: round(amt / self.targetFormula.num_atoms * 1000, 0)
                                      for el, amt in self.targetFormula.as_dict().items()}
        
        # Internal parameters to use during the processing steps
        self.availableCollections: List[str] = None
        self.availableObjectives: List[str] = []
        self.inputFiles: List[str] = None
        self.ks2022: np.ndarray = None
        self.figure: Figure = None
        self.clusterAffiliation: Dict[int, Dict[str, float]] = {}
        self.candidates: List[Dict[str, Union[str, float]]] = []
        self.finalCandidates: List[Dict[str, Union[str, float]]] = []

        print(f"Initialized CrystALLSearch for target formula: {Fore.MAGENTA}{Style.BRIGHT}{self.targetFormula.formula}{Style.RESET_ALL}")

        print(f"\nChecking for available models on the computer..")
        # Check if MPDD-ALIGNN default models are available
        self.alignn_models = True
        with resources.files('alignn') as alignn_dir:
            for model_dict in pretrained.get_default_models():
                if not os.path.exists(os.path.join(alignn_dir, model_dict['model'])):
                    self.alignn_models = False
                    print(f"{Style.DIM}✘ MPDD-ALIGNN models not found. Please run `download_models`.{Style.RESET_ALL}")
                    break
            if self.alignn_models:
                print(f"{Fore.GREEN}✔ MPDD-ALIGNN models found.{Style.RESET_ALL}")

        # Check if pySIPFENN models are available
        self.pysipfenn_models = True
        self.Calculator = calculator if calculator is not None else Calculator(autoLoad=False, verbose=False)
        
        existingCompatibleModels = set(self.Calculator.findCompatibleModels('KS2022')).intersection(set(self.Calculator.network_list_available))
        
        if not len(existingCompatibleModels) > 0:
            self.pysipfenn_models = False
            print(f"{Style.DIM}✘ No pySIPFENN models compatible with 'KS2022' descriptor found. Please run `download_models`.{Style.RESET_ALL}")
        else:
            print(f"{Fore.GREEN}✔ pySIPFENN models found.{Style.RESET_ALL}")
            self.update_objectives(self.Calculator)

    def update_objectives(
            self,
            calculator: Calculator
    ) -> None:

        assert isinstance(calculator, Calculator), "Calculator must be a Calculator object"
        
        loadedCompatibleModels = set(self.Calculator.findCompatibleModels('KS2022')).intersection(set(self.Calculator.loadedModels.keys()))

        if not len(loadedCompatibleModels) > 0:
            alternative_model = list(set(self.Calculator.findCompatibleModels('KS2022')).intersection(set(self.Calculator.network_list_available)))[0]
            print(f"\nNo pySIPFENN models compatible with 'KS2022' were specified in the Calculator. Loading available model: {alternative_model}.")
            self.Calculator.loadModels(network = alternative_model)
        
        self.objectives = list(set(calculator.findCompatibleModels('KS2022')).intersection(set(self.Calculator.loadedModels.keys())))
        self.objectives += ['ALIGNN-MP Formation Energy [eV/atom]', 'ALIGNN-JARVIS Formation Energy [eV/atom]'] if self.alignn_models else []

        print(f"\nObjectives set for calculation: {Fore.GREEN}{Style.BRIGHT}{', '.join(self.objectives)}{Style.RESET_ALL}\nTo select another set, use `update_objectives`.")

    def check_objectives(
            self
    ) -> None:
        """
        Checks and returns the available objectives.

        Raises:
            AssertionError: If no objectives are available.

        Returns:
            List[str]: List of common objectives between self.objectives and self.availableObjectives.
        """

        assert isinstance(self.objectives, list) and len(self.objectives) >= 1, "Objectives must be a list of strings with at least one objective."
        
        if any(value not in self.availableObjectives for value in self.objectives):
            existing_objectives = list(set(self.availableObjectives) & set(self.objectives))
            print("\nSome objectives do not have predictions calculated and will not be considered for clustering.")
            for obj in self.objectives:
                if obj not in existing_objectives:
                    print(f"{Style.DIM}✘ {Style.BRIGHT}{obj}{Style.RESET_ALL}")
                else:
                    print(f"{Fore.GREEN}✔ {Style.BRIGHT}{obj}{Style.RESET_ALL}")
            objectives = existing_objectives
        else:
            objectives = self.objectives
       
        return objectives

    def download_models(
            self
    ) -> None:
        """
        Downloads the default models for both pySIPFENN and MPDD-ALIGNN.
        """        
        
        print(f"Downloading models for pysipfenn..")
        self.Calculator.downloadModels()
        print(f"Downloading models for MPDD-alignn..")
        pretrained.download_default_models()
        print(f"Done! Models are ready to use.")

    def connect_MPDD(
            self,
            MPDD: Database = None,
            collections: List[str] = ['curated']
    ) -> None:
        """
        Establishes a connection to the MPDD database and sets the collections to search in.

        Args:
            MPDD: A pymongo Database object to connect to. If not provided, uses the existing connection.
            collections: List of collection names to search in. Defaults to ['curated'].

        Raises:
            AssertionError: If MPDD is not a pymongo Database object, if no collections are found in the database, if any specified collection is not in the database, or if no collections are specified.
        """

        self.MPDD = MPDD if MPDD is not None else self.MPDD
        assert isinstance(self.MPDD, Database), 'MPDD must be a pymongo Database object'
        self.availableCollections = self.MPDD.list_collection_names()
        assert len(self.availableCollections) > 0, 'No collections found in the database or the database connection has not been established properly'
        assert set(collections).issubset(self.availableCollections), 'At least one of the specified collections is not available in the database'
        print(f"\nConnected to database {Fore.MAGENTA}{Style.BRIGHT}{self.MPDD.name}{Style.RESET_ALL} with collections: {Fore.MAGENTA}{Style.BRIGHT}{', '.join(self.availableCollections)}{Style.RESET_ALL}")
        assert len(collections) > 0, 'At least one collection must be specified'
        self.collections = collections
        print(f"Using collections: {Fore.GREEN}{Style.BRIGHT}{', '.join(collections)}{Style.RESET_ALL}")
        
    def generate_MPDD(
            self,
            limit: int = 1000,
            devs: Dict[str, int] = None
    ) -> None:
        """
        Generates candidate materials from the Material-Property-Descriptor Database (MPDD) based on the target formula specified in the `targetFormula` attribute of the class.

        Before calling this method, a connection to the MPDD must be established using the `connect_MPDD` method.

        Raises:
            AssertionError: If no connection to the MPDD has been established, or if no collections have been specified.

        Args:
            limit: The maximum number of candidates to generate. Defaults to 1000.
        """
        assert self.MPDD is not None, "No database connection identified. Please run ``connect_MPDD`` first."
        assert self.collections, "No collections identified. Please provide a list of collections to search in."
        assert devs is None or isinstance(devs, dict), "Deviations must be a dictionary with element symbols as keys and integer values as deviations"

        if devs is None:
            self.candidatePath = run_MPDD(self.targetFormula, self.MPDD, self.collections, limit)
        else:
            self.candidatePath = run_compositional_MPDD(self.targetCompositionDict, devs, self.MPDD, self.collections, limit)
        self.inputFiles = natsorted(os.listdir(self.candidatePath))
        self.candidates = [{"index": i, "name": file} for i, file in enumerate(self.inputFiles)]

    def generate_local(
            self,
            candidatePath: str = None
    ) -> None:
        """
        Generates candidates from a local directory containing POSCAR or CIF files.

        Args:
            candidatePath: Path to the directory with candidate files. If not provided, uses the existing path.

        Raises:
            AssertionError: If no candidate directory is specified.
        """

        self.candidatePath = candidatePath if candidatePath is not None else self.candidatePath
        assert self.candidatePath, "No candidates folder identified. Please provide a path to the candidates folder."
        self.candidatePath = run_local(self.targetFormula, self.candidatePath)
        self.inputFiles = natsorted(os.listdir(self.candidatePath))
        self.candidates = [{"index": i, "name": file} for i, file in enumerate(self.inputFiles)]

    def run_pysipfenn(
            self
    ) -> None:
        """
        Predicts properties for the candidate materials using pySIPFENN models and updates the candidates with the predictions.

        Raises:
            AssertionError: If no candidate directory is specified, if the candidate directory path is not a string, if no pySIPFENN models are found, or if the number of candidates does not match the number of property predictions.
        """

        assert self.candidatePath is not None, "No candidates folder identified. Please run ``generate_MPDD`` or ``generate_local`` first."
        assert isinstance(self.candidatePath, str), "Candidates folder path must be a string"
        assert self.pysipfenn_models, "pySIPFENN models not found. Please run ``download_models`` first."

        for model in set(self.Calculator.findCompatibleModels('KS2022')).intersection(set(self.objectives)):
            if model not in self.availableObjectives:
                self.availableObjectives.append(model)
        
        self.Calculator.runFromDirectory(self.candidatePath, 'KS2022', mode='parallel', max_workers=8)
        resultDicts, self.ks2022 = self.Calculator.get_resultDicts(), np.array(self.Calculator.descriptorData)
        assert len(self.candidates) == len(resultDicts), "Number of candidates does not match number of property predictions"
        
        for i in range(len(resultDicts)):
            self.candidates[i].update(resultDicts[i])

        # Update average of the predictions
        self.calculate_average()
        
        print(f"Predicted properties for {len(self.ks2022)} candidates using pySIPFENN")
    
    def run_alignn(
            self
    ) -> None:
        """
        Predicts properties for the candidate materials using MPDD-ALIGNN models and updates the candidates with the predictions.

        Raises:
            AssertionError: If no candidate directory is specified, if the candidate directory path is not a string, if no MPDD-ALIGNN models are found, or if the number of candidates does not match the number of property predictions.
        """

        assert self.candidatePath is not None, "No candidates folder identified. Please run ``generate_MPDD`` or ``generate_local`` first."
        assert isinstance(self.candidatePath, str), "Candidates folder path must be a string"
        assert self.alignn_models, "MPDD-ALIGNN models not found. Please run ``download_models`` first."
        
        for model_dict in pretrained.get_default_models():
            if model_dict['name'] in self.objectives and model_dict['name'] not in self.availableObjectives:
                self.availableObjectives.append(model_dict['name'])
        
        print("\nModels that will be run: ['ALIGNN-MP Formation Energy [eV/atom]', 'ALIGNN-JARVIS Formation Energy [eV/atom]']")
        print('Calculating descriptors...')
        resultDicts = pretrained.run_models_from_directory(self.candidatePath, mode='parallel')
        assert len(self.candidates) == len(resultDicts), "Number of candidates does not match number of property predictions"

        for i in range(len(resultDicts)):
            self.candidates[i].update(resultDicts[i])
        
        # Update average of the predictions
        self.calculate_average()
        
        print(f"Predicted properties for {len(resultDicts)} candidates using ALIGNN")

    def calculate_average(
            self
    ) -> None:
        """
        Calculates the average of the predicted properties for each candidate material.

        Raises:
            AssertionError: If no features are identified or if less than two models are available for averaging.
        """

        assert len(self.candidates) > 0, "No candidates identified. Please run ``run_pysipfenn`` or ``run_alignn`` first."
        assert self.availableObjectives, "No objective predictions identified. Please run ``run_pysipfenn`` or ``run_alignn`` first."

        # Calculate the average of the predictions
        for i, candidate in enumerate(self.candidates):
            average_prediction = sum(candidate[pred] for pred in self.availableObjectives) / len(self.availableObjectives)
            self.candidates[i].update({f'Avg of Formation Energy': average_prediction})

    def filter_candidates(
            self,
            max_candidates: int,
            cutoff: float = 0.5
    ) -> None:
        """
        Filters the candidates based on the Formation Energy cutoff value provided. Must be run AFTER featurizarion / property prediction by ``run_pysipfenn``.
        """

        assert len(self.candidates) > 0, "No candidates identified. Please run ``generate_MPDD`` or ``generate_local`` first."
        assert len(self.ks2022) > 0, "No features identified. Please run ``run_pysipfenn`` first."

        if max_candidates:
            self.candidates.sort(key=lambda candidate: candidate['Avg of Formation Energy'])
            self.candidates = self.candidates[:max_candidates]
        else:
            min_energy = min(candidate['Avg of Formation Energy'] for candidate in self.candidates)
            self.candidates = [candidate for candidate in self.candidates if candidate['Avg of Formation Energy'] < cutoff + min_energy]

        # Update the 'index' key of each dictionary in self.candidates to its position in the list
        for i, candidate in enumerate(self.candidates):
            candidate['index'] = i

        # Update self.ks2022 to only include the values at the positions specified by the "index" key in the filtered self.candidates
        indices = [candidate['index'] for candidate in self.candidates]
        self.ks2022 = self.ks2022[indices]

        print(f"Filtered candidates to {len(self.candidates)} based on the maximum number of candidates") if max_candidates else print(f"Filtered candidates to {len(self.candidates)} based on the cutoff value of {cutoff}")

    def embed_features(
            self,
            sameComposition: bool = False
    ) -> None:
        """
        Embeds the features in 2D t-SNE space.

        Args:
            sameComposition: If True, only structural features are used. Defaults to False.

        Raises:
            AssertionError: If no features are identified.
        """

        assert self.ks2022 is not None, "No features identified. Please run ``run_pysipfenn`` first."

        if sameComposition:
            features2use = np.array([onlyStructural(v) for v in self.ks2022])
        else:
            features2use = self.ks2022

        # Select, Scale and Embed features in 2D t-SNE space
        scaler = MinMaxScaler()
        self.scaledks2022 = scaler.fit_transform(features2use)
        tsne = TSNE(n_components = 2, perplexity = 4, init = 'pca', n_iter = 2000, metric = 'correlation', angle = 0.2, n_jobs = -1, random_state = 580)
        self.embeddedks2022 = tsne.fit_transform(self.scaledks2022)
        tsne_3d = TSNE(n_components = 3, perplexity = 4, init = 'pca', n_iter = 2000, metric = 'correlation', angle = 0.2, n_jobs = -1, random_state = 580)
        self.embeddedks2022_3d = tsne_3d.fit_transform(self.scaledks2022)
        
    def cluster_candidates_KMeans(
            self,
            n_clusters: int = None,
            sameComposition: bool = False
    ) -> None:
        """
        Clusters the candidates using KMeans, assigns each data point to its cluster, and selects the best candidates.

        Args:
            n_clusters: Number of clusters. If not provided, the optimal number is determined automatically using silhouette score.
            sameComposition: If True, only candidates with the same composition are considered. Defaults to False.

        Raises:
            AssertionError: If no features are identified, if the number of clusters is not an integer, or if any objective is not allowed.
        """

        assert self.ks2022 is not None, "No features identified. Please run ``run_pysipfenn`` first."
        assert isinstance(n_clusters, int) or n_clusters is None, "Number of clusters must be an integer"
        assert n_clusters is None or n_clusters <= len(self.candidates), "Number of clusters must be less or equal than the number of candidates"

        objectives = self.check_objectives()

        # Select, Scale and Embed features in 2D t-SNE space
        self.embed_features(sameComposition)

        print("\nNo number of clusters has been provided. Determining the optimal number through `silhouette score`. This may take a few minutes..\n") if n_clusters is None else None
        
        # Determine the optimal number of clusters using silhouette score
        sil_dict = {}
        k_range = [n_clusters] if n_clusters is not None else np.unique(np.geomspace(2, len(self.embeddedks2022) - 1, 40).astype(int))
        for i, k in enumerate(k_range):
            print(f"Processing.. {round(100*i/len(k_range),2)} %") if i != 0 else None
            kmeans = KMeans(n_clusters=k, n_init=500, max_iter=500, random_state=580)
            clusterData = kmeans.fit(self.embeddedks2022)
            sil_score = silhouette_score(self.embeddedks2022, kmeans.labels_)
            sil_dict[sil_score] = clusterData

        # Determine the optimal number of clusters as the one with the highest silhouette score
        clusterData = sil_dict[max(sil_dict.keys())]

        print(f"\nClustered candidates into {clusterData.n_clusters} clusters using KMeans in 2D t-SNE embedded space.\n")

        # Assign each data point to its cluster
        clusters = {}
        for i, label in enumerate(clusterData.labels_):
            self.candidates[i].update({'cluster': label + 1})
            clusters.setdefault(label + 1, []).append(self.candidates[i])

        # Select the best candidate per cluster and Create a list of final candidates
        self.__select_candidates(clusters, objectives)

    def cluster_candidates_DBSCAN(
            self,
            eps: float = 0.3,
            min_samples: int = 1,
            p: int = 1,
            sameComposition: bool = False
    ) -> None:
        """
        Clusters the candidates using DBSCAN, assigns each data point to its cluster, and selects the best candidates.

        Args:
            eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other. Defaults to 0.3.
            min_samples: The number of samples in a neighborhood for a point to be considered as a core point. Defaults to 1.
            p: The power parameter for the Minkowski metric. Defaults to 1.
            sameComposition: If True, only candidates with the same composition are considered. Defaults to False.

        Raises:
            AssertionError: If no features are identified, if eps is not a float, if min_samples is not an integer, or if any objective is not allowed.
        """

        assert self.ks2022 is not None, "No features identified. Please run ``run_pysipfenn`` first."
        assert isinstance(eps, float), "eps must be a float"
        assert isinstance(min_samples, int), "min_samples must be an integer"

        objectives = self.check_objectives()

        # Select, Scale and Embed features in 2D t-SNE space
        self.embed_features(sameComposition)

        # Cluster the candidates using DBSCAN
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, p=p)
        dbscan.fit(self.scaledks2022)

        # Determine the number of clusters identified by DBSCAN
        n_clusters = len(set(dbscan.labels_)) - (1 if -1 in dbscan.labels_ else 0)
        print(f"\nClustered candidates into {n_clusters} clusters using DBSCAN.\n")
        
        # Assign each data point to its cluster
        clusters = {}
        for i, label in enumerate(dbscan.labels_):
            self.candidates[i].update({'cluster': label + 1})
            clusters.setdefault(label + 1, []).append(self.candidates[i])
      
        # Select the best candidate per cluster and Create a list of final candidates
        self.__select_candidates(clusters, objectives)

    def __select_candidates(
            self,
            clusters: dict,
            objectives: List[str],
    ) -> None:
        """
        Selects the best candidate per cluster for each objective and stores them in self.finalCandidates.

        Args:
            clusters: A dictionary mapping cluster labels to lists of candidate dictionaries.
            objectives: A list of objectives to consider when selecting candidates.

        Returns:
            List[dict]: A list of dictionaries representing the selected candidates.
        """

        print(f"List of candidates selected per energy of each cluster:")
        
        # Select the best candidate per cluster based on the minimum predicted property
        self.clusterAffiliation = {}
        for objective in objectives:
            print(f"\nObjective: {objective}")
            self.clusterAffiliation[objective] = {}
            for label, cluster in sorted(clusters.items()):
                candidate = min(cluster, key=lambda x: x[objective])
                self.clusterAffiliation[objective].update({candidate['cluster']: candidate['index']})
                print(f"           Cluster {label}: {candidate['name']}")
        
        # Create a list of final candidates
        self.finalCandidates, i = [], 0
        for objective in objectives:
            for index in self.clusterAffiliation[objective].values():
                self.finalCandidates.append(self.candidates[index].copy())
                self.finalCandidates[i].update({'objective': objective})
                i += 1

        return self.finalCandidates

    def export_table(
            self,
            content: List[str] = ['allCandidates', 'finalCandidates'],
            file_prefix: str = ''
    ) -> None:
        """
        Exports a CSV file with the specified content.

        Args:
            content: Specifies the type of data to export. Can be 'allCandidates' and/or 'finalCandidates'. 
                                        Defaults to ['allCandidates', 'finalCandidates'].
            file_prefix: A prefix that will be added to the file name. Defaults to ''.

        Raises:
            AssertionError: If no final candidates are identified or if an invalid content type is provided.

        Outputs:
            CSV files: Writes CSV files for 'allCandidates' and/or 'finalCandidates' depending on the content argument. 
                        The files are named based on the content type and the file_prefix (if provided).
        """

        assert self.finalCandidates, "No final candidates identified. Please run ``cluster_candidates_KMeans`` or ``cluster_candidates_DBSCAN`` first."
        assert not any(value not in ['allCandidates', 'finalCandidates'] for value in content), "Content should contain only 'candidates' and/or 'finalCandidates'"

        objectives = self.check_objectives()

        if 'allCandidates' in content:
            file_name = 'allCandidates.csv' if file_prefix == '' else file_prefix + '_allCandidates.csv'
            with open(file_name, 'w', newline='') as output_file:
                dict_writer = csv.DictWriter(output_file, self.candidates[0].keys())
                dict_writer.writeheader()
                dict_writer.writerows(self.candidates)
            
            print(f"\nExported {len(self.candidates)} candidates to {file_name}")
        
        if 'finalCandidates' in content:
            file_name = 'finalCandidates.csv' if file_prefix == '' else file_prefix + '_finalCandidates.csv'
            finalCandidates_exported = [d for d in self.finalCandidates if d['objective'] in objectives]
            
            with open('finalCandidates.csv', 'w', newline='') as output_file:
                dict_writer = csv.DictWriter(output_file, finalCandidates_exported[0].keys())
                dict_writer.writeheader()
                dict_writer.writerows(finalCandidates_exported)
            
            print(f"Exported {len(finalCandidates_exported)} final candidates to {file_name}")
  
    def make_plot(
            self,
            objective: Union[str, List[str]] = None
    ) -> None:
        """
        Creates a scatter plot of clustered candidates in 2D t-SNE space.

        Args:
            objective: The objective to consider for the plot. Must be a key in self.availableObjectives. Defaults to ``self.objectives[0]``.

        Raises:
            AssertionError: If no clusters are identified or if the objective is not in self.availableObjectives.
        """
        
        objective = objective if objective is not None else self.objectives[0]

        assert self.finalCandidates, "No final candidates identified. Please run ``cluster_candidates_KMeans`` or ``cluster_candidates_DBSCAN`` first."
        assert objective in self.availableObjectives, "Objective should be one of the calculated objectives and/or their average: " + ', '.join(self.availableObjectives)
        assert isinstance(objective, str) or isinstance(objective, list), "Objective must be a single value specified as a string or a list with a single string"
        assert not (isinstance(objective, list) and len(objective) != 1), "If objective is given as a list, it must contain exactly one string"

        fig = px.scatter(
            x=np.transpose(self.embeddedks2022)[0],
            y=np.transpose(self.embeddedks2022)[1],
            hover_name=['<br>'.join(['Cluster: ' + str(cl)] + ['-'.join(n[0:3])] + ['-'.join(n[3:5])] + [objective + ': '] + [str(round(p, 4))]) 
                        for c in self.candidates 
                        for cl, n, p in [(c['cluster'], c['name'].replace('permutation_', 'P').replace('.POSCAR', '').split('-'), c[objective])]],
            color=[
                'Top Candidate' if i in self.clusterAffiliation[objective].values() else 
                'Other Cluster Members' 
                for i in range(len(self.embeddedks2022))
                ],
            color_discrete_map={
                'Top Candidate': '#ff7f0e',
                'Other Cluster Members': '#1f77b4'
            },
            opacity=0.85,
            color_discrete_sequence=px.colors.qualitative.Dark24,
            template='plotly_white',
            labels={'x': f'{len(self.scaledks2022[0])}D->2D TSNE1',
                    'y': f'{len(self.scaledks2022[0])}D->2D TSNE2',
                    'color': f'Embedded Candidates'},
            height=400,
            width=800
            )
        
        self.figure = fig
        print(f"Plot created for `{objective}` objective. Run ``plot`` to display the plot.")

    def plot(
            self,
            save: bool = False,
            file_name: str = 'clusterCandidates'
    ) -> None:
        """
        Displays the plot of clustered candidates in 2D t-SNE space and optionally saves it as an HTML file.

        Args:
            save: If True, the plot is saved as an HTML file. Defaults to False.
            file_name: The file name of the file to be saved. Defaults to 'clusterCandidates'.
        
        Raises:
            AssertionError: If no plot is identified. Please run ``make_plot`` first.
        """
        
        assert self.figure is not None, "No plot identified. Please run ``export_plot`` first."

        if save:
            self.figure.write_html(file_name + '.html')
            print(f"Plot saved as {file_name}.html")

        else:
            self.figure.show()
