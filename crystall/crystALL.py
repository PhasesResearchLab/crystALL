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
from typing import List, Dict, Union
from pymongo.database import Database
from pymongo import MongoClient
from colorama import Fore, Style
from pysipfenn import Calculator
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

    #Substitute over all condidates over all of their permutations
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
        collections: List[str]
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
        ]):

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
        
        # Internal parameters to use during the processing steps
        self.availableCollections: List[str] = None
        self.inputFiles: List[str] = None
        self.ks2022: np.ndarray = None
        self.figure: Figure = None
        self.available_modes = []
        self.clusterAffiliation: Dict[int, Dict[str, float]] = {}
        self.candidates: List[Dict[str, Union[str, float]]] = []
        self.finalCandidates: List[Dict[str, Union[str, float]]] = []
        
        # Check if MPDD-ALIGNN default models are available
        print(f"Checking for available models..")
        self.alignn_models = True
        with resources.files('alignn') as alignn_dir:
            for model_dict in pretrained.get_default_models():
                if not os.path.exists(os.path.join(alignn_dir, model_dict['model'])):
                    self.alignn_models = False
                    print(f"{Style.DIM}x {model_dict['name']}{Style.RESET_ALL}")
                    print(f"MPDD-ALIGNN models not found. To use these models, you will need to download them using the `download_models` method.")
                    break
                else:
                    print(f"{Fore.GREEN}✔ {model_dict['name']}{Style.RESET_ALL}")

        # Check if pySIPFENN models are available
        self.pysipfenn_models = True
        self.Calculator = calculator if calculator is not None else Calculator(autoLoad=False, verbose=False)
        if not len(self.Calculator.network_list_available) > 0:
            self.pysipfenn_models = False
            print(f"pySIPFENN models not found. To use these models, you will need to download them using the `download_models` method.")

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
            MPDD (Database, optional): A pymongo Database object to connect to. If not provided, uses the existing connection.
            collections (List[str], optional): List of collection names to search in. Defaults to ['curated'].

        Raises:
            AssertionError: If MPDD is not a pymongo Database object, if no collections are found in the database, if any specified collection is not in the database, or if no collections are specified.
        """

        self.MPDD = MPDD if MPDD is not None else self.MPDD
        assert isinstance(self.MPDD, Database), 'MPDD must be a pymongo Database object'
        self.availableCollections = self.MPDD.list_collection_names()
        assert len(self.availableCollections) > 0, 'No collections found in the database or the database connection has not been established properly'
        assert set(collections).issubset(self.availableCollections), 'At least one of the specified collections is not available in the database'
        print(f"Connected to database {Fore.MAGENTA}{Style.BRIGHT}{self.MPDD.name}{Style.RESET_ALL} with collections: {Fore.MAGENTA}{Style.BRIGHT}{', '.join(self.availableCollections)}{Style.RESET_ALL}")
        assert len(collections) > 0, 'At least one collection must be specified'
        self.collections = collections
        print(f"Using collections: {Fore.GREEN}{Style.BRIGHT}{', '.join(collections)}{Style.RESET_ALL}")
        
    def generate_MPDD(
            self,
            limit: int = 1000
    ) -> None:
        """
        Generates candidate materials from the Material-Property-Descriptor Database (MPDD) based on the target formula specified in the `targetFormula` attribute of the class.

        Before calling this method, a connection to the MPDD must be established using the `connect_MPDD` method.

        Raises:
            AssertionError: If no connection to the MPDD has been established, or if no collections have been specified.

        Args:
            limit (int, optional): The maximum number of candidates to generate. Defaults to 1000.
        """
        assert self.MPDD is not None, "No database connection identified. Please run ``connect_MPDD`` first."
        assert self.collections, "No collections identified. Please provide a list of collections to search in."
        self.candidatePath = run_MPDD(self.targetFormula, self.MPDD, self.collections, limit)
        self.inputFiles = natsorted(os.listdir(self.candidatePath))
        self.candidates = [{"index": i, "name": file} for i, file in enumerate(self.inputFiles)]

    def generate_local(
            self,
            candidatePath: str = None
    ) -> None:
        """
        Generates candidates from a local directory containing POSCAR or CIF files.

        Args:
            candidatePath (str, optional): Path to the directory with candidate files. If not provided, uses the existing path.

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
        Uses pySIPFENN models to predict properties for the candidate materials.

        Raises:
            AssertionError: If no candidate directory is specified, if the candidate directory path is not a string, if no pySIPFENN models are found, or if the number of candidates does not match the number of property predictions.
        """

        assert self.candidatePath is not None, "No candidates folder identified. Please run ``generate_MPDD`` or ``generate_local`` first."
        assert isinstance(self.candidatePath, str), "Candidates folder path must be a string"
        assert self.pysipfenn_models, "pySIPFENN models not found. Please run ``download_models`` first."

        self.Calculator.loadModels(network = 'SIPFENN_Krajewski2022_NN30')
        for model in self.Calculator.loadedModels:
            if model not in self.available_modes:
                self.available_modes.append(model)
        
        self.Calculator.runFromDirectory(self.candidatePath, 'KS2022', mode='parallel', max_workers=8)
        resultDicts, self.ks2022 = self.Calculator.get_resultDicts(), np.array(self.Calculator.descriptorData)
        assert len(self.candidates) == len(resultDicts), "Number of candidates does not match number of property predictions"
        
        for i in range(len(resultDicts)):
            self.candidates[i].update(resultDicts[i])

        print(f"Predicted properties for {len(self.ks2022)} candidates using pySIPFENN")
    
    def run_alignn(
            self
    ) -> None:
        """
        Uses MPDD-ALIGNN models to predict properties for the candidate materials.

        Raises:
            AssertionError: If no candidate directory is specified, if the candidate directory path is not a string, if no MPDD-ALIGNN models are found, or if the number of candidates does not match the number of property predictions.
        """

        assert self.candidatePath is not None, "No candidates folder identified. Please run ``generate_MPDD`` or ``generate_local`` first."
        assert isinstance(self.candidatePath, str), "Candidates folder path must be a string"
        assert self.alignn_models, "MPDD-ALIGNN models not found. Please run ``download_models`` first."
        
        for model_dict in pretrained.get_default_models():
            if model_dict['name'] not in self.available_modes:
                self.available_modes.append(model_dict['name'])
        
        print('Importing structures...')
        resultDicts = pretrained.run_models_from_directory(self.candidatePath, mode='parallel')
        assert len(self.candidates) == len(resultDicts), "Number of candidates does not match number of property predictions"

        for i in range(len(resultDicts)):
            self.candidates[i].update(resultDicts[i])
        
        print(f"Predicted properties for {len(resultDicts)} candidates using ALIGNN")

    def cluster_candidates_KMeans(
            self,
            n_clusters: int = None,
            modes: List[str] = None
    ) -> None:
        """
        Clusters the candidates in 2D t-SNE space and selects the best ones based on the predicted property.

        Args:
            n_clusters (int, optional): Number of clusters. If not provided, it's determined automatically.
            modes (List[str], optional): List of modes to consider. If not provided, all allowed modes are used.

        Raises:
            AssertionError: If no features are identified, if the number of clusters is not an integer, or if any mode is not allowed.
        """

        assert self.ks2022 is not None, "No features identified. Please run ``run_pysipfenn`` first."
        assert isinstance(n_clusters, int) or n_clusters is None, "Number of clusters must be an integer"

        # Calculate the average of the predictions
        if len(self.available_modes) > 1:
            self.available_modes += ['Avg of Predictions'] if 'Avg of Predictions' not in self.available_modes else []
            for i, prediction in enumerate(self.candidates):
                average_prediction = sum(prediction[mode] for mode in self.available_modes if mode != 'Avg of Predictions') / (len(self.available_modes) - 1)
                self.candidates[i].update({'Avg of Predictions': average_prediction})

        modes = modes if modes is not None else self.available_modes
        assert not any(value not in self.available_modes for value in modes), "Modes should contain only models with calculated results and/or their average: " + ', '.join(self.available_modes)
        
        # Scale the features and embed them in 2D t-SNE space
        scaler = MinMaxScaler()
        self.scaledks2022 = scaler.fit_transform(self.ks2022)
        tsne = TSNE(n_components = 2, perplexity = 4, init = 'pca', n_iter = 2000, metric = 'correlation', angle = 0.2, n_jobs = -1, random_state = 580)
        self.embeddedks2022 = tsne.fit_transform(self.scaledks2022)

        print("This may take a few minutes.. Determining the optimal number of clusters..\n") if n_clusters is None else None
        
        # Determine the optimal number of clusters using silhouette score
        sil_dict = {}
        k_range = [n_clusters] if n_clusters is not None else np.geomspace(2, len(self.ks2022) - 1, 20).astype(int)
        for i, k in enumerate(np.unique(k_range)):
            print(f"Processing.. {100*i/20} %") if i != 0 else None
            kmeans = KMeans(n_clusters=k, n_init=500, max_iter=1000, random_state=580)
            clusterData = kmeans.fit(self.embeddedks2022)
            sil_score = silhouette_score(self.embeddedks2022, kmeans.labels_)
            sil_dict[sil_score] = clusterData

        # Determine the optimal number of clusters as the one with the highest silhouette score
        clusterData = sil_dict[max(sil_dict.keys())]

        print(f"\nClustered candidates into {clusterData.n_clusters} clusters using KMeans in 2D t-SNE embedded space.\n")

        # Create a list of lists for the clusters
        clusters = [[] for _ in range(clusterData.n_clusters)]

        # Assign each data point to its cluster
        for i, label in enumerate(clusterData.labels_):
            self.candidates[i].update({'cluster': label})
            clusters[label].append(self.candidates[i])

        print(f"List of candidates selected per energy of each cluster:")
    
        # Select the best candidate per cluster based on the minimum predicted property
        self.clusterAffiliation = {}
        for mode in modes:
            print(f"\nMode: {mode}")
            self.clusterAffiliation[mode] = {}
            for label, cluster in enumerate(clusters):
                candidate = min(cluster, key=lambda x: x[mode])
                self.clusterAffiliation[mode].update({candidate['cluster']: candidate['index']})
                print(f"      Cluster {label + 1}: {candidate['name']}")
        
        # Create a list of final candidates
        self.finalCandidates, i = [], 0
        for mode in modes:
            for index in self.clusterAffiliation[mode].values():
                self.finalCandidates.append(self.candidates[index].copy())
                self.finalCandidates[i].update({'mode': mode})
                i += 1

    def cluster_candidates_DBSCAN(
            self,
            eps: float = 0.3,
            min_samples: int = 2,
            p: int = 1,
            modes: List[str] = None
    ) -> None:
        """
        Clusters the candidates using DBSCAN and selects the best ones based on the predicted property.
        The function also scales the features and assigns each data point to its cluster.

        Args:
            eps (float, optional): The maximum distance between two samples for one to be considered as in the neighborhood of the other. Defaults to 0.3.
            min_samples (int, optional): The number of samples in a neighborhood for a point to be considered as a core point. Defaults to 2.
            modes (List[str], optional): List of modes to consider. If not provided, all allowed modes are used.

        Raises:
            AssertionError: If no features are identified, if eps is not a float, if min_samples is not an integer, or if any mode is not allowed.
        """

        assert self.ks2022 is not None, "No features identified. Please run ``run_pysipfenn`` first."
        assert isinstance(eps, float), "eps must be a float"
        assert isinstance(min_samples, int), "min_samples must be an integer"

        # Calculate the average of the predictions
        if len(self.available_modes) > 1:
            self.available_modes += ['Avg of Predictions'] if 'Avg of Predictions' not in self.available_modes else []
            for i, prediction in enumerate(self.candidates):
                average_prediction = sum(prediction[mode] for mode in self.available_modes if mode != 'Avg of Predictions') / (len(self.available_modes) - 1)
                self.candidates[i].update({'Avg of Predictions': average_prediction})

        modes = modes if modes is not None else self.available_modes
        assert not any(value not in self.available_modes for value in modes), "Modes should contain only models with calculated results and/or their average: " + ', '.join(self.available_modes)
        
        # Scale the features
        scaler = MinMaxScaler()
        self.scaledks2022 = scaler.fit_transform(self.ks2022)
        tsne = TSNE(n_components = 2, perplexity = 4, init = 'pca', n_iter = 2000, metric = 'correlation', angle = 0.2, n_jobs = -1, random_state = 580)
        self.embeddedks2022 = tsne.fit_transform(self.scaledks2022)

        # Cluster the candidates using DBSCAN
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, p=p)
        dbscan.fit(self.scaledks2022)

        # Determine the number of clusters identified by DBSCAN
        n_clusters = len(set(dbscan.labels_)) - (1 if -1 in dbscan.labels_ else 0)
        print(f"\nClustered candidates into {n_clusters} clusters using DBSCAN.\n")
        
        # Assign each data point to its cluster
        clusters = {}
        for i, label in enumerate(dbscan.labels_):
            self.candidates[i].update({'cluster': label})
            clusters.setdefault(label, []).append(self.candidates[i])

        print(f"List of candidates selected per energy of each cluster:")
        
        # Select the best candidate per cluster based on the minimum predicted property
        self.clusterAffiliation = {}
        for mode in modes:
            print(f"\nMode: {mode}")
            self.clusterAffiliation[mode] = {}
            for label, cluster in clusters.items():
                candidate = min(cluster, key=lambda x: x[mode])
                self.clusterAffiliation[mode].update({candidate['cluster']: candidate['index']})
                print(f"      Cluster {label + 1}: {candidate['name']}")
        
        # Create a list of final candidates
        self.finalCandidates, i = [], 0
        for mode in modes:
            for index in self.clusterAffiliation[mode].values():
                self.finalCandidates.append(self.candidates[index].copy())
                self.finalCandidates[i].update({'mode': mode})
                i += 1

    def export_table(
            self,
            content: List[str] = ['candidates', 'finalCandidates'],
            modes: List[str] = None
    ) -> None:
        """
        Exports a CSV file containing the candidates and/or final candidates with predicted properties.

        Args:
            content (List[str], optional): List of content types to export. Defaults to ['candidates', 'finalCandidates'].
            modes (List[str], optional): Modes to select 'finalCandidates'. Defaults to all models (``self.available_modes``).

        Raises:
            AssertionError: If no clusters are identified or if any content type is not allowed.
        """

        assert self.clusterAffiliation, "No clusters identified. Please run ``cluster_candidates_TSNE`` first."
        assert not any(value not in ['candidates', 'finalCandidates'] for value in content), "Content should contain only 'candidates' and/or 'finalCandidates'"

        modes = modes if modes is not None else self.available_modes
        
        if 'candidates' in content:
            with open('candidates.csv', 'w', newline='') as output_file:
                dict_writer = csv.DictWriter(output_file, self.candidates[0].keys())
                dict_writer.writeheader()
                dict_writer.writerows(self.candidates)
        
        if 'finalCandidates' in content:
            self.finalCandidates = [d for d in self.finalCandidates if d['mode'] in modes]
            
            with open('finalCandidates.csv', 'w', newline='') as output_file:
                dict_writer = csv.DictWriter(output_file, self.finalCandidates[0].keys())
                dict_writer.writeheader()
                dict_writer.writerows(self.finalCandidates)
  
    def make_plot(
            self,
            mode: str = 'Avg of Predictions'
    ) -> None:
        """
        Creates a scatter plot of clustered candidates in 2D t-SNE space and optionally saves it as an HTML file.

        Args:
            mode (str, optional): The mode to consider for the plot. Must be a key in self.clusterAffiliation. Defaults to 'Avg of Predictions'.
            save (bool, optional): If True, the plot is saved as an HTML file. Defaults to False.

        Raises:
            AssertionError: If no clusters are identified or if the mode is not in self.clusterAffiliation.
        """
        
        assert self.clusterAffiliation, "No clusters identified. Please run ``cluster_candidates_TSNE`` first."
        assert mode in self.clusterAffiliation.keys(), "Mode should be one of the calculated results and/or their average: " + ', '.join(self.clusterAffiliation.keys())
        
        fig = px.scatter(
            x=np.transpose(self.embeddedks2022)[0],
            y=np.transpose(self.embeddedks2022)[1],
            hover_name=['<br>'.join(['Cluster: ' + str(cl)] + ['-'.join([n[0]] + n[1:3] + [n[6]])] + ['-'.join(n[3:5])] + [mode + ': '] + [str(round(p, 4))]) 
                        for c in self.candidates 
                        for cl, n, p in [(c['cluster'], c['name'].replace('permutation_', 'P').replace('.POSCAR', '').split('-'), c[mode])]],
            color=[
                'Top Candidate' if i in self.clusterAffiliation[mode].values() else 
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
            labels={'x': '256D->2D TSNE1',
                    'y': '256D->2D TSNE2',
                    'color': f'Embedded Candidates'},
            height=400,
            width=800
            )
        
        self.figure = fig

    def plot(
            self,
            save: bool = False
    ) -> None:
        """
        Displays the plot of clustered candidates in 2D t-SNE space and optionally saves it as an HTML file.

        Args:
            save (bool, optional): If True, the plot is saved as an HTML file. Defaults to False.
        
        Raises:
            AssertionError: If no plot is identified. Please run ``make_plot`` first.
        """
        
        assert self.figure is not None, "No plot identified. Please run ``export_plot`` first."

        if save:
            self.figure.write_html('clusterCandidates.html')

        else:
            self.figure.show()
