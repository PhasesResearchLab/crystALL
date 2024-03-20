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


# Run candidate identification and elemental substitution for permutations within the anonymous formula.
def run_local(
        targetFormula: Union[str, mg.Composition], 
        candidatePath: str
    ) -> str:
    print('\nSearching for structure candidates for: '+targetFormula)
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
    sEL = sortedElementsList(mg.Composition(targetFormula))
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

    intPath = os.path.split(candidatePath)[1] + '-' + aFormula
    print('Intermediate Directory: ' + intPath)
    finPath = os.path.split(candidatePath)[1] + '-' + targetFormula
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
            sP.write_file(finPath + '/' + newName)
        allS = allS + p
        if allS % 100 == 0:
            print(str(allS) + ' substitutions complete.')
    print('\nGenerated ' + str(allS) + ' ' + targetFormula + ' candidates.')
    return finPath


# Run candidate identification and elemental substitution for permutations within the anonymous formula.
def run_MPDD(
        targetFormula: Union[str, mg.Composition],
        MPDD: Database,
        collections: List[str]
    ) -> str:
    print('\nSearching for structure candidates for: '+targetFormula)
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

    print('\nLooking for ' + targetFormula + ' in MPDD database collections.')

    natPath = '_'.join(collections) + '-native-' + targetFormula

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
            p.write_file(natPath + '/' + name + '.POSCAR')
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
    sEL = sortedElementsList(mg.Composition(targetFormula))
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

    finPath = '_'.join(collections) + '-' + targetFormula
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
                ).hint([('material.anonymizedFormula', 1)]):

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
                p = p + 1
                st = s.copy()
                for e in range(permut.__len__()):
                    d.update({mg.Element(permut[e]): mg.Element(sELtarget[e])})
                st.replace_species(d)
                if permuts.__len__() > 1:
                    name = nameRoot + '-' + str.replace(st.formula, ' ', '') + '-permutation_' + str(p) + '.POSCAR'
                else:
                    name = nameRoot + '-' + str.replace(st.formula, ' ', '') + '.POSCAR'
                sP = Poscar(st)
                sP.write_file(finPath + '/' + name)
            allS = allS + p
            if allS % 1000 == 0:
                print(str(allS) + ' substitutions complete.')

    print('\nGenerated ' + str(allS) + ' ' + targetFormula + ' candidates.')
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
            p.write_file(natPath + '/' + name + '.POSCAR')
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
                sP.write_file(finPath + '/' + name)
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
def compositionalManhattanDistanceAdd(structureFiles: str, csvfile: str, targetCompositionDict: dict):
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
def sortedElementsList(mgComp):
    fractions = re.sub(r'[A-z]+', '', mgComp.formula).split()
    elements = re.sub(r'[0-9]+', '', mgComp.formula).split()
    compl = []
    for i in range(elements.__len__()):
        compl.append([elements[i],int(fractions[i])])
    compl.sort(key=itemgetter(1))
    sortedEL = [[x for x,y in g]
           for k,g in  groupby(compl,key=itemgetter(1))]
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
def elementPermutations(sEL):
    generatingList = [[list(i) for i in permutations(lev)] for lev in sEL]
    outList = [[item for sublist in list(permutation) for item in sublist] for permutation in product(*generatingList)]
    return outList

# For a given (1) composition dictionary with permil fractions (1/1000)
# E.g. {'Nd':338, 'Sn':215, 'Bi':447}
# and a list of allowed composition deviations
# e.g. {'Nd':50, 'Sn':50, 'Bi':20}
# returns a MPDD query for ananymous composition space with N components and
def comp2acompQ(compDict: dict, devs: dict) -> dict:
    eVals = []
    for e in compDict:
        eVals.append([compDict[e], devs[e]])
    eVals.sort(key = lambda x: x[0], reverse=True)
    result = dict()
    for eVal, c in zip(eVals, range(eVals.__len__())):
        result.update({f'material.sortedFractionsDict.c{c+1}': {'$gte': (eVal[0]-eVal[1]), '$lte': (eVal[0]+eVal[1])}})
    result.update({f'material.nComponents': len(eVals)})
    return result

def comp2compQ(compDict: dict, devs: dict) -> dict:
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
def orderedAllowedSubstitutions(targetCompDict: dict, devs: dict, candidateCompDict: dict):
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