"""
Module for handling prototype identification, selection, and elemental substitution.
"""

import os
import re
import math
import json
import time
import threading
import concurrent.futures

from typing import Union, List, Dict, Literal, Tuple, Any, Iterable

from natsort import natsorted
from colorama import Fore, Style
from yaspin import yaspin

from itertools import groupby, permutations, product
from operator import itemgetter
from statistics import mean

from pymongo.database import Database
from pymatgen.core import Structure, Composition, Element
from pymatgen.io.vasp.inputs import Poscar
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


def _substitute_candidate(
    name_root: str,
    structure_dict: dict,
    substitutions: list,
    sELtarget: List[str],
    out_dir: str,
) -> int:
    st_orig = Structure.from_dict(structure_dict)
    count = 0
    p = 0
    for permut in substitutions:
        p += 1
        d = {}
        st = st_orig.copy()

        if isinstance(permut, dict):
            for orig_el, target_el in permut.items():
                d[Element(orig_el)] = Element(target_el)
        else:
            for e in range(len(permut)):
                d[Element(permut[e])] = Element(sELtarget[e])

        st.replace_species(d)

        if len(substitutions) > 1:
            name = name_root + "-p" + str(p) + ".vasp"
        else:
            name = name_root + ".vasp"

        sanitized_name = re.sub(r"[?:*|><\\/]", "-", name)
        sP = Poscar(st)
        sP.write_file(os.path.join(out_dir, sanitized_name))
        count += 1

    return count


def _generate_from_structures(
    target_formula: Union[str, Composition],
    candidates: Iterable[tuple[str, Structure, list]],
    out_dir: str,
    parallel: bool = True,
) -> int:
    """
    Apply elemental substitution to candidate structures and write .vasp files.

    Args:
        target_formula: Target composition used to derive the ordered target element list.
        candidates: Iterable of (name_root, structure, substitutions) tuples.
            name_root: sanitized base filename without extension or permutation suffix.
            structure: pymatgen Structure to substitute from.
            substitutions: list of ordered element symbol lists, one per permutation.
        out_dir: Existing output directory path.
        parallel: If True, process candidates in parallel using ProcessPoolExecutor.

    Returns:
        Total number of .vasp files written.
    """
    if isinstance(target_formula, str):
        composition = Composition(target_formula)
    else:
        composition = target_formula

    sEL = sorted_elements_list(composition)
    sELtarget = [item for sublist in sEL for item in sublist]

    candidates_list = list(candidates)
    if not candidates_list:
        return 0

    total_written = 0
    last_reported = 0

    if parallel and len(candidates_list) > 1:
        name_roots = [c[0] for c in candidates_list]
        struct_dicts = [c[1].as_dict() for c in candidates_list]
        subs = [c[2] for c in candidates_list]
        sELtargets = [sELtarget] * len(candidates_list)
        out_dirs = [out_dir] * len(candidates_list)

        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = executor.map(
                _substitute_candidate,
                name_roots,
                struct_dicts,
                subs,
                sELtargets,
                out_dirs,
            )
            for count in results:
                total_written += count
                if total_written - last_reported >= 100:
                    print(str(total_written) + " substitutions complete.")
                    last_reported = (total_written // 100) * 100
    else:
        for name_root, structure, substitutions in candidates_list:
            count = _substitute_candidate(
                name_root, structure.as_dict(), substitutions, sELtarget, out_dir
            )
            total_written += count
            if total_written - last_reported >= 100:
                print(str(total_written) + " substitutions complete.")
                last_reported = (total_written // 100) * 100

    return total_written


def run_local(target_formula: Union[str, Composition], prototype_path: str) -> str:
    """
    Run prototype identification and elemental substitution for permutations within the anonymous formula.

    Args:
        target_formula: The target formula as a string or pymatgen Composition object.
        prototype_path: Path to the directory containing prototype files.

    Returns:
        Path to the directory containing the generated prototypes.

    Raises:
        ValueError: If the target formula is not a string or pymatgen Composition object.
        AssertionError: If the prototype path is not a valid directory.
    """

    assert os.path.isdir(prototype_path), "Prototype path must be a valid directory"

    print(f"\nSearching for structure prototypes for: {target_formula}")
    print("Structure prototype dataset: " + prototype_path)

    if isinstance(target_formula, str):
        # Formula to pymatgen Composition
        composition = Composition(target_formula)
    elif isinstance(target_formula, Composition):
        composition = target_formula
    else:
        raise ValueError(
            "Target formula must be a string or a pymatgen Composition object"
        )
    assert os.path.isdir(prototype_path), "Prototype path must be a valid directory"

    aFormula = composition.anonymized_formula
    nElements = composition.to_data_dict["nelements"]

    # Initial prototype dataset scan – collect matching structures in memory only
    fileList = natsorted(os.listdir(prototype_path))
    total = len(fileList)

    allS = 0  # total files inspected
    nEl = 0  # structures with correct number of elements
    aF = 0  # structures matching anonymized formula
    # Keep both the original filename (base_name) and the Structure to avoid name collisions later
    prototypeList: List[Tuple[str, Structure]] = []

    for file in fileList:
        allS += 1
        ext = os.path.splitext(file)[1]
        if ext not in [".POSCAR", ".vasp"]:
            continue

        s = Structure.from_file(os.path.join(prototype_path, file))
        if s.n_elems == nElements:
            nEl += 1
            if Composition(s.formula).anonymized_formula == aFormula:
                base_name = os.path.splitext(os.path.basename(file))[0]
                prototypeList.append((base_name, s))
                aF += 1
                if aF % 100 == 0:
                    print(str(aF) + " prototype structures identified.")

        if total and allS % 5000 == 0:
            print(str(round(allS / total * 100, 1)) + "% database scan completed.")

    print("\nTotal structures detected: " + str(allS))
    print(str(nElements) + "-element structures: " + str(nEl))
    if total:
        print(
            str(nElements)
            + "-element structures fraction: "
            + str(round(nEl / total * 100))
            + "%\n"
        )
    else:
        print(str(nElements) + "-element structures fraction: 0%\n")
    print(aFormula + " Fraction from " + str(nElements) + "-element: " + str(aF))
    if nEl:
        print(
            aFormula
            + " Fraction from "
            + str(nElements)
            + "-element: "
            + str(round(aF / nEl * 100))
            + "%\n"
        )
    else:
        print(aFormula + " Fraction from " + str(nElements) + "-element: 0%\n")

    # Substitution elements combination.
    sEL = sorted_elements_list(composition)
    sELtarget = [item for sublist in sEL for item in sublist]

    # Verify need for permutations
    permN = 1
    for multiples in sEL:
        permN = permN * math.factorial(multiples.__len__())

    if permN > 1:
        print(
            "Target composition has degenerate anonymous formula thus requires permutations."
        )
        print(
            f"Up to {Fore.BLUE}{Style.BRIGHT}{str(permN)}{Style.RESET_ALL} per structure.\n"
        )
    else:
        print("Target composition indicates discrete element substitution.\n")

    # Standardized output directory naming: <formula>-local-prototypes
    formula_str = str(target_formula).replace(" ", "")
    out_dir = f"{formula_str}-local-prototypes"
    # Ensure we are not writing back into the prototype source directory
    assert os.path.abspath(out_dir) != os.path.abspath(prototype_path), (
        "Output directory must differ from prototype_path"
    )

    print(
        f"Final Prototype Directory: {Fore.LIGHTBLACK_EX}{Style.BRIGHT}{out_dir}{Style.RESET_ALL}"
    )
    try:
        os.mkdir(out_dir)
        print(f"{Fore.GREEN}Final directory created.{Style.RESET_ALL}")
    except FileExistsError:
        print(
            f"{Fore.RED}Final directory already exist: Prototypes will be overwritten/updated.{Style.RESET_ALL}\n"
        )

    # Substitute over all candidates over all of their permutations
    candidates = []
    for base_name, s in prototypeList:
        permuts = element_permutations(sorted_elements_list(Composition(s.formula)))
        spGroupA = SpacegroupAnalyzer(s)
        # Preserve original base_name to avoid overwriting files from different prototypes
        newNameRoot = "local-" + str(spGroupA.get_crystal_system()) + "-" + base_name
        candidates.append((newNameRoot, s, permuts))

    allS = _generate_from_structures(target_formula, candidates, out_dir)
    print(
        f"\nGenerated {Fore.GREEN}{Style.BRIGHT}{str(allS)}{Style.RESET_ALL} {target_formula} prototypes."
    )
    return out_dir


def run_MPDD(
    target_formula: Union[str, Composition],
    MPDD: Database,
    collections: List[str],
    target_prototypes: Literal["native", "non-native"],
    filter_prototypes: Literal["material.structID", "material.volNormStructID"],
    limit: int,
) -> str:
    """
    Run prototype identification and elemental substitution for permutations within the anonymous formula using the MPDD database.

    Args:
        target_formula: The target formula as a string or pymatgen Composition object.
        MPDD: A pymongo Database object corresponding to the MPDD database.
        collections: List of collection names to search in.
        target_prototypes: Specifies whether to use 'native' or 'non-native' compositions.
        filter_prototypes: Specifies the prototype filter to use.
        limit: The maximum number of prototypes to generate.

    Returns:
        Path to the directory containing the generated prototypes.

    Raises:
        ValueError: If the target formula is not a string or pymatgen Composition object.
        AssertionError: If no collections are specified or if MPDD is not a pymongo Database object.
    """

    assert len(collections) > 0, "At least one collection must be specified"
    assert isinstance(MPDD, Database), "MPDD must be a pymongo Database object"

    print(
        f"\nSearching for structure prototypes for: {Fore.YELLOW}{Style.BRIGHT}{target_formula}{Style.RESET_ALL}"
    )
    print(
        f"Using MPDD Database with collections: {Fore.MAGENTA}{Style.BRIGHT}{', '.join(collections)}{Style.RESET_ALL}"
    )

    if isinstance(target_formula, str):
        # Formula to pymatgen Composition
        composition = Composition(target_formula)
    elif isinstance(target_formula, Composition):
        composition = target_formula
    else:
        raise ValueError(
            "Target formula must be a string or a pymatgen Composition object"
        )
    assert len(collections) > 0, "At least one collection must be specified"
    assert isinstance(MPDD, Database), (
        "MPDD must be a pymongo Database object corresponding to the MPDD database or another database following (at least partially) the same schema"
    )

    aFormula = composition.anonymized_formula
    rFormula = composition.reduced_formula

    formula_str = str(target_formula).replace(" ", "")
    source_str = f"MPDD({'-'.join(collections)})"

    if target_prototypes == "native":
        out_dir = f"{formula_str}-{source_str}-prototypes(native)"

        try:
            os.mkdir(out_dir)
            print("Directory created for MPDD prototypes: " + out_dir)
        except FileExistsError:
            print(
                "Directory for MPDD prototypes already exist.\nPrototypes will be overwritten/updated.\n"
            )

        for col in collections:
            print("Looking through: " + col)

            pipeline = [
                {"$match": {"material.reducedFormula": rFormula}},
                {
                    "$group": {
                        "_id": "$" + filter_prototypes,
                        "document": {"$first": "$$ROOT"},
                    }
                },
                {"$sort": {"document.properties.EF_SIPFENN_NN30": 1}},
                {
                    "$facet": {
                        "data": [{"$limit": limit}],
                        "meta": [{"$count": "total"}],
                    }
                },
            ]

            _msgs = (
                "rɛplacing elɛments in thɛ prototypes",
                "replacing ɛlemɛnts in the prototypɛs",
                "rɛplacing ɛlɛments in the prototypes",
                "replacing elemɛnts in thɛ prototypɛs",
            )
            with yaspin(text=_msgs[0]) as sp:
                _stop_evt = threading.Event()

                def _cycle():
                    i = 0
                    while not _stop_evt.is_set():
                        sp.text = _msgs[i % len(_msgs)]
                        i += 1
                        time.sleep(0.6)

                _t = threading.Thread(target=_cycle, daemon=True)
                _t.start()
                try:
                    facet_out = list(MPDD[col].aggregate(pipeline, allowDiskUse=True))
                finally:
                    _stop_evt.set()
                    _t.join(timeout=2)

            facet = facet_out[0] if facet_out else {"data": [], "meta": []}
            aggregated_data = facet.get("data", [])
            total_groups = (
                facet.get("meta", [{}])[0].get("total", 0) if facet.get("meta") else 0
            )
            print(
                f"Total number of filtered structures: {Fore.GREEN}{Style.BRIGHT}{total_groups}{Style.RESET_ALL}"
                f"{f' (limited to {Fore.RED}{Style.BRIGHT}{limit}{Style.RESET_ALL})' if limit < total_groups else ''}"
            )

            for entry in aggregated_data:
                e = entry["document"]

                s = Structure.from_dict(json.loads(e["material"]["mgJSON"]))
                spGroupA = SpacegroupAnalyzer(s)

                name = (
                    e["material"]["formula"].replace(" ", "")
                    + "-"
                    + str(spGroupA.get_crystal_system())
                )

                if "parentDatabaseID" in e["metadata"]:
                    name = name + "-" + e["metadata"]["parentDatabaseID"]

                else:
                    name = (
                        name
                        + "-from-"
                        + e["metadata"]["name"]
                        + "-at-"
                        + e["metadata"]["affiliation"]
                    )

                p = Poscar(s)
                p.write_file(
                    out_dir + "/" + re.sub(r"[?:*|><\\/]", "-", name) + ".vasp"
                )
        print("Done.\n")

        return out_dir

    elif target_prototypes == "non-native":
        print("\nDetecting Matching Input Structures:")
        foundTotal = 0
        for col in collections:
            foundN = MPDD[col].count_documents({"material.anonymizedFormula": aFormula})
            foundTotal += foundN
            print(
                f"Found {Fore.GREEN}{Style.BRIGHT}{str(foundN)}{Style.RESET_ALL} in {Fore.MAGENTA}{Style.BRIGHT}{col}{Style.RESET_ALL}"
            )

        print(
            f"\nTotal of {Fore.GREEN}{Style.BRIGHT}{str(foundTotal)}{Style.RESET_ALL} structures available.\n"
        )

        # Substitution elements combination.
        sEL = sorted_elements_list(composition)
        sELtarget = [item for sublist in sEL for item in sublist]

        # Verify need for permutations
        permN = 1
        for multiples in sEL:
            permN = permN * math.factorial(multiples.__len__())

        if permN > 1:
            print(
                "Target composition has degenerate anonymous formula thus requires permutations."
            )
            print(
                f"Up to {Fore.BLUE}{Style.BRIGHT}{str(permN)}{Style.RESET_ALL} per structure.\n"
            )
        else:
            print("Target composition indicates discrete element substitution.\n")

        out_dir = f"{formula_str}-{source_str}-prototypes"
        print(
            f"Final Prototype Directory: {Fore.LIGHTBLACK_EX}{Style.BRIGHT}{out_dir}{Style.RESET_ALL}"
        )

        try:
            os.mkdir(out_dir)
            print(f"{Fore.GREEN}Final directory created.{Style.RESET_ALL}")
        except FileExistsError:
            print(
                f"{Fore.RED}Final directory already exist: Prototypes will be overwritten/updated.{Style.RESET_ALL}\n"
            )

        # Substitute over all prototypes over all of their permutations
        allS = 0
        for col in collections:
            print(
                f"Looking through: {Fore.MAGENTA}{Style.BRIGHT}{col}{Style.RESET_ALL}"
            )

            pipeline = [
                {"$match": {"material.anonymizedFormula": aFormula}},
                {
                    "$group": {
                        "_id": "$" + filter_prototypes,
                        "document": {"$first": "$$ROOT"},
                    }
                },
                {"$sort": {"document.properties.EF_SIPFENN_NN30": 1}},
                {
                    "$facet": {
                        "data": [{"$limit": limit}],
                        "meta": [{"$count": "total"}],
                    }
                },
            ]

            _msgs = (
                "rɛplacing elɛments in thɛ prototypes",
                "replacing ɛlemɛnts in the prototypɛs",
                "rɛplacing ɛlɛments in the prototypes",
                "replacing elemɛnts in thɛ prototypɛs",
            )
            with yaspin(text=_msgs[0]) as sp:
                _stop_evt = threading.Event()

                def _cycle():
                    i = 0
                    while not _stop_evt.is_set():
                        sp.text = _msgs[i % len(_msgs)]
                        i += 1
                        time.sleep(0.6)

                _t = threading.Thread(target=_cycle, daemon=True)
                _t.start()
                try:
                    facet_out = list(
                        MPDD[col].aggregate(
                            pipeline,
                            hint={"material.anonymizedFormula": 1},
                            allowDiskUse=True,
                        )
                    )
                finally:
                    _stop_evt.set()
                    _t.join(timeout=2)

            facet = facet_out[0] if facet_out else {"data": [], "meta": []}
            aggregated_data = facet.get("data", [])
            total_groups = (
                facet.get("meta", [{}])[0].get("total", 0) if facet.get("meta") else 0
            )

            print(
                f"Total number of filtered structures: {Fore.GREEN}{Style.BRIGHT}{total_groups}{Style.RESET_ALL}"
                f"{f' (limited to {Fore.RED}{Style.BRIGHT}{limit}{Style.RESET_ALL})' if limit < total_groups else ''}"
            )

            candidates = []
            for entry in aggregated_data:
                prototype = entry["document"]
                s = Structure.from_dict(json.loads(prototype["material"]["mgJSON"]))
                permuts = element_permutations(
                    sorted_elements_list(Composition(s.formula))
                )
                spGroupA = SpacegroupAnalyzer(s)

                nameRoot = (
                    prototype["material"]["reducedFormula"]
                    + "-"
                    + str(spGroupA.get_crystal_system())
                )

                if "parentDatabaseID" in prototype["metadata"]:
                    nameRoot = (
                        nameRoot
                        + "-"
                        + prototype["metadata"]["parentDatabaseID"]
                        .replace("/", "-")
                        .replace(":", "-")
                        .replace("-", "")
                    )

                candidates.append((nameRoot, s, permuts))

            allS += _generate_from_structures(target_formula, candidates, out_dir)

        print(
            f"Total number of unique prototypes: {Fore.GREEN}{Style.BRIGHT}{len(aggregated_data)}{Style.RESET_ALL} * {Fore.BLUE}{Style.BRIGHT}{permN}{Style.RESET_ALL} permutations each\n"
            f"\nGenerated {Fore.GREEN}{Style.BRIGHT}{str(allS)}{Style.RESET_ALL} {target_formula} prototypes."
        )

        return out_dir


def run_MPDD_devs(
    target_formula: Union[str, Composition],
    devs: Dict[str, int],
    MPDD: Database,
    collections: List[str],
    target_prototypes: Literal["native", "non-native"],
    filter_prototypes: Literal["material.structID", "material.volNormStructID"],
    limit: int,
) -> str:
    """
    Run prototype identification and elemental substitution for permutations within the anonymous composition fractions.

    Args:
        target_composition_dict: Dictionary with element symbols as keys and their amounts as values.
        devs: Dictionary with element symbols as keys and their allowed deviations as values.
        MPDD: A pymongo Database object corresponding to the MPDD database.
        collections: List of collection names to search in.
        target_prototypes: Specifies whether to use 'native' or 'non-native' compositions.
        filter_prototypes: Specifies the prototype filter to use.
        limit: The maximum number of prototypes to generate.

    Returns:
        Path to the directory containing the generated prototypes.
    """

    assert len(collections) > 0, "At least one collection must be specified"
    assert isinstance(MPDD, Database), "MPDD must be a pymongo Database object"

    if isinstance(target_formula, str):
        # Formula to pymatgen Composition
        composition = Composition(target_formula)
    elif isinstance(target_formula, Composition):
        composition = target_formula
    else:
        raise ValueError(
            "Target formula must be a string or a pymatgen Composition object"
        )

    # Target formula and standardized directory naming components
    rangesStr = f"({','.join(str(r) for r in devs.values())})"
    formula_str = f"{str(target_formula).replace(' ', '')}±{rangesStr}"
    source_str = f"MPDD({'-'.join(collections)})"

    target_composition_dict = {
        el: round(amt / composition.num_atoms * 1000, 0)
        for el, amt in composition.as_dict().items()
    }

    print("\nSearching for structure prototypes for: " + formula_str)
    print("Using MPDD Database with collections: " + ", ".join(collections))

    if target_prototypes == "native":
        # ************************* Try to find materials already in the MPDD database **********************************

        out_dir = f"{formula_str}-{source_str}-prototypes(native)"

        try:
            os.mkdir(out_dir)
            print("Directory created for MPDD prototypes: " + out_dir)
        except FileExistsError:
            print(
                "Directory for MPDD prototypes already exist: Prototypes will be overwritten/updated.\n"
            )

        # Query for matching MPDD entries
        nativeQuery = comp2comp_q(target_composition_dict, devs)

        targetPresent = 0
        for col in collections:
            print("Looking through: " + col)

            pipeline = [
                {"$match": nativeQuery},
                {
                    "$group": {
                        "_id": "$" + filter_prototypes,
                        "document": {"$first": "$$ROOT"},
                    }
                },
                {"$sort": {"document.properties.EF_SIPFENN_NN30": 1}},
                {
                    "$facet": {
                        "data": [{"$limit": limit}],
                        "meta": [{"$count": "total"}],
                    }
                },
            ]

            _msgs = (
                "rɛplacing elɛments in thɛ prototypes",
                "replacing ɛlemɛnts in the prototypɛs",
                "rɛplacing ɛlɛments in the prototypes",
                "replacing elemɛnts in thɛ prototypɛs",
            )
            with yaspin(text=_msgs[0]) as sp:
                _stop_evt = threading.Event()

                def _cycle():
                    i = 0
                    while not _stop_evt.is_set():
                        sp.text = _msgs[i % len(_msgs)]
                        i += 1
                        time.sleep(0.6)

                _t = threading.Thread(target=_cycle, daemon=True)
                _t.start()
                try:
                    facet_out = list(MPDD[col].aggregate(pipeline, allowDiskUse=True))
                finally:
                    _stop_evt.set()
                    _t.join(timeout=2)

            facet = facet_out[0] if facet_out else {"data": [], "meta": []}
            aggregated_data = facet.get("data", [])
            total_groups = (
                facet.get("meta", [{}])[0].get("total", 0) if facet.get("meta") else 0
            )
            print(
                f"Total number of filtered structures: {Fore.GREEN}{Style.BRIGHT}{total_groups}{Style.RESET_ALL}"
                f"{f' (limited to {Fore.RED}{Style.BRIGHT}{limit}{Style.RESET_ALL})' if limit < total_groups else ''}"
            )

            for entry in aggregated_data:
                e = entry["document"]

                s = Structure.from_dict(json.loads(e["material"]["mgJSON"]))
                spGroupA = SpacegroupAnalyzer(s)

                name = (
                    e["material"]["formula"].replace(" ", "")
                    + "-"
                    + str(spGroupA.get_crystal_system())
                )

                if "parentDatabaseID" in e["metadata"]:
                    name = name + "-" + e["metadata"]["parentDatabaseID"]

                else:
                    name = (
                        name
                        + "-from-"
                        + e["metadata"]["name"]
                        + "-at-"
                        + e["metadata"]["affiliation"]
                    )

                p = Poscar(s)
                p.write_file(
                    out_dir + "/" + re.sub(r"[?:*|><\\/]", "-", name) + ".vasp"
                )
                targetPresent += 1

        print("Done.")
        print(f"{targetPresent} target materials found in MPDD.\n")

        return out_dir

    elif target_prototypes == "non-native":
        # ******** Try to find materials from the MPDD database that can be adjusted to the target formula  **************

        # Sorted fractions dictionary query
        sortedFractionsDictQuery = comp2acomp_q(target_composition_dict, devs)

        # ************************************** Count prototypes ********************************************************

        print("\nDetecting Matching Input Structures:")
        foundTotal = 0
        for col in collections:
            foundN = MPDD[col].count_documents(sortedFractionsDictQuery)
            foundTotal += foundN
            print("Found " + str(foundN) + " in " + col)

        print("\nTotal of " + str(foundTotal) + " structures available.\n")

        # ********************************* Substitution elements combination ********************************************

        out_dir = f"{formula_str}-{source_str}-prototypes"
        print("Final Prototype Directory: " + out_dir)

        try:
            os.mkdir(out_dir)
            print("Final directory created.")
        except FileExistsError:
            print(
                "Final directory already exist.\nPrototypes will be overwritten/updated.\n"
            )

        # All generated prototypes
        allS = 0
        # List of permutation numbers
        permutNList = []

        # Search all collections using the compound index for the first 8 components
        for col in collections:
            print("Looking through: " + col)

            pipeline = [
                {"$match": sortedFractionsDictQuery},
                {
                    "$group": {
                        "_id": "$" + filter_prototypes,
                        "document": {"$first": "$$ROOT"},
                    }
                },
                {"$sort": {"document.properties.EF_SIPFENN_NN30": 1}},
                {
                    "$facet": {
                        "data": [{"$limit": limit}],
                        "meta": [{"$count": "total"}],
                    }
                },
            ]

            _msgs = (
                "rɛplacing elɛments in thɛ prototypes",
                "replacing ɛlemɛnts in the prototypɛs",
                "rɛplacing ɛlɛments in the prototypes",
                "replacing elemɛnts in thɛ prototypɛs",
            )
            with yaspin(text=_msgs[0]) as sp:
                _stop_evt = threading.Event()

                def _cycle():
                    i = 0
                    while not _stop_evt.is_set():
                        sp.text = _msgs[i % len(_msgs)]
                        i += 1
                        time.sleep(0.6)

                _t = threading.Thread(target=_cycle, daemon=True)
                _t.start()
                try:
                    facet_out = list(
                        MPDD[col].aggregate(
                            pipeline,
                            hint={
                                "material.sortedFractionsDict.c1": 1,
                                "material.sortedFractionsDict.c2": 1,
                                "material.sortedFractionsDict.c3": 1,
                                "material.sortedFractionsDict.c4": 1,
                                "material.sortedFractionsDict.c5": 1,
                                "material.sortedFractionsDict.c6": 1,
                                "material.sortedFractionsDict.c7": 1,
                                "material.sortedFractionsDict.c8": 1,
                            },
                            allowDiskUse=True,
                        )
                    )
                finally:
                    _stop_evt.set()
                    _t.join(timeout=2)

            facet = facet_out[0] if facet_out else {"data": [], "meta": []}
            aggregated_data = facet.get("data", [])
            total_groups = (
                facet.get("meta", [{}])[0].get("total", 0) if facet.get("meta") else 0
            )
            print(
                f"Total number of filtered structures: {Fore.GREEN}{Style.BRIGHT}{total_groups}{Style.RESET_ALL}"
                f"{f' (limited to {Fore.RED}{Style.BRIGHT}{limit}{Style.RESET_ALL})' if limit < total_groups else ''}"
            )

            candidates = []
            for entry in aggregated_data:
                prototype = entry["document"]

                s = Structure.from_dict(json.loads(prototype["material"]["mgJSON"]))
                c = Composition(s.formula)

                cDict = dict(c.fractional_composition.as_dict())
                for e in cDict:
                    cDict[e] = int(cDict[e] * 1000)

                elements = [element for element in cDict]

                oAS = ordered_allowed_substitutions(
                    target_composition_dict, devs, cDict
                )

                permutNList += [len(oAS)]

                spGroupA = SpacegroupAnalyzer(s)

                nameRoot = (
                    prototype["material"]["reducedFormula"]
                    + "-"
                    + str(spGroupA.get_crystal_system())
                )

                if "parentDatabaseID" in prototype["metadata"]:
                    nameRoot = (
                        nameRoot
                        + "-"
                        + prototype["metadata"]["parentDatabaseID"]
                        .replace("/", "-")
                        .replace(":", "-")
                        .replace("-", "")
                    )

                substitutions = []
                for sub in oAS:
                    substitutions.append({el: rep for el, rep in zip(elements, sub)})

                candidates.append((nameRoot, s, substitutions))

            allS += _generate_from_structures(target_formula, candidates, out_dir)

        if allS > 0:
            print(
                f"\nGenerated {Fore.GREEN}{Style.BRIGHT}{str(allS)}{Style.RESET_ALL} {formula_str} prototypes."
            )
            print(f"Mean number of permutations: {mean(permutNList)}")
            print(
                f"Min / Max number of permutations: {min(permutNList)} / {max(permutNList)}"
            )
        else:
            print("No prototypes were generated.")

        return out_dir


def sorted_elements_list(comp: Composition) -> List[List[str]]:
    """
    Return elements grouped by their occurrence in the formula, starting from the least common.

    Args:
        comp: A pymatgen Composition object.

    Returns:
        List of lists of element symbols grouped by their occurrence in the formula.
    """

    fractions = re.sub(r"[A-z]+", "", comp.formula).split()
    elements = re.sub(r"[0-9]+", "", comp.formula).split()
    compl = []
    for i in range(elements.__len__()):
        compl.append([elements[i], int(fractions[i])])
    compl.sort(key=itemgetter(1))
    sortedEL = [[x for x, _ in g] for _, g in groupby(compl, key=itemgetter(1))]
    return sortedEL


def element_permutations(sEL: List[List[str]]) -> List[List[str]]:
    """
    Return all permutations of elements within the groups that conserve the anonymous formula of the compound.

    Args:
        sEL: List of lists of element symbols grouped by their occurrence in the formula.

    Returns:
        List of lists of element symbols representing all permutations.
    """

    generatingList = [[list(i) for i in permutations(lev)] for lev in sEL]
    outList = [
        [item for sublist in list(permutation) for item in sublist]
        for permutation in product(*generatingList)
    ]
    return outList


def comp2acomp_q(
    comp_dict: Dict[str, int], devs: Dict[str, int]
) -> Dict[str, Union[Dict[str, int], int]]:
    """
    Return a MPDD query for anonymous composition space with N components.

    Args:
        comp_dict: Dictionary with element symbols as keys and their amounts as values.
        devs: Dictionary with element symbols as keys and their allowed deviations as values.

    Returns:
        Dictionary representing the MPDD query.
    """

    eVals = []
    for e in comp_dict:
        eVals.append([comp_dict[e], devs[e]])
    eVals.sort(key=lambda x: x[0], reverse=True)
    result = dict()
    for eVal, c in zip(eVals, range(eVals.__len__())):
        result.update(
            {
                f"material.sortedFractionsDict.c{c + 1}": {
                    "$gte": (eVal[0] - eVal[1]),
                    "$lte": (eVal[0] + eVal[1]),
                }
            }
        )
    result.update({f"material.nComponents": len(eVals)})
    return result


def comp2comp_q(
    comp_dict: Dict[str, int], devs: Dict[str, int]
) -> Dict[str, Union[Dict[str, int], int]]:
    """
    Return a MPDD query for the given composition dictionary with permil fractions.

    Args:
        comp_dict: Dictionary with element symbols as keys and their amounts as values.
        devs: Dictionary with element symbols as keys and their allowed deviations as values.

    Returns:
        Dictionary representing the MPDD query.
    """

    meanComposition = Composition(
        " ".join("".join((key, str(val))) for (key, val) in comp_dict.items())
    )
    mpddQuery = dict()
    mpddQuery.update({"material.system": meanComposition.chemical_system})
    for e in comp_dict:
        key = "material.compositionDictionary." + e
        mpddQuery.update(
            {
                key: {
                    "$gte": (comp_dict[e] - devs[e]) / 1000,
                    "$lte": (comp_dict[e] - devs[e]) / 1000,
                }
            }
        )
    return mpddQuery


def ordered_allowed_substitutions(
    target_comp_dict: Dict[str, int],
    devs: Dict[str, int],
    prototype_comp_dict: Dict[str, int],
) -> List[List[str]]:
    """
    Return elements grouped by their occurrence in the formula, starting from the least common.

    Args:
        target_comp_dict: Dictionary with element symbols as keys and their amounts as values.
        devs: Dictionary with element symbols as keys and their allowed deviations as values.
        prototype_comp_dict: Dictionary with element symbols as keys and their amounts as values for the prototype.

    Returns:
        List of lists of element symbols representing allowed substitutions.
    """

    # Finds what elements can be used for substitution
    allInRangeSubstitutions = []
    for c in prototype_comp_dict:
        inRangeSubstitutions = []
        val = prototype_comp_dict[c]
        for e in target_comp_dict:
            if (
                target_comp_dict[e] - devs[e] <= val
                and target_comp_dict[e] + devs[e] >= val
            ):
                inRangeSubstitutions += [str(e)]
        allInRangeSubstitutions.append(inRangeSubstitutions)

    # E.g. [['Al', 'Fe', 'N'], ['Al', 'Fe', 'N'], ['N']]
    allInRangeSubstitutionPermutations = list(product(*allInRangeSubstitutions))
    allAllowedSubstitutions = []
    for irsp in allInRangeSubstitutionPermutations:
        if len(irsp) == len(set(irsp)):
            allAllowedSubstitutions += [irsp]
    # E.g. [('Al', 'Fe', 'N'), ('Fe', 'Al', 'N')]

    return allAllowedSubstitutions


def run_optimade(
    target_formula: Union[str, Composition],
    base_urls: list[str],
    target_prototypes: Literal["native", "non-native"] = "non-native",
    limit: int = 1000,
    parallel: bool = True,
) -> str:
    """
    Run prototype identification and elemental substitution using OPTIMADE databases.
    """
    try:
        from optimade.client import OptimadeClient
        from optimade.models.utils import anonymize_formula
        from optimade.adapters import Structure as OptimadeStructure
        from optimade.adapters.structures.pymatgen import get_pymatgen_structure
    except ImportError as e:
        raise ImportError(
            "run_optimade requires the 'optimade[http_client]' package. "
            "Install it with: pip install 'optimade[http_client]'"
        ) from e

    if isinstance(target_formula, str):
        composition = Composition(target_formula)
    else:
        composition = target_formula

    formula_str = str(target_formula).replace(" ", "")
    nelements = len(composition.as_dict())

    if target_prototypes == "native":
        filter_str = f'chemical_formula_reduced = "{composition.reduced_formula}" AND nelements = {nelements}'
        out_dir = f"{formula_str}-optimade-prototypes(native)"
    else:
        anon = anonymize_formula(composition.formula)
        filter_str = (
            f'chemical_formula_anonymous = "{anon}" AND nelements = {nelements}'
        )
        out_dir = f"{formula_str}-optimade-prototypes"

    print(
        f"\nSearching for structure prototypes for: {Fore.YELLOW}{Style.BRIGHT}{target_formula}{Style.RESET_ALL}"
    )
    print(
        f"Using OPTIMADE providers: {Fore.MAGENTA}{Style.BRIGHT}{', '.join(base_urls)}{Style.RESET_ALL}"
    )
    print(
        f"Final Prototype Directory: {Fore.LIGHTBLACK_EX}{Style.BRIGHT}{out_dir}{Style.RESET_ALL}"
    )

    try:
        os.mkdir(out_dir)
        print(f"{Fore.GREEN}Final directory created.{Style.RESET_ALL}")
    except FileExistsError:
        print(
            f"{Fore.RED}Final directory already exist: Prototypes will be overwritten/updated.{Style.RESET_ALL}\n"
        )

    _msgs = (
        "fetching structures from OPTIMADE providers",
        "fɛtching structurɛs from OPTIMADE providɛrs",
        "fɛtching structurɛs from OPTIMADE providɛrs",
        "fetching structurɛs from OPTIMADE providɛrs",
    )
    with yaspin(text=_msgs[0]) as sp:
        _stop_evt = threading.Event()

        def _cycle():
            i = 0
            while not _stop_evt.is_set():
                sp.text = _msgs[i % len(_msgs)]
                i += 1
                time.sleep(0.6)

        _t = threading.Thread(target=_cycle, daemon=True)
        _t.start()
        try:
            client = OptimadeClient(
                base_urls=base_urls,
                max_results_per_provider=limit,
                silent=True,
            )
            client.structures.get(filter_str)
        finally:
            _stop_evt.set()
            _t.join(timeout=2)

    all_entries = []
    # all_results.get('structures', {}) structure depends on optimade version.
    # OptimadeClient.all_results is a dict like {'structures': {'filter_str': {base_url: {'data': [...]}}}}
    structures_results = client.all_results.get("structures", {}).get(filter_str, {})
    for base_url, query_results in structures_results.items():
        # query_results may be a QueryResults dataclass (optimade >= 1.x) or a plain dict
        if hasattr(query_results, "data"):
            entries = query_results.data if isinstance(query_results.data, list) else []
        else:
            entries = query_results.get("data", [])
        # Add provider domain info for naming
        domain = base_url.split("//")[-1].split("/")[0]
        for e in entries:
            e["_provider_domain"] = domain
        all_entries.extend(entries)

    print(
        f"Total number of filtered structures: {Fore.GREEN}{Style.BRIGHT}{len(all_entries)}{Style.RESET_ALL}"
    )

    if target_prototypes == "native":
        allS = 0
        for entry in all_entries:
            if allS >= limit:
                break
            try:
                opt_struct = OptimadeStructure(entry)
                s = get_pymatgen_structure(opt_struct)
            except Exception as exc:
                print(
                    f"[crystALL] Warning: skipping entry {entry.get('id', '?')} — {exc}"
                )
                continue

            # Post-fetch composition verification
            if s.composition.reduced_composition != composition.reduced_composition:
                continue

            provider_domain = entry.get("_provider_domain", "unknown")
            entry_id_sanitized = re.sub(r"[?:*|><\\/]", "-", entry.get("id", "unknown"))
            name = f"optimade-{provider_domain}-{entry_id_sanitized}.vasp"

            p = Poscar(s)
            p.write_file(os.path.join(out_dir, name))
            allS += 1

        print("Done.\n")
        return out_dir

    else:
        # Non-native branch
        candidates = []
        for entry in all_entries:
            try:
                opt_struct = OptimadeStructure(entry)
                s = get_pymatgen_structure(opt_struct)
            except Exception as exc:
                print(
                    f"[crystALL] Warning: skipping entry {entry.get('id', '?')} — {exc}"
                )
                continue

            spGroupA = SpacegroupAnalyzer(s)
            crystal_system = spGroupA.get_crystal_system()
            entry_id_sanitized = re.sub(r"[?:*|><\\/]", "-", entry.get("id", "unknown"))
            name_root = f"optimade-{crystal_system}-{entry_id_sanitized}"

            permuts = element_permutations(sorted_elements_list(Composition(s.formula)))
            candidates.append((name_root, s, permuts))

        allS = _generate_from_structures(
            target_formula, candidates, out_dir, parallel=parallel
        )
        print(
            f"\nGenerated {Fore.GREEN}{Style.BRIGHT}{str(allS)}{Style.RESET_ALL} {target_formula} prototypes."
        )
        return out_dir
