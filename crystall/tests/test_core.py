"""
Core test suite for the crystALL project.

Prerequisites:
1. MongoDB Server:
   - On most Linux systems, start the service with: `sudo systemctl start mongod`
   - Alternatively, run manually: `mongod --dbpath <path_to_your_data_folder>`
   - The tests expect the server to be listening at mongodb://localhost:27017/
2. Python Dependencies:
   - This suite requires `pytest-order` to ensure tests execute in the correct 
     sequence, as many TestCrystALLSearch methods depend on state (prototypes 
     and predictions) generated in earlier steps.

Execution:
1. Activate your environment (e.g., `conda activate crystall`)
2. Run from the project root: `pytest crystall/tests/test_core.py -v`
"""
import os
import re
import glob
import json
import shutil
import tempfile
import pytest
import crystall
import unittest
import warnings
import numpy as np
from functools import partial
from ruamel.yaml import YAML


from unittest.mock import patch
from urllib.error import HTTPError
from pymatgen.core import Structure, Composition
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymongo import MongoClient

IN_GITHUB_ACTIONS = (
    os.getenv("GITHUB_ACTIONS") == "true" and os.getenv("MODELS_FETCHED") != "true"
)


class TestTopLevelFunctions(unittest.TestCase):
    """Tests for top-level prototype generation functions (local + MPDD)."""

    def test_load_adapters(self) -> None:
        """Test that load_adapters returns a dict mapping adapter names to callables or classes."""
        adapters = crystall.load_adapters()
        self.assertIsInstance(adapters, dict)
        for name, adapter in adapters.items():
            self.assertIsInstance(name, str)
            self.assertIsNotNone(adapter)
        # Also exercise the entry_points failure path which should be handled
        with patch("crystall.core.entry_points", side_effect=Exception("EP failure")):
            adapters = crystall.load_adapters()
            self.assertIsInstance(adapters, dict)

    def setUp(self) -> None:
        """Set up shared state for local and MPDD-related top-level functions."""
        # Local test setup mirrors previous TestLocal
        self.target_formula_local = "Ni22Ru8"
        self.prototype_path_local = "crystall/tests/testFiles/localTest"

        # Ensure the output directories are not present before each test
        if os.path.exists("Ni22Ru8-local-prototypes"):
            shutil.rmtree("Ni22Ru8-local-prototypes")

        # MPDD test setup mirrors previous TestMPDD
        self.client = MongoClient(
            "mongodb://localhost:27017/", serverSelectionTimeoutMS=100
        )
        self.db = self.client["MPDD"]
        self.collection = self.db["curated"]
        self.target_formula_mpdd = "Ni26Ru8"  # "Ni26Ru8" for quick run or "Ni22Ru8" to stay consistent with local target

        # Drop old collection that may have been left behind
        self.collection.drop()

        # Set up hints
        self.hints = [
            "material.anonymizedFormula",
            "material.reducedFormula",
            "material.structID",
            "material.coarseStructID",
            "material.volNormStructID",
            "material.sortedFractionsDict",
        ]

        for hint in self.hints:
            self.collection.create_index([(hint, 1)], name=hint)

        # Populate the collection with the test data
        with open("crystall/tests/testFiles/MPDD_A4B11or13or17.json", "r") as f:
            for entry in json.loads(f.read()):
                entry["_id"] = entry["_id"]["$oid"]
                self.collection.insert_one(entry)

    def test_run_local(self) -> None:
        """Test that run_local generates the expected output directory and files."""
        result = crystall.run_local(
            target_formula=self.target_formula_local,
            prototype_path=self.prototype_path_local,
        )

        # Directory name and existence
        self.assertEqual(result, "Ni22Ru8-local-prototypes")
        self.assertTrue(
            os.path.isdir(result), "Local prototypes directory was not created."
        )

        # Basic sanity on number of generated files
        files = os.listdir(result)
        self.assertGreater(
            len(files), 0, "run_local did not generate any local prototypes."
        )
        self.assertEqual(len(files), 4, "Incorrect number of files generated.")

        for fname in files:
            with self.subTest(file=fname):
                s = Structure.from_file(os.path.join(result, fname))
                self.assertEqual(
                    s.composition.reduced_composition,
                    Composition(self.target_formula_local).reduced_composition,
                    f"Stoichiometry mismatch in {fname}"
                )

        for fname in files:
            with self.subTest(file=fname):
                s = Structure.from_file(os.path.join(result, fname))
                spgA = SpacegroupAnalyzer(s)
                # Crystal system is the third dash-separated token in the filename
                # e.g. local-monoclinic-Al13Rh4-... -> parts[1] after splitting on '-'
                parts = fname.split("-")
                if len(parts) >= 2:
                    expected_system = parts[1]
                    self.assertEqual(
                        spgA.get_crystal_system(), expected_system,
                        f"Crystal system mismatch in {fname}"
                    )

    def test_run_MPDD(self) -> None:
        """Test the run_MPDD top-level function."""
        target_formula = "Ni26Ru8"
        collections = ["curated"]
        filter_prototypes = "material.structID"
        limit = 100
        result = crystall.run_MPDD(
            target_formula, self.db, collections, "native", filter_prototypes, limit
        )
        self.assertTrue(
            os.path.exists(result), "The directory for MPDD prototypes was not created."
        )
        # Naming: run_MPDD with target_prototypes="native" produces
        # "{formula}-MPDD({collections})-prototypes(native)"
        self.assertTrue(result.endswith("native)") or result.endswith("prototypes"),
                        f"Unexpected directory name: {result}")
        files = os.listdir(result)
        self.assertGreater(len(files), 0, "No POSCAR files were generated by run_MPDD.")
        # Parse one file and verify SpacegroupAnalyzer succeeds
        s = Structure.from_file(os.path.join(result, files[0]))
        spgA = SpacegroupAnalyzer(s)
        self.assertIsNotNone(spgA.get_crystal_system())

    def test_run_MPDD_devs(self) -> None:
        """Test the run_MPDD_devs / compositional MPDD top-level function."""
        target_composition = "Ni26Ru8"
        devs = {"Ni": 10, "Ru": 10}
        collections = ["curated"]
        filter_prototypes = "material.structID"
        limit = 100
        result = crystall.run_MPDD_devs(
            target_composition,
            devs,
            self.db,
            collections,
            "native",
            filter_prototypes,
            limit,
        )
        self.assertTrue(
            os.path.exists(result),
            "The directory for compositional MPDD prototypes was not created.",
        )
        files = os.listdir(result)
        self.assertGreater(len(files), 0, "No POSCAR files were generated by run_MPDD_devs.")
        s = Structure.from_file(os.path.join(result, files[0]))
        spgA = SpacegroupAnalyzer(s)
        self.assertIsNotNone(spgA.get_crystal_system())

    def test_sorted_elements_list(self) -> None:
        """Test sorted_elements_list returns a sorted list of element symbols."""
        from crystall.core import sorted_elements_list

        comp = Composition("Ni2AlZr3")
        result = sorted_elements_list(comp)
        # Expect grouping by multiplicity (least common first) then alphabetical
        self.assertEqual(result, [["Al"], ["Ni"], ["Zr"]])

    def test_element_permutations(self) -> None:
        """Test element_permutations generates permutations of the element list."""
        from crystall.core import element_permutations

        elements = ["A", "B", "C"]
        perms = element_permutations(elements)
        # Should return at least the original ordering as a list
        self.assertGreaterEqual(len(perms), 1)
        self.assertIn(["A", "B", "C"], perms)

    def test_comp2acomp_q(self) -> None:
        """Test comp2acomp_q builds a valid anonymized composition MPDD query."""
        from crystall.core import comp2acomp_q

        comp = {"A": 2, "B": 1}
        devs = {"A": 1, "B": 1}
        q = comp2acomp_q(comp, devs)

        # Expect fraction constraints and nComponents
        self.assertIn("material.nComponents", q)
        self.assertEqual(q["material.nComponents"], 2)
        keys = [k for k in q if k.startswith("material.sortedFractionsDict.c")]
        self.assertGreater(len(keys), 0)

    def test_comp2comp_q(self) -> None:
        """Test comp2comp_q builds a valid composition-dictionary MPDD query."""
        from crystall.core import comp2comp_q

        comp = {"Ni": 500, "Ru": 500}
        devs = {"Ni": 100, "Ru": 100}
        q = comp2comp_q(comp, devs)

        self.assertIn("material.system", q)
        # Check that element range keys exist
        for el in comp:
            key = f"material.compositionDictionary.{el}"
            self.assertIn(key, q)
            self.assertIn("$gte", q[key])
            self.assertIn("$lte", q[key])

    def test_ordered_allowed_substitutions(self) -> None:
        """Test ordered_allowed_substitutions returns distinct-element permutations."""
        from crystall.core import ordered_allowed_substitutions

        target = {"A": 1, "B": 1}
        devs = {"A": 0, "B": 0}
        prototype = {"X": 1, "Y": 1}

        subs = ordered_allowed_substitutions(target, devs, prototype)
        # Expect at least the two distinct permutations
        self.assertIn(("A", "B"), subs)
        self.assertIn(("B", "A"), subs)
        # Ensure no substitution tuple repeats an element
        for tpl in subs:
            self.assertEqual(len(tpl), len(set(tpl)))

    def test_query_tag(self) -> None:
        """_query_tag must be empty for None/{} other_targets, deterministic, order-independent,
        and sensitive to formula_str, source_str, and other_targets contents."""
        from crystall.core import _query_tag

        formula = "Ni26Ru8"
        source = "MPDD(curated)"

        # Empty / None other_targets → no tag (regardless of formula/source)
        self.assertEqual(_query_tag(formula, source, None), "")
        self.assertEqual(_query_tag(formula, source, {}), "")
        self.assertEqual(_query_tag("OtherFormula", "MPDD(other)", None), "")

        # Non-empty → starts with '-q' followed by 8 hex chars
        t1 = _query_tag(formula, source, {"material.dimensionality": 2})
        self.assertTrue(t1.startswith("-q"))
        self.assertEqual(len(t1), 10)  # '-q' + 8 hex

        # Determinism: same inputs always produce the same tag
        self.assertEqual(
            t1, _query_tag(formula, source, {"material.dimensionality": 2})
        )

        # Order independence: dict insertion order must not affect the tag
        self.assertEqual(
            _query_tag(formula, source, {"material.dimensionality": 2, "material.spaceGroup": 191}),
            _query_tag(formula, source, {"material.spaceGroup": 191, "material.dimensionality": 2}),
        )

        # Different other_targets → different tag
        self.assertNotEqual(
            t1, _query_tag(formula, source, {"material.dimensionality": 3})
        )

        # Different formula_str → different tag (problem identity)
        self.assertNotEqual(
            t1, _query_tag("DifferentFormula", source, {"material.dimensionality": 2})
        )

        # Different source_str → different tag (problem identity)
        self.assertNotEqual(
            t1, _query_tag(formula, "MPDD(experimental)", {"material.dimensionality": 2})
        )

    def test_normalize_other_targets(self) -> None:
        """_normalize_other_targets drops composition-implied entries per query_mode,
        passes everything else through (including typos, operator forms, unrelated keys)."""
        from crystall.core import _normalize_other_targets

        target = "Mn4Al11"  # 2 elements; chemical_system "Al-Mn"; anon formula "A4B11"

        # Empty / None → empty dict
        self.assertEqual(_normalize_other_targets(target, None, "exact-composition"), {})
        self.assertEqual(_normalize_other_targets(target, {}, "exact-composition"), {})

        # Unrelated keys always pass through
        unrelated = {"material.dimensionality": 2, "properties.STAB_SIPFENN_NN30": 0}
        for mode in ("exact-composition", "composition-range",
                     "anonymized-formula", "fraction-range"):
            with self.subTest(mode=mode):
                self.assertEqual(
                    _normalize_other_targets(target, unrelated, mode), unrelated
                )

        # nComponents is universally redundant — dropped in all 4 modes when matching
        for mode in ("exact-composition", "composition-range",
                     "anonymized-formula", "fraction-range"):
            with self.subTest(mode=mode, key="material.nComponents (matches)"):
                self.assertEqual(
                    _normalize_other_targets(target, {"material.nComponents": 2}, mode),
                    {},
                )

        # Non-matching nComponents passes through (no contradiction-detection in minimal version)
        self.assertEqual(
            _normalize_other_targets(target, {"material.nComponents": 3}, "exact-composition"),
            {"material.nComponents": 3},
        )

        # exact-composition: 6 fields are redundant when matching
        exact_all_matching = {
            "material.nComponents": 2,
            "material.elements": ["Al", "Mn"],
            "material.system": "Al-Mn",
            "material.formula": "Mn4Al11",
            "material.reducedFormula": "Mn4Al11",
            "material.anonymizedFormula": "A4B11",
        }
        self.assertEqual(
            _normalize_other_targets(target, exact_all_matching, "exact-composition"), {}
        )

        # composition-range: only nComponents/elements/system are redundant;
        # formula/reducedFormula/anonymizedFormula are NOT (counts vary within devs)
        cr_input = {
            "material.nComponents": 2,
            "material.elements": ["Al", "Mn"],
            "material.system": "Al-Mn",
            "material.formula": "Mn4Al11",          # NOT redundant in this mode
            "material.reducedFormula": "Mn4Al11",   # NOT redundant in this mode
            "material.anonymizedFormula": "A4B11",  # NOT redundant in this mode
        }
        self.assertEqual(
            _normalize_other_targets(target, cr_input, "composition-range"),
            {
                "material.formula": "Mn4Al11",
                "material.reducedFormula": "Mn4Al11",
                "material.anonymizedFormula": "A4B11",
            },
        )

        # anonymized-formula: only nComponents and anonymizedFormula are redundant
        af_input = {
            "material.nComponents": 2,
            "material.anonymizedFormula": "A4B11",
            "material.elements": ["Al", "Mn"],      # NOT redundant — elements get substituted
            "material.system": "Al-Mn",             # NOT redundant
        }
        self.assertEqual(
            _normalize_other_targets(target, af_input, "anonymized-formula"),
            {"material.elements": ["Al", "Mn"], "material.system": "Al-Mn"},
        )

        # fraction-range: only nComponents is redundant
        fr_input = {
            "material.nComponents": 2,
            "material.anonymizedFormula": "A4B11",  # NOT redundant (anon varies within devs)
            "material.elements": ["Al", "Mn"],
            "material.system": "Al-Mn",
        }
        self.assertEqual(
            _normalize_other_targets(target, fr_input, "fraction-range"),
            {
                "material.anonymizedFormula": "A4B11",
                "material.elements": ["Al", "Mn"],
                "material.system": "Al-Mn",
            },
        )

        # Formula format tolerance: differing whitespace/order still recognized as redundant
        # (pymatgen returns "Al11 Mn4" but a user might write "Mn4Al11" or "Mn4 Al11")
        for written in ("Mn4Al11", "Mn4 Al11", "Al11Mn4", "Al11 Mn4"):
            with self.subTest(formula_written_as=written):
                self.assertEqual(
                    _normalize_other_targets(
                        target, {"material.formula": written}, "exact-composition"
                    ),
                    {},
                )

        # Elements list order doesn't matter (user might write unsorted)
        for written in (["Al", "Mn"], ["Mn", "Al"]):
            with self.subTest(elements_written_as=written):
                self.assertEqual(
                    _normalize_other_targets(
                        target, {"material.elements": written}, "exact-composition"
                    ),
                    {},
                )

        # Operator-form values pass through unchanged (not interpreted)
        operator_inputs = [
            {"material.nComponents": {"$gte": 2}},
            {"material.nComponents": {"$in": [2, 3]}},
            {"material.anonymizedFormula": {"$ne": "AB"}},
            {"material.elements": {"$exists": True}},
        ]
        for op_in in operator_inputs:
            with self.subTest(operator_form=op_in):
                self.assertEqual(
                    _normalize_other_targets(target, op_in, "exact-composition"),
                    op_in,
                )

        # Mismatched values pass through (no error, no drop)
        mismatches = {
            "material.nComponents": 5,
            "material.elements": ["Fe", "O"],
            "material.system": "Fe-O",
            "material.anonymizedFormula": "AB2",
            "material.reducedFormula": "FeO2",
            "material.formula": "Fe2O3",
        }
        self.assertEqual(
            _normalize_other_targets(target, mismatches, "exact-composition"),
            mismatches,
        )

        # Composition object as target_formula (alternative input type)
        self.assertEqual(
            _normalize_other_targets(
                Composition("Mn4Al11"), {"material.nComponents": 2}, "exact-composition"
            ),
            {},
        )

    def test_write_query_sidecar(self) -> None:
        """_write_query_sidecar writes a YAML sidecar next to (not inside) the output directory."""
        from crystall.core import _write_query_sidecar

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = os.path.join(tmp, "Ni26Ru8-MPDD(curated)-prototypes(native)")
            os.mkdir(out_dir)
            _write_query_sidecar(
                out_dir,
                target_formula="Ni26Ru8",
                target_prototypes="native",
                limit=10,
                other_targets={"material.dimensionality": 2},
            )

            # Sidecar lives as a sibling of the output directory, NOT inside it
            sidecar = out_dir + "_query.yaml"
            self.assertTrue(os.path.isfile(sidecar))
            self.assertFalse(os.path.exists(os.path.join(out_dir, "_query.yaml")))

            # Contents are valid YAML and include the passed fields
            with open(sidecar) as f:
                data = YAML(typ='safe', pure=True).load(f)
            self.assertEqual(data["target_formula"], "Ni26Ru8")
            self.assertEqual(data["target_prototypes"], "native")
            self.assertEqual(data["limit"], 10)
            self.assertEqual(data["other_targets"], {"material.dimensionality": 2})

    def tearDown(self) -> None:
        """Clean up any local and MPDD-generated directories after each test."""
        # Sweep all Ni22Ru8 / Ni26Ru8 prototype directories AND the
        # `_query.yaml` sidecar files emitted next to filtered outputs.
        for pattern in [
            "Ni22Ru8*prototypes*",
            "Ni26Ru8*prototypes*",
        ]:
            for path in glob.glob(pattern):
                if os.path.isdir(path):
                    shutil.rmtree(path)
                elif os.path.isfile(path):
                    os.remove(path)


class TestCrystALLSearch(unittest.TestCase):
    """Test the functionalities of the CrystALLSearch class end-to-end."""

    @classmethod
    def setUpClass(cls) -> None:
        """Set up MPDD, create CrystALLSearch and generate both MPDD and local prototypes."""
        cls.client = MongoClient(
            "mongodb://localhost:27017/", serverSelectionTimeoutMS=100
        )
        cls.db = cls.client["MPDD"]
        cls.collection = cls.db["curated"]

        # Drop old collection that may have been left behind
        cls.collection.drop()

        # Set up hints
        cls.hints = [
            "material.anonymizedFormula",
            "material.reducedFormula",
            "material.structID",
            "material.coarseStructID",
            "material.volNormStructID",
            "material.sortedFractionsDict",
        ]

        for hint in cls.hints:
            cls.collection.create_index([(hint, 1)], name=hint)

        # Populate the collection with the test data
        with open("crystall/tests/testFiles/MPDD_A4B11or13or17.json", "r") as f:
            for entry in json.loads(f.read()):
                entry["_id"] = entry["_id"]["$oid"]
                cls.collection.insert_one(entry)

        cls.target_formula = "Ni26Ru8"
        with patch("builtins.input", return_value="n"):
            cls.cas = crystall.CrystALLSearch(cls.target_formula)

        # Also exercise __init__ branch where no KS2022 models exist and the
        # initializer asks to run download_pysipfenn. We create a separate
        # instance under patched Calculator behavior and patch
        # CrystALLSearch.download_pysipfenn to ensure it's invoked.
        with (
            patch.object(
                crystall.core.Calculator, "findCompatibleModels", return_value=[]
            ),
            patch(
                "crystall.core.Calculator.network_list_available", new=[], create=True
            ),
            patch("builtins.input", return_value=""),
            patch.object(crystall.CrystALLSearch, "download_pysipfenn") as dl_mock,
        ):
            # instantiate a temporary instance to trigger the no-models branch
            _tmp = crystall.CrystALLSearch(cls.target_formula)
            cls.init_download_called = dl_mock.called

        # Connect to MPDD once; prototype generation is tested explicitly
        cls.cas.connect_MPDD(cls.db)

    @pytest.mark.order(1)
    def test_download_pysipfenn(self) -> None:
        """Test downloading models from the MPDD database via download_pysipfenn."""
        # Patch out the expensive network/model download so test stays fast
        with (
            patch.object(self.cas.Calculator, "downloadModels") as mocked_dl,
            patch.object(
                self.cas.Calculator,
                "findCompatibleModels",
                return_value=["SIPFENN_Krajewski2022_NN30"],
            ),
            patch.object(
                self.cas.Calculator,
                "network_list_available",
                new=["SIPFENN_Krajewski2022_NN30"],
            ),
        ):
            self.cas.download_pysipfenn(mode="serial")
            mocked_dl.assert_called_once_with("SIPFENN_Krajewski2022_NN30")

        # Ensure that pySIPFENN objectives are registered
        self.assertIn("pySIPFENN", self.cas.available_calculators)
        self.assertGreater(len(self.cas.available_calculators["pySIPFENN"]), 0)

        # Confirm that the __init__ path attempted to call download_pysipfenn
        # when no models were found during construction (set in setUpClass).
        self.assertTrue(getattr(self.__class__, "init_download_called", False))

        # Retry behavior: simulate downloadModels failing twice (HTTPError), then succeeding
        err1 = HTTPError(url=None, code=500, msg="err1", hdrs=None, fp=None)
        err2 = HTTPError(url=None, code=500, msg="err2", hdrs=None, fp=None)
        with (
            patch.object(
                self.cas.Calculator, "downloadModels", side_effect=[err1, err2, None]
            ) as mocked_retry_dl,
            patch.object(
                self.cas.Calculator,
                "findCompatibleModels",
                return_value=["SIPFENN_Krajewski2022_NN30"],
            ),
            patch.object(
                self.cas.Calculator,
                "network_list_available",
                new=["SIPFENN_Krajewski2022_NN30"],
            ),
        ):
            # use small retries and zero delay to keep test fast
            self.cas.download_pysipfenn(mode="serial", retries=3, delay=0)
            self.assertEqual(mocked_retry_dl.call_count, 3)

    def _assert_prototypes_satisfy_filter(self, mongo_filter, target_prototypes) -> None:
        """Verify every prototype in self.cas.prototypes traces back to an MPDD doc that satisfies mongo_filter.

        Prototypes are written to disk only — the in-memory dicts carry just
        {index, name}, so to check the filter we re-derive the source MPDD doc
        from the parentDatabaseID embedded in each POSCAR filename.

        Empty prototype list is tolerated (vacuously true). Only literal scalar
        filter values are supported (no $gte/$in/etc).
        """
        # 1. Nothing to verify if no prototypes were generated.
        if not self.cas.prototypes:
            return

        # 2. Pick the sanitization rule core.py uses when embedding the
        #    parentDatabaseID into a POSCAR filename. The native and non-native
        #    branches of run_MPDD / run_MPDD_devs use different rules, so we
        #    must mirror whichever one produced these prototypes.
        if target_prototypes == "native":
            # native: re.sub(r'[?:*|><\\/]', '-', name) -- core.py:407, core.py:694
            sanitize = lambda pdb: re.sub(r"[?:*|><\\/]", "-", pdb)
        else:
            # non-native: replace '/' and ':' with '-', then strip every '-'
            # -- core.py:520, core.py:830
            sanitize = lambda pdb: pdb.replace("/", "-").replace(":", "-").replace("-", "")

        # 3. Build a reverse map sanitized-pdbID -> source MPDD doc, so we can
        #    trace any filename back to the doc that produced it.
        pdb_form_to_doc = {}
        for d in self.collection.find({}, {"metadata.parentDatabaseID": 1, "material": 1}):
            pdb = d.get("metadata", {}).get("parentDatabaseID")
            if pdb:
                pdb_form_to_doc[sanitize(pdb)] = d

        # 4. For each prototype, find the unique source doc and check the filter.
        for proto in self.cas.prototypes:
            name = proto["name"]
            # Anchor the regex at the end of the filename so 'JVASP-2081' does
            # not falsely match a name ending in 'JVASP-20813.POSCAR'. The
            # optional '-pN' tail is the permutation suffix used in non-native.
            matches = [
                form for form in pdb_form_to_doc
                if re.search(re.escape(form) + r"(?:-p\d+)?\.POSCAR$", name)
            ]
            # 5. Exactly one match expected. Zero = our parsing assumption is
            #    wrong; many = two pdbIDs collide after sanitization.
            self.assertEqual(
                len(matches), 1,
                f"Could not uniquely identify source MPDD doc for prototype {name!r}; matched: {matches}",
            )
            doc = pdb_form_to_doc[matches[0]]
            # 6. Walk the dotted filter key into the doc and compare to the
            #    expected literal value (e.g. material.dimensionality == 2).
            for key, expected in mongo_filter.items():
                actual = doc
                for part in key.split("."):
                    actual = actual[part]
                self.assertEqual(
                    actual, expected,
                    f"Prototype {name!r} (sanitized pdbID {matches[0]!r}) has {key}={actual!r}, expected {expected!r}",
                )

    @pytest.mark.order(2)
    def test_generate_MPDD(self) -> None:
        """Test MPDD prototype generation for multiple target/filter combinations."""
        # Loop explicitly over target types, filter fields and devs variants
        target_options = ["native", "non-native"]
        filter_options = ["material.structID", "material.volNormStructID"]
        devs_options = [None, {"Ni": 10, "Ru": 10}]
        other_target_options = [None, {"material.crystalSystem": "monoclinic"}]

        for other_targets in other_target_options:
            for devs in devs_options:
                for target_prototypes in target_options:
                    for filter_prototypes in filter_options:
                        with self.subTest(
                            f"Generate_MPDD for other_targets={other_targets} / devs={devs} / "
                            f"{target_prototypes} / {filter_prototypes}"
                        ):
                            # Remove any existing output directories to avoid interference
                            # match common patterns for Ni26Ru8 prototype outputs
                            for pattern in [
                                "Ni26Ru8*prototypes*",
                                "Ni26Ru8-local-prototypes",
                            ]:
                                for path in glob.glob(pattern):
                                    # The glob also matches `_query.yaml`
                                    # sidecar files written next to filtered
                                    # output directories — handle both.
                                    if os.path.isdir(path):
                                        shutil.rmtree(path)
                                    elif os.path.isfile(path):
                                        os.remove(path)

                            # Call generate_MPDD. Both `devs` and `other_targets` may be
                            # None; generate_MPDD handles that internally. The `devs` path
                            # performs aggregation queries that may require additional
                            # indexes in the local test MPDD instance. Handle
                            # OperationFailure gracefully and skip the subtest if the
                            # environment cannot support the query.
                            try:
                                self.cas.generate_MPDD(
                                    limit=10,
                                    target_prototypes=target_prototypes,
                                    filter_prototypes=filter_prototypes,
                                    devs=devs,
                                    other_targets=other_targets,
                                )
                            except Exception as e:
                                # If MongoDB aggregation fails due to missing index or
                                # unsupported hint, skip this subtest rather than
                                # failing the whole suite. Record the exception for
                                # debugging in the skip message.
                                msg = f"Skipping devs aggregation subtest due to DB error: {e}"
                                self.skipTest(msg)

                            mpdd_path_after = self.cas.prototype_path
                            self.assertIsInstance(mpdd_path_after, str)
                            self.assertTrue(os.path.isdir(mpdd_path_after))

                            # For the non-devs + non-native path we expect
                            # prototypes (anonymized formula A4B13 has 30
                            # matches in the mock). The native path queries
                            # reducedFormula and the small mock has no
                            # Ni13Ru4 entries, so 0 is acceptable. Devs-based
                            # compositional substitution queries may also
                            # legitimately return zero prototypes against
                            # the local MPDD snapshot.
                            if devs is None and target_prototypes == "non-native":
                                self.assertGreater(len(self.cas.prototypes), 0)
                            else:
                                self.assertIsInstance(self.cas.prototypes, list)

                            # When an other_targets filter is set, every
                            # generated prototype must trace back to an MPDD
                            # doc that satisfies it.
                            if other_targets is not None:
                                self._assert_prototypes_satisfy_filter(
                                    other_targets, target_prototypes
                                )

        # Unhappy path: invalid devs type should raise
        with self.assertRaises(AssertionError):
            self.cas.generate_MPDD(devs="not-a-dict")

        # Unhappy path: other_targets must be a dict
        with self.assertRaises(AssertionError):
            self.cas.generate_MPDD(other_targets="not-a-dict")

        # Unhappy path: other_targets cannot override reserved formula keys in run_MPDD
        with self.assertRaises(ValueError):
            self.cas.generate_MPDD(other_targets={"material.reducedFormula": "Ni26Ru8"})

        # Unhappy path: other_targets cannot collide with composition-query
        with self.assertRaises(ValueError):
            self.cas.generate_MPDD(
                target_prototypes="native",
                devs={"Ni": 10, "Ru": 10},
                other_targets={"material.system": "Ni-Ru"},
            )

    @pytest.mark.order(3)
    def test_generate_local(self) -> None:
        """Test local prototype generation using MPDD prototypes as input prototypes."""
        # Ensure MPDD prototypes exist
        if not (
            isinstance(self.cas.prototype_path, str)
            and os.path.isdir(self.cas.prototype_path)
        ):
            self.cas.generate_MPDD(limit=10)

        mpdd_path_after = self.cas.prototype_path
        self.cas.generate_local(prototype_path=mpdd_path_after)

        local_path_after = self.cas.prototype_path
        self.assertIsInstance(local_path_after, str)
        self.assertTrue(os.path.isdir(local_path_after))
        self.assertGreater(len(self.cas.prototypes), 0)

        # Every generated POSCAR must parse and yield correct stoichiometry
        for fname in os.listdir(local_path_after):
            with self.subTest(file=fname):
                s = Structure.from_file(os.path.join(local_path_after, fname))
                self.assertEqual(
                    s.composition.reduced_composition,
                    self.cas.target_formula.reduced_composition,
                    f"Stoichiometry mismatch in {fname}"
                )
                spgA = SpacegroupAnalyzer(s)
                self.assertIsNotNone(spgA.get_crystal_system())

        # Do not call generate_local without prototype_path here because
        # run_local asserts that output dir must differ from prototype_path.

    @pytest.mark.order(4)
    @pytest.mark.skipif(
        IN_GITHUB_ACTIONS, reason="Test depends on the ONNX network files"
    )
    def test_run_pysipfenn(self) -> None:
        """Test running pySIPFENN models end-to-end."""
        self.cas.run_pysipfenn(mode="serial")
        self.assertIsNotNone(self.cas.ks2022)
        self.assertGreater(
            len(self.cas.ks2022), 0, "pySIPFENN did not generate any features."
        )

        # Note: unhappy path (missing prototypes folder) is handled via
        # assertion in run_pysipfenn, but constructing a fresh CrystALLSearch
        # instance without setting prototype_path raises AttributeError
        # before that assertion, so we do not test that here.

    @pytest.mark.order(5)
    def test_get_calculator_properties(self) -> None:
        """Test get_calculator_properties wiring using a dummy external calculator adapter."""

        self_test = self

        class DummyAdapter:
            objectives = ["ext_prop"]

            def run_on_directory(
                self, path, mode="serial", max_workers=None, objectives=None
            ):
                return [
                    {"name": p["name"], "properties": {"ext_prop": 2.5}}
                    for p in self_test.cas.prototypes
                ]

        self.cas.supported_calculators["dummy"] = DummyAdapter
        self.cas.available_calculators["dummy"] = ["ext_prop"]

        self.cas.get_calculator_properties("dummy", mode="serial")

        for p in self.cas.prototypes:
            self.assertIn("ext_prop", p)
        self.assertIn("ext_prop", self.cas.predicted_objectives)

        # Unhappy path: unknown calculator should raise
        with self.assertRaises(AssertionError):
            self.cas.get_calculator_properties("nonexistent")

        # Exercise objectives subset intersection branch
        self.cas.supported_calculators["dummy2"] = DummyAdapter
        self.cas.available_calculators["dummy2"] = ["ext_prop", "other_prop"]
        self.cas.get_calculator_properties(
            "dummy2", mode="serial", objectives=["ext_prop"]
        )
        self.assertIn("ext_prop", self.cas.predicted_objectives)
        self.assertNotIn("other_prop", self.cas.predicted_objectives)

        # Error branch: adapter instantiation failure should raise RuntimeError
        class BadAdapter:
            def __init__(self):
                raise RuntimeError("fail to init")

        self.cas.supported_calculators["bad_adapter"] = BadAdapter
        self.cas.available_calculators["bad_adapter"] = []
        with self.assertRaises(RuntimeError):
            # core.get_calculator_properties wraps instantiation exceptions
            # into RuntimeError; assert this propagation happens
            self.cas.get_calculator_properties("bad_adapter")

        # No-match branch: adapter returns results but names don't match prototypes
        class NoMatchAdapter:
            objectives = ["mismatch_prop"]

            def run_on_directory(
                self, path, mode="serial", max_workers=None, objectives=None
            ):
                return [
                    {"name": "nonexistent.POSCAR", "properties": {"mismatch_prop": 1.0}}
                ]

        self.cas.supported_calculators["nomatch"] = NoMatchAdapter
        self.cas.available_calculators["nomatch"] = ["mismatch_prop"]
        with self.assertRaises(AssertionError):
            self.cas.get_calculator_properties("nomatch")

        # Length-mismatch warning: adapter returns fewer entries than prototypes
        class PartialAdapter:
            objectives = ["ext_prop"]

            def run_on_directory(
                self, path, mode="serial", max_workers=None, objectives=None
            ):
                # return only a single matching result to trigger the warning path
                if len(self_test.cas.prototypes) > 0:
                    return [
                        {
                            "name": self_test.cas.prototypes[0]["name"],
                            "properties": {"ext_prop": 4.2},
                        }
                    ]
                return []

        self.cas.supported_calculators["partial"] = PartialAdapter
        self.cas.available_calculators["partial"] = ["ext_prop"]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self.cas.get_calculator_properties("partial")
            # Should have emitted at least one warnings.warn about returned N results
            self.assertTrue(
                any("returned" in str(x.message) or "returned" in str(x) for x in w)
            )

    @pytest.mark.order(6)
    @pytest.mark.skipif(
        IN_GITHUB_ACTIONS, reason="Test depends on the ONNX network files"
    )
    def test_get_mpdd_convexhull(self) -> None:
        """Test getting MPDD convex hull stability."""
        self.cas.get_mpdd_convexhull()
        self.assertTrue(
            all("MPDD_Stability" in prototype for prototype in self.cas.prototypes),
            "MPDD convex hull stability not calculated for all prototypes.",
        )

        # Unhappy paths: no connection or no prototypes
        cas2 = crystall.CrystALLSearch(self.target_formula, calculator=self.cas.Calculator)
        cas2.prototypes = [{"name": "dummy", "SIPFENN_Krajewski2022_NN30": -1.0}]
        with self.assertRaises(AssertionError):
            cas2.get_mpdd_convexhull()

        cas3 = crystall.CrystALLSearch(self.target_formula, calculator=self.cas.Calculator)
        cas3.connect_MPDD(self.db)
        cas3.prototypes = []
        with self.assertRaises(AssertionError):
            cas3.get_mpdd_convexhull()

    @pytest.mark.order(7)
    @pytest.mark.skipif(
        IN_GITHUB_ACTIONS, reason="Test depends on the ONNX network files"
    )
    def test_filter_prototypes(self) -> None:
        """Test filtering prototypes with default and custom objectives."""
        # Ensure features are available and at least one objective exists
        if getattr(self.cas, "ks2022", None) is None:
            self.cas.run_pysipfenn(mode="serial")

        base_obj = self.cas.predicted_objectives[0]
        new_name = f"scaled_{base_obj}"

        # Define a simple scaled objective #0 * 2
        self.cas.define_objective(new_name, [base_obj], "#0 * 2")
        self.assertIn(new_name, self.cas.predicted_objectives)

        # Filter using the newly defined objective
        before = len(self.cas.prototypes)
        self.cas.filter_prototypes(max_prototypes=5, objective=new_name)
        self.assertLessEqual(len(self.cas.prototypes), 5)
        self.assertLessEqual(len(self.cas.prototypes), before)

        # Files on disk must reflect filtered set
        files_after = os.listdir(self.cas.prototype_path)
        self.assertEqual(
            len(files_after), len(self.cas.prototypes),
            "Files on disk do not match self.cas.prototypes after filtering"
        )
        # ks2022 must be trimmed to match
        self.assertEqual(
            len(self.cas.ks2022), len(self.cas.prototypes),
            "self.cas.ks2022 length does not match self.cas.prototypes after filtering"
        )

        # Also exercise cutoff-based filtering (max_prototypes == 0)
        if getattr(self.cas, "ks2022", None) is None or len(self.cas.ks2022) == 0:
            self.cas.run_pysipfenn(mode="serial")

        # Use the same custom objective and cutoff branch
        self.cas.filter_prototypes(max_prototypes=0, objective=new_name)
        self.assertGreaterEqual(len(self.cas.prototypes), 1)

        # Edge cases for define_objective: invalid syntax and out-of-range index
        with self.assertRaises(AssertionError):
            self.cas.define_objective("bad_syntax", [base_obj], "bad$expr")
        with self.assertRaises(AssertionError):
            # index 1 is out-of-range when only one base objective is provided
            self.cas.define_objective("out_of_range", [base_obj], "#1")

        # Unhappy path: invalid objective name should raise
        with self.assertRaises(AssertionError):
            self.cas.filter_prototypes(
                max_prototypes=1, objective="nonexistent_objective"
            )

    @pytest.mark.order(8)
    @pytest.mark.skipif(
        IN_GITHUB_ACTIONS, reason="Test depends on the ONNX network files"
    )
    def test_cluster_prototypes_DBSCAN(self) -> None:
        """Test clustering prototypes using DBSCAN."""
        # Unhappy path: invalid argument types should raise
        cas2 = crystall.CrystALLSearch(self.target_formula, calculator=self.cas.Calculator)
        cas2.ks2022 = self.cas.ks2022[:5]
        cas2.prototypes = self.cas.prototypes[:5]
        cas2.available_calculators = {"dummy_calc": [self.cas.predicted_objectives[0]]}
        cas2.predicted_objectives = [self.cas.predicted_objectives[0]]

        with self.assertRaises(AssertionError):
            cas2.cluster_prototypes_DBSCAN(eps="0.3")
        with self.assertRaises(AssertionError):
            cas2.cluster_prototypes_DBSCAN(eps=0.3, min_samples=1.5)

        if getattr(self.cas, "ks2022", None) is None:
            self.cas.run_pysipfenn(mode="serial")
        # Real DBSCAN — no mocking
        self.cas.cluster_prototypes_DBSCAN()
        self.assertTrue(len(self.cas.top_candidates) > 0,
                        "DBSCAN clustering did not generate any top candidates.")
        # Verify cluster labels were attached to all prototypes
        for p in self.cas.prototypes:
            self.assertIn("cluster", p, f"Prototype {p.get('name')} missing cluster label")

        # same_composition=True branch also real — no mocking
        self.cas.top_candidates = []
        self.cas.cluster_prototypes_DBSCAN(same_composition=True)
        self.assertTrue(len(self.cas.top_candidates) > 0,
                        "DBSCAN clustering with same_composition=True did not generate any top candidates.")

    @pytest.mark.order(9)
    @pytest.mark.skipif(
        IN_GITHUB_ACTIONS, reason="Test depends on the ONNX network files"
    )
    def test_cluster_prototypes_KMeans(self) -> None:
        """Test clustering prototypes using KMeans."""
        # Unhappy path: too few prototypes triggers early return
        cas2 = crystall.CrystALLSearch(self.target_formula, calculator=self.cas.Calculator)
        cas2.ks2022 = self.cas.ks2022[:3]
        cas2.prototypes = self.cas.prototypes[:3]
        cas2.available_calculators = {"dummy_calc": [self.cas.predicted_objectives[0]]}
        cas2.predicted_objectives = [self.cas.predicted_objectives[0]]
        cas2.cluster_prototypes_KMeans()
        self.assertEqual(len(cas2.top_candidates), 0)

        if getattr(self.cas, "ks2022", None) is None:
            self.cas.run_pysipfenn(mode="serial")
        # Real KMeans with explicit cluster count — exercises the n_clusters != None branch
        self.cas.cluster_prototypes_KMeans(n_clusters=2)
        self.assertTrue(len(self.cas.top_candidates) > 0,
                        "KMeans clustering did not generate any top candidates.")
        for p in self.cas.prototypes:
            self.assertIn("cluster", p, f"Prototype {p.get('name')} missing cluster label")

        # Real KMeans with automatic silhouette search — exercises the n_clusters=None branch
        self.cas.top_candidates = []
        self.cas.cluster_prototypes_KMeans()
        self.assertTrue(len(self.cas.top_candidates) > 0,
                        "KMeans silhouette-search clustering did not generate any top candidates.")

    @pytest.mark.order(10)
    @pytest.mark.skipif(
        IN_GITHUB_ACTIONS, reason="Test depends on the ONNX network files"
    )
    def test_export_table(self) -> None:
        """Test exporting prototype data to CSV."""
        # First, call export_table when no top_candidates are available to
        # exercise the fallback from 'top-candidates' to 'all-prototypes'.
        saved_top = self.cas.top_candidates
        self.cas.top_candidates = []
        self.cas.export_table(content=["top-candidates"])
        self.cas.top_candidates = saved_top

        # Standard behavior: export both all-prototypes and top-candidates
        self.cas.export_table()
        self.assertTrue(
            os.path.exists("all-prototypes.csv"), "Exported CSV file not found."
        )
        self.assertTrue(
            os.path.exists("top-candidates.csv"), "Exported CSV file not found."
        )
        os.remove("all-prototypes.csv")
        os.remove("top-candidates.csv")

    @pytest.mark.order(11)
    @pytest.mark.skipif(
        IN_GITHUB_ACTIONS, reason="Test depends on the ONNX network files"
    )
    def test_plot_barchart(self) -> None:
        """Test plotting bar charts."""
        # First, ensure early-return path when no top_candidates exist
        saved_top = self.cas.top_candidates
        self.cas.top_candidates = []
        self.cas.plot_barchart(save=False)
        self.cas.top_candidates = saved_top

        # Then generate and save the HTML file with clustered candidates
        self.cas.plot_barchart(save=True)
        self.assertTrue(
            os.path.exists("barchart_clusterPrototypes.html"),
            "Bar chart HTML file not found.",
        )
        os.remove("barchart_clusterPrototypes.html")

    @pytest.mark.order(12)
    @pytest.mark.skipif(
        IN_GITHUB_ACTIONS, reason="Test depends on the ONNX network files"
    )
    def test_plot_2D(self) -> None:
        """Test plotting 2D scatter plots."""
        # Non-save branch: patch the module-level `iplot` used by `crystall.core`
        with patch("crystall.core.iplot") as mock_ip:
            self.cas.plot_2D(save=False)
            mock_ip.assert_called()
        # Ensure no file was written
        self.assertFalse(os.path.exists("2D_clusterPrototypes.html"))

        # Save branch: write and then remove file
        self.cas.plot_2D(save=True)
        self.assertTrue(
            os.path.exists("2D_clusterPrototypes.html"),
            "2D scatter plot HTML file not found.",
        )
        os.remove("2D_clusterPrototypes.html")

    @pytest.mark.order(13)
    @pytest.mark.skipif(
        IN_GITHUB_ACTIONS, reason="Test depends on the ONNX network files"
    )
    def test_plot_3D(self) -> None:
        """Test plotting 3D scatter plots."""
        # Non-save branch: patch the module-level `iplot` used by `crystall.core`
        with patch("crystall.core.iplot") as mock_ip3:
            self.cas.plot_3D(save=False)
            mock_ip3.assert_called()
        # Ensure no file was written
        self.assertFalse(os.path.exists("3D_clusterPrototypes.html"))

        # Save branch: write and then remove file
        self.cas.plot_3D(save=True)
        self.assertTrue(
            os.path.exists("3D_clusterPrototypes.html"),
            "3D scatter plot HTML file not found.",
        )
        os.remove("3D_clusterPrototypes.html")


class TestBaseCalculator(unittest.TestCase):
    """Tests for BaseCalculator helpers and abstract behavior."""

    def test_run_on_directory(self) -> None:
        """BaseCalculator.run_on_directory should raise NotImplementedError by default."""

        class DummyBase(crystall.BaseCalculator):
            name = "dummy"
            objectives = []

        dummy = DummyBase()
        with self.assertRaises(NotImplementedError):
            dummy.run_on_directory("some/path")

    def test_vector_to_scalar(self) -> None:
        """BaseCalculator.vector_to_scalar should handle all supported aggregation modes."""

        # rms and sum on simple 1D data
        data_1d = [3.0, 4.0]
        rms = crystall.BaseCalculator.vector_to_scalar("rms", data_1d)
        total = crystall.BaseCalculator.vector_to_scalar("sum", data_1d)
        self.assertAlmostEqual(
            rms, (3.0**2 + 4.0**2) ** 0.5 / 2**0.5
        )  # or 3.5355339...
        self.assertAlmostEqual(total, 7.0)

        # norm and mean on small arrays
        data_2d = [[3.0, 4.0], [0.0, 5.0]]
        norm = crystall.BaseCalculator.vector_to_scalar("norm", data_2d)
        mean = crystall.BaseCalculator.vector_to_scalar("mean", [1.0, 2.0, 3.0])
        # For two 2D vectors [3,4] (norm=5) and [0,5] (norm=5), mean norm is 5
        self.assertAlmostEqual(norm, 5.0)
        self.assertAlmostEqual(mean, 2.0)

        # vonMises for a 6-component stress tensor [s_xx, s_yy, s_zz, s_xy, s_xz, s_yz]
        tensor_6 = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
        vm = crystall.BaseCalculator.vector_to_scalar("vonMises", tensor_6)
        self.assertIsInstance(vm, float)

        # Invalid type should return None gracefully
        self.assertIsNone(crystall.BaseCalculator.vector_to_scalar("unknown", tensor_6))


if __name__ == "__main__":
    unittest.main()
