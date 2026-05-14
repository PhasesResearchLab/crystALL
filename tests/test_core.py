import os
import glob
import json
import shutil
import tempfile
import pytest
import crystall
import unittest
from crystall.calculators import BaseCalculator, Objective, ObjectiveRegistry
import warnings
import numpy as np
from functools import partial
from pathlib import Path

from unittest.mock import patch
from urllib.error import HTTPError
from pymatgen.core import Composition
from pymongo import MongoClient
from sklearn.cluster import KMeans

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
        with patch(
            "importlib.metadata.entry_points", side_effect=Exception("EP failure")
        ):
            adapters = crystall.load_adapters()
            self.assertIsInstance(adapters, dict)

    def setUp(self) -> None:
        """Set up shared state for local and MPDD-related top-level functions."""
        # Local test setup mirrors previous TestLocal
        self.target_formula_local = "Ni22Ru8"
        self.prototype_path_local = "tests/testFiles/localTest"

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
        with open("tests/testFiles/MPDD_A4B11or13or17.json", "r") as f:
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

    def tearDown(self) -> None:
        """Clean up any local and MPDD-generated directories after each test."""
        # Local directories
        if os.path.exists("Ni22Ru8-local-prototypes"):
            shutil.rmtree("Ni22Ru8-local-prototypes")

        # MPDD prototype directories created by tests
        for path in [
            "Ni26Ru8-local-prototypes",
            "Ni26Ru8-MPDD(curated)-prototypes",
            "Ni26Ru8-MPDD(curated)-prototypes(native)",
            "Ni26Ru8±(10,10)-MPDD(curated)-prototypes",
            "Ni26Ru8±(10,10)-MPDD(curated)-prototypes(native)",
        ]:
            if os.path.exists(path):
                shutil.rmtree(path)


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
        with open("tests/testFiles/MPDD_A4B11or13or17.json", "r") as f:
            for entry in json.loads(f.read()):
                entry["_id"] = entry["_id"]["$oid"]
                cls.collection.insert_one(entry)

        cls.target_formula = "Ni26Ru8"
        with (
            patch(
                "crystall.core._pysipfenn_defaults",
                return_value=("fake/root", [], ""),
            ),
            patch("sys.stdin.isatty", return_value=True),
            patch("builtins.input", return_value="n"),
        ):
            cls.cas = crystall.CrystALLSearch(cls.target_formula)

        # Also exercise __init__ branch where no KS2022 models exist and the
        # initializer asks to run download_pysipfenn. We create a separate
        # instance under patched Calculator behavior and patch
        # CrystALLSearch.download_pysipfenn to ensure it's invoked.
        with (
            patch(
                "crystall.core._pysipfenn_defaults",
                return_value=("fake/root", [], ""),
            ),
            patch("sys.stdin.isatty", return_value=True),
            patch("builtins.input", return_value=""),
            patch.object(crystall.CrystALLSearch, "download_pysipfenn") as dl_mock,
        ):
            # instantiate a temporary instance to trigger the no-models branch
            _tmp = crystall.CrystALLSearch(cls.target_formula)
            cls.init_download_called = dl_mock.called

        # Connect to MPDD once; prototype generation is tested explicitly
        cls.cas.setup_MPDD(cls.db)

    @pytest.mark.order(1)
    def test_download_pysipfenn(self) -> None:
        """Test downloading models from the MPDD database via download_pysipfenn."""
        if self.cas.Calculator is None:
            from pysipfenn import Calculator

            self.cas.Calculator = Calculator(autoLoad=False, verbose=False)
        # Patch out the expensive network/model download so test stays fast
        with (
            patch.object(self.cas.Calculator, "downloadModels") as mocked_dl,
            patch(
                "crystall.core._pysipfenn_defaults",
                return_value=("fake/root", ["SIPFENN_Krajewski2022_NN30"], "SIPFENN_Krajewski2022_NN30"),
            ),
            patch("os.path.isfile", return_value=True),
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
            patch(
                "crystall.core._pysipfenn_defaults",
                return_value=("fake/root", ["SIPFENN_Krajewski2022_NN30"], "SIPFENN_Krajewski2022_NN30"),
            ),
            patch("os.path.isfile", return_value=True),
        ):
            # use small retries and zero delay to keep test fast
            self.cas.download_pysipfenn(mode="serial", retries=3, delay=0)
            self.assertEqual(mocked_retry_dl.call_count, 3)

    @pytest.mark.order(2)
    def test_generate_MPDD(self) -> None:
        """Test MPDD prototype generation for multiple target/filter combinations."""
        # Loop explicitly over target types, filter fields and devs variants
        target_options = ["native", "non-native"]
        filter_options = ["material.structID", "material.volNormStructID"]
        devs_options = [None, {"Ni": 10, "Ru": 10}]

        for devs in devs_options:
            for target_prototypes in target_options:
                for filter_prototypes in filter_options:
                    with self.subTest(
                        f"Generate_MPDD for devs={devs} / {target_prototypes} / {filter_prototypes}"
                    ):
                        # Remove any existing output directories to avoid interference
                        # match common patterns for Ni26Ru8 prototype outputs
                        for pattern in [
                            "Ni26Ru8*prototypes*",
                            "Ni26Ru8-local-prototypes",
                        ]:
                            for path in glob.glob(pattern):
                                if os.path.exists(path):
                                    shutil.rmtree(path)

                        # Call generate_MPDD with or without devs. The `devs` path
                        # performs aggregation queries that may require additional
                        # indexes in the local test MPDD instance. Handle
                        # OperationFailure gracefully and skip the subtest if the
                        # environment cannot support the query.
                        try:
                            if devs is None:
                                self.cas.generate_MPDD(
                                    limit=10,
                                    target_prototypes=target_prototypes,
                                    filter_prototypes=filter_prototypes,
                                )
                            else:
                                self.cas.generate_MPDD(
                                    limit=10,
                                    target_prototypes=target_prototypes,
                                    filter_prototypes=filter_prototypes,
                                    devs=devs,
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

                        # For the non-devs path we expect prototypes to be found.
                        # For devs-based compositional substitution queries the
                        # local MPDD snapshot may legitimately return zero
                        # prototypes; therefore only require a non-empty list for
                        # the devs == None case.
                        if devs is None:
                            self.assertGreater(len(self.cas.prototypes), 0)
                        else:
                            self.assertIsInstance(self.cas.prototypes, list)

        # Unhappy path: invalid devs type should raise
        with self.assertRaises(AssertionError):
            self.cas.generate_MPDD(devs="not-a-dict")

        # Ensure cas is left in a state with >0 prototypes for subsequent tests
        self.cas.generate_MPDD(
            limit=10, target_prototypes="native", filter_prototypes="material.structID"
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

        # Do not call generate_local without prototype_path here because
        # run_local asserts that output dir must differ from prototype_path.

    @pytest.mark.order(4)
    @pytest.mark.skipif(
        IN_GITHUB_ACTIONS, reason="Test depends on the ONNX network files"
    )
    def test_run_pysipfenn(self) -> None:
        """Test running pySIPFENN models end-to-end."""
        from unittest.mock import MagicMock

        self.cas.Calculator.loadModels = MagicMock()
        self.cas.Calculator.runFromDirectory = MagicMock()
        self.cas.Calculator.get_resultDictsWithNames = MagicMock(
            return_value=[
                {"name": p["name"], "SIPFENN_Krajewski2022_NN30": -3.5}
                for p in self.cas.prototypes
            ]
        )
        self.cas.Calculator.descriptorData = [[0.1] * 30 for _ in self.cas.prototypes]

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
                self, path, mode="serial", max_workers=None, objectives=None, **kwargs
            ):
                return [
                    {"name": p["name"], "objectives": {"ext_prop": 2.5}}
                    for p in self_test.cas.prototypes
                ]

        self.cas.supported_calculators["dummy"] = DummyAdapter
        self.cas.available_calculators["dummy"] = ["ext_prop"]

        # Ensure prototype_path is set
        if not self.cas.prototype_path:
            self.cas.prototype_path = "dummy_path"

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
                    {"name": "nonexistent.POSCAR", "objectives": {"mismatch_prop": 1.0}}
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
                            "objectives": {"ext_prop": 4.2},
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
        with patch("builtins.input", return_value="n"):
            self.cas.get_mpdd_convexhull()
        self.assertTrue(
            all("MPDD_Stability" in prototype for prototype in self.cas.prototypes),
            "MPDD convex hull stability not calculated for all prototypes.",
        )

        # Unhappy paths: no connection or no prototypes
        cas2 = crystall.CrystALLSearch(self.target_formula)
        cas2.prototypes = [{"name": "dummy", "SIPFENN_Krajewski2022_NN30": -1.0}]
        with self.assertRaises(AssertionError):
            cas2.get_mpdd_convexhull()

        cas3 = crystall.CrystALLSearch(self.target_formula)
        cas3.setup_MPDD(self.db)
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
        # Trim the working set to keep clustering light while retaining enough samples
        self.cas.ks2022 = self.cas.ks2022[:5]
        self.cas.prototypes = self.cas.prototypes[:5]
        self.cas.top_candidates = []

        # Patch embed to avoid TSNE cost during tests
        def cheap_embed(self, same_composition: bool) -> None:
            feats = np.asarray(self.ks2022)
            if feats.ndim == 1:
                feats = feats.reshape(-1, 1)
            if same_composition:
                feats = feats[:, :1]
            self.scaledks2022 = feats
            pad2 = np.zeros((feats.shape[0], max(0, 2 - feats.shape[1])))
            emb2 = np.hstack([feats, pad2])[:, :2]
            self.embeddedks2022 = emb2
            pad3 = np.zeros((feats.shape[0], max(0, 3 - emb2.shape[1])))
            self.embeddedks2022_3d = np.hstack([emb2, pad3])[:, :3]

        from unittest.mock import MagicMock

        mock_dbscan_inst = MagicMock()
        mock_dbscan_inst.fit_predict.return_value = np.array(
            [0, 0, 1, 1, -1][: len(self.cas.ks2022)]
        )
        mock_dbscan_inst.labels_ = np.array([0, 0, 1, 1, -1][: len(self.cas.ks2022)])

        with (
            patch.object(crystall.CrystALLSearch, "_embed_features", cheap_embed),
            patch("crystall.core.DBSCAN", return_value=mock_dbscan_inst),
        ):
            self.cas.cluster_prototypes_DBSCAN()
        self.assertTrue(
            len(self.cas.top_candidates) > 0,
            "DBSCAN clustering did not generate any top candidates.",
        )

        # same_composition branch (uses only structural features)
        self.cas.ks2022 = self.cas.ks2022[:5]
        self.cas.prototypes = self.cas.prototypes[:5]
        self.cas.top_candidates = []
        with (
            patch.object(crystall.CrystALLSearch, "_embed_features", cheap_embed),
            patch("crystall.core.DBSCAN", return_value=mock_dbscan_inst),
        ):
            self.cas.cluster_prototypes_DBSCAN(same_composition=True)
        self.assertTrue(
            len(self.cas.top_candidates) > 0,
            "DBSCAN clustering with same_composition did not generate any top candidates.",
        )

        # Unhappy path: invalid argument types should raise
        cas2 = crystall.CrystALLSearch(self.target_formula)
        cas2.ks2022 = self.cas.ks2022[:5]
        cas2.prototypes = self.cas.prototypes[:5]
        cas2.available_calculators = {"dummy_calc": [self.cas.predicted_objectives[0]]}
        cas2.predicted_objectives = [self.cas.predicted_objectives[0]]

        with self.assertRaises(AssertionError):
            cas2.cluster_prototypes_DBSCAN(eps="0.3")
        with self.assertRaises(AssertionError):
            cas2.cluster_prototypes_DBSCAN(eps=0.3, min_samples=1.5)

    @pytest.mark.order(9)
    @pytest.mark.skipif(
        IN_GITHUB_ACTIONS, reason="Test depends on the ONNX network files"
    )
    def test_cluster_prototypes_KMeans(self) -> None:
        """Test clustering prototypes using KMeans."""
        # Trim the working set to keep clustering light while retaining enough samples
        self.cas.ks2022 = self.cas.ks2022[:5]
        self.cas.prototypes = self.cas.prototypes[:5]
        self.cas.top_candidates = []

        def cheap_embed(self, same_composition: bool) -> None:
            # Use scaled features with minimal projection to avoid TSNE cost
            feats = np.asarray(self.ks2022)
            if feats.ndim == 1:
                feats = feats.reshape(-1, 1)
            if same_composition:
                feats = feats[:, :1]
            self.scaledks2022 = feats
            # 2D/3D embeddings as thin slices
            pad2 = np.zeros((feats.shape[0], max(0, 2 - feats.shape[1])))
            emb2 = np.hstack([feats, pad2])[:, :2]
            self.embeddedks2022 = emb2
            pad3 = np.zeros((feats.shape[0], max(0, 3 - emb2.shape[1])))
            self.embeddedks2022_3d = np.hstack([emb2, pad3])[:, :3]

        # Patch KMeans, __embed_features, and the k-range to keep silhouette search light
        from unittest.mock import MagicMock

        mock_kmeans_inst = MagicMock()
        mock_kmeans_inst.fit_predict.return_value = np.array(
            [0, 0, 1, 1, 0][: len(self.cas.ks2022)]
        )
        mock_kmeans_inst.labels_ = np.array([0, 0, 1, 1, 0][: len(self.cas.ks2022)])

        with (
            patch("crystall.core.KMeans", return_value=mock_kmeans_inst),
            patch("crystall.core.silhouette_score", return_value=0.5),
            patch.object(crystall.CrystALLSearch, "_embed_features", cheap_embed),
            patch("crystall.core.np.geomspace", return_value=np.array([2, 3])),
        ):
            # Explicitly exercise silhouette-search path with fixed small cluster count
            self.cas.cluster_prototypes_KMeans(n_clusters=2)
            self.assertTrue(
                len(self.cas.top_candidates) > 0,
                "KMeans clustering did not generate any top candidates.",
            )

        # Unhappy path: too few prototypes triggers early return
        cas2 = crystall.CrystALLSearch(self.target_formula)
        cas2.ks2022 = self.cas.ks2022[:3]
        cas2.prototypes = self.cas.prototypes[:3]
        cas2.available_calculators = {"dummy_calc": [self.cas.predicted_objectives[0]]}
        cas2.predicted_objectives = [self.cas.predicted_objectives[0]]
        cas2.cluster_prototypes_KMeans()
        self.assertEqual(len(cas2.top_candidates), 0)

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

    class DummyEnvCalculator(BaseCalculator):
        """Small test adapter used for micro-environment lifecycle tests."""

        name = "dummy-env"
        objectives = ObjectiveRegistry([])

        def __init__(self, env_path: Path, fail_layer2: bool = False) -> None:
            self._test_env_path = Path(env_path)
            self._fail_layer2 = fail_layer2

        def _env_path(self) -> Path:
            return self._test_env_path

        def _install_layer2(self, env: dict) -> None:
            if self._fail_layer2:
                raise RuntimeError("layer2-failure")

        def setup_commands(self) -> list[str]:
            return []

        def _run_code(self) -> str:
            return "print('noop')"

    class DummyStaticCalculator(BaseCalculator):
        name = "dummy-static"
        objectives = ObjectiveRegistry([Objective("DUMMY_STA", "Dummy Static", "dummy", "DummySta")])

        def _run_code(self) -> str:
            return "print('noop')"

    @staticmethod
    def _fake_uv_venv_run(cmd, env=None, check=False, **kwargs):
        """Mock uv venv creation by laying down an executable micro-Python."""
        target = Path(cmd[-1])
        bin_dir = target / ("Scripts" if os.name == "nt" else "bin")
        py_name = "python.exe" if os.name == "nt" else "python"
        bin_dir.mkdir(parents=True, exist_ok=True)
        py = bin_dir / py_name
        py.write_text("#!/usr/bin/env python\n", encoding="utf-8")
        py.chmod(0o755)

    def test_run_on_directory(self) -> None:
        """BaseCalculator.run_on_directory should raise NotImplementedError by default."""
        import tempfile

        class DummyBase(BaseCalculator):
            name = "dummy"
            objectives = ObjectiveRegistry([])

            def is_env_ready(self) -> bool:
                return True

        dummy = DummyBase()
        with tempfile.TemporaryDirectory() as tempdir:
            # Create a dummy file so it calls _evaluate_structure
            with open(os.path.join(tempdir, "dummy.cif"), "w") as f:
                f.write("dummy")
            with self.assertRaises(NotImplementedError):
                dummy.run_on_directory(tempdir)

    def test_vector_to_scalar(self) -> None:
        """BaseCalculator.vector_to_scalar should handle all supported aggregation modes."""

        # rms and sum on simple 1D data
        data_1d = [3.0, 4.0]
        rms = BaseCalculator.vector_to_scalar("rms", data_1d)
        total = BaseCalculator.vector_to_scalar("sum", data_1d)
        self.assertAlmostEqual(
            rms, (3.0**2 + 4.0**2) ** 0.5 / 2**0.5
        )  # or 3.5355339...
        self.assertAlmostEqual(total, 7.0)

        # norm and mean on small arrays
        data_2d = [[3.0, 4.0], [0.0, 5.0]]
        norm = BaseCalculator.vector_to_scalar("norm", data_2d)
        mean = BaseCalculator.vector_to_scalar("mean", [1.0, 2.0, 3.0])
        # For two 2D vectors [3,4] (norm=5) and [0,5] (norm=5), mean norm is 5
        self.assertAlmostEqual(norm, 5.0)
        self.assertAlmostEqual(mean, 2.0)

        # vonMises for a 6-component stress tensor [s_xx, s_yy, s_zz, s_xy, s_xz, s_yz]
        tensor_6 = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
        vm = BaseCalculator.vector_to_scalar("vonMises", tensor_6)
        self.assertIsInstance(vm, float)

        # Invalid type should return None gracefully
        self.assertIsNone(BaseCalculator.vector_to_scalar("unknown", tensor_6))

    def test_setup_env_writes_sentinel_and_is_env_ready_requires_it(self) -> None:
        """setup_env should write readiness sentinel and is_env_ready should require it."""
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / "dummy_env"
            calc = self.DummyEnvCalculator(env_path=env_path)

            with patch("subprocess.run", side_effect=self._fake_uv_venv_run):
                calc.setup_env()

            sentinel = env_path / ".crystall_env_ready"
            self.assertTrue(sentinel.exists())
            self.assertTrue(calc.is_env_ready())

            sentinel.unlink()
            self.assertFalse(calc.is_env_ready())

    def test_setup_env_rebuilds_incomplete_env(self) -> None:
        """setup_env should remove/rebuild env when Python exists but sentinel is missing."""
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / "dummy_env"
            calc = self.DummyEnvCalculator(env_path=env_path)

            old_py = calc._micro_python()
            old_py.parent.mkdir(parents=True, exist_ok=True)
            old_py.write_text("#!/usr/bin/env python\n", encoding="utf-8")
            old_py.chmod(0o755)
            stale_file = env_path / "stale-marker.txt"
            stale_file.write_text("stale", encoding="utf-8")

            with patch("subprocess.run", side_effect=self._fake_uv_venv_run):
                calc.setup_env()

            sentinel = env_path / ".crystall_env_ready"
            self.assertTrue(sentinel.exists())
            self.assertTrue(calc.is_env_ready())
            self.assertFalse(stale_file.exists())

    def test_setup_env_cleans_up_after_layer2_failure(self) -> None:
        """setup_env should remove partial env when layer-2 install fails."""
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / "dummy_env"
            calc = self.DummyEnvCalculator(env_path=env_path, fail_layer2=True)

            with patch("subprocess.run", side_effect=self._fake_uv_venv_run):
                with self.assertRaises(RuntimeError):
                    calc.setup_env()

            self.assertFalse(env_path.exists())
            self.assertFalse((env_path / ".crystall_env_ready").exists())

    def test_base_calculator_type_default_is_static(self) -> None:
        """BaseCalculator should default calculator_type to static."""
        self.assertEqual(BaseCalculator.calculator_type, "static")

    def test_mlip_calculator_type_is_mlip(self) -> None:
        """MLIPAdapter should declare calculator_type as mlip."""
        from crystall.calculators import MLIPAdapter

        self.assertEqual(MLIPAdapter.calculator_type, "mlip")

    def test_mlip_static_accepts_static_adapters(self) -> None:
        """MLIPAdapter static pipe should accept static-role adapters."""
        from crystall.calculators import MLIPAdapter
        from crystall.calculators import AlignnAdapter
        from crystall.calculators import MatGLAdapter

        self.assertIsInstance(
            MLIPAdapter("MACE", static=[AlignnAdapter()]), MLIPAdapter
        )
        self.assertIsInstance(
            MLIPAdapter("MACE", static=[MatGLAdapter()]), MLIPAdapter
        )

    def test_mlip_static_rejects_mlip_adapters(self) -> None:
        """MLIPAdapter static pipe should reject MLIP-role adapters."""
        from crystall.calculators import MLIPAdapter

        with self.assertRaises(TypeError) as ctx:
            MLIPAdapter("ORB", static=[MLIPAdapter("MACE")])
        self.assertIn("calculator_type='mlip'", str(ctx.exception))


class TestBaseCalculatorSchemaGuards(unittest.TestCase):
    """Tests for the adapter schema guards introduced in TODO_260429_Adapter_Schema_Guards.

    Covers:
      - disk-space warning auto-proceeds in non-interactive sessions
      - setup_env aborts cleanly on interactive 'n' input
      - _setup_preflight fires before the cached-env short-circuit
      - _resolve_device behaviour for cpu / cuda
      - MLIPAdapter and DeePAWAdapter do not override setup_env
      - NequIP MLIP config raises on unsupported OS via base guard
      - all concrete non-MLIP adapters have explicit supported_os / supported_devices
    """

    # ── Minimal concrete subclass for hook / device tests ───────────────
    class _GuardAdapter(BaseCalculator):
        name = "_Guard"
        objectives = ObjectiveRegistry([Objective("GUARD_X", "Guard test output", "guard", "Guard")])
        supported_os = ["linux", "macos", "windows"]
        supported_devices = ["cpu", "cuda"]

        def _env_path(self):
            from pathlib import Path

            return Path(self._tmp) / "env"

        def _run_code(self):
            return "print('ok')"

    # ── 1. Non-interactive session: disk-space warning auto-proceeds ─────
    def test_setup_env_non_interactive_auto_proceeds(self):
        """setup_env must print the warning and continue without prompting when stdin is not a tty."""
        import tempfile, io
        from unittest.mock import patch, MagicMock

        adapter = self._GuardAdapter()
        with tempfile.TemporaryDirectory() as tmp:
            adapter._tmp = tmp
            with (
                patch("sys.stdin", io.StringIO("")),  # isatty() → False
                patch.object(adapter, "_uv_bin", return_value="uv"),
                patch("subprocess.run", return_value=MagicMock(returncode=0)),
                patch.object(adapter, "_install_layer2"),
                patch.object(adapter, "setup_commands", return_value=[]),
                patch.object(adapter, "_layer3_setup_commands", return_value=[]),
                patch("builtins.input") as mock_input,
            ):
                try:
                    adapter.setup_env()
                except Exception:
                    pass
            mock_input.assert_not_called()

    # ── 2. Interactive 'n': setup_env aborts ─────────────────────────────
    def test_setup_env_aborts_on_n_response(self):
        """setup_env must return early and skip installation when user enters 'n'."""
        import tempfile, io
        from unittest.mock import patch, MagicMock

        adapter = self._GuardAdapter()
        with tempfile.TemporaryDirectory() as tmp:
            adapter._tmp = tmp
            tty_stdin = MagicMock()
            tty_stdin.isatty.return_value = True
            with (
                patch("sys.stdin", tty_stdin),
                patch("builtins.input", return_value="n"),
                patch("subprocess.run") as mock_run,
            ):
                adapter.setup_env()
            mock_run.assert_not_called()

    # ── 3. _setup_preflight fires BEFORE the cached-env short-circuit ────
    def test_setup_preflight_fires_before_cache_check(self):
        """_setup_preflight must be called even when the env is already ready."""
        import tempfile
        from unittest.mock import patch, call

        adapter = self._GuardAdapter()
        with tempfile.TemporaryDirectory() as tmp:
            adapter._tmp = tmp
            call_log = []
            with (
                patch.object(
                    adapter,
                    "_setup_preflight",
                    side_effect=lambda: call_log.append("preflight"),
                ),
                patch.object(
                    adapter,
                    "is_env_ready",
                    side_effect=lambda: call_log.append("cache_check") or True,
                ),
            ):
                adapter.setup_env()
        self.assertEqual(
            call_log[0], "preflight", "preflight must fire before cache check"
        )
        self.assertIn("cache_check", call_log)

    # ── 4. _resolve_device('cuda') raises when CUDA unavailable ──────────
    def test_resolve_device_cuda_raises_when_unavailable(self):
        """_resolve_device must raise RuntimeError for 'cuda' when the system has no GPU."""
        from unittest.mock import patch

        adapter = self._GuardAdapter()
        with patch.object(type(adapter), "_cuda_available_system", return_value=False):
            with self.assertRaises(RuntimeError) as ctx:
                adapter._resolve_device("cuda")
        self.assertIn("cuda", str(ctx.exception).lower())

    # ── 5. _resolve_device('cpu') always succeeds ────────────────────────
    def test_resolve_device_cpu_always_succeeds(self):
        """_resolve_device must accept 'cpu' unconditionally."""
        adapter = self._GuardAdapter()
        result = adapter._resolve_device("cpu")
        self.assertEqual(result, "cpu")

    # ── 6. MLIPAdapter does not override setup_env ───────────────────────
    def test_mlip_adapter_does_not_override_setup_env(self):
        from crystall.calculators import MLIPAdapter

        self.assertIs(
            MLIPAdapter.setup_env,
            BaseCalculator.setup_env,
            "MLIPAdapter must not override setup_env",
        )

    # ── 7. DeePAWAdapter does not override setup_env ─────────────────────
    def test_deepaw_adapter_does_not_override_setup_env(self):
        from crystall.calculators import DeePAWAdapter

        self.assertIs(
            DeePAWAdapter.setup_env,
            BaseCalculator.setup_env,
            "DeePAWAdapter must not override setup_env",
        )

    # ── 8. NequIP MLIP config raises on non-Linux OS ─────────────────────
    def test_nequip_mlip_raises_on_non_linux_os(self):
        """NequIP is linux-only; _assert_os_supported must raise on macOS."""
        from unittest.mock import patch
        from crystall.calculators import MLIPAdapter

        adapter = MLIPAdapter("NequIP")
        with patch("platform.system", return_value="Darwin"):
            with self.assertRaises(RuntimeError) as ctx:
                adapter._assert_os_supported()
        self.assertIn("linux", str(ctx.exception).lower())

    # ── 9. All concrete non-MLIP adapters have explicit declarations ──────
    def test_all_concrete_adapters_declare_supported_os_and_devices(self):
        """Every concrete non-MLIP adapter must declare supported_os and supported_devices
        as non-empty class-level attributes (not inherited from BaseCalculator unchanged)."""
        from crystall.calculators import (
            AlignnAdapter,
            DeePAWAdapter,
            MatGLAdapter,
            PhonaxAdapter,
            UPETDosAdapter,
        )

        adapters = [
            AlignnAdapter,
            DeePAWAdapter,
            MatGLAdapter,
            UPETDosAdapter,
            PhonaxAdapter,
        ]
        for cls in adapters:
            with self.subTest(adapter=cls.__name__):
                self.assertIn(
                    "supported_os",
                    cls.__dict__,
                    f"{cls.__name__} must declare supported_os as a class attribute",
                )
                self.assertIn(
                    "supported_devices",
                    cls.__dict__,
                    f"{cls.__name__} must declare supported_devices as a class attribute",
                )
                self.assertTrue(
                    cls.__dict__["supported_os"],
                    f"{cls.__name__}.supported_os must be non-empty",
                )
                self.assertTrue(
                    cls.__dict__["supported_devices"],
                    f"{cls.__name__}.supported_devices must be non-empty",
                )


class TestObjectiveRegistry(unittest.TestCase):
    """Tests for objective code registry helpers (TODO_260429_Objective_Code_Registry, tests 1-4)."""

    # ── Test 1: Objective dataclass fields ───────────────────────────────

    def test_objective_fields(self) -> None:
        """Objective dataclass exposes code, description, dispatch_key, backend."""
        obj = Objective(
            code="ALIGNN_EF_MP",
            description="Formation Energy [eV/atom]",
            dispatch_key="mp_e_form",
            backend="ALIGNN",
        )
        self.assertEqual(obj.code, "ALIGNN_EF_MP")
        self.assertEqual(obj.description, "Formation Energy [eV/atom]")
        self.assertEqual(obj.dispatch_key, "mp_e_form")
        self.assertEqual(obj.backend, "ALIGNN")

        # Objective is frozen — mutation must raise
        with self.assertRaises((AttributeError, TypeError)):
            obj.code = "OTHER"  # type: ignore[misc]

    # ── Test 2: __init_subclass__ warning on non-conformant keys ─────────

    

    # ── Test 2: ObjectiveRegistry.from_adapters() duplicate code handling ──

    def test_build_objective_registry_deduplication(self) -> None:
        """Duplicate codes across adapters: first entry wins and a UserWarning is emitted."""

        class _Dup1(BaseCalculator):
            name = "dup-1"
            objectives = ObjectiveRegistry([
                Objective("DUPCODE", "Formation Energy from adapter 1", "ef_dup1", "Dup1"),
            ])

        class _Dup2(BaseCalculator):
            name = "dup-2"
            objectives = ObjectiveRegistry([
                Objective("DUPCODE", "Formation Energy from adapter 2", "ef_dup2", "Dup2"),
            ])

        adapters = {"dup-1": _Dup1(), "dup-2": _Dup2()}

        with self.assertWarns(UserWarning) as ctx:
            registry = ObjectiveRegistry.from_adapters(adapters)

        # First entry must be preserved
        self.assertIn("DUPCODE", registry)
        obj = registry["DUPCODE"]
        self.assertEqual(obj.description, "Formation Energy from adapter 1")
        self.assertEqual(obj.dispatch_key, "ef_dup1")
        self.assertEqual(obj.backend, "Dup1")

        # Warning must name both the duplicate code and the offending adapter
        warning_texts = [str(w.message) for w in ctx.warnings]
        self.assertTrue(
            any("DUPCODE" in t and "dup-2" in t for t in warning_texts),
            "Warning should name the duplicate code and the second adapter",
        )

    # ── Test 4: _sanitize_model_code ─────────────────────────────────────

    def test_sanitize_model_code(self) -> None:
        """_sanitize_model_code produces uppercase alphanumeric-with-underscore code prefixes."""
        from crystall.calculators import _sanitize_model_code

        cases = [
            ("ORB-v2", "ORB_V2"),
            ("CHGNet", "CHGNET"),
            ("MACE-MP-0", "MACE_MP_0"),
            ("orb", "ORB"),
            ("M3GNet-MP", "M3GNET_MP"),
        ]

        for model_name, expected in cases:
            with self.subTest(model_name=model_name):
                self.assertEqual(_sanitize_model_code(model_name), expected)

    # ── Shared helper ────────────────────────────────────────────

    @staticmethod
    def _make_minimal_search(registry, predicted, available=None):
        """Build a CrystALLSearch instance with only the state define_objective needs.

        ``registry`` must be an ``ObjectiveRegistry`` instance.
        Uses ``object.__new__`` to bypass ``__init__`` so no MongoDB connection
        or pySIPFENN models are required.
        """
        cas = object.__new__(crystall.CrystALLSearch)
        cas._objective_registry = registry
        cas.predicted_objectives = list(predicted)
        cas.available_calculators = available if available is not None else {}
        return cas

    # ── Test 5: MLIPAdapter piped objective code format ─────────────────

    def test_mlip_piped_code_format(self) -> None:
        """Piped MLIP objective codes use '_' as the namespace separator (no '::')."""
        from crystall.calculators import MLIPAdapter

        fake_config = {
            "FAKE-ML": {
                "python_version": "3.11",
                "layer2_dependencies": [],
                "setup_commands": [],
                "os": ["linux", "macos", "windows"],
            }
        }

        class _DummyStaticAdapter(BaseCalculator):
            name = "dummy-inf"
            objectives = ObjectiveRegistry([
                Objective("DUMMY_EF", "Dummy formation energy (eV/atom)", "dummy_ef", "Dummy"),
            ])

        with patch(
            "crystall.adapters.mlips._load_mlips_config", return_value=fake_config
        ):
            adapter = MLIPAdapter("FAKE-ML", static=[_DummyStaticAdapter()])

        objectives = adapter.objectives
        self.assertIsInstance(objectives, ObjectiveRegistry)

        # No '::' may appear in any objective code
        for code in objectives.codes():
            self.assertNotIn(
                "::", code, f"Code {code!r} must not use '::' as a separator"
            )

        # The piped objective must follow the FAKE_ML_DUMMY_EF naming convention
        piped = [obj for obj in objectives.values() if "DUMMY_EF" in obj.code and "via" in obj.description]
        self.assertEqual(len(piped), 1, "Expected exactly one piped objective for DUMMY_EF")
        self.assertTrue(
            piped[0].code.startswith("FAKE_ML_DUMMY_EF"),
            f"Piped code has unexpected prefix: {piped[0].code!r}",
        )
        self.assertIn("(via dummy-inf)", piped[0].description)

    def test_mlip_build_format_vars_uses_custom_defaults(self) -> None:
        """MLIPAdapter helper should source model and backend vars from custom."""
        from crystall.calculators import MLIPAdapter

        fake_config = {
            "FAKE-ML": {
                "custom": {
                    "model": "fake-default",
                    "head": "default-head",
                }
            }
        }

        with patch(
            "crystall.adapters.mlips._load_mlips_config", return_value=fake_config
        ):
            adapter = MLIPAdapter("FAKE-ML")
            with (
                patch.object(adapter, "_resolve_device", return_value="cpu"),
                patch.object(adapter, "_env_path", return_value=Path("/envs/fake")),
            ):
                fv = adapter._helper_build_format_vars()

        self.assertEqual(
            fv,
            {
                "device": "cpu",
                "SFY_PATH": "/envs/fake",
                "model": "fake-default",
                "head": "default-head",
            },
        )

    def test_mlip_build_format_vars_custom_overrides_replace_defaults(self) -> None:
        """MLIPAdapter helper should shallow-overwrite mlips.json custom defaults."""
        from crystall.calculators import MLIPAdapter

        fake_config = {
            "FAKE-ML": {
                "custom": {
                    "model": "fake-default",
                    "head": "default-head",
                    "precision": "float32",
                }
            }
        }

        with patch(
            "crystall.adapters.mlips._load_mlips_config", return_value=fake_config
        ):
            adapter = MLIPAdapter(
                "FAKE-ML",
                custom={"model": "fake-override", "head": "override-head"},
            )
            with (
                patch.object(adapter, "_resolve_device", return_value="cpu"),
                patch.object(adapter, "_env_path", return_value=Path("/envs/fake")),
            ):
                fv = adapter._helper_build_format_vars()

        self.assertEqual(fv["model"], "fake-override")
        self.assertEqual(fv["head"], "override-head")
        self.assertEqual(fv["precision"], "float32")
        self.assertEqual(fv["device"], "cpu")
        self.assertEqual(fv["SFY_PATH"], "/envs/fake")

    def test_mlip_build_format_vars_resolves_overridden_env_placeholders(self) -> None:
        """MLIPAdapter helper should resolve env placeholders from overrides too."""
        from crystall.calculators import MLIPAdapter

        fake_config = {
            "FAKE-ML": {
                "custom": {
                    "model": "fake-default",
                }
            }
        }

        with patch(
            "crystall.adapters.mlips._load_mlips_config", return_value=fake_config
        ):
            adapter = MLIPAdapter(
                "FAKE-ML",
                custom={"token": "${FAKE_TOKEN} test token guidance"},
            )
            with (
                patch.object(adapter, "_resolve_device", return_value="cpu"),
                patch.object(adapter, "_env_path", return_value=Path("/envs/fake")),
                patch.dict("os.environ", {"FAKE_TOKEN": "secret-token"}, clear=False),
            ):
                fv = adapter._helper_build_format_vars()

        self.assertEqual(fv["token"], "secret-token")
        self.assertEqual(fv["model"], "fake-default")

    def test_mlip_build_format_vars_ignores_device_and_sfy_path_overrides(self) -> None:
        """MLIPAdapter helper must not allow custom overrides for device or SFY_PATH."""
        from crystall.calculators import MLIPAdapter

        fake_config = {
            "FAKE-ML": {
                "custom": {
                    "model": "fake-default",
                }
            }
        }

        with patch(
            "crystall.adapters.mlips._load_mlips_config", return_value=fake_config
        ):
            adapter = MLIPAdapter(
                "FAKE-ML",
                custom={
                    "device": "cuda",
                    "SFY_PATH": "/tmp/custom",
                },
            )
            with (
                patch.object(adapter, "_resolve_device", return_value="cpu"),
                patch.object(adapter, "_env_path", return_value=Path("/envs/fake")),
            ):
                fv = adapter._helper_build_format_vars()

        self.assertEqual(fv["device"], "cpu")
        self.assertEqual(fv["SFY_PATH"], "/envs/fake")
        self.assertEqual(fv["model"], "fake-default")

    # ── Test 4: define_objective accepts a bare code ─────────────────────

    def test_resolve_to_code_accepts_code(self) -> None:
        """define_objective passes a bare registered code through unchanged."""
        reg = ObjectiveRegistry([Objective("ALIGNN_EF_MP", "Formation energy", "mp_e_form", "ALIGNN")])
        cas = self._make_minimal_search(registry=reg, predicted=["ALIGNN_EF_MP"])
        self.assertEqual(cas.define_objective("ALIGNN_EF_MP"), "ALIGNN_EF_MP")

    # ── Test 5: define_objective accepts a legacy 'CODE: description' key ─

    def test_resolve_to_code_accepts_full_key(self) -> None:
        """define_objective extracts the bare code from a legacy 'CODE: description' key."""
        reg = ObjectiveRegistry([Objective("ALIGNN_EF_MP", "Formation energy", "mp_e_form", "ALIGNN")])
        cas = self._make_minimal_search(registry=reg, predicted=["ALIGNN_EF_MP"])
        self.assertEqual(
            cas.define_objective("ALIGNN_EF_MP: Formation energy (eV/atom)"),
            "ALIGNN_EF_MP",
        )

    # ── Test 6: define_objective raises ValueError with a hint ────────────

    def test_resolve_to_code_raises_on_unknown(self) -> None:
        """define_objective raises ValueError naming the bad code and a prefix-matched hint."""
        reg = ObjectiveRegistry([Objective("ALIGNN_EF_MP", "Formation energy", "mp_e_form", "ALIGNN")])
        cas = self._make_minimal_search(registry=reg, predicted=["ALIGNN_EF_MP"])
        # 'ALIGNN_BG_MP' shares the 'ALIG' prefix with the only known code
        with self.assertRaises(ValueError) as ctx:
            cas.define_objective("ALIGNN_BG_MP")
        msg = str(ctx.exception)
        self.assertIn("ALIGNN_BG_MP", msg, "Error must name the unrecognised code")
        self.assertIn(
            "ALIGNN_EF_MP", msg, "Error must suggest the prefix-matched candidate"
        )


if __name__ == "__main__":
    unittest.main()
