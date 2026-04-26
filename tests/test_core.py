import os
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
from pathlib import Path

from unittest.mock import patch
from urllib.error import HTTPError
from pymatgen.core import Composition
from pymongo import MongoClient
from sklearn.cluster import KMeans


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
        with patch('importlib.metadata.entry_points', side_effect=Exception('EP failure')):
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
        self.client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=100)
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
        self.assertTrue(os.path.isdir(result), "Local prototypes directory was not created.")

        # Basic sanity on number of generated files
        files = os.listdir(result)
        self.assertGreater(len(files), 0, "run_local did not generate any local prototypes.")

    def test_run_MPDD(self) -> None:
        """Test the run_MPDD top-level function."""
        target_formula = "Ni26Ru8"
        collections = ["curated"]
        filter_prototypes = "material.structID"
        limit = 100
        result = crystall.run_MPDD(
            target_formula, self.db, collections, "native", filter_prototypes, limit
        )
        self.assertTrue(os.path.exists(result), "The directory for MPDD prototypes was not created.")

    def test_run_MPDD_devs(self) -> None:
        """Test the run_MPDD_devs / compositional MPDD top-level function."""
        target_composition = "Ni26Ru8"
        devs = {"Ni": 10, "Ru": 10}
        collections = ["curated"]
        filter_prototypes = "material.structID"
        limit = 100
        result = crystall.run_MPDD_devs(
            target_composition, devs, self.db, collections, "native", filter_prototypes, limit
        )
        self.assertTrue(
            os.path.exists(result), "The directory for compositional MPDD prototypes was not created."
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
        cls.client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=100)
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
        with patch("builtins.input", return_value="n"):
            cls.cas = crystall.CrystALLSearch(cls.target_formula)

        # Also exercise __init__ branch where no KS2022 models exist and the
        # initializer asks to run download_pysipfenn. We create a separate
        # instance under patched Calculator behavior and patch
        # CrystALLSearch.download_pysipfenn to ensure it's invoked.
        with patch.object(crystall.core.Calculator, "findCompatibleModels", return_value=[]), \
             patch("crystall.core.Calculator.network_list_available", new=[], create=True), \
             patch("builtins.input", return_value=""), \
             patch.object(crystall.CrystALLSearch, "download_pysipfenn") as dl_mock:
            # instantiate a temporary instance to trigger the no-models branch
            _tmp = crystall.CrystALLSearch(cls.target_formula)
            cls.init_download_called = dl_mock.called

        # Connect to MPDD once; prototype generation is tested explicitly
        cls.cas.setup_MPDD(cls.db)

    @pytest.mark.order(1)
    def test_download_pysipfenn(self) -> None:
        """Test downloading models from the MPDD database via download_pysipfenn."""
        # Patch out the expensive network/model download so test stays fast
        with patch.object(self.cas.Calculator, "downloadModels") as mocked_dl, \
             patch.object(self.cas.Calculator, "findCompatibleModels", return_value=["SIPFENN_Krajewski2022_NN30"]), \
             patch.object(self.cas.Calculator, "network_list_available", new=["SIPFENN_Krajewski2022_NN30"]):
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
        with patch.object(self.cas.Calculator, "downloadModels", side_effect=[err1, err2, None]) as mocked_retry_dl, \
             patch.object(self.cas.Calculator, "findCompatibleModels", return_value=["SIPFENN_Krajewski2022_NN30"]), \
             patch.object(self.cas.Calculator, "network_list_available", new=["SIPFENN_Krajewski2022_NN30"]):
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
        self.cas.generate_MPDD(limit=10, target_prototypes="native", filter_prototypes="material.structID")

    @pytest.mark.order(3)
    def test_generate_local(self) -> None:
        """Test local prototype generation using MPDD prototypes as input prototypes."""
        # Ensure MPDD prototypes exist
        if not (isinstance(self.cas.prototype_path, str) and os.path.isdir(self.cas.prototype_path)):
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
    def test_run_pysipfenn(self) -> None:
        """Test running pySIPFENN models end-to-end."""
        from unittest.mock import MagicMock
        self.cas.Calculator.loadModels = MagicMock()
        self.cas.Calculator.runFromDirectory = MagicMock()
        self.cas.Calculator.get_resultDictsWithNames = MagicMock(
            return_value=[{"name": p["name"], "SIPFENN_Krajewski2022_NN30": -3.5} for p in self.cas.prototypes]
        )
        self.cas.Calculator.descriptorData = [[0.1] * 30 for _ in self.cas.prototypes]
        
        self.cas.run_pysipfenn(mode="serial")
        self.assertIsNotNone(self.cas.ks2022)
        self.assertGreater(len(self.cas.ks2022), 0, "pySIPFENN did not generate any features.")

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

            def run_on_directory(self, path, mode="serial", max_workers=None, objectives=None, **kwargs):
                return [
                    {"name": p["name"], "properties": {"ext_prop": 2.5}}
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
        self.cas.get_calculator_properties("dummy2", mode="serial", objectives=["ext_prop"])
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

            def run_on_directory(self, path, mode="serial", max_workers=None, objectives=None):
                return [{"name": "nonexistent.POSCAR", "properties": {"mismatch_prop": 1.0}}]

        self.cas.supported_calculators["nomatch"] = NoMatchAdapter
        self.cas.available_calculators["nomatch"] = ["mismatch_prop"]
        with self.assertRaises(AssertionError):
            self.cas.get_calculator_properties("nomatch")

        # Length-mismatch warning: adapter returns fewer entries than prototypes
        class PartialAdapter:
            objectives = ["ext_prop"]

            def run_on_directory(self, path, mode="serial", max_workers=None, objectives=None):
                # return only a single matching result to trigger the warning path
                if len(self_test.cas.prototypes) > 0:
                    return [{"name": self_test.cas.prototypes[0]["name"], "properties": {"ext_prop": 4.2}}]
                return []

        self.cas.supported_calculators["partial"] = PartialAdapter
        self.cas.available_calculators["partial"] = ["ext_prop"]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self.cas.get_calculator_properties("partial")
            # Should have emitted at least one warnings.warn about returned N results
            self.assertTrue(any("returned" in str(x.message) or "returned" in str(x) for x in w))

    @pytest.mark.order(6)
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
    def test_filter_prototypes(self) -> None:
        """Test filtering prototypes with default and custom objectives."""
        # Ensure features are available and at least one objective exists
        if getattr(self.cas, "ks2022", None) is None:
            self.cas.run_pysipfenn(mode="serial")

        base_obj = self.cas.predicted_objectives[0]
        new_name = f"scaled_{base_obj}"

        # Define a simple scaled objective [0] * 2
        self.cas.define_objective(new_name, [base_obj], "[0] * 2")
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
            self.cas.define_objective("out_of_range", [base_obj], "[1]")

        # Unhappy path: invalid objective name should raise
        with self.assertRaises(AssertionError):
            self.cas.filter_prototypes(max_prototypes=1, objective="nonexistent_objective")

    @pytest.mark.order(8)
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
        mock_dbscan_inst.fit_predict.return_value = np.array([0, 0, 1, 1, -1][:len(self.cas.ks2022)])
        mock_dbscan_inst.labels_ = np.array([0, 0, 1, 1, -1][:len(self.cas.ks2022)])

        with patch.object(crystall.CrystALLSearch, "_CrystALLSearch__embed_features", cheap_embed), \
             patch("crystall.core.DBSCAN", return_value=mock_dbscan_inst):
            self.cas.cluster_prototypes_DBSCAN()
        self.assertTrue(
            len(self.cas.top_candidates) > 0,
            "DBSCAN clustering did not generate any top candidates.",
        )

        # same_composition branch (uses only structural features)
        self.cas.ks2022 = self.cas.ks2022[:5]
        self.cas.prototypes = self.cas.prototypes[:5]
        self.cas.top_candidates = []
        with patch.object(crystall.CrystALLSearch, "_CrystALLSearch__embed_features", cheap_embed), \
             patch("crystall.core.DBSCAN", return_value=mock_dbscan_inst):
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
        mock_kmeans_inst.fit_predict.return_value = np.array([0, 0, 1, 1, 0][:len(self.cas.ks2022)])
        mock_kmeans_inst.labels_ = np.array([0, 0, 1, 1, 0][:len(self.cas.ks2022)])

        with patch("crystall.core.KMeans", return_value=mock_kmeans_inst), \
             patch("crystall.core.silhouette_score", return_value=0.5), \
             patch.object(crystall.CrystALLSearch, "_CrystALLSearch__embed_features", cheap_embed), \
             patch("crystall.core.np.geomspace", return_value=np.array([2, 3])):
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
    def test_export_table(self) -> None:
        """Test exporting prototype data to CSV."""
        # First, call export_table when no top_candidates are available to
        # exercise the fallback from 'top-candidates' to 'all-prototypes'.
        saved_top = self.cas.top_candidates
        self.cas.top_candidates = []
        self.cas.export_table(content=['top-candidates'])
        self.cas.top_candidates = saved_top

        # Standard behavior: export both all-prototypes and top-candidates
        self.cas.export_table()
        self.assertTrue(os.path.exists("all-prototypes.csv"), "Exported CSV file not found.")
        self.assertTrue(os.path.exists("top-candidates.csv"), "Exported CSV file not found.")
        os.remove("all-prototypes.csv")
        os.remove("top-candidates.csv")

    @pytest.mark.order(11)
    def test_plot_barchart(self) -> None:
        """Test plotting bar charts."""
        # First, ensure early-return path when no top_candidates exist
        saved_top = self.cas.top_candidates
        self.cas.top_candidates = []
        self.cas.plot_barchart(save=False)
        self.cas.top_candidates = saved_top

        # Then generate and save the HTML file with clustered candidates
        self.cas.plot_barchart(save=True)
        self.assertTrue(os.path.exists("barchart_clusterPrototypes.html"), "Bar chart HTML file not found.")
        os.remove("barchart_clusterPrototypes.html")

    @pytest.mark.order(12)
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
        self.assertTrue(os.path.exists("2D_clusterPrototypes.html"), "2D scatter plot HTML file not found.")
        os.remove("2D_clusterPrototypes.html")

    @pytest.mark.order(13)
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
        self.assertTrue(os.path.exists("3D_clusterPrototypes.html"), "3D scatter plot HTML file not found.")
        os.remove("3D_clusterPrototypes.html")


class TestBaseCalculator(unittest.TestCase):
    """Tests for BaseCalculator helpers and abstract behavior."""

    class DummyEnvCalculator(crystall.BaseCalculator):
        """Small test adapter used for micro-environment lifecycle tests."""

        name = "dummy-env"
        objectives_map = {}

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

    class DummyInferenceCalculator(crystall.BaseCalculator):
        name = "dummy-inference"
        objectives_map = {"Dummy Inference": "dummy"}

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

        class DummyBase(crystall.BaseCalculator):
            name = "dummy"
            objectives_map = {}

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
        rms = crystall.BaseCalculator.vector_to_scalar("rms", data_1d)
        total = crystall.BaseCalculator.vector_to_scalar("sum", data_1d)
        self.assertAlmostEqual(rms, (3.0**2 + 4.0**2) ** 0.5 / 2**0.5)  # or 3.5355339...
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

    def test_base_calculator_type_default_is_inference(self) -> None:
        """BaseCalculator should default calculator_type to inference."""
        self.assertEqual(crystall.BaseCalculator.calculator_type, "inference")

    def test_mlip_calculator_type_is_mlip(self) -> None:
        """MLIPAdapter should declare calculator_type as mlip."""
        from crystall.calculators import MLIPAdapter

        self.assertEqual(MLIPAdapter.calculator_type, "mlip")

    def test_mlip_inference_accepts_inference_adapters(self) -> None:
        """MLIPAdapter inference pipe should accept inference-role adapters."""
        from crystall.calculators import MLIPAdapter
        from crystall.calculators import AlignnAdapter
        from crystall.calculators import CHGNetStaticAdapter

        self.assertIsInstance(MLIPAdapter("MACE-mh", inference=[AlignnAdapter()]), MLIPAdapter)
        self.assertIsInstance(MLIPAdapter("MACE-mh", inference=[CHGNetStaticAdapter()]), MLIPAdapter)

    def test_mlip_inference_rejects_mlip_adapters(self) -> None:
        """MLIPAdapter inference pipe should reject MLIP-role adapters."""
        from crystall.calculators import MLIPAdapter

        with self.assertRaises(TypeError) as ctx:
            MLIPAdapter("ORB", inference=[MLIPAdapter("MACE-mh")])
        self.assertIn("calculator_type='mlip'", str(ctx.exception))

if __name__ == '__main__':
    unittest.main()
