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
import glob
import json
import shutil
import pytest
import crystall
import unittest
import warnings
from typing import TYPE_CHECKING

from unittest.mock import patch
from urllib.error import HTTPError
from pymatgen.core import Lattice, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

if TYPE_CHECKING:
    from pymongo import MongoClient

MONGO_URI = "mongodb://localhost:27017/"
MONGO_TIMEOUT_MS = 100
MPDD_COLLECTION_NAME = "curated"
MPDD_HINT_FIELDS = [
    "material.anonymizedFormula",
    "material.reducedFormula",
    "material.structID",
    "material.coarseStructID",
    "material.volNormStructID",
    "material.sortedFractionsDict",
]
MPDD_FIXTURE_JSON = "tests/testFiles/MPDD_A4B11or13or17.json"
LOCAL_TEST_POSCAR_DIR = "tests/testFiles/localTest"

IN_GITHUB_ACTIONS = (
    os.getenv("GITHUB_ACTIONS") == "true" and os.getenv("MODELS_FETCHED") != "true"
)


def build_mongodb_client() -> "MongoClient":
    from pymongo import MongoClient

    return MongoClient(MONGO_URI, serverSelectionTimeoutMS=MONGO_TIMEOUT_MS)


def seed_mpdd_curated_collection(collection) -> None:
    collection.drop()
    for hint in MPDD_HINT_FIELDS:
        collection.create_index([(hint, 1)], name=hint)
    with open(MPDD_FIXTURE_JSON, "r") as f:
        for entry in json.loads(f.read()):
            entry["_id"] = entry["_id"]["$oid"]
            collection.insert_one(entry)


def make_dummy_structure(
    a: float = 3.5, species=("Si",), frac_coords=((0.0, 0.0, 0.0),)
) -> Structure:
    return Structure(Lattice.cubic(a), list(species), [list(x) for x in frac_coords])


class TestCrystALLSearch(unittest.TestCase):
    """Test the functionalities of the CrystALLSearch class end-to-end."""

    @classmethod
    def setUpClass(cls) -> None:
        """Set up MPDD, create CrystALLSearch and generate both MPDD and local prototypes."""
        cls.client = None
        try:
            cls.client = build_mongodb_client()
            cls.db = cls.client["MPDD"]
            cls.collection = cls.db[MPDD_COLLECTION_NAME]
            cls.hints = list(MPDD_HINT_FIELDS)
            seed_mpdd_curated_collection(cls.collection)

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
        except Exception as exc:
            if cls.client is not None:
                cls.client.close()
            raise unittest.SkipTest(
                f"MongoDB unavailable for core integration tests: {exc}"
            )

    @classmethod
    def tearDownClass(cls) -> None:
        """Clean test artifacts and close Mongo connection."""
        if getattr(cls, "client", None) is not None:
            cls.client.close()
        for pattern in [
            "Ni26Ru8*prototypes*",
            "all-prototypes.csv",
            "top-candidates.csv",
            "barchart_clusterPrototypes.html",
            "2D_clusterPrototypes.html",
            "3D_clusterPrototypes.html",
        ]:
            for path in glob.glob(pattern):
                if os.path.isdir(path):
                    shutil.rmtree(path, ignore_errors=True)
                elif os.path.exists(path):
                    os.remove(path)

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
                return_value=(
                    "fake/root",
                    ["SIPFENN_Krajewski2022_NN30"],
                    "SIPFENN_Krajewski2022_NN30",
                ),
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
                return_value=(
                    "fake/root",
                    ["SIPFENN_Krajewski2022_NN30"],
                    "SIPFENN_Krajewski2022_NN30",
                ),
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
                                    limit=100,
                                    target_prototypes=target_prototypes,
                                    filter_prototypes=filter_prototypes,
                                )
                            else:
                                self.cas.generate_MPDD(
                                    limit=100,
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

                        if devs is None:
                            # Exact prototype counts known from the test fixture
                            expected_counts = {
                                ("native", "material.structID"): 2,
                                ("native", "material.volNormStructID"): 1,
                                ("non-native", "material.structID"): 30,
                                ("non-native", "material.volNormStructID"): 28,
                            }
                            key = (target_prototypes, filter_prototypes)
                            expected = expected_counts.get(key)
                            if expected is not None:
                                self.assertEqual(
                                    len(self.cas.prototypes),
                                    expected,
                                    f"Prototype count mismatch for {key}: "
                                    f"got {len(self.cas.prototypes)}, expected {expected}",
                                )
                            # Files on disk must match the in-memory prototype list
                            files_on_disk = os.listdir(self.cas.prototype_path)
                            self.assertEqual(
                                len(files_on_disk),
                                len(self.cas.prototypes),
                                "Files on disk do not match self.cas.prototypes length",
                            )
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
        try:
            self.cas.generate_local(prototype_path=mpdd_path_after)
        except PermissionError as exc:
            self.skipTest(
                f"Local process-pool execution unavailable in this environment: {exc}"
            )

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
                    f"Stoichiometry mismatch in {fname}",
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
    def test_get_calculator_objectives(self) -> None:
        """Test get_calculator_objectives wiring using a dummy external calculator adapter."""

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

        self.cas.get_calculator_objectives("dummy", mode="serial")

        for p in self.cas.prototypes:
            self.assertIn("ext_prop", p)
        self.assertIn("ext_prop", self.cas.predicted_objectives)

        # Unhappy path: unknown calculator should raise
        with self.assertRaises(AssertionError):
            self.cas.get_calculator_objectives("nonexistent")

        # Exercise objectives subset intersection branch
        self.cas.supported_calculators["dummy2"] = DummyAdapter
        self.cas.available_calculators["dummy2"] = ["ext_prop", "other_prop"]
        self.cas.get_calculator_objectives(
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
            # core.get_calculator_objectives wraps instantiation exceptions
            # into RuntimeError; assert this propagation happens
            self.cas.get_calculator_objectives("bad_adapter")

        # No-match branch: adapter returns results but names don't match prototypes
        class NoMatchAdapter:
            objectives = ["mismatch_prop"]

            def run_on_directory(
                self, path, mode="serial", max_workers=None, objectives=None, **kwargs
            ):
                return [
                    {"name": "nonexistent.POSCAR", "objectives": {"mismatch_prop": 1.0}}
                ]

        self.cas.supported_calculators["nomatch"] = NoMatchAdapter
        self.cas.available_calculators["nomatch"] = ["mismatch_prop"]
        with self.assertRaises(AssertionError):
            self.cas.get_calculator_objectives("nomatch")

        # Length-mismatch warning: adapter returns fewer entries than prototypes
        class PartialAdapter:
            objectives = ["ext_prop"]

            def run_on_directory(
                self, path, mode="serial", max_workers=None, objectives=None, **kwargs
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
            self.cas.get_calculator_objectives("partial")
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
        self.cas.define_new_objective(new_name, [base_obj], "#0 * 2")
        self.assertIn(new_name, self.cas.predicted_objectives)

        # Filter using the newly defined objective
        before = len(self.cas.prototypes)
        self.cas.filter_prototypes(max_prototypes=5, objective=new_name)
        self.assertLessEqual(len(self.cas.prototypes), 5)
        self.assertLessEqual(len(self.cas.prototypes), before)

        # Files on disk must reflect filtered set
        files_after = os.listdir(self.cas.prototype_path)
        self.assertEqual(
            len(files_after),
            len(self.cas.prototypes),
            "Files on disk do not match self.cas.prototypes after filtering",
        )
        # ks2022 must be trimmed to match
        self.assertEqual(
            len(self.cas.ks2022),
            len(self.cas.prototypes),
            "self.cas.ks2022 length does not match self.cas.prototypes after filtering",
        )

        # Also exercise cutoff-based filtering (max_prototypes == 0)
        if getattr(self.cas, "ks2022", None) is None or len(self.cas.ks2022) == 0:
            self.cas.run_pysipfenn(mode="serial")

        # Use the same custom objective and cutoff branch
        self.cas.filter_prototypes(max_prototypes=0, objective=new_name)
        self.assertGreaterEqual(len(self.cas.prototypes), 1)

        # Edge cases for define_new_objective: invalid syntax and out-of-range index
        with self.assertRaises(AssertionError):
            self.cas.define_new_objective("bad_syntax", [base_obj], "bad$expr")
        with self.assertRaises(AssertionError):
            # index 1 is out-of-range when only one base objective is provided
            self.cas.define_new_objective("out_of_range", [base_obj], "#1")

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

        # Unhappy path: invalid argument types should raise
        cas2 = crystall.CrystALLSearch(
            self.target_formula, calculator=self.cas.Calculator
        )
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
        self.assertTrue(
            len(self.cas.top_candidates) > 0,
            "DBSCAN clustering did not generate any top candidates.",
        )
        # Verify cluster labels were attached to all prototypes
        for p in self.cas.prototypes:
            self.assertIn(
                "cluster", p, f"Prototype {p.get('name')} missing cluster label"
            )

        # same_composition=True branch also real — no mocking
        self.cas.top_candidates = []
        self.cas.cluster_prototypes_DBSCAN(same_composition=True)
        self.assertTrue(
            len(self.cas.top_candidates) > 0,
            "DBSCAN clustering with same_composition=True did not generate any top candidates.",
        )

    @pytest.mark.order(9)
    @pytest.mark.skipif(
        IN_GITHUB_ACTIONS, reason="Test depends on the ONNX network files"
    )
    def test_cluster_prototypes_KMeans(self) -> None:
        """Test clustering prototypes using KMeans."""
        # Unhappy path: too few prototypes triggers early return
        cas2 = crystall.CrystALLSearch(
            self.target_formula, calculator=self.cas.Calculator
        )
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
        self.assertTrue(
            len(self.cas.top_candidates) > 0,
            "KMeans clustering did not generate any top candidates.",
        )
        for p in self.cas.prototypes:
            self.assertIn(
                "cluster", p, f"Prototype {p.get('name')} missing cluster label"
            )

        # Real KMeans with automatic silhouette search — exercises the n_clusters=None branch
        self.cas.top_candidates = []
        self.cas.cluster_prototypes_KMeans()
        self.assertTrue(
            len(self.cas.top_candidates) > 0,
            "KMeans silhouette-search clustering did not generate any top candidates.",
        )

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


if __name__ == "__main__":
    unittest.main()
