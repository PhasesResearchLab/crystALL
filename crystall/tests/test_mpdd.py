import unittest
import os
import json
import shutil

import crystall

from pymatgen.core import Structure, Composition
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from pymongo import MongoClient

class TestMPDD(unittest.TestCase):
    
    def setUp(self) -> None:
        """Connects to the local MongoDB instance at port 27017 and uploads ``testFiles/MPDD_A4B11or13or17.json`` to a mocked ``curated`` collection in ``MPDD`` database"""
        self.client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=100)
        self.db = self.client["MPDD"]
        self.collection = self.db["curated"]
        self.targetFormula = "Ni26Ru8" # "Ni26Ru8" for quick run or "Ni22Ru8" to stay consistent with local target (running on different, smaller dataset)

        with self.subTest("Upload the test file to the database"):
            # Drop old collection that may have been left behind
            self.collection.drop()

            # Set up hints 
            self.hints = [
                'material.anonymizedFormula',
                'material.reducedFormula',
                'material.structID',
                'material.coarseStructID',
                'material.volNormStructID',
                'material.sortedFractionsDict'
            ]

            for hint in self.hints:
                self.collection.create_index([(hint, 1)], name=hint)
            
            # Populate the collection with the test data
            with open("crystall/tests/testFiles/MPDD_A4B11or13or17.json", "r") as f:
                for entry in json.loads(f.read()):
                    entry["_id"] = entry["_id"]["$oid"]
                    self.collection.insert_one(entry)

    def test_somethingUploaded(self) -> None:
        """Check that the database is not empty."""
        self.assertNotEqual(self.collection.count_documents({}), 0, "The database is empty.")
        print(self.collection.find_one())

    def test_CrystALLSearch(self) -> None:
        """Test the ability to set up the MongoDB connection and perform a search."""
        cas = crystall.CrystALLSearch("Ni26Ru8")
        with self.subTest("Check that disconnected MongoDB raises an error"):
            self.assertRaises(AssertionError, cas.generate_MPDD)

        with self.subTest("Check that connected MongoDB can be connected to and used to run"):
            cas.connect_MPDD(self.db)
            cas.generate_MPDD()

        with self.subTest("Assert that one of the expected prototypes is present"):
            self.assertEqual(cas.prototype_path, "Ni26Ru8-curated-prototypes", "The prototype path is incorrect.")
            self.assertIn("Al13Rh4-monoclinic-aflow880c7a1f43ceb5f8.POSCAR", os.listdir("Ni26Ru8-curated-prototypes"), "The expected prototype was not generated.")

        with self.subTest('Assert one of the expected prototypes has the correct composition and symmetry'):
            structure = Structure.from_file("Ni26Ru8-curated-prototypes/Al13Rh4-monoclinic-aflow880c7a1f43ceb5f8.POSCAR")
            spgA = SpacegroupAnalyzer(structure)
            self.assertEqual(structure.composition.reduced_formula, Composition(self.targetFormula).reduced_formula, "The composition of the prototype is incorrect.")
            self.assertEqual(spgA.get_crystal_system(), "monoclinic", "The crystal system of the prototype is incorrect.")
            self.assertEqual(spgA.get_space_group_number(), 12, "The space group number of the prototype is incorrect.")

        test_cases = [
            ("native", "material.structID", (2, 2)),
            ("native", "material.volNormStructID", (1, 1)),
            ("non-native", "material.structID", (30, 30)),
            ("non-native", "material.volNormStructID", (28, 28)),
        ]

        for target_prototypes, filter_prototypes, expected_counts in test_cases:
            with self.subTest(f"Assert that the correct number of prototypes was generated for {target_prototypes} and {filter_prototypes}"):
                natPath = "Ni26Ru8-curated-prototypes" if target_prototypes == "non-native" else "Ni26Ru8-curated-native-prototypes"
                os.path.exists(natPath) and shutil.rmtree(natPath)
                cas.generate_MPDD(target_prototypes=target_prototypes, filter_prototypes=filter_prototypes)
                self.assertEqual(len(os.listdir(natPath)), expected_counts[0], f"The number of prototypes generated is incorrect ({filter_prototypes}), based on the local files.")
                self.assertEqual(len(cas.prototypes), expected_counts[1], f"The number of prototype dictionaries listed by CAS is incorrect ({filter_prototypes}).")

    def test_run_MPDD(self) -> None:
        """Test the run_MPDD function."""
        targetFormula = "Ni26Ru8"
        collections = ["curated"]
        target_prototypes = "native"
        filter_prototypes = "material.structID"
        limit = 1000
        result = crystall.run_MPDD(targetFormula, self.db, collections, target_prototypes, filter_prototypes, limit)
        self.assertTrue(os.path.exists(result), "The directory for MPDD prototypes was not created.")

    def test_run_compositional_MPDD(self) -> None:
        """Test the run_compositional_MPDD function."""
        targetCompositionDict = {"Ni": 260, "Ru": 80}
        devs = {"Ni": 10, "Ru": 10}
        collections = ["curated"]
        target_prototypes = "native"
        filter_prototypes = "material.structID"
        limit = 1000
        result = crystall.run_compositional_MPDD(targetCompositionDict, devs, self.db, collections, target_prototypes, filter_prototypes, limit)
        self.assertTrue(os.path.exists(result), "The directory for compositional MPDD prototypes was not created.")

if __name__ == '__main__':
    unittest.main()