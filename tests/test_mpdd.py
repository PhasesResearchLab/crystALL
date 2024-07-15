import unittest
import pytest
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
            with open("tests/testFiles/MPDD_A4B11or13or17.json", "r") as f:
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

        with self.subTest("Assert that one of the expected candidates is present"):
            self.assertEqual(cas.candidatePath, "curated-Ni26Ru8", "The candidate path is incorrect.")
            self.assertIn("Al13Rh4-monoclinic-12-aflow-880c7a1f43ceb5f8-Ni39Ru12.POSCAR", os.listdir("curated-Ni26Ru8"), "The expected candidate was not generated.")

        with self.subTest('Assert one of the expected candidates has the correct composition and symmetry'):
            structure = Structure.from_file("curated-Ni26Ru8/Al13Rh4-monoclinic-12-aflow-880c7a1f43ceb5f8-Ni39Ru12.POSCAR")
            spgA = SpacegroupAnalyzer(structure)
            self.assertEqual(structure.composition.reduced_formula, Composition(self.targetFormula).reduced_formula, "The composition of the candidate is incorrect.")
            self.assertEqual(spgA.get_crystal_system(), "monoclinic", "The crystal system of the candidate is incorrect.")
            self.assertEqual(spgA.get_space_group_number(), 12, "The space group number of the candidate is incorrect.")

        with self.subTest("Assert that the correct number of candidates was generated"):
            self.assertEqual(len(os.listdir("curated-Ni26Ru8")), 30, "The number of candidates generated is incorrect, based on the local files.")
            self.assertEqual(len(cas.inputFiles), 30, "The number of candidates identified as input files by CAS is incorrect.")
            self.assertEqual(len(cas.candidates), 30, "The number of candidate dictionaries listed by CAS is incorrect.")

        with self.subTest("Download pySIPFENN (SIPFENN_Krajewski2022_NN30) model and ALIGNN models; then run featurization"):
            cas.download_models()
            #cas.run_pysipfenn()
            #cas.run_alignn()
