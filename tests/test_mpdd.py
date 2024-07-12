import unittest
import pytest
import os
import json
import shutil

import crystall

from pymatgen.core import Structure, Composition

import crystall.crystALL

from pymongo import MongoClient

class TestMPDD(unittest.TestCase):
    
    def setUp(self) -> None:
        """Connects to the local MongoDB instance at port 27017 and uploads ``testFiles/MPDD_A4B11or13or17.json`` to a mocked ``curated`` collection in ``MPDD`` database"""
        self.client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=100)
        self.db = self.client["MPDD"]
        self.collection = self.db["curated"]
        
        with self.subTest("Upload the test file to the database"):
            self.collection.drop()
            with open("testFiles/MPDD_A4B11or13or17.json", "r") as f:
                for entry in json.loads(f.read()):
                    entry["_id"] = entry["_id"]["$oid"]
                    self.collection.insert_one(entry)

    def test_somethingUploaded(self) -> None:
        """Check that the database is not empty."""
        self.assertNotEqual(self.collection.count_documents({}), 0, "The database is empty.")
        print(self.collection.find_one())
