import unittest
from unittest.mock import patch

import os
import json
import pytest

import crystall

from pymongo import MongoClient
from colorama import just_fix_windows_console

class TestCrystALLSearch(unittest.TestCase):
    """Test the functionalities of the CrystALLSearch class."""

    @classmethod
    def setUpClass(cls) -> None:
        """Set up the test environment once before any tests are run."""
        cls.client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=100)
        cls.db = cls.client["MPDD"]
        cls.collection = cls.db["curated"]
        
        # Drop old collection that may have been left behind
        cls.collection.drop()

        # Set up hints 
        cls.hints = [
            'material.anonymizedFormula',
            'material.reducedFormula',
            'material.structID',
            'material.coarseStructID',
            'material.volNormStructID',
            'material.sortedFractionsDict'
        ]

        for hint in cls.hints:
            cls.collection.create_index([(hint, 1)], name=hint)
        
        # Populate the collection with the test data
        with open("crystall/tests/testFiles/MPDD_A4B11or13or17.json", "r") as f:
            for entry in json.loads(f.read()):
                entry["_id"] = entry["_id"]["$oid"]
                cls.collection.insert_one(entry)
        
        cls.targetFormula = "Ni26Ru8"
        with patch('builtins.input', return_value='n'):
            cls.cas = crystall.CrystALLSearch(cls.targetFormula)
            cls.cas.connect_MPDD(cls.db)
            cls.cas.generate_MPDD()

    def setUp(self) -> None:
        just_fix_windows_console()

    @pytest.mark.order(1)
    def test_download_models(self) -> None:
        """Test downloading models from the MPDD database."""
        self.cas.download_models(mode='serial')
        self.assertTrue(self.cas.model_status['pySIPFENN'], "pySIPFENN models not downloaded.")
        self.assertTrue(self.cas.model_status['MPDD-ALIGNN'], "ALIGNN models not downloaded.")
    
    @pytest.mark.order(2)
    def test_run_pysipfenn(self) -> None:
        """Test running pySIPFENN models."""
        self.cas.run_pysipfenn()
        self.assertTrue(len(self.cas.ks2022) > 0, "pySIPFENN did not generate any features.")

    @pytest.mark.order(3)
    def test_get_alignn_properties(self) -> None:
        """Test running ALIGNN models."""
        self.cas.get_alignn_properties()
        self.assertTrue(len(self.cas.prototypes) > 0, "ALIGNN did not generate any properties.")

    @pytest.mark.order(4)
    def test_get_mpdd_convexhull(self) -> None:
        """Test getting MPDD convex hull stability."""
        self.cas.get_mpdd_convexhull()
        self.assertTrue(all('MPDD_Stability' in prototype for prototype in self.cas.prototypes), "MPDD convex hull stability not calculated for all prototypes.")

    @pytest.mark.order(5)
    def test_filter_prototypes(self) -> None:
        """Test filtering prototypes based on formation energy."""
        self.cas.filter_prototypes(max_prototypes=10)
        self.assertTrue(len(self.cas.prototypes) <= 10, "Filtering did not reduce the number of prototypes correctly.")

    @pytest.mark.order(6)
    def test_cluster_prototypes_KMeans(self) -> None:
        """Test clustering prototypes using KMeans."""
        self.cas.top_candidates = []
        self.cas.cluster_prototypes_KMeans()
        self.assertTrue(len(self.cas.top_candidates) > 0, "KMeans clustering did not generate any top candidates.")

    @pytest.mark.order(7)
    def test_cluster_prototypes_DBSCAN(self) -> None:
        """Test clustering prototypes using DBSCAN."""
        self.cas.top_candidates = []
        self.cas.cluster_prototypes_DBSCAN()
        self.assertTrue(len(self.cas.top_candidates) > 0, "DBSCAN clustering did not generate any top candidates.")

    @pytest.mark.order(8)
    def test_export_table(self) -> None:
        """Test exporting prototype data to CSV."""
        self.cas.export_table()
        self.assertTrue(os.path.exists("all-prototypes.csv"), "Exported CSV file not found.")
        self.assertTrue(os.path.exists("top-candidates.csv"), "Exported CSV file not found.")
        os.remove("all-prototypes.csv")
        os.remove("top-candidates.csv")

    @pytest.mark.order(9)
    def test_plot_barchart(self) -> None:
        """Test plotting bar charts."""
        self.cas.plot_barchart(save=True)
        self.assertTrue(os.path.exists("barchart_clusterPrototypes.html"), "Bar chart HTML file not found.")
        os.remove("barchart_clusterPrototypes.html")

    @pytest.mark.order(10)
    def test_plot_2D(self) -> None:
        """Test plotting 2D scatter plots."""
        self.cas.plot_2D(save=True)
        self.assertTrue(os.path.exists("2D_clusterPrototypes.html"), "2D scatter plot HTML file not found.")
        os.remove("2D_clusterPrototypes.html")

    @pytest.mark.order(11)
    def test_plot_3D(self) -> None:
        """Test plotting 3D scatter plots."""
        self.cas.plot_3D(save=True)
        self.assertTrue(os.path.exists("3D_clusterPrototypes.html"), "3D scatter plot HTML file not found.")
        os.remove("3D_clusterPrototypes.html")

if __name__ == '__main__':
    unittest.main()