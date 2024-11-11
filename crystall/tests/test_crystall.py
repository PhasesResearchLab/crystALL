import unittest
import os
import shutil

import crystall

from pymatgen.core import Structure, Composition
from pymongo import MongoClient

class TestCrystALLSearch(unittest.TestCase):
    """Test the functionalities of the CrystALLSearch class."""

    def setUp(self) -> None:
        """Set up the test environment."""
        self.client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=100)
        self.db = self.client["MPDD"]
        self.targetFormula = "Ni26Ru8"
        self.cas = crystall.CrystALLSearch(self.targetFormula)
        self.cas.connect_MPDD(self.db)

    def test_generate_local(self) -> None:
        """Test the generation of prototypes from a local directory."""
        local_path = "tests/testFiles/local_prototypes"
        os.makedirs(local_path, exist_ok=True)
        with open(os.path.join(local_path, "test.POSCAR"), "w") as f:
            f.write("POSCAR content")
        self.cas.generate_local(local_path)
        self.assertTrue(os.path.exists(self.cas.prototypePath), "Local prototype path does not exist.")
        shutil.rmtree(local_path)

    def test_run_pysipfenn(self) -> None:
        """Test running pySIPFENN models."""
        self.cas.generate_MPDD()
        self.cas.download_models()
        self.cas.run_pysipfenn()
        self.assertTrue(len(self.cas.ks2022) > 0, "pySIPFENN did not generate any features.")

    def test_get_alignn_properties(self) -> None:
        """Test running ALIGNN models."""
        self.cas.generate_MPDD()
        self.cas.download_models()
        self.cas.get_alignn_properties()
        self.assertTrue(len(self.cas.prototypes) > 0, "ALIGNN did not generate any properties.")

    def test_cluster_prototypes_KMeans(self) -> None:
        """Test clustering prototypes using KMeans."""
        self.cas.generate_MPDD()
        self.cas.download_models()
        self.cas.run_pysipfenn()
        self.cas.cluster_prototypes_KMeans()
        self.assertTrue(len(self.cas.topCandidates) > 0, "KMeans clustering did not generate any top candidates.")

    def test_cluster_prototypes_DBSCAN(self) -> None:
        """Test clustering prototypes using DBSCAN."""
        self.cas.generate_MPDD()
        self.cas.download_models()
        self.cas.run_pysipfenn()
        self.cas.cluster_prototypes_DBSCAN()
        self.assertTrue(len(self.cas.topCandidates) > 0, "DBSCAN clustering did not generate any top candidates.")

    def test_filter_prototypes(self) -> None:
        """Test filtering prototypes based on formation energy."""
        self.cas.generate_MPDD()
        self.cas.download_models()
        self.cas.run_pysipfenn()
        self.cas.filter_prototypes(max_prototypes=10)
        self.assertTrue(len(self.cas.prototypes) <= 10, "Filtering did not reduce the number of prototypes correctly.")

    def test_export_table(self) -> None:
        """Test exporting prototype data to CSV."""
        self.cas.generate_MPDD()
        self.cas.download_models()
        self.cas.run_pysipfenn()
        self.cas.export_table()
        self.assertTrue(os.path.exists("allPrototypes.csv"), "Exported CSV file not found.")
        os.remove("allPrototypes.csv")

    def test_plot_barchart(self) -> None:
        """Test plotting bar charts."""
        self.cas.generate_MPDD()
        self.cas.download_models()
        self.cas.run_pysipfenn()
        self.cas.cluster_prototypes_KMeans()
        self.cas.plot_barchart(save=True)
        self.assertTrue(os.path.exists("barchart_clusterPrototypes.html"), "Bar chart HTML file not found.")
        os.remove("barchart_clusterPrototypes.html")

    def test_plot_2D(self) -> None:
        """Test plotting 2D scatter plots."""
        self.cas.generate_MPDD()
        self.cas.download_models()
        self.cas.run_pysipfenn()
        self.cas.cluster_prototypes_KMeans()
        self.cas.plot_2D(save=True)
        self.assertTrue(os.path.exists("2D_clusterPrototypes.html"), "2D scatter plot HTML file not found.")
        os.remove("2D_clusterPrototypes.html")

    def test_plot_3D(self) -> None:
        """Test plotting 3D scatter plots."""
        self.cas.generate_MPDD()
        self.cas.download_models()
        self.cas.run_pysipfenn()
        self.cas.cluster_prototypes_KMeans()
        self.cas.plot_3D(save=True)
        self.assertTrue(os.path.exists("3D_clusterPrototypes.html"), "3D scatter plot HTML file not found.")
        os.remove("3D_clusterPrototypes.html")

    def test_get_mpdd_convexhull(self) -> None:
        """Test getting MPDD convex hull stability."""
        self.cas.generate_MPDD()
        self.cas.download_models()
        self.cas.run_pysipfenn()
        self.cas.get_mpdd_convexhull()
        self.assertTrue(all('MPDD_Stability' in prototype for prototype in self.cas.prototypes), "MPDD convex hull stability not calculated for all prototypes.")

if __name__ == '__main__':
    unittest.main()