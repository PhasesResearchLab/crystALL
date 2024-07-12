import unittest
import pytest
import os
import shutil

import crystall

from pymatgen.core import Structure, Composition

class TestLocal(unittest.TestCase):
    """Test the local runtime of crystALL based on the ``testFiles/localTest`` directory with A8B22 sigma phase structure candidates."""

    def setUp(self) -> None:
        """Set targets and ensure that the output directories ``localTest-A4B11`` and ``localTest-Ni22Ru8`` are not present."""
        self.targetFormula = "Ni22Ru8"
        self.prototypePath = "tests/testFiles/localTest"
        
        with self.subTest("Check the existence of the output directories"):
            assert not os.path.exists("localTest-Ni22Ru8"), "localTest-Ni22Ru8 directory is present and results may be overwritten if the test proceeds."
            assert not os.path.exists("localTest-A4B11"), "localTest-A4B11 directory is present and results may be overwritten if the test proceeds."

        with self.subTest("Runtime attempt"):
            crystall.run_local(
                targetFormula=self.targetFormula,
                prototypePath=self.prototypePath,
            )

    def test_file_generation(self) -> None:
        self.assertTrue(os.path.exists("localTest-Ni22Ru8"), "localTest-Ni22Ru8 directory was not created.")
        self.assertTrue(os.path.exists("localTest-A4B11"), "localTest-A4B11 directory was not created.")

        # Check the number of files in the directories, should be 4 in each. If 0, then no results were generated. If 5, the filtering was not applied correctly.
        with self.subTest("Check the number of files in the Ni22Ru8"):
            n1 = len(os.listdir("localTest-Ni22Ru8"))
            if n1 == 0:
                raise RuntimeError("No results were generated in localTest-Ni22Ru8.")
            elif n1 == 5:
                raise RuntimeError("The filtering was not applied correctly in localTest-Ni22Ru8.")
            self.assertEqual(n1, 4, "The number of results in localTest-Ni22Ru8 is incorrect.")

        with self.subTest("Check the number of files in the A4B11"):
            n2 = len(os.listdir("localTest-A4B11"))
            if n2 == 0:
                raise RuntimeError("No results were generated in localTest-A4B11.")
            elif n2 == 5:
                raise RuntimeError("The filtering was not applied correctly in localTest-A4B11.")
            self.assertEqual(n2, 4, "The number of results in localTest-A4B11 is incorrect.")

    def test_filtering_didnt_modify(self) -> None:
        """For each file in the localTest-A4B11 directory (filtering-only), verifies that it was present in the starting test files set based on either hash matching or Structure matching."""
        sourceContents = set()
        sourceStructures = list()
        for file in os.listdir(self.prototypePath):
            with open(self.prototypePath+"/"+file, "r") as f:
                contents = f.read()
                sourceContents.add(hash(contents))
                sourceStructures.append(Structure.from_file(self.prototypePath+"/"+file))
        
        for file in os.listdir("localTest-A4B11"):
            with self.subTest("Check the presence of "+file+" in the prototype files"):
                with open("localTest-A4B11/"+file, "r") as f:
                    contents = f.read()
                    if hash(contents) in sourceContents:
                        continue
                    else:
                        structure = Structure.from_file("localTest-A4B11/"+file)
                        for s in sourceStructures:
                            if s == structure:
                                break
                        else:
                            raise RuntimeError(file+" was not present in the starting test files set based on neither hash matching nor Structure matching.")
                        
    def test_substituted_stoichiometry(self) -> None:
        """For each generated candidate in the localTest-Ni22Ru8 directory, verifies that the stoichiometry is correct."""
        for file in os.listdir("localTest-Ni22Ru8"):
            with self.subTest("Check the stoichiometry of "+file):
                structure = Structure.from_file("localTest-Ni22Ru8/"+file)
                self.assertEqual(structure.composition, Composition(self.targetFormula), "The stoichiometry of "+file+" is incorrect.")


    def tearDown(self) -> None:
        """Clean up the output directories after the test. Even if not empty."""
        shutil.rmtree("localTest-Ni22Ru8")
        shutil.rmtree("localTest-A4B11")