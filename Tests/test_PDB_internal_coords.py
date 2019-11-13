# Copyright 2017 by Bernhard Thiel.  All rights reserved.
# This file is part of the Biopython distribution and governed by your
# choice of the "Biopython License Agreement" or the "BSD 3-Clause License".
# Please see the LICENSE file that should have been included as part of this
# package.

"""Unit tests for PARTS of the parse_pdb_header module of Bio.PDB."""


import unittest
import re

try:
    import numpy  # noqa F401
except ImportError:
    from Bio import MissingPythonDependencyError

    raise MissingPythonDependencyError("Install NumPy if you want to use Bio.PDB.")

from Bio.PDB.ic_rebuild import structure_rebuild_test
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio._py3k import StringIO
from Bio.PDB.SCADIO import write_SCAD
from Bio.File import as_handle


class Rebuild(unittest.TestCase):
    """Read PDB and mmCIF structures, convert to/from internal coordinates."""

    PDB_parser = PDBParser(PERMISSIVE=True, QUIET=True)
    CIF_parser = MMCIFParser(QUIET=True)
    pdb_1LCD = PDB_parser.get_structure("1LCD", "PDB/1LCD.pdb")
    cif_3JQH = CIF_parser.get_structure("3JQH", "PDB/3JQH.cif")

    def test_rebuild_multichain_missing_residues(self):
        """Convert multichain protein to internal coordinates and back."""
        # 1lcd has missing residues
        r = structure_rebuild_test(self.pdb_1LCD, False)
        # print(r)
        self.assertEqual(r["residues"], 153)
        self.assertEqual(r["rCount"], 360)
        self.assertEqual(r["rMatchCount"], 360)
        self.assertEqual(r["aCount"], 1491)
        self.assertEqual(r["disAtmCount"], 0)
        self.assertEqual(r["aCoordMatchCount"], 1491)
        self.assertEqual(len(r["chains"]), 3)
        self.assertTrue(r["pass"])

    def test_rebuild_disordered_atoms_residues(self):
        """Convert disordered protein to internal coordinates and back."""
        # 3jqh has both disordered residues
        # and disordered atoms in ordered residues
        r = structure_rebuild_test(self.cif_3JQH, False)
        # print(r)
        self.assertEqual(r["residues"], 26)
        self.assertEqual(r["rCount"], 47)
        self.assertEqual(r["rMatchCount"], 47)
        self.assertEqual(r["aCount"], 217)
        self.assertEqual(r["disAtmCount"], 50)
        self.assertEqual(r["aCoordMatchCount"], 217)
        self.assertEqual(len(r["chains"]), 1)
        self.assertTrue(r["pass"])

    def test_write_SCAD(self):
        sf = StringIO()
        write_SCAD(
            self.cif_3JQH, sf, 10.0, pdbid="3jqh", backboneOnly=True, includeCode=False
        )
        sf.seek(0)
        next_one = False
        with as_handle(sf, mode="r") as handle:
            for aline in handle.readlines():
                if "// (1_P_N_A_0.83, 1_P_CA_A_0.83, 1_P_C_A_0.83)" in aline:
                    m = re.search(r"\[\s+(\d+\.\d+)\,", aline)
                    if m:
                        # test correctly scaled altloc residue/atom bond length
                        self.assertAlmostEqual(float(m.group(1)), 14.65214)
                elif '[ 25, "23L",' in aline:
                    next_one = True
                elif next_one:
                    next_one = False
                    # test last residue transform looks roughly correct
                    # some differences due to sorting issues on different python
                    # versions
                    target = [-335.294, -24.402, -15.566, 1.0]
                    ms = re.findall(r"\s+(-?\d+\.\d+)\s+\]", aline)
                    if ms:
                        for i in range(0, 3):
                            self.assertAlmostEqual(float(ms[i]), target[i], places=0)


if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2)
    unittest.main(testRunner=runner)
