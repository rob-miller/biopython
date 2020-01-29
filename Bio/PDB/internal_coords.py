# Copyright 2019 by Robert T. Miller.  All rights reserved.
# This file is part of the Biopython distribution and governed by your
# choice of the "Biopython License Agreement" or the "BSD 3-Clause License".
# Please see the LICENSE file that should have been included as part of this
# package.

"""Classes to support internal coordinates for protein structures.

Internal coordinates comprise Psi, Phi and Omega dihedral angles along the
protein backbone, Chi angles along the sidechains, and all 3-atom angles and
bond lengths comprising a protein chain.  These routines can compute internal
coordinates from atom XYZ coordinates, and compute atom XYZ coordinates from
internal coordinates.

Internal coordinates are defined on sequences of atoms which span
residues or follow accepted nomenclature along sidechains.  To manage these
sequences and support Biopython's disorder mechanisms, AtomKey specifiers are
implemented to capture residue, atom and variant identification in a single
object.  A Hedron object is specified as three sequential AtomKeys, comprising
two bond lengths and the bond angle between them.  A Dihedron consists of four
sequential AtomKeys, linking two Hedra with a dihedral angle between them.

A Protein Internal Coordinate (.pic) file format is defined to capture
sufficient detail to reproduce a PDB file from chain starting coordinates
(first residue N, Ca, C XYZ coordinates) and remaining internal coordinates.
These files are used internally to verify that a given structure can be
regenerated from its internal coordinates.

Internal coordinates may also be exported as OpenSCAD data arrays for
generating 3D printed protein models.  OpenSCAD software is provided as
proof-of-concept for generating such models.

The following classes comprise the core functionality for processing internal
coordinates and are sufficiently related and coupled to place them together in
this module:

IC_Chain: Extends Biopython Chain on .internal_coords attribute.
    Manages connected sequence of residues and chain breaks; methods generally
    apply IC_Residue methods along chain.

IC_Residue: Extends for Biopython Residue on .internal_coords attribute.
    Most control and methods of interest are in this class, see API.

Dihedron: four joined atoms forming a dihedral angle.
    Dihedral angle, homogeneous atom coordinates in local coordinate space,
    references to relevant Hedra and IC_Residue.  Methods to compute
    residue dihedral angles, bond angles and bond lengths.

Hedron: three joined atoms forming a plane.
    Contains homogeneous atom coordinates in local coordinate space as well as
    bond lengths and angle between them.

Edron: base class for Hedron and Dihedron classes.
    Tuple of AtomKeys comprising child, string ID, mainchain membership boolean
    and other routines common for both Hedra and Dihedra.  Implements rich
    comparison.

AtomKey: keys (dictionary and string) for referencing atom sequences.
    Capture residue and disorder/occupancy information, provides a
    no-whitespace key for .pic files, and implements rich comparison.

Custom exception classes: HedronMatchError and MissingAtomError
"""

import re
from collections import deque, namedtuple

try:
    import numpy  # type: ignore
except ImportError:
    from Bio import MissingPythonDependencyError

    raise MissingPythonDependencyError(
        "Install NumPy to build proteins from internal coordinates."
    )

from Bio.PDB.Atom import Atom, DisorderedAtom
from Bio.PDB.Polypeptide import three_to_one

from Bio.PDB.vectors import homog_rot_mtx, coord_space, calc_dihedral, Vector
from Bio.PDB.ic_data import ic_data_backbone, ic_data_sidechains
from Bio.PDB.ic_data import ic_data_sidechain_extras, residue_atom_bond_state

# for type checking only
from typing import (
    List,
    Dict,
    Set,
    TextIO,
    Union,
    Tuple,
    cast,
    Type,
    TYPE_CHECKING,
    Optional,
)

if TYPE_CHECKING:
    from Bio.PDB.Residue import Residue
    from Bio.PDB.Chain import Chain

HKT = Tuple["AtomKey", "AtomKey", "AtomKey"]  # Hedron key tuple
DKT = Tuple["AtomKey", "AtomKey", "AtomKey", "AtomKey"]  # Dihedron Key Tuple
EKT = Union[HKT, DKT]  # Edron Key Tuple
BKT = Tuple["AtomKey", "AtomKey"]  # Bond Key Tuple

HACS = Tuple[numpy.array, numpy.array, numpy.array]  # Hedron Atom Coord Set
DACS = Tuple[
    numpy.array, numpy.array, numpy.array, numpy.array
]  # Dihedron Atom Coord Set


class IC_Chain:
    """Class to extend Biopython Chain with internal coordinate data.

    Attributes
    ----------
    chain: biopython Chain object reference
        The Chain object this extends
    initNCaC: AtomKey indexed dictionary of N, Ca, C atom coordinates.
        NCaCKeys start chain segments (first residue or after chain break).
        These 3 atoms define the coordinate space for a contiguous chain segment,
        as initially specified by PDB or mmCIF file.
    MaxPeptideBond: Class attribute to detect chain breaks.
        Override for fully contiguous chains with some very long bonds - e.g.
        for 3D printing (OpennSCAD output) a structure with fully disordered
        (missing) residues.
    ordered_aa_ic_list: list of IC_Residue objects
        IC_Residue objects ic algorithms can process (e.g. no waters)

    Methods
    -------
    internal_to_atom_coordinates(verbose, start, fin)
        Process ic data to Residue/Atom coordinates; calls assemble_residues()
        followed by coords_to_structure()
    assemble_residues(verbose, start, fin)
        Generate IC_Residue atom coords from internal coordinates
    coords_to_structure()
        update Biopython Residue.Atom coords from IC_Residue coords for all
        Residues with IC_Residue attributes
    atom_to_internal_coordinates(verbose, allBonds)
        Calculate dihedrals, angles, bond lengths (internal coordinates) for
        Atom data
    link_residues()
        Call link_dihedra() on each IC_Residue (needs rprev, rnext set)
    render_dihedra()
        Call render_hedra() and render_dihedra() on each IC_Residue
    set_residues()
        Add .internal_coord attribute for all Residues in parent Chain, populate
        ordered_aa_ic_list, set IC_Residue rprev, rnext or initNCaC coordinates
    write_SCAD()
        Write OpenSCAD matrices for internal coordinate data comprising chain

    """

    MaxPeptideBond = 1.4  # larger C-N distance than this is chain break

    def __init__(self, parent: "Chain") -> None:
        """Initialize IC_Chain object, with or without residue/Atom data.

        :param parent: Biopython Chain object
            Chain object this extends
        """
        # type hinting parent as Chain leads to import cycle
        self.chain = parent
        self.ordered_aa_ic_list: List[IC_Residue] = []
        self.initNCaC: Dict[Tuple[str], Dict["AtomKey", numpy.array]] = {}
        self.sqMaxPeptideBond = IC_Chain.MaxPeptideBond * IC_Chain.MaxPeptideBond
        self.set_residues()  # no effect if no residues loaded

    def _peptide_check(self, prev: "Residue", curr: "Residue") -> bool:
        if 0 == len(curr.child_dict):
            # curr residue with no atoms => reading pic file, no break
            return True
        if (0 != len(curr.child_dict)) and (0 == len(prev.child_dict)):
            # prev residue with no atoms, curr has atoms => reading pic file,
            # have break
            return False

        # both biopython Resdiues have Atoms, so check distance
        Natom = curr.child_dict.get("N", None)
        pCatom = prev.child_dict.get("C", None)
        if Natom is None or pCatom is None:
            return False

        # confirm previous residue has all backbone atoms
        pCAatom = prev.child_dict.get("CA", None)
        pNatom = prev.child_dict.get("N", None)
        if pNatom is None or pCAatom is None:
            return False

        if Natom.is_disordered():
            Natom = Natom.selected_child
        if pCatom.is_disordered():
            pCatom = pCatom.selected_child
        diff = curr["N"].coord - prev["C"].coord
        sum = 0
        for axis in diff:
            if axis > self.MaxPeptideBond:
                return False
            sum += axis * axis
        if sum > self.sqMaxPeptideBond:
            return False
        return True

    def _add_residue(self, res: "Residue", last_res: List, last_ord_res: List) -> bool:
        """Set rprev, rnext, determine chain break."""
        if not res.internal_coord:
            res.internal_coord = IC_Residue(res)
        if (
            0 < len(last_res)
            and last_ord_res == last_res
            and self._peptide_check(last_ord_res[0].residue, res)
        ):
            # no chain break
            for prev in last_ord_res:
                prev.rnext.append(res.internal_coord)
                res.internal_coord.rprev.append(prev)
            return True
        elif all(atm in res.child_dict for atm in ("N", "CA", "C")):
            # chain break, save coords for restart
            initNCaC: Dict["AtomKey", numpy.array] = {}
            ric = res.internal_coord
            for atm in ("N", "CA", "C"):
                bpAtm = res.child_dict[atm]
                if bpAtm.is_disordered():
                    for altAtom in bpAtm.child_dict.values():
                        ak = AtomKey(ric, altAtom)
                        initNCaC[ak] = IC_Residue.atm241(altAtom.coord)
                else:
                    ak = AtomKey(ric, bpAtm)
                    initNCaC[ak] = IC_Residue.atm241(bpAtm.coord)
            self.initNCaC[ric.rbase] = initNCaC
            return True
        elif (
            0 == len(res.child_list)
            and self.chain.child_list[0].id == res.id
            and res.internal_coord.is20AA
        ):
            # this is first residue, no atoms at all, is std amino acid
            # conclude reading pic file with no N-Ca-C coords
            return True
        # chain break but do not have N, Ca, C coords to restart from
        return False

    def set_residues(self) -> None:
        """Initialize internal_coord data for loaded Residues.

        Add IC_Residue as .internal_coord attribute for each Residue in parent
        Chain; populate ordered_aa_ic_list with IC_Residue references for residues
        which can be built (amino acids and some hetatms); set rprev and rnext
        on each sequential IC_Residue, populate initNCaC at start and after
        chain breaks.
        """
        # ndx = 0
        last_res: List["IC_Residue"] = []
        last_ord_res: List["IC_Residue"] = []
        for res in self.chain.get_residues():
            # select only not hetero or accepted hetero
            if res.id[0] == " " or res.id[0] in IC_Residue.accept_resnames:
                this_res: List["IC_Residue"] = []
                if 2 == res.is_disordered():
                    # print('disordered res:', res.is_disordered(), res)
                    for r in res.child_dict.values():
                        if self._add_residue(r, last_res, last_ord_res):
                            this_res.append(r.internal_coord)
                else:
                    if self._add_residue(res, last_res, last_ord_res):
                        this_res.append(res.internal_coord)

                if 0 < len(this_res):
                    self.ordered_aa_ic_list.extend(this_res)
                    last_ord_res = this_res
                last_res = this_res

    def link_residues(self) -> None:
        """link_dihedra() for each IC_Residue; needs rprev, rnext set."""
        for ric in self.ordered_aa_ic_list:
            ric.link_dihedra()

    def render_dihedra(self) -> None:
        """Set dihedron local coords for each IC_Residue."""
        for ric in self.ordered_aa_ic_list:
            ric.render_dihedra()

    def assemble_residues(
        self,
        verbose: bool = False,
        start: Optional[int] = None,
        fin: Optional[int] = None,
    ) -> None:
        """Generate IC_Residue atom coords from internal coordinates.

        Filter positions between start and fin if set, find appropriate start
        coordinates for each residue and pass to IC_Residue.assemble()

        :param verbose bool: default False
            describe runtime problems
        :param: start, fin lists
            sequence position, insert code for begin, end of subregion to
            process

        """
        for ric in self.ordered_aa_ic_list:
            if not hasattr(ric, "NCaCKey"):
                if verbose:
                    print(
                        f"no assembly for {str(ric)} due to missing N, Ca and/or C atoms"
                    )
                continue
            respos = ric.residue.id[1]
            if start and start > respos:
                continue
            if fin and fin < respos:
                continue

            ric.atom_coords = cast(
                Dict[AtomKey, numpy.array], ric.assemble(verbose=verbose)
            )
            ric.ak_set = set(ric.atom_coords.keys())

    def coords_to_structure(self) -> None:
        """Promote all ic atom_coords to Biopython Residue/Atom coords.

        IC atom_coords are homogeneous [4][1], Biopython atom coords are XYZ [3].
        """
        self.ndx = 0
        for res in self.chain.get_residues():
            if 2 == res.is_disordered():
                for r in res.child_dict.values():
                    if r.internal_coord:
                        r.internal_coord.coords_to_residue()
            elif res.internal_coord:
                res.internal_coord.coords_to_residue()

    def internal_to_atom_coordinates(
        self,
        verbose: bool = False,
        start: Optional[int] = None,
        fin: Optional[int] = None,
    ) -> None:
        """Complete process, IC data to Residue/Atom coords.
        
        :param: start, fin lists
            sequence position, insert code for begin, end of subregion to
            process
        :param verbose bool: default False
            describe runtime problems

        """
        self.assemble_residues(
            verbose=verbose, start=start, fin=fin
        )  # internal to XYZ coordinates
        self.coords_to_structure()  # promote to BioPython Residue/Atom

    def atom_to_internal_coordinates(
        self, verbose: bool = False, allBonds: bool = False
    ) -> None:
        """Calculate dihedrals, angles, bond lengths for Atom data."""
        for ric in self.ordered_aa_ic_list:
            ric.atom_to_internal_coordinates(verbose=verbose, allBonds=allBonds)

    @staticmethod
    def _write_mtx(fp: TextIO, mtx: numpy.array) -> None:
        fp.write("[ ")
        rowsStarted = False
        for row in mtx:
            if rowsStarted:
                fp.write(", [ ")
            else:
                fp.write("[ ")
                rowsStarted = True
            colsStarted = False
            for col in row:
                if colsStarted:
                    fp.write(", " + str(col))
                else:
                    fp.write(str(col))
                    colsStarted = True
            fp.write(" ]")  # close row
        fp.write(" ]")

    @staticmethod
    def _writeSCAD_dihed(
        fp: TextIO, d: "Dihedron", transformations, hedraNdx: Dict, hedraSet: Set[str]
    ) -> None:
        fp.write(
            "[ {:9.5f}, {}, {}, {}, ".format(
                d.dihedral1,
                hedraNdx[d.h1key],
                hedraNdx[d.h2key],
                (1 if d.reverse else 0),
            )
        )
        fp.write(
            "{}, {}, ".format(
                (0 if d.h1key in hedraSet else 1), (0 if d.h2key in hedraSet else 1)
            )
        )
        fp.write(
            "    // {} [ {} -- {} ] {}\n".format(
                d.id, d.hedron1.id, d.hedron2.id, ("reversed" if d.reverse else "")
            )
        )
        fp.write("        ")
        mtx = transformations[d.id3]
        IC_Chain._write_mtx(fp, mtx)
        fp.write(" ]")  # close residue array of dihedra entry

    def write_SCAD(self, fp: TextIO, backboneOnly: bool) -> None:
        """Write self to file fp as OpenSCAD data matrices.

        Works with write_SCAD() and embedded OpenSCAD routines in SCADIO.py.
        The OpenSCAD code explicitly creates spheres and cylinders to
        represent atoms and bonds in a 3D model.  Options are available
        to support rotatable bonds and magnetic hydrogen bonds.

        Matrices are written to link, enumerate and describe residues,
        dihedra, hedra, and chains, mirroring contents of the relevant IC_*
        data structures.

        The OpenSCAD matrix of hedra has additional information as follows:

        * the atom and bond state (single, double, resonance) are logged
          so that covalent radii may be used for atom spheres in the 3D models

        * bonds and atoms are tracked so that each is only created once

        * bond options for rotation and magnet holders for hydrogen bonds
          may be specified

        Note the application of IC_Chain attribute MaxPeptideBond: missing
        residues may be linked (joining chain segments with arbitrarily long
        bonds) by setting this to a large value.

        All ALTLOC (disordered) residues and atoms are written to the output model.
        """
        fp.write('   "{}", // chain id\n'.format(self.chain.id))

        # generate dict for all hedra to eliminate redundant references
        hedra = {}
        for ric in self.ordered_aa_ic_list:
            respos, resicode = ric.residue.id[1:]
            for k, h in ric.hedra.items():
                hedra[k] = h
        atomSet: Set[AtomKey] = set()
        bondDict: Dict = {}  # set()
        hedraSet: Set[str] = set()
        ndx = 0
        hedraNdx = {}

        for hk in sorted(hedra):
            hedraNdx[hk] = ndx
            ndx += 1

        # write residue dihedra table

        fp.write("   [  // residue array of dihedra")
        resNdx = {}
        dihedraNdx = {}
        ndx = 0
        chnStarted = False
        for ric in self.ordered_aa_ic_list:
            resNdx[ric] = ndx
            if chnStarted:
                fp.write("\n     ],")
            else:
                chnStarted = True
            fp.write(
                "\n     [ // "
                + str(ndx)
                + " : "
                + str(ric.residue.id)
                + " "
                + ric.lc
                + " backbone\n"
            )
            ndx += 1
            # assemble with no start position, return transform matrices
            transformations = ric.assemble(transforms=True, resetLocation=True)
            ndx2 = 0
            for i in range(1 if backboneOnly else 2):
                if i == 1:
                    fp.write(
                        ",\n       // "
                        + str(ric.residue.id)
                        + " "
                        + ric.lc
                        + " sidechain\n"
                    )
                started = False
                for dk, d in sorted(ric.dihedra.items()):
                    if d.h2key in hedraNdx and (
                        (i == 0 and d.is_backbone()) or (i == 1 and not d.is_backbone())
                    ):
                        if started:
                            fp.write(",\n")
                        else:
                            started = True
                        fp.write("      ")
                        IC_Chain._writeSCAD_dihed(
                            fp, d, transformations, hedraNdx, hedraSet
                        )
                        dihedraNdx[dk] = ndx2
                        hedraSet.add(str(d.h1key))
                        hedraSet.add(str(d.h2key))
                        ndx2 += 1
        fp.write("   ],")  # end of residue entry dihedra table
        fp.write("\n  ],\n")  # end of all dihedra table

        # write hedra table

        fp.write("   [  //hedra\n")
        for hk in sorted(hedra):
            hed = hedra[hk]
            fp.write("     [ ")
            fp.write("{:9.5f}, {:9.5f}, {:9.5f}".format(hed.len1, hed.angle2, hed.len3))
            atom_str = ""  # atom and bond state
            atom_done_str = ""  # create each only once
            akndx = 0
            for ak in hed.aks:
                atm = ak.akl[ak.fields.atm]
                res = ak.akl[ak.fields.resname]
                # try first for generic backbone/Cbeta atoms
                ab_state_res = residue_atom_bond_state["X"]
                ab_state = ab_state_res.get(atm, None)
                if "H" == atm[0]:
                    ab_state = "Hsb"
                if ab_state is None:
                    # not found above, must be sidechain atom
                    ab_state_res = residue_atom_bond_state.get(res, None)
                    if ab_state_res is not None:
                        ab_state = ab_state_res.get(atm, "")
                    else:
                        ab_state = ""
                atom_str += ', "' + ab_state + '"'

                if ak in atomSet:
                    atom_done_str += ", 0"
                elif hk in hedraSet:
                    if (
                        hasattr(hed, "flex_female_1") or hasattr(hed, "flex_male_1")
                    ) and akndx != 2:
                        if akndx == 0:
                            atom_done_str += ", 0"
                        elif akndx == 1:
                            atom_done_str += ", 1"
                            atomSet.add(ak)
                    elif (
                        hasattr(hed, "flex_female_2") or hasattr(hed, "flex_male_2")
                    ) and akndx != 0:
                        if akndx == 2:
                            atom_done_str += ", 0"
                        elif akndx == 1:
                            atom_done_str += ", 1"
                            atomSet.add(ak)
                    else:
                        atom_done_str += ", 1"
                        atomSet.add(ak)
                else:
                    atom_done_str += ", 0"
                akndx += 1
            fp.write(atom_str)
            fp.write(atom_done_str)

            # specify bond options

            bond = []
            bond.append(hed.aks[0].id + "-" + hed.aks[1].id)
            bond.append(hed.aks[1].id + "-" + hed.aks[2].id)
            b0 = True
            for b in bond:
                wstr = ""
                if b in bondDict and bondDict[b] == "StdBond":
                    wstr = ", 0"
                elif hk in hedraSet:
                    bondType = "StdBond"
                    if b0:
                        if hasattr(hed, "flex_female_1"):
                            bondType = "FemaleJoinBond"
                        elif hasattr(hed, "flex_male_1"):
                            bondType = "MaleJoinBond"
                        elif hasattr(hed, "skinny_1"):
                            bondType = "SkinnyBond"
                        elif hasattr(hed, "hbond_1"):
                            bondType = "HBond"
                    else:
                        if hasattr(hed, "flex_female_2"):
                            bondType = "FemaleJoinBond"
                        elif hasattr(hed, "flex_male_2"):
                            bondType = "MaleJoinBond"
                        # elif hasattr(hed, 'skinny_2'):  # unused
                        #     bondType = 'SkinnyBond'
                        elif hasattr(hed, "hbond_2"):
                            bondType = "HBond"
                    if b in bondDict:
                        bondDict[b] = "StdBond"
                    else:
                        bondDict[b] = bondType
                    wstr = ", " + str(bondType)
                else:
                    wstr = ", 0"
                fp.write(wstr)
                b0 = False
            akl = hed.aks[0].akl
            fp.write(
                ', "'
                + akl[ak.fields.resname]
                + '", '
                + akl[ak.fields.respos]
                + ', "'
                + hed.dh_class
                + '"'
            )
            fp.write(" ], // " + str(hk) + "\n")
        fp.write("   ],\n")  # end of hedra table

        # write chain table

        fp.write("\n[  // chain - world transform for each residue\n")
        chnStarted = False
        for ric in self.ordered_aa_ic_list:
            # handle start / end
            for NCaCKey in sorted(ric.NCaCKey):
                if 0 < len(ric.rprev):
                    for rpr in ric.rprev:
                        acl = [rpr.atom_coords[ak] for ak in NCaCKey]
                        mt, mtr = coord_space(acl, True)
                else:
                    mtr = numpy.identity(4, dtype=numpy.float64)
                if chnStarted:
                    fp.write(",\n")
                else:
                    chnStarted = True
                fp.write("     [ " + str(resNdx[ric]) + ', "' + str(ric.residue.id[1]))
                fp.write(ric.lc + '", //' + str(NCaCKey) + "\n")
                IC_Chain._write_mtx(fp, mtr)
                fp.write(" ]")
        fp.write("\n   ]\n")


class IC_Residue(object):
    """Class to extend Biopython Residue with internal coordinate data.

    Attributes
    ----------
    residue: Biopython Residue object reference
        The Residue object this extends
    hedra: dict indexed by 3-tuples of AtomKeys
        Hedra forming this residue
    dihedra: dict indexed by 4-tuples of AtomKeys
        Dihedra forming (overlapping) this residue
    rprev, rnext: lists of IC_Residue objects
        References to adjacent (bonded, not missing, possibly disordered)
        residues in chain
    atom_coords: AtomKey indexed dict of numpy [4][1] arrays
        Local copy of atom homogeneous coordinates [4][1] for work
        distinct from Bopython Residue/Atom values
    alt_ids: list of char
        AltLoc IDs from PDB file
    bfactors: dict
        AtomKey indexed B-factors as read from PDB file
    NCaCKey: List of tuples of AtomKeys
        List of tuples of N, Ca, C backbone atom AtomKeys; usually only 1
        but more if backbone altlocs
    is20AA: bool
        True if residue is one of 20 standard amino acids, based on
        Residue resname
    accept_atoms: tuple
        list of PDB atom names to use when generatiing internal coordinates.
        Default is:

        `accept_atoms = accept_mainchain + accept_hydrogens`

        to exclude hydrogens in internal coordinates and generated PDB files,
        override as:

        `IC_Residue.accept_atoms = IC_Residue.accept_mainchain`

        to get only backbone atoms plus amide proton, use:

        `IC_Residue.accept_atoms = IC_Residue.accept_mainchain + ('H',)`

        to convert D atoms to H, set `AtomKey.d2h = True` and use:

        `IC_Residue.accept_atoms = accept_mainchain + accept_hydrogens + accept_deuteriums`

        There is currently no option to output internal coordinates with D
        instead of H

    accept_resnames: tuple
        list of 3-letter residue names for HETATMs to accept when generating
        internal coordinates from atoms.  HETATM sidechain will be ignored, but normal
        backbone atoms (N, CA, C, O, CB) will be included.  Currently only
        CYG, YCM and UNK; override at your own risk.  To generate
        sidechain, add appropriate entries to ic_data_sidechains in
        ic_data.py and support in atom_to_internal_coordinates()
    gly_Cbeta: bool default False
        override class variable to True to generate internal coordinates for
        glycine CB atoms in atom_to_internal_coordinates().

        `IC_Residue.gly_Cbeta = True`

    scale: optional float
        used for OpenSCAD output to generate gly_Cbeta bond length

    Parameters (__init__)
    ---------------------
    parent: biopython Residue object this class extends
    NO_ALTLOC: bool default False
    Disable processing of ALTLOC atoms if True, use only selected atoms.


    Methods
    -------
    applyMtx()
        multiply all IC_Residue atom_cords by passed matrix
    assemble(atomCoordsIn, transforms, verbose)
        Compute atom coordinates for this residue from internal coordinates
    atm241(coord)
        Convert 1x3 cartesian coords to 4x1 homogeneous coords
    coords_to_residue()
        Convert homogeneous atom_coords to Biopython cartesian Atom coords
    atom_to_internal_coordinates(verbose, allBonds)
        Create hedra and dihedra for atom coordinates
    get_angle()
        Return angle for passed key
    get_length()
        Return bond length for specified pair
    link_dihedra()
        Link dihedra to this residue, form id3_dh_index
    load_PIC(edron)
        Process parsed (di-/h-)edron data from PIC file
    pick_angle()
        Find Hedron or Dihedron for passed key
    pick_length()
        Find hedra for passed AtomKey pair
    rak(atom info)
        Residue AtomKey - per residue AtomKey result cache
    render_dihedra()
        Call init_pos for each dihedron in dihedra
    render_hedra()
        Call init_pos for each hedron in hedra
    set_angle()
        Set angle for passed key (no position updates)
    set_length()
        Set bond length in all relevant hedra for specified pair
    write_PIC(pdbid, chainId, s)
        Generate PIC format strings for this residue

    """

    # add 3-letter residue name here for non-standard residues with
    # normal backbone.  CYG for test case 4LGY (1305 residue contiguous
    # chain)
    accept_resnames = ("CYG", "YCM", "UNK")

    def __init__(self, parent: "Residue", NO_ALTLOC: bool = False) -> None:
        """Initialize IC_Residue with parent Biopython Residue.

        :param parent: Biopython Residue object
            The Biopython Residue this object extends
        :param NO_ALTLOC: bool default False
            Option to disable processing altloc disordered atoms, use selected.
        """
        # NO_ALTLOC=True will turn off alotloc positions and just use selected
        self.residue = parent
        # self.ndx = ndx
        # dict of hedron objects indexed by hedron keys
        self.hedra: Dict[HKT, Hedron] = {}
        # dict of dihedron objects indexed by dihedron keys
        self.dihedra: Dict[DKT, Dihedron] = {}
        # map of dihedron key (first 3 atom keys) to dihedron
        # for all dihedra in Residue
        # set by Residue.link_dihedra()
        self.akc: Dict[str, AtomKey] = {}
        # cache of AtomKey results for rak()
        self.id3_dh_index: Dict[HKT, List[Dihedron]] = {}
        # set of AtomKeys involved in dihedra, used by split_akl
        self.ak_set: Set[AtomKey] = set()
        # reference to adjacent residues in chain
        self.rprev: List[IC_Residue] = []
        self.rnext: List[IC_Residue] = []
        # local copy, homogeneous coordinates for atoms, numpy [4][1]
        # generated from dihedra include some i+1 atoms
        # or initialised here from parent residue if loaded from coordinates
        self.atom_coords: Dict["AtomKey", numpy.array] = {}
        # bfactors copied from PDB file
        self.bfactors: Dict[str, float] = {}
        self.alt_ids: Union[List[str], None] = None if NO_ALTLOC else []
        #       if NO_ALTLOC:
        #           self.alt_ids = None
        ##       else:
        #           self.alt_ids: List[str] = []
        self.is20AA = True
        # rbase = position, insert code or none, resname (1 letter if in 20)
        rid = parent.id
        rbase = [rid[1], rid[2] if " " != rid[2] else None, parent.resname]
        try:
            rbase[2] = three_to_one(rbase[2]).upper()
        except KeyError:
            self.is20AA = False

        self.rbase = tuple(rbase)
        self.lc = rbase[2]

        if self.is20AA or rbase[2] in self.accept_resnames:
            # self.NCaCKey = []
            self.NCaCKey = [(self.rak("N"), self.rak("CA"), self.rak("C"))]

            for atom in parent.get_atoms():
                if hasattr(atom, "child_dict"):
                    if NO_ALTLOC:
                        self._add_atom(atom.selected_child)
                    else:
                        for atm in atom.child_dict.values():
                            self._add_atom(atm)
                else:
                    self._add_atom(atom)

            # print(self.atom_coords)

    def rak(self, atm: Union[str, Atom]) -> "AtomKey":
        """Cache calls to AtomKey for this residue."""
        # I have read this is the internal representation of Dict.get(), so
        # do it explicitly here so need not check result is None
        try:
            ak = self.akc[atm]
        except (KeyError):
            ak = self.akc[atm] = AtomKey(self, atm)
        return ak

    accept_mainchain = (
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "CG1",
        "OG1",
        "OG",
        "SG",
        "CG2",
        "CD",
        "CD1",
        "SD",
        "OD1",
        "ND1",
        "CD2",
        "ND2",
        "CE",
        "CE1",
        "NE",
        "OE1",
        "NE1",
        "CE2",
        "OE2",
        "NE2",
        "CE3",
        "CZ",
        "NZ",
        "CZ2",
        "CZ3",
        "OD2",
        "OH",
        "CH2",
        "OXT",
    )
    accept_hydrogens = (
        "H",
        "H1",
        "H2",
        "H3",
        "HA",
        "HA2",
        "HA3",
        "HB",
        "HB1",
        "HB2",
        "HB3",
        "HG2",
        "HG3",
        "HD2",
        "HD3",
        "HE2",
        "HE3",
        "HZ1",
        "HZ2",
        "HZ3",
        "HG11",
        "HG12",
        "HG13",
        "HG21",
        "HG22",
        "HG23",
        "HZ",
        "HD1",
        "HE1",
        "HD11",
        "HD12",
        "HD13",
        "HG",
        "HG1",
        "HD21",
        "HD22",
        "HD23",
        "NH1",
        "NH2",
        "HE",
        "HH11",
        "HH12",
        "HH21",
        "HH22",
        "HE21",
        "HE22",
        "HE2",
        "HH",
        "HH2",
    )
    accept_deuteriums = (
        "D",
        "D1",
        "D2",
        "D3",
        "DA",
        "DA2",
        "DA3",
        "DB",
        "DB1",
        "DB2",
        "DB3",
        "DG2",
        "DG3",
        "DD2",
        "DD3",
        "DE2",
        "DE3",
        "DZ1",
        "DZ2",
        "DZ3",
        "DG11",
        "DG12",
        "DG13",
        "DG21",
        "DG22",
        "DG23",
        "DZ",
        "DD1",
        "DE1",
        "DD11",
        "DD12",
        "DD13",
        "DG",
        "DG1",
        "DD21",
        "DD22",
        "DD23",
        "ND1",
        "ND2",
        "DE",
        "DH11",
        "DH12",
        "DH21",
        "DH22",
        "DE21",
        "DE22",
        "DE2",
        "DH",
        "DH2",
    )
    accept_atoms = accept_mainchain + accept_hydrogens

    gly_Cbeta = False

    @staticmethod
    def atm241(coord: numpy.array) -> numpy.array:
        """Convert 1x3 cartesian coordinates to 4x1 homogeneous coordinates."""
        arr41 = numpy.append(coord, [1])
        return numpy.array(arr41, dtype=numpy.float64)[numpy.newaxis].transpose()

    def _add_atom(self, atm: Atom) -> None:
        """Filter Biopython Atom with accept_atoms; set atom_coords, ak_set.

        Arbitrarily renames O' and O'' to O and OXT
        """
        if "O'" == atm.name:
            atm.name = "O"
        if "O''" == atm.name:
            atm.name = "OXT"

        if atm.name not in self.accept_atoms:
            # print('skip:', atm.name)
            return
        ak = self.rak(atm)  # passing Atom here not string
        self.atom_coords[ak] = IC_Residue.atm241(atm.coord)
        self.ak_set.add(ak)

    def __str__(self) -> str:
        """Print string is parent Residue ID."""
        return str(self.residue.full_id)

    def load_PIC(self, edron: Dict[str, str]) -> None:
        """Process parsed (di-/h-)edron data from PIC file.

        :param edron: parse dictionary from Edron.edron_re
        """

        if edron["a4"] is None:
            ek = (AtomKey(edron["a1"]), AtomKey(edron["a2"]), AtomKey(edron["a3"]))
            self.hedra[ek] = Hedron(ek, **edron)
        else:
            ek = (
                AtomKey(edron["a1"]),
                AtomKey(edron["a2"]),
                AtomKey(edron["a3"]),
                AtomKey(edron["a4"]),
            )
            self.dihedra[ek] = Dihedron(ek, **edron)

    def link_dihedra(self, verbose: bool = False) -> None:
        """Housekeeping after loading all residues and dihedra.

        - Link dihedra to this residue
        - form id3_dh_index
        - form ak_set
        - fix NCaCKey to be available AtomKeys
        """
        id3i: Dict[HKT, List[Dihedron]] = {}
        for dh in self.dihedra.values():
            dh.IC_Residue = self  # each dihedron can find its IC_Residue
            id3 = dh.id3
            if id3 not in id3i:
                id3i[id3] = []
            id3i[id3].append(dh)
            self.ak_set.update(id3)
        # map to find each dihedron from atom tokens 1-3
        self.id3_dh_index = id3i
        # more efficient to catch NCaC KeyError later, but fixing here
        # avoids having to test/find altloc problem in future code
        # newNCaCKey: List[Tuple["AtomKey", "AtomKey", "AtomKey"]] = []
        newNCaCKey: List[Tuple["AtomKey", ...]] = []
        """
        for tpl in sorted(self.NCaCKey):
            newNCaCKey.extend(self._split_akl(tpl))
        self.NCaCKey = tuple(newNCaCKey)
        """
        try:
            for tpl in sorted(self.NCaCKey):
                newNCaCKey.extend(self._split_akl(tpl))
            self.NCaCKey = cast(
                List[Tuple["AtomKey", "AtomKey", "AtomKey"]], newNCaCKey
            )
        except AttributeError:
            if verbose:
                print(
                    f"Missing N, Ca and/or C atoms for residue {str(self.residue)} chain {self.residue.parent.id}"
                )
            pass

        pass

    def render_dihedra(self) -> None:
        """Set hedron-space atom coordinates for each dihedron."""
        for d in self.dihedra.values():
            # if d.atoms_updated:   # sorry, not fully implemented
            d.init_pos()

    def set_flexible(self) -> None:
        """For OpenSCAD, mark N-CA and CA-C bonds to be flexible joints."""
        for h in self.hedra.values():
            if h.dh_class == "NCAC":
                setattr(h, "flex_female_1", True)
                setattr(h, "flex_female_2", True)
            elif h.dh_class.endswith("NCA"):
                setattr(h, "flex_male_2", True)
            elif h.dh_class.startswith("CAC") and h.aks[1].akl[3] == "C":
                setattr(h, "flex_male_1", True)
            elif h.dh_class == "CBCAC":
                setattr(h, "skinny_1", True)  # CA-CB bond interferes with flex join

    def set_hbond(self) -> None:
        """For OpenSCAD, mark H-N and C-O bonds to be hbonds (magnets)."""
        for h in self.hedra.values():
            if h.dh_class == "HNCA":
                setattr(h, "hbond_1", True)
            elif h.dh_class == "CACO":
                setattr(h, "hbond_2", True)

    def get_startpos(self) -> Dict["AtomKey", numpy.array]:
        """Find N-Ca-C coordinates to build this residue from."""
        if 0 < len(self.rprev):
            # if there is a previous residue, build on from it
            startPos = {}
            # nb akl for this res n-ca-c in rp (prev res) dihedra
            akl: List[AtomKey] = []
            for tpl in self.NCaCKey:
                akl.extend(tpl)
            for ak in akl:
                for rp in self.rprev:
                    rpak = rp.atom_coords.get(ak, None)
                    if rpak is not None:
                        startPos[ak] = rpak
            if 3 > len(startPos):
                startPos = {}
        else:
            # get atom posns already added by load_structure
            sp = self.residue.parent.internal_coord.initNCaC.get(self.rbase, None)
            if sp is None:
                startPos = {}
            else:
                startPos = cast(Dict["AtomKey", numpy.array], sp)

        if startPos == {}:
            # fallback: use N-CA-C initial coords from creating dihedral
            startPos = {}
            dlist0 = [self.id3_dh_index[akl2] for akl2 in sorted(self.NCaCKey)]
            # https://stackoverflow.com/questions/11264684/flatten-list-of-lists
            dlist = [val for sublist in dlist0 for val in sublist]
            # dlist = self.id3_dh_index[NCaCKey]
            for d in dlist:
                for i, a in enumerate(d.aks):
                    startPos[a] = d.initial_coords[i]
        return startPos

    def assemble(
        self,
        transforms: bool = False,
        resetLocation: bool = False,
        verbose: bool = False,
    ) -> Union[Dict["AtomKey", numpy.array], Dict[HKT, numpy.array]]:
        """Compute atom coordinates for this residue from internal coordinates.

        Join dihedrons starting from N-CA-C and N-CA-CB hedrons, computing protein
        space coordinates for backbone and sidechain atoms

        **Algorithm**

        form double-ended queue, start with c-ca-n, o-c-ca, n-ca-cb, n-ca-c.

        if resetLocation=True, use initial coords from generating dihedron
        for n-ca-c initial positions (result in dihedron coordinate space)

        while queue not empty
            get 3-atom hedron key

            for each dihedron starting with hedron key (1st hedron of dihedron)

                if have coordinates for all 4 atoms already
                    add 2nd hedron key to back of queue
                else if have coordinates for 1st 3 atoms
                    compute forward and reverse transforms to take 1st 3 atoms
                    to/from dihedron initial coordinate space

                    use reverse transform to get position of 4th atom in
                    current coordinates from dihedron initial coordinates

                    add 2nd hedron key to back of queue
                else
                    ordering failed, put hedron key at back of queue and hope
                    next time we have 1st 3 atom positions (should not happen)

        loop terminates (queue drains) as hedron keys which do not start any
        dihedra are removed without action

        :param transforms: bool default False
            - Option to return transformation matrices for each hedron instead
            of coordinates.

        :param resetLocation: bool default False
            - Option to ignore start location and orient so N-Ca-C hedron
            at origin.

        :returns:
            Homogeneous atom coords for residue in protein space relative to
            previous residue **OR** table of transformation matrices if
            **transforms** = True

        """
        dbg = False

        transformations = {}
        NCaCKey = sorted(self.NCaCKey)

        if transforms:
            for akl1 in NCaCKey:
                transformations[akl1] = numpy.identity(4, dtype=numpy.float64)

        startLst = []
        for lst in [
            (self.rak("C"), self.rak("CA"), self.rak("N")),
            (self.rak("N"), self.rak("CA"), self.rak("CB")),
            (self.rak("O"), self.rak("C"), self.rak("CA")),
        ]:
            startLst.extend(self._split_akl(lst))
        startLst.extend(NCaCKey)

        q = deque(startLst)

        # get initial coords from previous residue or IC_Chain info
        # or default coords
        if resetLocation:
            # use N-CA-C initial coords from creating dihedral
            atomCoords = {}
            dlist0 = [self.id3_dh_index[akl] for akl in NCaCKey]
            # https://stackoverflow.com/questions/11264684/flatten-list-of-lists
            dlist = [val for sublist in dlist0 for val in sublist]
            # dlist = self.id3_dh_index[NCaCKey]
            for d in dlist:
                for i, a in enumerate(d.aks):
                    atomCoords[a] = d.initial_coords[i]
        else:
            atomCoords = self.get_startpos()

        while q:  # deque is not empty
            if dbg:
                print("assemble loop start q=", q)
            h1k = cast(HKT, q.pop())
            dihedra = self.id3_dh_index.get(h1k, None)
            if dbg:
                print(
                    "  h1k:",
                    h1k,
                    "len dihedra: ",
                    len(dihedra) if dihedra is not None else "None",
                )
            if dihedra is not None:
                for d in dihedra:
                    if 4 == len(d.initial_coords) and d.initial_coords[3] is not None:
                        # skip incomplete dihedron if don't have 4th atom due
                        # to missing input data
                        d_h2key = d.hedron2.aks
                        akl = d.aks
                        acount = 0
                        if dbg:
                            print("    process", d, d_h2key, akl)
                        for a in akl:
                            if a in atomCoords and atomCoords[a] is not None:
                                acount += 1
                        if 4 == acount:  # and not need_transform:
                            # dihedron already done, queue 2nd hedron key
                            q.appendleft(d_h2key)
                            # print("    4- already done, append left")
                            if transforms and not (h1k in transformations):
                                # can happen for altloc atoms
                                mt, mtr = coord_space(
                                    [atomCoords[a] for a in akl[:3]], True
                                )
                                transformations[h1k] = mtr
                        elif 3 == acount:  # or need_transform:
                            # print("    3- call coord_space")
                            mt, mtr = coord_space(
                                [atomCoords[a] for a in akl[:3]], True
                            )
                            if transforms:
                                transformations[h1k] = mtr
                            if dbg:
                                print(
                                    "        initial_coords[3]=",
                                    d.initial_coords[3].transpose(),
                                )
                            acak3 = mtr.dot(d.initial_coords[3])
                            if dbg:
                                print("        acak3=", acak3.transpose())
                            for i in range(3):
                                acak3[i][0] = set_accuracy_83(acak3[i][0])
                            atomCoords[akl[3]] = acak3
                            if dbg:
                                print(
                                    "        3- finished, ak:",
                                    akl[3],
                                    "coords:",
                                    atomCoords[akl[3]].transpose(),
                                )
                            q.appendleft(d_h2key)
                        else:
                            if verbose:
                                print("no coords to start", d)
                            pass
                    else:
                        if verbose:
                            print("no initial coords for", d)
                        pass
        # print('coord_space returning')
        if transforms:
            return transformations
        else:
            return atomCoords

    def _split_akl(
        self, lst: Union[Tuple["AtomKey", ...], List["AtomKey"]]
    ) -> List[Tuple["AtomKey", ...]]:
        """Get AtomKeys for this residue (ak_set) given generic list of AtomKeys.

        Given a list of AtomKeys (aks) for a Hedron or Dihedron,
          return:
                list of matching aks that have id3_dh in this residue
                (ak may change if occupancy != 1.00)

            or
                multiple lists of matching aks expanded for all atom altlocs

            or
                empty list if any of atom_coord(ak) missing

        :param lst: list[3] or [4] of AtomKeys
            non-altloc AtomKeys to match to specific AtomKeys for this residue
        """
        altloc_ndx = AtomKey.fields.altloc

        # step 1
        # given a list of AtomKeys (aks)
        #  form a new list of same aks with coords or diheds in this residue
        #      plus lists of matching altloc aks in coords or diheds
        edraLst: List[Tuple[AtomKey, ...]] = []
        altlocs = set()
        posnAltlocs: Dict["AtomKey", Set[str]] = {}
        akMap = {}
        for ak in lst:
            posnAltlocs[ak] = set()
            if ak in self.ak_set:
                edraLst.append((ak,))  # tuple of ak
            else:
                ak2_lst = []
                for ak2 in self.ak_set:
                    if ak.altloc_match(ak2):
                        # print(key)
                        ak2_lst.append(ak2)
                        akMap[ak2] = ak
                        altloc = ak2.akl[altloc_ndx]
                        if altloc is not None:
                            altlocs.add(altloc)
                            posnAltlocs[ak].add(altloc)
                edraLst.append(tuple(ak2_lst))

        # step 2
        # check and finish for
        #   missing atoms
        #   simple case no altlocs
        # else form new AtomKey lists covering all altloc permutations
        maxc = 0
        for akl in edraLst:
            lenAKL = len(akl)
            if 0 == lenAKL:
                return []  # atom missing in atom_coords, cannot form object
            elif maxc < lenAKL:
                maxc = lenAKL
        if 1 == maxc:  # simple case no altlocs for any ak in list
            newAKL = []
            for akl in edraLst:
                newAKL.append(akl[0])
            return [tuple(newAKL)]
        else:
            new_edraLst = []
            for al in altlocs:
                # form complete new list for each altloc
                alhl = []
                for akl in edraLst:
                    lenAKL = len(akl)
                    if 1 == lenAKL:
                        alhl.append(akl[0])  # not all atoms will have altloc
                    # elif (lenAKL < maxc
                    #      and al not in posnAltlocs[akMap[akl[0]]]):
                    elif al not in posnAltlocs[akMap[akl[0]]]:
                        # this postion has fewer altlocs than other positions
                        # and this position does not have this al,
                        # so just grab first to form angle as could be any
                        alhl.append(sorted(akl)[0])
                    else:
                        for ak in akl:
                            if ak.akl[altloc_ndx] == al:
                                alhl.append(ak)
                new_edraLst.append(tuple(alhl))

            # print(new_edraLst)
            return new_edraLst

    def _gen_edra(self, lst: Union[Tuple["AtomKey", ...], List["AtomKey"]]) -> None:
        """Populate hedra/dihedra given edron ID tuple.

        Given list of AtomKeys defining hedron or dihedron
          convert to AtomKeys with coordinates in this residue
          add appropriately to self.di/hedra, expand as needed atom altlocs

        :param lst: tuple of AtomKeys
            Specifies Hedron or Dihedron
        """

        lenLst = len(lst)
        if 4 > lenLst:
            dct, obj = self.hedra, Hedron
        else:
            dct, obj = self.dihedra, Dihedron  # type: ignore

        if isinstance(lst, List):
            tlst = tuple(lst)
        else:
            tlst = lst

        if all(ak in self.atom_coords for ak in lst):
            hl = [tlst]
        else:
            hl = self._split_akl(tlst)

        for nlst in hl:
            # do not add edron if split_akl() made something shorter
            if len(nlst) == lenLst:
                # if edron already exists, then update not replace with new
                tnlst = tuple(nlst)
                if tnlst not in dct:
                    dct[tnlst] = obj(nlst)  # type: ignore
                dct[tnlst].atoms_updated = False  # type: ignore

    def atom_to_internal_coordinates(
        self, verbose: bool = False, allBonds: bool = False
    ) -> None:
        """Create hedra and dihedra for atom coordinates.

        :param allBonds: bool default False
            For OpenSCAD, generate explicit hedra covering all bonds if True.
        """
        sN, sCA, sC = self.rak("N"), self.rak("CA"), self.rak("C")
        sCB = self.rak("CB")

        if 0 < len(self.rnext):
            # atom_coords, hedra and dihedra for backbone dihedra
            # which reach into next residue
            for rn in self.rnext:
                nN, nCA, nC = rn.rak("N"), rn.rak("CA"), rn.rak("C")

                for ak in (nN, nCA, nC):
                    if ak in rn.atom_coords:
                        self.atom_coords[ak] = rn.atom_coords[ak]
                        self.ak_set.add(ak)
                    else:
                        for rn_ak in rn.atom_coords.keys():
                            if rn_ak.altloc_match(ak):
                                self.atom_coords[rn_ak] = rn.atom_coords[rn_ak]
                                self.ak_set.add(rn_ak)

                self._gen_edra((sN, sCA, sC, nN))  # psi
                self._gen_edra((sCA, sC, nN, nCA))  # omega i+1
                self._gen_edra((sC, nN, nCA, nC))  # phi i+1
                self._gen_edra((sCA, sC, nN))
                self._gen_edra((sC, nN, nCA))
                self._gen_edra((nN, nCA, nC))

        if 0 == len(self.rprev):
            self._gen_edra((sN, sCA, sC))

        # standard backbone atoms independent of neighbours
        backbone = ic_data_backbone
        for edra in backbone:
            r_edra = [self.rak(atom) for atom in edra]
            self._gen_edra(r_edra[0:4])  # [4] is label on some table entries

        # sidechain hedra and dihedra
        if self.lc is not None:
            sidechain = ic_data_sidechains.get(self.lc, [])
            for edra in sidechain:
                r_edra = [self.rak(atom) for atom in edra]
                # [4] is label on some table entries
                self._gen_edra(r_edra[0:4])
            if allBonds:
                sidechain = ic_data_sidechain_extras.get(self.lc, [])
                for edra in sidechain:
                    r_edra = [self.rak(atom) for atom in edra]
                    self._gen_edra(r_edra[0:4])

        self.link_dihedra(verbose)

        if self.gly_Cbeta and "G" == self.lc and sCB not in self.atom_coords:
            # do here because atom_to_internal_coordinates() may need
            self.atom_coords[sCB] = None  # so _gen_edra will complete

        for d in self.dihedra.values():
            # populate values and hedra for dihedron ojects
            d.dihedron_from_atoms()
        for h in self.hedra.values():
            # miss redundant hedra above, needed for some chi1 angles
            # also miss if missing atoms means hedron not in dihedra
            if (not hasattr(h, "len1")) or (not h.atoms_updated):
                # print(h)
                h.hedron_from_atoms(self.atom_coords)

        if self.gly_Cbeta and "G" == self.lc and self.atom_coords[sCB] is None:
            # add C-beta for Gly

            # data averaged from Sep 2019 Dunbrack cullpdb_pc20_res2.2_R1.0
            # restricted to structures with amide protons.
            # Ala avg rotation of OCCACB from NCACO query:
            # select avg(g.rslt) as avg_rslt, stddev(g.rslt) as sd_rslt, count(*)
            # from
            # (select f.d1d, f.d2d,
            # (case when f.rslt > 0 then f.rslt-360.0 else f.rslt end) as rslt
            # from (select d1.dihedral1 as d1d, d2.dihedral1 as d2d,
            # (d2.dihedral1 - d1.dihedral1) as rslt from dihedron d1,
            # dihedron d2 where d1.rdh_class='AOACACAACB' and
            # d2.rdh_class='ANACAACAO' and d1.pdb=d2.pdb and d1.chn=d2.chn
            # and d1.res=d2.res) as f) as g
            # +-------------------+------------------+---------+
            # | avg_rslt          | sd_rslt          | count   |
            # |-------------------+------------------+---------|
            # | -122.682194862932 | 5.04403040513919 | 14098   |
            # +-------------------+------------------+---------+

            Ca_Cb_Len = 1.53363
            if hasattr(self, "scale"):
                Ca_Cb_len *= self.scale  # type: ignore

            # main orientation comes from O-C-Ca-Cb so make Cb-Ca-C hedron
            sO = self.rak("O")
            htpl = (sCB, sCA, sC)
            self._gen_edra(htpl)
            h = self.hedra[htpl]
            h.len3 = self.hedra[(sCA, sC, sO)].len1
            h.angle2 = 110.17513
            h.len1 = Ca_Cb_Len
            h.init_pos()

            # generate dihedral based on N-Ca-C-O offset from db query above
            dtpl = (sO, sC, sCA, sCB)
            self._gen_edra(dtpl)
            d = self.dihedra[dtpl]
            d.IC_Residue = self
            d._set_hedra()
            sN = self.rak("N")
            refval = self.dihedra.get((sN, sCA, sC, sO), None)
            if refval:
                d.dihedral1 = 122.68219 + refval.dihedral1
                if d.dihedral1 > 180.0:
                    d.dihedral1 -= 360.0
            else:
                d.dihedral1 = 120
            del self.atom_coords[sCB]  # remove None so now must populate
            d.init_pos()

            self.link_dihedra(verbose)  # re-run for new dihedra

    @staticmethod
    def _pdb_atom_string(atm: Atom) -> str:
        """Generate PDB ATOM record.

        :param atm: Biopython Atom object reference
        """
        if 2 == atm.is_disordered():
            s = ""
            for a in atm.child_dict.values():
                s += IC_Residue._pdb_atom_string(a)
            return s
        else:
            res = atm.parent
            chn = res.parent
            s = (
                "{:6}{:5d} {:4}{:1}{:3} {:1}{:4}{:1}   {:8.3f}{:8.3f}{:8.3f}"
                "{:6.2f}{:6.2f}        {:>4}\n"
            ).format(
                "ATOM",
                atm.serial_number,
                atm.fullname,
                atm.altloc,
                res.resname,
                chn.id,
                res.id[1],
                res.id[2],
                atm.coord[0],
                atm.coord[1],
                atm.coord[2],
                atm.occupancy,
                atm.bfactor,
                atm.element,
            )
            # print(s)
        return s

    @staticmethod
    def _residue_string(res: "Residue") -> str:
        """Generate PIC Residue string.

        Enough to create Biopython Residue object without actual Atoms.

        :param res: Biopython Residue object reference
        """
        segid = res.get_segid()
        if segid.isspace() or "" == segid:
            segid = ""
        else:
            segid = " [" + segid + "]"
        return str(res.get_full_id()) + " " + res.resname + segid + "\n"

    def _write_pic_bfac(self, atm: Atom, s: str, col: int) -> Tuple[str, int]:
        ak = self.rak(atm)
        if 0 == col % 5:
            s += "BFAC:"
        s += " " + ak.id + " " + "{:6.2f}".format(atm.get_bfactor())
        col += 1
        if 0 == col % 5:
            s += "\n"
        return s, col

    def write_PIC(self, pdbid: str = "0PDB", chainid: str = "A", s: str = "") -> str:
        """Write PIC format lines for this residue.

        :param str pdbid: PDB idcode string; default 0PDB
        :param str chainid: PDB Chain ID character; default A
        :param str s: result string to add to
        """
        if pdbid is None:
            pdbid = "0PDB"
        if chainid is None:
            chainid = "A"
        s += IC_Residue._residue_string(self.residue)
        if 0 == len(self.rprev):
            NCaChedron = self.pick_angle("tau")
            if NCaChedron is not None and NCaChedron.atoms_updated:
                try:
                    ts = IC_Residue._pdb_atom_string(self.residue["N"])
                    ts += IC_Residue._pdb_atom_string(self.residue["CA"])
                    ts += IC_Residue._pdb_atom_string(self.residue["C"])
                    s += ts  # only if no exception, have all 3 atoms
                except KeyError:
                    pass

        base = pdbid + " " + chainid + " "
        for h in sorted(self.hedra.values()):
            try:
                s += (
                    base
                    + h.id
                    + " "
                    + "{:9.5f} {:9.5f} {:9.5f}".format(h.len1, h.angle2, h.len3)
                )
            except KeyError:
                pass
            s += "\n"
        for d in sorted(self.dihedra.values()):
            try:
                s += base + d.id + " " + "{:9.5f}".format(d.dihedral1)
            except KeyError:
                pass
            s += "\n"

        col = 0
        for a in sorted(self.residue.get_atoms()):
            if 2 == a.is_disordered():  # hasattr(a, 'child_dict'):
                if self.alt_ids is None:
                    s, col = self._write_pic_bfac(a.selected_child, s, col)
                else:
                    for atm in a.child_dict.values():
                        s, col = self._write_pic_bfac(atm, s, col)
            else:
                s, col = self._write_pic_bfac(a, s, col)
        if 0 != col % 5:
            s += "\n"

        return s

    def coords_to_residue(self) -> None:
        """Convert self.atom_coords to biopython Residue Atom coords.

        Change homogeneous IC_Residue atom_coords to self.residue cartesian
        Biopython Atom coords.
        """
        respos, icode = self.residue.id[1:3]
        respos = str(respos)
        spNdx, icNdx, resnNdx, atmNdx, altlocNdx, occNdx = AtomKey.fields

        Res = self.residue
        ndx = Res.parent.internal_coord.ndx

        for ak in sorted(self.atom_coords):
            # print(ak)
            if respos == ak.akl[spNdx] and (
                (icode == " " and ak.akl[icNdx] is None) or icode == ak.akl[icNdx]
            ):

                ac = self.atom_coords[ak]
                atm_coords = ac[:3].transpose()[0]
                akl = ak.akl
                atm, altloc = akl[atmNdx], akl[altlocNdx]

                Atm = None
                newAtom = None

                if Res.has_id(atm):
                    Atm = Res[atm]

                if Atm is None or (
                    2 == Atm.is_disordered() and not Atm.disordered_has_id(altloc)
                ):
                    # print('new', ak)
                    occ = akl[occNdx]
                    aloc = akl[altlocNdx]
                    bfac = self.bfactors.get(ak.id, None)
                    newAtom = Atom(
                        atm,
                        atm_coords,
                        (0.0 if bfac is None else bfac),
                        (1.00 if occ is None else float(occ)),
                        (" " if aloc is None else aloc),
                        atm,
                        ndx,
                        atm[0],
                    )
                    ndx += 1
                    if Atm is None:
                        if altloc is None:
                            Res.add(newAtom)
                        else:
                            disordered_atom = DisorderedAtom(atm)
                            Res.add(disordered_atom)
                            disordered_atom.disordered_add(newAtom)
                            Res.flag_disordered()
                    else:
                        Atm.disordered_add(newAtom)
                else:
                    # Atm is not None, might be disordered with altloc
                    # print('update', ak)
                    if 2 == Atm.is_disordered() and Atm.disordered_has_id(altloc):
                        Atm.disordered_select(altloc)
                    Atm.set_coord(atm_coords)
                    ndx = Atm.get_serial_number() + 1

        Res.parent.internal_coord.ndx = ndx

        # print(ak, ac)

    def _get_ak_tuple(self, ak_str: str) -> Tuple["AtomKey", ...]:
        """Convert atom pair string to AtomKey tuple.

        :param ak_str: str
            Two atom names separated by ':', e.g. 'N:CA'
            Optional position specifier relative to self,
            e.g. '-1C:N' for preceding peptide bond.
        """
        AK = AtomKey
        S = self
        angle_key2 = []
        for a in ak_str.split(":"):
            m = self.relative_atom_re.match(a)
            if m:
                if m.group(1) == "-1":
                    if 0 < len(S.rprev):
                        angle_key2.append(AK(S.rprev[0], m.group(2)))
                elif m.group(1) == "1":
                    if 0 < len(S.rnext):
                        angle_key2.append(AK(S.rnext[0], m.group(2)))
                elif m.group(1) == "0":
                    angle_key2.append(self.rak(m.group(2)))
            else:
                angle_key2.append(self.rak(a))
        return tuple(angle_key2)

    relative_atom_re = re.compile(r"^(-?[10])([A-Z]+)$")

    def _get_angle_for_tuple(
        self, angle_key: EKT
    ) -> Optional[Union["Hedron", "Dihedron"]]:
        len_mkey = len(angle_key)
        rval: Optional[Union["Hedron", "Dihedron"]]
        if 4 == len_mkey:
            rval = self.dihedra.get(cast(DKT, angle_key), None)
        elif 3 == len_mkey:
            rval = self.hedra.get(cast(HKT, angle_key), None)
        return rval

    def pick_angle(
        self, angle_key: Union[EKT, str]
    ) -> Optional[Union["Hedron", "Dihedron"]]:
        """Get Hedron or Dihedron for angle_key.

        :param angle_key:
            - tuple of 3 or 4 AtomKeys
            - string of atom names ('CA') separated by :'s
            - string of [-1, 0, 1]<atom name> separated by :'s, -1 is
              previous residue, 0 is this residue, 1 is next residue
            - psi, phi, omg, omega, chi1, chi2, chi3, chi4, chi5
            - tau (N-CA-C angle) see Richardson1981
            - except for tuples, no option to access alternate disordered atoms

        :return: Matching Hedron, Dihedron, or None.
        """
        rval: Optional[Union["Hedron", "Dihedron"]] = None
        if isinstance(angle_key, tuple):
            rval = self._get_angle_for_tuple(angle_key)
        elif ":" in angle_key:
            angle_key = cast(EKT, self._get_ak_tuple(cast(str, angle_key)))
            rval = self._get_angle_for_tuple(angle_key)
        elif "psi" == angle_key:
            if 0 == len(self.rnext):
                return None
            rn = self.rnext[0]
            sN, sCA, sC = self.rak("N"), self.rak("CA"), self.rak("C")
            nN = rn.rak("N")
            rval = self.dihedra.get((sN, sCA, sC, nN), None)
        elif "phi" == angle_key:
            if 0 == len(self.rprev):
                return None
            rp = self.rprev[0]
            pC, sN, sCA = rp.rak("C"), self.rak("N"), self.rak("CA")
            sC = self.rak("C")
            rval = rp.dihedra.get((pC, sN, sCA, sC), None)
        elif "omg" == angle_key or "omega" == angle_key:
            if 0 == len(self.rprev):
                return None
            rp = self.rprev[0]
            pCA, pC, sN = rp.rak("CA"), rp.rak("C"), self.rak("N")
            sCA = self.rak("CA")
            rval = rp.dihedra.get((pCA, pC, sN, sCA), None)
        elif "tau" == angle_key:
            sN, sCA, sC = self.rak("N"), self.rak("CA"), self.rak("C")
            rval = self.hedra.get((sN, sCA, sC), None)
            if rval is None and 0 != len(self.rprev):
                rp = self.rprev[0]  # tau in prev residue for all but first
                rval = rp.hedra.get((sN, sCA, sC), None)
        elif angle_key.startswith("chi"):
            sclist = ic_data_sidechains.get(self.lc, None)
            if sclist is None:
                return None
            for akl in sclist:
                if 5 == len(akl):
                    if akl[4] == angle_key:
                        klst = [self.rak(a) for a in akl[0:4]]
                        tklst = cast(DKT, tuple(klst))
                        rval = self.dihedra.get(tklst, None)

        return rval

    def get_angle(self, angle_key: Union[EKT, str]) -> Optional[float]:
        """Get dihedron or hedron angle for specified key.

        See pick_angle() for key specifications.
        """
        edron = self.pick_angle(angle_key)
        if edron:
            return edron.get_angle()
        return None

    def set_angle(self, angle_key: Union[EKT, str], v: float):
        """Set dihedron or hedron angle for specified key.

        See pick_angle() for key specifications.
        """
        rval = self.pick_angle(angle_key)
        if rval is not None:
            rval.set_angle(v)

    def pick_length(self, ak_spec: Union[str, BKT]) -> Tuple[List["Hedron"], BKT]:
        """Get list of hedra containing specified atom pair.

        :param ak_spec:
            - tuple of two AtomKeys
            - string: two atom names separated by ':', e.g. 'N:CA' with
              optional position specifier relative to self, e.g. '-1C:N' for
              preceding peptide bond.

        :return: list of hedra containing specified atom pair, tuple of atom keys
        """
        rlst: List[Hedron] = []
        # if ":" in ak_spec:
        if isinstance(ak_spec, str):
            ak_spec = cast(BKT, self._get_ak_tuple(ak_spec))

        for hed_key, hed_val in self.hedra.items():
            if all(ak in hed_key for ak in ak_spec):
                rlst.append(hed_val)
        return rlst, ak_spec

    def get_length(self, ak_spec: Union[str, BKT]) -> Optional[float]:
        """Get bond length for specified atom pair.

        See pick_length() for ak_spec.
        """
        hed_lst, ak_spec = self.pick_length(ak_spec)
        for hed in hed_lst:
            val = hed.get_length(ak_spec)
            if val is not None:
                return val
        return None

    def set_length(self, ak_spec: Union[str, BKT], val: float) -> None:
        """Set bond length for specified atom pair.

        See pick_length() for ak_spec.
        """
        hed_lst, ak_spec = self.pick_length(ak_spec)
        for hed in hed_lst:
            hed.set_length(ak_spec, val)

    def applyMtx(self, mtx: numpy.array) -> None:
        """Apply matrix to atom_coords for this IC_Residue."""
        for ak, ac in self.atom_coords.items():
            # self.atom_coords[ak] = mtx @ ac
            self.atom_coords[ak] = mtx.dot(ac)


class Edron(object):
    """Base class for Hedron and Dihedron classes.

    Supports rich comparison based on lists of AtomKeys.

    Attributes
    ----------
    aks: tuple
        3 (hedron) or 4 (dihedron) AtomKeys defining this di/hedron
    id: str
        ':'-joined string of AtomKeys for this di/hedron
    atoms_updated: bool
        indicates hedron local atom_coords reflect current di/hedron angle and
        length values in hedron local coordinate space
    dh_class: str
        sequence of atoms (no position or residue) comprising di/hedron
        for statistics
    rdh_class: str
        sequence of residue, atoms comprising di/hedron for statistics
    edron_re: compiled regex (Class Attribute)
        A compiled regular expression matching string IDs for Hedron
        and Dihedron objects

    Methods
    -------
    gen_key([AtomKey, ...] or AtomKey, ...) (Static Method)
        generate a ':'-joined string of AtomKey Ids
    gen_acs(atom_coords)
        generate tuple of atom coords for keys in self.aks
    is_backbone()
        Return True if all aks atoms are N, Ca, C or O

    """

    # regular expresion to capture hedron and dihedron specifications, as in
    #  .pic files
    edron_re = re.compile(
        # pdbid and chain id
        r"^(?P<pdbid>\w+)?\s(?P<chn>[\w|\s])?\s"
        # 3 atom specifiers for hedron
        r"(?P<a1>[\w\-\.]+):(?P<a2>[\w\-\.]+):(?P<a3>[\w\-\.]+)"
        # 4th atom specifier for dihedron
        r"(:(?P<a4>[\w\-\.]+))?"
        r"\s+"
        # len-angle-len for hedron
        r"(((?P<len1>\S+)\s+(?P<angle2>\S+)\s+(?P<len3>\S+)\s*$)|"
        # dihedral angle for dihedron
        r"((?P<dihedral1>\S+)\s*$))"
    )

    @staticmethod
    def gen_key(lst: Union[List[str], List["AtomKey"]]) -> str:
        """Generate string of ':'-joined AtomKey strings from input.

        :param lst: list of AtomKey objects or id strings
        """
        if isinstance(lst[0], str):
            lst = cast(List[str], lst)
            return ":".join(lst)
        else:
            lst = cast(List[AtomKey], lst)
            return ":".join(ak.id for ak in lst)

    def __init__(self, *args: Union[List["AtomKey"], EKT], **kwargs: str) -> None:
        """Initialize Edron with sequence of AtomKeys.

        Acceptable input:

            [ AtomKey, ... ]  : list of AtomKeys
            AtomKey, ...      : sequence of AtomKeys as args
            {'a1': str, 'a2': str, ... }  : dict of AtomKeys as 'a1', 'a2' ...
        """
        aks: List[AtomKey] = []
        for arg in args:
            if isinstance(arg, list):
                aks = arg
            elif isinstance(arg, tuple):
                aks = list(arg)
            else:
                if arg is not None:
                    aks.append(arg)
        if [] == aks and all(k in kwargs for k in ("a1", "a2", "a3")):
            aks = [AtomKey(kwargs["a1"]), AtomKey(kwargs["a2"]), AtomKey(kwargs["a3"])]
            if "a4" in kwargs and kwargs["a4"] is not None:
                aks.append(AtomKey(kwargs["a4"]))

        # if args are atom key strings instead of AtomKeys
        # for i in range(len(aks)):
        #    if not isinstance(aks[i], AtomKey):
        #        aks[i] = AtomKey(aks[i])

        self.aks = tuple(aks)
        self.id = Edron.gen_key(aks)
        self._hash = hash(self.aks)

        # flag indicating that atom coordinates are up to date
        # (do not need to be recalculated from dihedral1)
        self.atoms_updated = False

        # no residue or position, just atoms
        self.dh_class = ""
        # same but residue specific
        self.rdh_class = ""

        atmNdx = AtomKey.fields.atm
        resNdx = AtomKey.fields.resname
        for ak in aks:
            akl = ak.akl
            self.dh_class += akl[atmNdx]
            self.rdh_class += akl[resNdx] + akl[atmNdx]

    def gen_acs(
        self, atom_coords: Dict["AtomKey", numpy.array]
    ) -> Tuple[numpy.array, ...]:
        """Generate tuple of atom coord arrays for keys in self.aks.

        :param atom_coords: AtomKey dict of atom coords for residue
        :raises: MissingAtomError any atoms in self.aks missing coordinates
        """
        aks = self.aks
        acs: List[numpy.array] = []
        estr = ""
        for ak in aks:
            ac = atom_coords[ak]
            if ac is None:
                estr += str(ak) + " "
            else:
                acs.append(ac)
        if estr != "":
            raise MissingAtomError("%s missing coordinates for %s" % (self, estr))
        return tuple(acs)

    def is_backbone(self) -> bool:
        """Report True for contains only N, C, CA, O, H atoms."""
        atmNdx = AtomKey.fields.atm
        if all(
            atm in ("N", "C", "CA", "O", "H")
            for atm in (ak.akl[atmNdx] for ak in self.aks)
        ):
            return True
        return False

    def __repr__(self) -> str:
        """Tuple of AtomKeys is default repr string."""
        return str(self.aks)

    def __hash__(self) -> int:
        """Hash calculated at init from aks tuple."""
        return self._hash

    def _cmp(self, other: "Edron") -> Union[Tuple["AtomKey", "AtomKey"], bool]:
        """Comparison function ranking self vs. other. False on equal"""
        for ak_s, ak_o in zip(self.aks, other.aks):
            if ak_s != ak_o:
                return ak_s, ak_o
        return False

    def __eq__(self, other: object) -> bool:
        """Test for equality."""
        if not isinstance(other, type(self)):
            return NotImplemented
        return self.id == other.id

    def __ne__(self, other: object) -> bool:
        """Test for inequality."""
        if not isinstance(other, type(self)):
            return NotImplemented
        return self.id != other.id

    def __gt__(self, other: object) -> bool:
        """Test greater than."""
        if not isinstance(other, type(self)):
            return NotImplemented
        rslt = self._cmp(other)
        if rslt:
            rslt = cast(Tuple[AtomKey, AtomKey], rslt)
            return rslt[0] > rslt[1]
        return False

    def __ge__(self, other: object) -> bool:
        """Test greater or equal."""
        if not isinstance(other, type(self)):
            return NotImplemented
        rslt = self._cmp(other)
        if rslt:
            rslt = cast(Tuple[AtomKey, AtomKey], rslt)
            return rslt[0] >= rslt[1]
        return True

    def __lt__(self, other: object) -> bool:
        """Test less than."""
        if not isinstance(other, type(self)):
            return NotImplemented
        rslt = self._cmp(other)
        if rslt:
            rslt = cast(Tuple[AtomKey, AtomKey], rslt)
            return rslt[0] < rslt[1]
        return False

    def __le__(self, other: object) -> bool:
        """Test less or equal."""
        if not isinstance(other, type(self)):
            return NotImplemented
        rslt = self._cmp(other)
        if rslt:
            rslt = cast(Tuple[AtomKey, AtomKey], rslt)
            return rslt[0] <= rslt[1]
        return True


class Hedron(Edron):
    """Class to represent three joined atoms forming a plane.

    Contains atom coordinates in local coordinate space, central atom
    at origin.  Stored in two orientations, with the 3rd (forward) or
    first (reversed) atom on the +Z axis.

    Attributes
    ----------
    len1: float
        distance between 1st and 2nd atom
    angle2: float
        angle (degrees) formed by 3 atoms
    len3: float
        distance between 2nd and 3rd atoms
    atoms: tuple[3] of numpy arrays [4][1]
        3 atoms comprising hedron, 1st on XZ, 2nd at origin, 3rd on +Z
    atomsR: tuple[3] of numpy arrays [4][1]
        atoms reversed, 1st on +Z, 2nd at origin, 3rd on XZ plane

    Methods
    -------
    init_pos()
        Create hedron space atom coordinate numpy arrays.
    hedron_from_atoms()
        Compute length, angle, length for hedron from IC_Residue atom coords
    get_angle()
        return angle2
    set_angle()
        update angle2 with supplied value
    set_length()
        update length for specified atom pair

    """

    def __init__(self, *args: Union[List["AtomKey"], HKT], **kwargs: str) -> None:
        """Initialize Hedron with sequence of AtomKeys, kwargs.

        Acceptable input:
            As for Edron, plus optional 'len1', 'angle2', 'len3'
            keyworded values.
        """
        super(Hedron, self).__init__(*args, **kwargs)

        # print('initialising', self.id)

        # 3 matrices specifying hedron space coordinates of constituent atoms,
        # initially atom3 on +Z axis
        self.atoms: HACS
        # 3 matrices, hedron space coordinates, reversed order
        # initially atom1 on +Z axis
        self.atomsR: HACS

        if "len1" in kwargs:
            # distance between 1st and 2nd atom
            self.len1 = float(kwargs["len1"])
            # angle formed between 3 atoms
            self.angle2 = float(kwargs["angle2"])
            # distance between 2nd and 3rd atoms
            self.len3 = float(kwargs["len3"])

            self.init_pos()
        # else:
        #    self.len1 = None
        #    self.angle2 = None
        #    self.len3 = None

        # print(self)

    def __str__(self) -> str:
        """Print string for Hedron object."""
        return f"3-{self.id} {self.rdh_class} {str(self.len1)} {str(self.angle2)} {str(self.angle2)}"

    def init_pos(self) -> None:
        """Initialize Hedron by creating atom coordinate numpy arrays."""
        if not hasattr(self, "len1"):
            raise AttributeError("%s missing length and angle settings" % (self))

        # build hedron with a2 on +Z axis, a1 at origin,
        # a0 in -Z at angle n XZ plane
        atoms: List[numpy.array] = []
        for _ in range(3):
            # note this initializes a1 to 0,0,0 origin
            atoms.append(
                numpy.array([[0], [0], [0], [1]], dtype=numpy.float64)
            )  # 4x1 array

        # supplementary angle radian: angles which add to 180 are supplementary
        sar = numpy.deg2rad(180.0 - self.angle2)

        # a2 is len3 up from a2 on Z axis, X=Y=0
        atoms[2][2][0] = self.len3
        # a0 X is sin( sar ) * len1
        atoms[0][0][0] = numpy.sin(sar) * self.len1
        # a0 Z is -(cos( sar ) * len1)
        # (assume angle2 always obtuse, so a0 is in -Z)
        atoms[0][2][0] = -(numpy.cos(sar) * self.len1)

        self.atoms = cast(HACS, tuple(atoms))

        atomsR: List[numpy.array] = []
        # same again but 'reversed' : a0 on Z axis, a1 at origin, a2 in -Z
        for _ in range(3):
            # atom[1] to 0,0,0 origin
            atomsR.append(numpy.array([[0], [0], [0], [1]], dtype=numpy.float64))

        # a0r is len1 up from a1 on Z axis, X=Y=0
        atomsR[0][2][0] = self.len1
        # a2r X is sin( sar ) * len3
        atomsR[2][0][0] = numpy.sin(sar) * self.len3
        # a2r Z is -(cos( sar ) * len3)
        atomsR[2][2][0] = -(numpy.cos(sar) * self.len3)

        self.atomsR = cast(HACS, tuple(atomsR))

        self.atoms_updated = True

    @staticmethod
    def _get_dad(acs: HACS) -> Tuple[float, float, float]:
        """Get distance, angle, distance for 3 atoms.

        :param acs: list[3] of homogeneous atom coords as numpy arrays [4][[1]]
        """
        a0 = acs[0].squeeze()
        a1 = acs[1].squeeze()
        a2 = acs[2].squeeze()

        a0a1 = numpy.linalg.norm(a0 - a1)
        a1a2 = numpy.linalg.norm(a1 - a2)
        a0a2 = numpy.linalg.norm(a0 - a2)

        a0a1a2 = numpy.rad2deg(
            numpy.arccos(
                ((a0a1 * a0a1) + (a1a2 * a1a2) - (a0a2 * a0a2)) / (2 * a0a1 * a1a2)
            )
        )
        return a0a1, a0a1a2, a1a2

    def hedron_from_atoms(self, atom_coords: Dict["AtomKey", numpy.array]) -> None:
        """Compute length, angle, length for hedron for residue atom coords."""
        acs = cast(HACS, self.gen_acs(atom_coords))

        len1, angle2, len3 = Hedron._get_dad(acs)
        self.len1 = set_accuracy_95(len1)
        self.angle2 = set_accuracy_95(angle2)
        self.len3 = set_accuracy_95(len3)

        # self.atoms_updated = False
        self.init_pos()

    def get_angle(self) -> float:
        """Get this hedron angle."""
        return self.angle2

    def set_angle(self, angle_deg) -> None:
        """Set this hedron angle; clears atoms_updated."""
        self.angle2 = set_accuracy_95(angle_deg)
        self.atoms_updated = False

    def get_length(self, ak_tpl: BKT) -> Optional[float]:
        """Get bond length for specified atom pair.

        :param ak_tpl: tuple of AtomKeys
            pair of atoms in this Hedron
        """
        if 2 > len(ak_tpl):
            return None
        if all(ak in self.aks[:2] for ak in ak_tpl):
            return self.len1
        if all(ak in self.aks[1:] for ak in ak_tpl):
            return self.len3
        return None

    def set_length(self, ak_tpl: BKT, newLength: float):
        """Set bond length for specified atom pair; clears atoms_updated.

        :param ak_tpl: tuple of AtomKeys
            pair of atoms in this Hedron
        """
        if 2 > len(ak_tpl):
            raise TypeError("Require exactly 2 AtomKeys: %s" % (str(ak_tpl)))
        elif all(ak in self.aks[:2] for ak in ak_tpl):
            self.len1 = newLength
        elif all(ak in self.aks[1:] for ak in ak_tpl):
            self.len1 = newLength
        else:
            raise TypeError("%s not found in %s" % (str(ak_tpl), self))
            return
        self.atoms_updated = False


class Dihedron(Edron):
    """Class to represent four joined atoms forming a dihedral angle.

    Attributes
    ----------
    dihedral1: float
        Measurement or specification of dihedral angle
    hedron1, hedron2: Hedron object references
        The two hedra which form the dihedral angle
    h1key, h2key: tuples of AtomKeys
        Hash keys for hedron1 and hedron2
    id3,id32: tuples of AtomKeys
        First 3 and second 3 atoms comprising dihedron; hxkey orders may differ
    initial_coords: tuple[4] of numpy arrays [4][1]
        Local atom coords for 4 atoms, [0] on XZ plane, [1] at origin,
        [2] on +Z, [3] rotated by dihedral1
    a4_pre_rotation: numpy array [4][1]
        4th atom of dihedral aligned to XZ plane (dihedral1 not applied)
    IC_Residue: IC_Residue object reference
        IC_Residue object containing this dihedral
    reverse: bool
        Indicates order of atoms in dihedron is reversed from order of atoms
        in hedra (configured by _set_hedra())

    Methods
    -------
    init_pos()
        Find Hedron objects for self.IC_Residue, set initial_coords
        and a4_pre_rotation
    dihedron_from_atoms()
        Compute dihedral and bond lengths, angles from IC_Residue atom_coords
    get_angle()
        return dihedral1
    set_angle()
        Store new dihedral angle and update initial_coords accordingly

    """

    def __init__(self, *args: Union[List["AtomKey"], DKT], **kwargs: str) -> None:
        """Initialize Dihedron with sequence of AtomKeys and optional dihedral angle.

        Acceptable input:
            As for Edron, plus optional 'dihedral1' keyworded angle value.
        """
        super(Dihedron, self).__init__(*args, **kwargs)

        # hedra making up this dihedron; set by self:_set_hedra()
        self.hedron1: Hedron  # = None
        self.hedron2: Hedron  # = None

        self.h1key: HKT  # = None
        self.h2key: HKT  # = None

        self.id3: HKT = cast(HKT, tuple(self.aks[0:3]))
        self.id32: HKT = cast(HKT, tuple(self.aks[1:4]))

        # 4 matrices specifying hedron space coordinates of constituent atoms,
        # in this space atom 3 is on on +Z axis
        # see coord_space()
        self.initial_coords: DACS
        self.a4_pre_rotation: numpy.array  # = None

        # IC_Residue object which includes this dihedron;
        # set by Residue:linkDihedra()
        self.IC_Residue: IC_Residue  #  = None
        # order of atoms in dihedron is reversed from order of atoms in hedra
        self.reverse = False

        if "dihedral1" in kwargs:
            self.dihedral1 = float(kwargs["dihedral1"])
            # self.init_pos()  # can't do here because need adjacent residues
        # else:
        #    self.dihedral1 = None

        # print(self, self.dclass)

    def __str__(self) -> str:
        """Print string for Dihedron object."""
        return f"4-{str(self.id)} {self.rdh_class} {str(self.dihedral1)} {str(self.IC_Residue)}"

    @staticmethod
    def _get_hedron(ic_res: IC_Residue, id3: HKT) -> Optional[Hedron]:
        """Find specified hedron on this residue or its adjacent neighbors."""
        hedron = ic_res.hedra.get(id3, None)
        if not hedron and 0 < len(ic_res.rprev):
            for rp in ic_res.rprev:
                hedron = rp.hedra.get(id3, None)
                if hedron is not None:
                    break
        if not hedron and 0 < len(ic_res.rnext):
            for rn in ic_res.rnext:
                hedron = rn.hedra.get(id3, None)
                if hedron is not None:
                    break
        return hedron

    def _set_hedra(self) -> Tuple[bool, Hedron, Hedron]:
        """Work out hedra keys and set rev flag."""
        rev = False
        res = self.IC_Residue
        h1key = self.id3
        hedron1 = Dihedron._get_hedron(res, h1key)
        if not hedron1:
            rev = True
            h1key = cast(HKT, tuple(self.aks[2::-1]))
            hedron1 = Dihedron._get_hedron(res, h1key)
            h2key = cast(HKT, tuple(self.aks[3:0:-1]))
        else:
            h2key = self.id32

        if not hedron1:
            raise HedronMatchError(
                "can't find 1st hedron for key %s dihedron %s" % (h1key, self)
            )

        hedron2 = Dihedron._get_hedron(res, h2key)

        if not hedron2:
            raise HedronMatchError(
                "can't find 2nd hedron for key %s dihedron %s" % (h2key, self)
            )

        self.hedron1 = hedron1
        self.h1key = h1key
        self.hedron2 = hedron2
        self.h2key = h2key

        self.reverse = rev

        return rev, hedron1, hedron2

    def init_pos(self, updating: bool = False) -> None:
        """Set hedron-space atom coords with dihedral1 applied.

        :param updating: bool
            skip _set_hedra if True
        """

        if hasattr(self, "hedron1"):
            hedron1 = self.hedron1

            if updating:
                rev = self.reverse
                hedron2 = self.hedron2
        else:
            rev, hedron1, hedron2 = self._set_hedra()

        acount = 0
        for a in hedron1.atoms:
            if a is not None:
                acount += 1
        for a in hedron2.atoms:
            if a is not None:
                acount += 1
        if 6 > acount:
            raise MissingAtomError("dihedron: hedra missing atoms: " + str(self))

        initial = []

        if not rev:
            initial.append(hedron1.atoms[0].copy())
            initial.append(hedron1.atoms[1].copy())
            initial.append(hedron1.atoms[2].copy())

            a4_pre_rotation = hedron2.atomsR[2].copy()
            a4shift = hedron2.len1
        else:
            initial.append(hedron1.atomsR[2].copy())
            initial.append(hedron1.atomsR[1].copy())
            initial.append(hedron1.atomsR[0].copy())

            a4_pre_rotation = hedron2.atoms[0].copy()
            a4shift = hedron2.len3

        # a4 to +Z
        a4_pre_rotation[2][0] *= -1
        # hedron2 shift up so a2 at 0,0,0
        a4_pre_rotation[2][0] += a4shift

        mrz = homog_rot_mtx(numpy.deg2rad(self.dihedral1), "z")
        # initial.append(mrz @ a4_pre_rotation)
        initial.append(mrz.dot(a4_pre_rotation))

        self.initial_coords = cast(DACS, tuple(initial))
        self.a4_pre_rotation = a4_pre_rotation

        self.atoms_updated = True

    def set_angle(self, dangle_deg: float) -> None:
        """Save new dihedral angle and update initial_coords.

        :param dangle_deg: float
            New dihedral angle in degrees
        """
        self.dihedral1 = dangle_deg
        self.atoms_updated = False

    def get_angle(self):
        """Get this object dihedral angle."""
        return self.dihedral1

    @staticmethod
    def _get_dadad(acs: DACS) -> Tuple[float, float, float, float, float]:
        """Get distance, angle, distance, angle, distance for 4 atoms.

        :param acs: list[4] of numpy [4][1] array
            Atom coordinates
        """
        a0 = acs[0].squeeze()
        a1 = acs[1].squeeze()
        a2 = acs[2].squeeze()
        a3 = acs[3].squeeze()

        a0a1 = numpy.linalg.norm(a0 - a1)
        a1a2 = numpy.linalg.norm(a1 - a2)
        a2a3 = numpy.linalg.norm(a2 - a3)

        a0a2 = numpy.linalg.norm(a0 - a2)
        a1a3 = numpy.linalg.norm(a1 - a3)

        sqr_a1a2 = a1a2 * a1a2

        a0a1a2 = numpy.rad2deg(
            numpy.arccos(((a0a1 * a0a1) + sqr_a1a2 - (a0a2 * a0a2)) / (2 * a0a1 * a1a2))
        )

        a1a2a3 = numpy.rad2deg(
            numpy.arccos((sqr_a1a2 + (a2a3 * a2a3) - (a1a3 * a1a3)) / (2 * a1a2 * a2a3))
        )

        return a0a1, a0a1a2, a1a2, a1a2a3, a2a3

    def dihedron_from_atoms(self) -> None:
        """Compute residue dihedral, bond angles, bond lengths.

        Source data is Biopython Residue.Atom coords.
        Call link_dihedra before this so can find Residue.Atom coords.
        Updates hedron and dihedron values, then all local atom coords
        for both hedra and this dihedron.
        """
        rev, hed1, hed2 = self._set_hedra()

        atom_coords = self.IC_Residue.atom_coords
        acs = cast(DACS, self.gen_acs(atom_coords))
        mt = coord_space(acs[:3])
        # do4 = mt @ acs[3]
        do4 = mt.dot(acs[3])

        dh1r = numpy.rad2deg(numpy.arctan2(do4[1][0], do4[0][0]))

        self.dihedral1 = dh1r

        """
        # for testing
        bp_dihed = numpy.rad2deg(
            calc_dihedral(
                Vector(acs[0][0][0], acs[0][1][0], acs[0][2][0]),
                Vector(acs[1][0][0], acs[1][1][0], acs[1][2][0]),
                Vector(acs[2][0][0], acs[2][1][0], acs[2][2][0]),
                Vector(acs[3][0][0], acs[3][1][0], acs[3][2][0]),
            )
        )
        print("dihed: ", dh1r, " ", bp_dihed)
        """

        a0a1, a0a1a2, a1a2, a1a2a3, a2a3 = Dihedron._get_dadad(acs)

        if not rev:
            hed1.len1 = set_accuracy_95(a0a1)
            hed1.len3 = hed2.len1 = set_accuracy_95(a1a2)
            hed2.len3 = set_accuracy_95(a2a3)
        else:
            hed1.len3 = set_accuracy_95(a0a1)
            hed1.len1 = hed2.len3 = set_accuracy_95(a1a2)
            hed2.len1 = set_accuracy_95(a2a3)

        hed1.angle2 = set_accuracy_95(a0a1a2)
        hed2.angle2 = set_accuracy_95(a1a2a3)

        hed1.init_pos()
        hed2.init_pos()

        self.init_pos(True)


class AtomKey(object):
    """Class for dict keys to reference atom coordinates.

    AtomKeys capture residue and disorder information together, and
    provide a no-whitespace string key for .pic files.

    Supports rich comparison and multiple ways to instantiate.

    AtomKeys contain:
     residue position, insertion code, 1 or 3 char residue name,
     atom name, altloc, and occupancy

    Attributes
    ----------
    akl: tuple
        All six fields of AtomKey
    fieldNames: tuple (Class Attribute)
        Mapping of key index positions to names
    fields: namedtuple (Class Attribute)
        Mapping of field names to index positions
    id: str
        '_'-joined AtomKey fields, excluding 'None' fields
    atom_re: compiled regex (Class Attribute)
        A compiled regular expression matching the string form of the key
    d2h: bool
        Convert D atoms to H on input; must also modify IC_Residue.accept_atoms

    Methods
    -------
    altloc_match(other)
        Returns True if this AtomKey matches other AtomKey excluding altloc
        and occupancy fields

    """

    atom_re = re.compile(
        r"^(?P<respos>-?\d+)(?P<icode>[A-Za-z])?"
        r"_(?P<resname>[a-zA-Z]+)_(?P<atm>[A-Za-z0-9]+)"
        r"(?:_(?P<altloc>\w))?(?:_(?P<occ>-?\d\.\d?\d?))?$"
    )

    # PDB altLoc = Character = [\w ] (any non-ctrl ASCII incl space)
    # PDB iCode = AChar = [A-Za-z]

    fieldNames = ("respos", "icode", "resname", "atm", "altloc", "occ")
    fieldsDef = namedtuple(
        "fieldsDef", ["respos", "icode", "resname", "atm", "altloc", "occ"]
    )
    fields = fieldsDef(0, 1, 2, 3, 4, 5)

    d2h = False  # convert D Deuterium to H Hydrogen on input

    def __init__(
        self, *args: Union[IC_Residue, Atom, List, Dict, str], **kwargs: str
    ) -> None:
        """Initialize AtomKey with residue and atom data.

        Examples of acceptable input:
            (<IC_Residue>, 'CA', ...)    : IC_Residue with atom info
            (<IC_Residue>, <Atom>)       : IC_Residue with Biopython Atom
            ([52, None, 'G', 'CA', ...])  : list of ordered data fields
            (52, None, 'G', 'CA', ...)    : multiple ordered arguments
            ({respos: 52, icode: None, atm: 'CA', ...}) : dict with fieldNames
            (respos: 52, icode: None, atm: 'CA', ...) : kwargs with fieldNames
            52_G_CA, 52B_G_CA, 52_G_CA_0.33, 52_G_CA_B_0.33  : id strings
        """
        akl: List[Optional[str]] = []
        # self.id = None
        for arg in args:
            if isinstance(arg, IC_Residue):
                if [] != akl:
                    raise Exception("Atom Key init Residue not first argument")
                akl += arg.rbase
            elif isinstance(arg, Atom):
                if 3 != len(akl):
                    raise Exception("Atom Key init Atom before Residue info")
                akl.append(arg.name)
                altloc = arg.altloc
                akl.append(altloc if altloc != " " else None)
                occ = float(arg.occupancy)
                akl.append(str(occ) if occ != 1.00 else None)
            elif isinstance(arg, list):
                akl += arg
            elif isinstance(arg, dict):
                for k in self.fieldNames:
                    akl.append(arg.get(k, None))
            elif "_" in arg:
                # got atom key string, recurse with regex parse
                m = self.atom_re.match(arg)
                if m is not None:
                    if [] != akl:
                        raise Exception(
                            "Atom Key init full key not first argument: " + arg
                        )
                    for fn in AtomKey.fieldNames:
                        akl.append(m.group(fn))
            else:
                akl.append(arg)

        # process kwargs, initialize occ and altloc to None
        # if not specified above
        for i in range(6):
            if len(akl) <= i:
                fld = kwargs.get(self.fieldNames[i])
                if fld is not None:
                    akl.append(fld)

        # tweak local akl to generate id string
        akl[0] = str(akl[0])  # numeric residue position to string

        # occNdx = self.fields.occ
        # if akl[occNdx] is not None:
        #    akl[occNdx] = str(akl[occNdx])  # numeric occupancy to string

        if self.d2h:
            atmNdx = self.fields.atm
            if akl[atmNdx][0] == "D":
                akl[atmNdx] = re.sub("D", "H", akl[atmNdx], count=1)

            # unused option:
            # (self.respos, self.icode, self.resname, self.atm, self.occ,
            #    self.altloc) = akl

        self.id = "_".join(
            [
                "".join(filter(None, akl[:2])),
                str(akl[2]),  # exclude None
                "_".join(filter(None, akl[3:])),
            ]
        )

        self.akl = tuple(akl)
        self._hash = hash(self.akl)

    def __repr__(self) -> str:
        """Repr string from id."""
        return self.id

    def __hash__(self) -> int:
        """Hash calculated at init from akl tuple."""
        return self._hash

    _backbone_sort_keys = {"N": 0, "CA": 1, "C": 2, "O": 3}

    _sidechain_sort_keys = {
        "CB": 1,
        "CG": 2,
        "CG1": 2,
        "OG": 2,
        "OG1": 2,
        "SG": 2,
        "CG2": 3,
        "CD": 4,
        "CD1": 4,
        "SD": 4,
        "OD1": 4,
        "ND1": 4,
        "CD2": 5,
        "ND2": 5,
        "OD2": 5,
        "CE": 6,
        "NE": 6,
        "CE1": 6,
        "OE1": 6,
        "NE1": 6,
        "CE2": 7,
        "OE2": 7,
        "NE2": 7,
        "CE3": 8,
        "CZ": 9,
        "CZ2": 9,
        "NZ": 9,
        "NH1": 10,
        "OH": 10,
        "CZ3": 10,
        "CH2": 11,
        "NH2": 11,
        "OXT": 12,
        "H": 13,
    }

    _greek_sort_keys = {"A": 0, "B": 1, "G": 2, "D": 3, "E": 4, "Z": 5, "H": 6}

    def altloc_match(self, other: "AtomKey") -> bool:
        """Test AtomKey match other discounting occupancy and altloc."""
        if isinstance(other, type(self)):
            return self.akl[:4] == other.akl[:4]
        else:
            return NotImplemented

    def _cmp(self, other: "AtomKey") -> Tuple[int, int]:
        """Comparison function ranking self vs. other."""
        akl_s = self.akl
        akl_o = other.akl
        atmNdx = self.fields.atm
        occNdx = self.fields.occ
        rsNdx = self.fields.respos
        rsnNdx = self.fields.resname
        for i in range(6):
            s, o = akl_s[i], akl_o[i]
            if s != o:
                # insert_code, altloc can be None, deal with first
                if s is None and o is not None:
                    # no insert code before named insert code
                    return 0, 1
                elif o is None and s is not None:
                    return 1, 0
                # now we know s, o not None
                s, o = cast(str, s), cast(str, o)

                if atmNdx != i:
                    # only sorting complications at atom level, occ.
                    # otherwise respos, insertion code will trigger
                    # before residue name
                    if occNdx == i:
                        oi = int(float(s) * 100)
                        si = int(float(o) * 100)
                        return si, oi  # swap so higher occupancy comes first
                    elif rsNdx == i:
                        return int(s), int(o)
                    else:  # resname or altloc
                        return ord(s), ord(o)

                # backbone atoms before sidechain atoms

                sb = self._backbone_sort_keys.get(s, None)
                ob = self._backbone_sort_keys.get(o, None)
                if sb is not None and ob is not None:
                    return sb, ob
                elif sb is not None and ob is None:
                    return 0, 1
                elif sb is None and ob is not None:
                    return 1, 0
                # finished backbone and backbone vs. sidechain atoms
                # now hydrogens after sidechain
                # s0, o0 = s[0], o[0]
                # if (s0 == 'H' and o0 != 'H'):
                #    return 1, 0
                # elif (s0 != 'H' and o0 == 'H'):
                #    return 0, 1

                ss = self._sidechain_sort_keys.get(s, None)
                os = self._sidechain_sort_keys.get(o, None)
                if ss is not None and os is not None:
                    return ss, os
                elif ss is not None and os is None:
                    return 0, 1
                elif ss is None and os is not None:
                    return 1, 0

                s0, s1, o0, o1 = s[0], s[1], o[0], o[1]
                s1d, o1d = s1.isdigit(), o1.isdigit()
                # if "H" == s0 == o0: # breaks cython
                if ("H" == s0) and ("H" == o0):
                    alphaS = s.rstrip("0123456789")
                    alphaO = o.rstrip("0123456789")
                    if (s1 == o1) or (s1d and o1d):
                        numS = int(s[len(alphaS) :])
                        numO = int(s[len(alphaO) :])
                        return numS, numO
                    elif s1d:
                        return 0, 1
                    elif o1d:
                        return 1, 0
                    else:
                        return (self._greek_sort_keys[s1], self._greek_sort_keys[o1])
                return int(s), int(o)  # raise exception?
        return 1, 1

    def __ne__(self, other: object) -> bool:
        """Test for inequality."""
        if isinstance(other, type(self)):
            return self.akl != other.akl
        else:
            return NotImplemented

    def __eq__(self, other: object) -> bool:  # type: ignore
        """Test for equality."""
        if isinstance(other, type(self)):
            return not self.__ne__(other)
        else:
            return NotImplemented

    def __gt__(self, other: object) -> bool:
        """Test greater than."""
        if isinstance(other, type(self)):
            rslt = self._cmp(other)
            return rslt[0] > rslt[1]
        else:
            return NotImplemented

    def __ge__(self, other: object) -> bool:
        """Test greater or equal."""
        if isinstance(other, type(self)):
            rslt = self._cmp(other)
            return rslt[0] >= rslt[1]
        else:
            return NotImplemented

    def __lt__(self, other: object) -> bool:
        """Test less than."""
        if isinstance(other, type(self)):
            rslt = self._cmp(other)
            return rslt[0] < rslt[1]
        else:
            return NotImplemented

    def __le__(self, other: object) -> bool:
        """Test less or equal."""
        if isinstance(other, type(self)):
            rslt = self._cmp(other)
            return rslt[0] <= rslt[1]
        else:
            return NotImplemented


def set_accuracy_95(num: float) -> float:
    """Reduce floating point accuracy to 9.5 (xxxx.xxxxx).

    Used by Hedron and Dihedron classes.
    :param float num: input number
    :returns: float with specified accuracy
    """
    return float("{:9.5f}".format(num))


def set_accuracy_83(num: float) -> float:
    """Reduce floating point accuracy to 8.3 (xxxxx.xxx).

    Used by IC_Residue class, matches PDB output format.
    :param float num: input number
    :returns: float with specified accuracy
    """
    return float("{:8.3f}".format(num))


# internal coordinates construction Exceptions
class HedronMatchError(Exception):
    """Cannot find hedron in residue for given key."""

    pass


class MissingAtomError(Exception):
    """Missing atom coordinates for hedron or dihedron."""

    pass
