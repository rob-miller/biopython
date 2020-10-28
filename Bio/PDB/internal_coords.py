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

IC_Chain: Extends Biopython Chain on .internal_coord attribute.
    Manages connected sequence of residues and chain breaks; methods generally
    apply IC_Residue methods along chain.

IC_Residue: Extends for Biopython Residue on .internal_coord attribute.
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
import copy

try:
    import numpy as np  # type: ignore
except ImportError:
    from Bio import MissingPythonDependencyError

    raise MissingPythonDependencyError(
        "Install numpy to build proteins from internal coordinates."
    )

from Bio.PDB.Atom import Atom, DisorderedAtom
from Bio.PDB.Polypeptide import three_to_one

from Bio.PDB.vectors import multi_coord_space, multi_rot_Z, multi_rot_Y
from Bio.PDB.vectors import coord_space

# , calc_dihedral, Vector
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
    TYPE_CHECKING,
    Optional,
    ByteString,
)

if TYPE_CHECKING:
    from Bio.PDB.Residue import Residue
    from Bio.PDB.Chain import Chain

HKT = Tuple["AtomKey", "AtomKey", "AtomKey"]  # Hedron key tuple
DKT = Tuple["AtomKey", "AtomKey", "AtomKey", "AtomKey"]  # Dihedron Key Tuple
EKT = Union[HKT, DKT]  # Edron Key Tuple
BKT = Tuple["AtomKey", "AtomKey"]  # Bond Key Tuple

# HACS = Tuple[np.array, np.array, np.array]  # Hedron Atom Coord Set
HACS = np.array  # Hedron Atom Coord Set
DACS = Tuple[np.array, np.array, np.array, np.array]  # Dihedron Atom Coord Set


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
        rtm: list of tuple(N, Ca, C atomKeys)

    MaxPeptideBond: **Class** attribute to detect chain breaks.
        Override for fully contiguous chains with some very long bonds - e.g.
        for 3D printing (OpenSCAD output) a structure with fully disordered
        (missing) residues.

    ParallelAssembleResidues: **Class** attribute affecting internal_to_atom_coords.
        The overhead for processing long (1200 residue) chains can outweigh the
        improvement of calculating dihedra in parallel where possible, so clearing
        flag (set to False) will switch to the serial algorithm

    ordered_aa_ic_list: list of IC_Residue objects
        IC_Residue objects ic algorithms can process (e.g. no waters)

    hedra: dict indexed by 3-tuples of AtomKeys
        Hedra forming residues in this chain

    hedraLen: int length of hedra dict

    hedraNdx: dict mapping hedra AtomKeys to numpy array data

    dihedra: dict indexed by 4-tuples of AtomKeys
        Dihedra forming (overlapping) this residue

    dihedraLen: int length of dihedra dict

    dihedraNdx: dict mapping dihedra AtomKeys to numpy array data

    atomArray: numpy array of homogeneous atom coords for chain

    atomArrayIndex: dict mapping AtomKeys to atomArray indexes

    numpy arrays for vector processing of chain di/hedra:

    hedraIC: length-angle-length entries for each hedron

    new:  rtm
    hedraL12
    hedraAngle
    hedraL23

    hAtoms: homogeneous atom coordinates (3x4) of hedra, central atom at origin

    hAtomsR: hAtoms in reverse order

    hAtoms_needs_update: booleans indicating whether hAtoms represent hedraL12/A/L23

    dihedraAngle: dihedral angles for each dihedron

    dAtoms: homogeneous atom coordinates (4x4) of dihedra, second atom at origin

    dAtoms_needs_update: booleans indicating whether dAtoms represent dihedraAngle

    dCoordSpace: forward and reverse transform matrices standardising positions
        of first hedron

    dcs_valid: booleans indicating dCoordSpace up to date

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
    atom_to_internal_coordinates(verbose)
        Calculate dihedrals, angles, bond lengths (internal coordinates) for
        Atom data
    link_residues()
        Call link_dihedra() on each IC_Residue (needs rprev, rnext set)
    set_residues()
        Add .internal_coord attribute for all Residues in parent Chain, populate
        ordered_aa_ic_list, set IC_Residue rprev, rnext or initNCaC coordinates
    write_SCAD()
        Write OpenSCAD matrices for internal coordinate data comprising chain

    """

    MaxPeptideBond = 1.4  # larger C-N distance than this is chain break
    ParallelAssembleResidues = True  # parallel internal_to_atom, slow for long chains

    # for assemble_residues
    dihedraSelect = np.array([True, True, True, False])
    dihedraOK = np.array([True, True, True, True])

    def __init__(self, parent: "Chain", verbose: bool = False) -> None:
        """Initialize IC_Chain object, with or without residue/Atom data.

        :param parent: Biopython Chain object
            Chain object this extends
        """
        # type hinting parent as Chain leads to import cycle
        self.chain = parent
        self.ordered_aa_ic_list: List[IC_Residue] = []
        # self.initNCaC: Dict[Tuple[str], Dict["AtomKey", np.array]] = {}
        self.initNCaCs = []

        self.sqMaxPeptideBond = IC_Chain.MaxPeptideBond * IC_Chain.MaxPeptideBond
        # need init here for _gen_edra():
        self.hedra = {}
        # self.hedraNdx = {}
        self.dihedra = {}
        # self.dihedraNdx = {}

        # cache of AtomKey results for cak()
        # self.akc: Dict[Tuple(IC_Residue, str), AtomKey] = {}

        self.atomArray: np.array
        self.atomArrayIndex: Dict["AtomKey", int] = {}

        self.set_residues(verbose)  # no effect if no residues loaded

    def __deepcopy__(self, memo) -> "IC_Chain":
        existing = memo.get(id(self), False)
        if existing:
            return existing
        dup = type(self).__new__(self.__class__)
        memo[id(self)] = dup
        dup.chain = memo[id(self.chain)]
        dup.chain.child_dict = copy.deepcopy(self.chain.child_dict, memo)
        # now have all res and ic_res but ic_res not complete
        dup.chain.child_list = copy.deepcopy(self.chain.child_list, memo)
        dup.akset = copy.deepcopy(self.akset, memo)
        dup.aktuple = copy.deepcopy(self.aktuple, memo)
        # now have all ak w/.ric
        dup.ordered_aa_ic_list = copy.deepcopy(self.ordered_aa_ic_list, memo)

        dup.atomArrayIndex = self.atomArrayIndex.copy()
        dup.atomArrayValid = self.atomArrayValid.copy()
        dup.atomArray = self.atomArray.copy()

        dup.hedra = copy.deepcopy(self.hedra, memo)
        dup.dihedra = copy.deepcopy(self.dihedra, memo)

        # update missing items in ic_residues and
        # set all bp residue atom coords to be views on dup.atomArray
        # [similar in buildAtomArray() but does not copy from bpAtoms
        # or modify atomArrayValid, and accesses dup]
        def setAtomVw(res, atm):
            ak = AtomKey(res.internal_coord, atm)
            ndx = dup.atomArrayIndex[ak]
            atm.coord = dup.atomArray[ndx, 0:3]  # make view on atomArray

        def setResAtmVws(res):
            for atm in res.get_atoms():
                if atm.is_disordered():
                    for altAtom in atm.child_dict.values():
                        setAtomVw(res, altAtom)
                else:
                    setAtomVw(res, atm)

        for ric in dup.ordered_aa_ic_list:
            setResAtmVws(ric.residue)
            ric.rprev = copy.deepcopy(ric.rprev, memo)
            ric.rnext = copy.deepcopy(ric.rnext, memo)
            ric.ak_set = copy.deepcopy(ric.ak_set, memo)
            ric.akc = copy.deepcopy(ric.akc, memo)
            ric.dihedra = copy.deepcopy(ric.dihedra, memo)
            ric.hedra = copy.deepcopy(ric.hedra, memo)

        dup.sqMaxPeptideBond = self.sqMaxPeptideBond

        dup.hedraLen = self.hedraLen
        dup.hedraL12 = self.hedraL12.copy()
        dup.hedraAngle = self.hedraAngle.copy()
        dup.hedraL23 = self.hedraL23.copy()
        dup.hedraNdx = copy.deepcopy(self.hedraNdx, memo)

        dup.dihedraLen = self.dihedraLen
        dup.dihedraAngle = self.dihedraAngle.copy()
        dup.dihedraAngleRads = self.dihedraAngleRads.copy()
        dup.dihedraNdx = copy.deepcopy(self.dihedraNdx, memo)

        dup.a2da_map = self.a2da_map.copy()
        dup.a2d_map = self.a2d_map.copy()
        dup.d2a_map = self.d2a_map.copy()

        dup.dH1ndx = self.dH1ndx.copy()
        dup.dH2ndx = self.dH2ndx.copy()

        dup.hAtoms = self.hAtoms.copy()
        dup.hAtomsR = self.hAtomsR.copy()
        dup.hAtoms_needs_update = self.hAtoms_needs_update.copy()

        dup.dRev = self.dRev.copy()
        dup.dFwd = self.dFwd.copy()
        dup.dAtoms_needs_update = self.dAtoms_needs_update.copy()

        dup.dAtoms = self.dAtoms.copy()
        dup.a4_pre_rotation = self.a4_pre_rotation.copy()

        dup.dCoordSpace = self.dCoordSpace.copy()
        dup.dcsValid = self.dcsValid.copy()

        return dup

    # return True if a0, a1 within supplied cutoff
    def _atm_dist_chk(self, a0: Atom, a1: Atom, cutoff: float, sqCutoff: float) -> bool:
        return sqCutoff > np.sum(np.square(a0.coord - a1.coord))

    # return a string describing issue, or None if OK
    def _peptide_check(self, prev: "Residue", curr: "Residue") -> Optional[str]:
        if 0 == len(curr.child_dict):
            # curr residue with no atoms => reading pic file, no break
            return None
        if (0 != len(curr.child_dict)) and (0 == len(prev.child_dict)):
            # prev residue with no atoms, curr has atoms => reading pic file,
            # have break
            return "PIC data missing atoms"

        # handle non-standard AA not marked as HETATM (1KQF, 1NTH)
        if not prev.internal_coord.isAccept:
            return "previous residue not standard/accepted amino acid"

        # both biopython Residues have Atoms, so check distance
        Natom = curr.child_dict.get("N", None)
        pCatom = prev.child_dict.get("C", None)
        if Natom is None or pCatom is None:
            return f"missing {'previous C' if pCatom is None else 'N'} atom"

        # confirm previous residue has all backbone atoms
        pCAatom = prev.child_dict.get("CA", None)
        pNatom = prev.child_dict.get("N", None)
        if pNatom is None or pCAatom is None:
            return "previous residue missing N or Ca"

        if not Natom.is_disordered() and not pCatom.is_disordered():
            dc = self._atm_dist_chk(
                Natom, pCatom, IC_Chain.MaxPeptideBond, self.sqMaxPeptideBond
            )
            if dc:
                return None
            else:
                return f"MaxPeptideBond ({IC_Chain.MaxPeptideBond} angstroms) exceeded"

        # drop through for else Natom or pCatom is disordered:

        Nlist: List[Atom] = []
        pClist: List[Atom] = []
        if Natom.is_disordered():
            Nlist.extend(Natom.child_dict.values())
        else:
            Nlist = [Natom]
        if pCatom.is_disordered():
            pClist.extend(pCatom.child_dict.values())
        else:
            pClist = [pCatom]

        for n in Nlist:
            for c in pClist:
                if self._atm_dist_chk(
                    Natom, pCatom, IC_Chain.MaxPeptideBond, self.sqMaxPeptideBond
                ):
                    return None
        return f"MaxPeptideBond ({IC_Chain.MaxPeptideBond} angstroms) exceeded"

    def clear_ic(self):
        """Clear residue internal_coord settings for this chain."""
        for res in self.chain.get_residues():
            res.internal_coord = None

    def _add_residue(
        self, res: "Residue", last_res: List, last_ord_res: List, verbose: bool = False
    ) -> bool:
        """Set rprev, rnext, manage chain break.

        Returns True for no chain break or residue has sufficient data to restart
        at this position after a chain break (sets initNCaC coordinates in this
        case).  False return means insufficient data to extend chain with this
        residue.
        """
        if not res.internal_coord:
            res.internal_coord = IC_Residue(res)
            res.internal_coord.cic = self
        if (
            0 < len(last_res)
            and last_ord_res == last_res
            and self._peptide_check(last_ord_res[0].residue, res) is None
        ):
            # no chain break
            for prev in last_ord_res:
                prev.rnext.append(res.internal_coord)
                res.internal_coord.rprev.append(prev)
            return True
        elif all(atm in res.child_dict for atm in ("N", "CA", "C")):
            # chain break, save coords for restart
            if verbose and len(last_res) != 0:  # not first residue
                if last_ord_res != last_res:
                    reason = f"disordered residues after {last_ord_res.pretty_str()}"
                else:
                    reason = cast(
                        str, self._peptide_check(last_ord_res[0].residue, res)
                    )
                print(
                    f"chain break at {res.internal_coord.pretty_str()} due to {reason}"
                )
            # initNCaC: Dict["AtomKey", np.array] = {}
            initNCaC = []
            ric = res.internal_coord
            for atm in ("N", "CA", "C"):
                bpAtm = res.child_dict[atm]
                if bpAtm.is_disordered():
                    for altAtom in bpAtm.child_dict.values():
                        ak = AtomKey(ric, altAtom)
                        # initNCaC[ak] = IC_Residue.atm241(altAtom.coord)
                else:
                    ak = AtomKey(ric, bpAtm)
                    initNCaC.append(ak)
                    # initNCaC[ak] = IC_Residue.atm241(bpAtm.coord)
            # self.initNCaC[ric.rbase] = initNCaC
            self.initNCaCs.append(tuple(initNCaC))
            return True
        elif (
            0 == len(res.child_list)
            and self.chain.child_list[0].id == res.id
            and (res.internal_coord.isAccept)
        ):
            # this is first residue, no atoms at all, is std amino acid
            # conclude reading pic file with no N-Ca-C coords
            return True
        # chain break but do not have N, Ca, C coords to restart from
        return False

    def set_residues(self, verbose: bool = False) -> None:
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

        # atomCoordDict = {}
        akset = set()
        for res in self.chain.get_residues():
            # select only not hetero or accepted hetero
            if res.id[0] == " " or res.id[0] in IC_Residue.accept_resnames:
                this_res: List["IC_Residue"] = []
                if 2 == res.is_disordered():
                    # print('disordered res:', res.is_disordered(), res)
                    for r in res.child_dict.values():
                        if self._add_residue(r, last_res, last_ord_res, verbose):
                            this_res.append(r.internal_coord)
                            akset.update(r.internal_coord.ak_set)
                else:
                    if self._add_residue(res, last_res, last_ord_res, verbose):
                        this_res.append(res.internal_coord)
                        akset.update(res.internal_coord.ak_set)

                if 0 < len(this_res):
                    self.ordered_aa_ic_list.extend(this_res)
                    last_ord_res = this_res
                    # for ric in this_res:  # rtm think not needed after vectorise:
                    #    atomCoordDict.update(ric.atom_coords_vw)

                last_res = this_res

        self.akset = akset
        # wait for gcb - self.aktuple = tuple(sorted(akset))

        # if last_ord_res != []:
        #    self.build_atomArray(akset) # do this after adding gcb's
        # aa = np.array(tuple(atomCoordDict.values()))
        # self.atomArray = np.insert(aa, 3, 1, axis=1)  # make homogeneous
        # # self.atomVwArray = aa  # rtm:BpAtmVw
        # siz = len(atomCoordDict)
        # self.atomArrayIndex = dict(zip(atomCoordDict.keys(), range(siz)))
        # self.atomArrayValid = np.ones(siz, dtype=bool)

        # set all ric.atom_coords to be views on chain atomArray
        # rtm think not needed after vectorise:
        # for ric in self.ordered_aa_ic_list:
        #    for ak in ric.atom_coords_vw.keys():
        #        ric.atom_coords[ak] = self.atomArray[self.atomArrayIndex[ak]]
        # rtm temp option
        # for ric in self.ordered_aa_ic_list:
        #    for ak in ric.ak_set:
        #        ric.atom_coords[ak] = self.atomArray[self.atomArrayIndex[ak]]

    def link_residues(self) -> None:
        """link_dihedra() for each IC_Residue; needs rprev, rnext set.

        Called by PICIO:read_PIC() after finished reading chain
        """
        for ric in self.ordered_aa_ic_list:
            ric.cic = self
            ric.link_dihedra()

    def build_atomArray(self) -> None:
        """Create Chain numpy coordinate array from biopython atoms."""

        def setAtom(res, atm):
            ak = AtomKey(res.internal_coord, atm)
            try:
                ndx = self.atomArrayIndex[ak]
            except KeyError:
                return
            self.atomArray[ndx, 0:3] = atm.coord
            atm.coord = self.atomArray[ndx, 0:3]  # make view on atomArray
            self.atomArrayValid[ndx] = True

        def setResAtms(res):
            for atm in res.get_atoms():
                if atm.is_disordered():
                    for altAtom in atm.child_dict.values():
                        setAtom(res, altAtom)
                else:
                    setAtom(res, atm)

        self.AAsiz = len(self.akset)
        # sorted(akset) needed here for pdb atom serial number and to maintain
        # consistency between a2ic and i2ac
        self.aktuple = tuple(sorted(self.akset))
        self.atomArrayIndex = dict(zip(self.aktuple, range(self.AAsiz)))
        self.atomArrayValid = np.zeros(self.AAsiz, dtype=bool)
        self.atomArray = np.zeros((self.AAsiz, 4), dtype=np.float64)
        self.atomArray[:, 3] = 1.0

        for ric in self.ordered_aa_ic_list:
            setResAtms(ric.residue)
            if ric.akc == {}:  # pic file read
                ric.build_rak_cache()

    def build_edraArrays(self) -> None:
        """Build chain level hedra and dihedra arrays."""
        # dihedra coord space
        self.dCoordSpace: np.ndarray = np.empty(
            (2, self.dihedraLen, 4, 4), dtype=np.float64
        )
        self.dcsValid: np.ndarray = np.zeros((self.dihedraLen), dtype=np.bool)

        # hedra atoms
        self.hAtoms: np.ndarray = np.zeros((self.hedraLen, 3, 4), dtype=np.float64)
        self.hAtoms[:, :, 3] = 1.0  # homogeneous
        self.hAtomsR: np.ndarray = np.copy(self.hAtoms)
        self.hAtoms_needs_update = np.full(self.hedraLen, True)

        # maps between hAtoms and atomArray
        a2ha_map = {}
        self.a2h_map = [[] for _ in range(self.AAsiz)]
        # bond_map = {}

        h2aa = [[] for _ in range(self.hedraLen)]
        for hk, hndx in self.hedraNdx.items():
            hstep = hndx * 3
            for i in range(3):
                ndx = self.atomArrayIndex[hk[i]]
                a2ha_map[hstep + i] = ndx
            self.hedra[hk].ndx = hndx
            for ak in self.hedra[hk].aks:
                akndx = self.atomArrayIndex[ak]
                h2aa[hndx].append(akndx)
                self.a2h_map[akndx].append(hndx)
            # a, b, c = hk[:]
            # j, k, l = a2ha_map[hstep : hstep + 3]
            # t0 = ((a, b) if (a <= b) else (b, a))
            # t1 = ((b, c) if (b <= c) else (c, b))
            # t2 = ((a, c) if (a <= c) else (c, a))
            # bond_map[t0] = (j, k)
            # bond_map[t1] = (k, l)
            # bond_map[t2] = (j, l)
        self.a2ha_map = np.array(tuple(a2ha_map.values()))
        self.h2aa = np.array(h2aa)

        # dihedra atoms
        self.dAtoms: np.ndarray = np.empty((self.dihedraLen, 4, 4), dtype=np.float64)
        self.dAtoms[:, :, 3] = 1.0  # homogeneous
        self.a4_pre_rotation = np.empty((self.dihedraLen, 4))

        # maps between dAtoms and atomArray
        # hedra and dihedra
        # dihedra forward/reverse data
        a2da_map = {}
        a2d_map = [[[], []] for _ in range(self.AAsiz)]
        self.dRev: np.ndarray = np.zeros((self.dihedraLen), dtype=np.bool)

        self.dH1ndx = np.empty(self.dihedraLen, dtype=np.int)
        self.dH2ndx = np.empty(self.dihedraLen, dtype=np.int)
        self.h1d_map = [[] for _ in range(self.hedraLen)]
        # self.id3_dh_index = dict((k[0:3], []) for k in self.dihedraNdx.keys())
        self.id3_dh_index = {k[0:3]: [] for k in self.dihedraNdx.keys()}
        d2aa = [[] for _ in range(self.dihedraLen)]
        for dk, dndx in self.dihedraNdx.items():
            # build map between atomArray and dAtoms
            dstep = dndx * 4
            did3 = dk[0:3]
            d = self.dihedra[dk]
            for i in range(4):
                ndx = self.atomArrayIndex[dk[i]]
                a2da_map[dstep + i] = ndx
                a2d_map[ndx][0].append(dndx)
                a2d_map[ndx][1].append(i)

            try:
                d.h1key = did3
                d.h2key = dk[1:4]
                h1ndx = self.hedraNdx[d.h1key]
            except KeyError:
                d.h1key = dk[2::-1]
                d.h2key = dk[3:0:-1]
                h1ndx = self.hedraNdx[d.h1key]
                self.dRev[dndx] = True
                d.reverse = True

            h2ndx = self.hedraNdx[d.h2key]
            d.hedron1 = self.hedra[d.h1key]
            d.hedron2 = self.hedra[d.h2key]
            self.dH1ndx[dndx] = h1ndx
            self.dH2ndx[dndx] = h2ndx
            self.h1d_map[h1ndx].append(dndx)

            d.ndx = dndx
            d.cst = self.dCoordSpace[0][dndx]
            d.rcst = self.dCoordSpace[1][dndx]
            for ak in d.aks:
                d2aa[dndx].append(self.atomArrayIndex[ak])
            self.id3_dh_index[did3].append(dk)

        self.a2da_map = np.array(tuple(a2da_map.values()))
        self.d2a_map = self.a2da_map.reshape(-1, 4)
        self.dFwd = self.dRev != True  # noqa: E712
        self.d2aa = np.array(d2aa)

        # manually create np.where(atom in dihedral)
        self.a2d_map = [(np.array(xi[0]), np.array(xi[1])) for xi in a2d_map]
        self.dAtoms_needs_update = np.full(self.dihedraLen, True)

    def hedraDict2chain(
        self,
        hl12: Dict[str, float],
        ha: Dict[str, float],
        hl23: Dict[str, float],
        da: Dict[str, float],
        bfacs: Dict[str, float],
    ) -> None:
        """Generate chain numpy arrays from PICIO read_PIC() dicts.

        On entry chain internal_coord has ordered_aa_ic_list built, akset;
        residues have rnext, rprev, ak_set and di/hedra dicts initialised.
        Chain, residues do NOT have NCaC info, id3_dh_index
        Di/hedra have cic, aks set
        Dihedra do NOT have valid reverse flag, h1/2 info
        """
        # self.set_residues()

        # do in build_atomArray - self.aktuple = tuple(sorted(self.akset))

        for ric in self.ordered_aa_ic_list:
            # log chain starts - beginning and after breaks
            # chain starts are only atom coords in pic files
            # assume valid pic files with all 3 of N, Ca, C coords
            initNCaC = []
            for atm in ric.residue.get_atoms():
                if 2 == atm.is_disordered():
                    for altAtom in atm.child_dict.values():
                        if altAtom.coord is not None:
                            initNCaC.append(AtomKey(ric, altAtom))
                elif atm.coord is not None:
                    initNCaC.append(AtomKey(ric, atm))
            if initNCaC != []:
                self.initNCaCs.append(tuple(initNCaC))

            # next residue NCaCKeys so can do per-residue assemble()
            ric.NCaCKey = []
            ric.NCaCKey.extend(
                ric._split_akl(
                    (AtomKey(ric, "N"), AtomKey(ric, "CA"), AtomKey(ric, "C"))
                )
            )
            pass

        # set any supplied coordinates from biopython atoms
        # just loaded pic file so only start/chain break residues
        # will have atoms
        self.build_atomArray()

        # now create all biopython atoms for chain, setting coords to be view
        # on atomArray entry
        spNdx, icNdx, resnNdx, atmNdx, altlocNdx, occNdx = AtomKey.fields
        sn = None
        for ak, ndx in self.atomArrayIndex.items():
            res = ak.ric.residue  # read_PIC inits with IC_Residue
            atm, altloc = ak.akl[atmNdx], ak.akl[altlocNdx]
            occ = 1.00 if ak.akl[occNdx] is None else float(ak.akl[occNdx])
            bfac = bfacs.get(ak.id, 0.0)
            sn = sn + 1 if sn is not None else ndx + 1
            bpAtm = None
            if res.has_id(atm):
                bpAtm = res[atm]
            if bpAtm is None or (
                2 == bpAtm.is_disordered() and not bpAtm.disordered_has_id(altloc)
            ):
                # print('new', ak)
                newAtom = Atom(
                    atm,
                    self.atomArray[ndx][0:3],  # init as view on atomArray
                    bfac,
                    occ,
                    (" " if altloc is None else altloc),
                    atm,
                    sn,
                    atm[0],
                )
                if bpAtm is None:
                    if altloc is None:
                        res.add(newAtom)
                    else:
                        disordered_atom = DisorderedAtom(atm)
                        res.add(disordered_atom)
                        disordered_atom.disordered_add(newAtom)
                        res.flag_disordered()
                else:
                    bpAtm.disordered_add(newAtom)

            else:
                if 2 == bpAtm.is_disordered() and bpAtm.disordered_has_id(altloc):
                    bpAtm.disordered_select(altloc)
                # bpAtm.set_coord(self.atomArray[ndx][0:3])  # done in build_atomArray
                bpAtm.set_bfactor(bfac)
                bpAtm.set_occupancy(occ)
                sn = bpAtm.get_serial_number()

            # pass
            # ak.ric.akc[bpAtm] = ak

        # hedra
        self.hedraLen = len(ha)
        self.hedraL12 = np.fromiter(hl12.values(), dtype=np.float64)
        self.hedraAngle = np.fromiter(ha.values(), dtype=np.float64)
        self.hedraL23 = np.fromiter(hl23.values(), dtype=np.float64)
        self.hedraNdx = dict(zip(ha.keys(), range(self.hedraLen)))

        # dihedra
        self.dihedraLen = len(da)
        self.dihedraAngle = np.fromiter(da.values(), dtype=np.float64)
        self.dihedraAngleRads = np.deg2rad(self.dihedraAngle)
        self.dihedraNdx = dict(zip(da.keys(), range(self.dihedraLen)))

        self.build_edraArrays()

        pass  # rtm

    # @profile
    def ar2(self, verbose: bool = False) -> None:
        """Generate atom coords from internal coords, vectorised.

        Starting with dihedra already formed by init_atom_coords(), transform
        each from dihedron build coordinate space into protein chain coordinate
        space.  Iterate until all dependencies satisfied.

        Does not update dCoordSpace as assemble _residues() does - call
        update_dCoordSpace() if needed.  Faster to do in single operation once
        all atom coordinates finished.

        :param verbose bool: default False
            report number of iterations to compute changed dihedra
        """
        # dihedron atom positions of chain atom ndxs, maps atomArray to dihedra
        a2da_map = self.a2da_map  # 8468 x int
        # each chain atom to list of [dihedron], [dihedron_position]
        a2d_map = self.a2d_map  # 2000 x ([int], [int])
        # every dihedron atom to chain atoms
        d2a_map = self.d2a_map  # 2117 x [4] ints
        # all chain atoms
        # rtm test code
        # self.atomArrayValid[:] = False
        # self.atomArrayValid[0:3] = True
        atomArray = self.atomArray  # 2000
        # bool markers for chain atoms with valid coordinates
        atomArrayValid = self.atomArrayValid  # 2000

        """ """
        # rtm
        # atomArrayValid[...] = False
        # atomArrayValid[0:3] = True
        """
        atomArray[3:2000] = [
            0,
            0,
            0,
            1,
        ]  # np.zeros((self.AAsiz, 4), dtype=np.float64)
        # atomArray[:, 3] = 1.0
        """

        # complete array of dihedra atoms
        dAtoms = self.dAtoms  # 2117 x [4][4] float
        # coordinate space transformations optionally supplied
        dCoordSpace1 = self.dCoordSpace[1]
        dcsValid = self.dcsValid

        # rtm
        # foo = self.ordered_aa_ic_list[0].assemble()
        # if

        # dSet is 4-atom arrays for every dihedral, multiple copies of
        # many atoms as the dihedra overlap
        dSet = atomArray[a2da_map].reshape(-1, 4, 4)
        # dSetValid indicates accurate atom positions in each dSet dihedral
        dSetValid = atomArrayValid[a2da_map].reshape(-1, 4)

        # clear any transforms for dihedrals with outdated atoms
        workSelector = (dSetValid == self.dihedraOK).all(axis=1)
        # rtm this breaks rebuild-test copyCoordSpace
        self.dcsValid[np.logical_not(workSelector)] = False

        if verbose:
            dihedraWrk = workSelector.size - workSelector.sum()

        # mask for dihedral with 3 valid atoms in dSet, ready to be processed:
        targ = IC_Chain.dihedraSelect
        # select the dihedrals ready for processing
        workSelector = (dSetValid == targ).all(axis=1)

        loopCount = 0
        while np.any(workSelector):
            # indexes of dihedra in dset to update
            workNdxs = np.where(workSelector)
            # subset of dihedra to update
            workSet = dSet[workSelector]
            # will update coordinates of 4th atom in each workSet dihedron
            updateMap = d2a_map[workNdxs, 3][0]

            # get all coordSpace transforms
            if np.all(dcsValid[workSelector]):
                cspace = dCoordSpace1[workSelector]
            else:
                cspace = multi_coord_space(workSet, np.sum(workSelector), True)[1]
            # foo = self.dCoordSpace[1][workSelector]

            # generate new coords for 4th atoms in workSet dihedra
            initCoords = dAtoms[workSelector].reshape(-1, 4, 4)
            """
            atomArray[updateMap] = np.round(
                np.einsum("ijk,ik->ij", cspace, initCoords[:, 3]), 3
            )  # must round to PDB resolution here or get coordinate drift along chain
            """
            # rtm temporary no rounding
            atomArray[updateMap] = np.einsum("ijk,ik->ij", cspace, initCoords[:, 3])

            # mark new computed atom positions as valid
            atomArrayValid[updateMap] = True

            workSelector[:] = False
            # atoms may map to multiple dihedrals so seems like this has to be
            # list of lists processing, not array
            for a in updateMap:
                # copy new atom positions into dihedra atom array
                dSet[a2d_map[a]] = atomArray[a]
                # build new workSelector from only updated dihedra
                adlist = a2d_map[a]
                for d in adlist[0]:
                    dvalid = atomArrayValid[d2a_map[d]]
                    workSelector[d] = (dvalid == targ).all()
                """
                for d, p in zip(adlist[0], adlist[1]):
                    if p != 3:  # if a is not atom4 for this d (that we just placed)
                        atomArrayValid[
                            d2a_map[d]
                        ] = False  # then need to update rest in d
                        atomArrayValid[a] = True
                        atomArrayValid[updateMap] = True
                    # dvalid = atomArrayValid[d2a_map[d]]
                    # workSelector[d] = (dvalid == targ).all()
                    dSetValid = atomArrayValid[a2da_map].reshape(-1, 4)
                    workSelector = (dSetValid == targ).all(axis=1)
                """

            loopCount += 1

        if verbose:
            cid = self.chain.full_id
            print(
                f"{cid[0]} {cid[2]} coordinates for {dihedraWrk} dihedra updated in {loopCount} iterations"
            )

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
        # for ric in self.ordered_aa_ic_list:
        #    ric.clear_transforms()
        self.dcsValid[:] = False

        for ric in self.ordered_aa_ic_list:
            # if not hasattr(ric, "NCaCKey"):
            #    if verbose:
            #        print(
            #            f"no assembly for {str(ric)} due to missing N, Ca and/or C atoms"
            #        )
            #    continue
            respos = ric.residue.id[1]
            if start and start > respos:
                continue
            if fin and fin < respos:
                continue

            # rtm
            ric.atom_coords = cast(
                Dict[AtomKey, np.array], ric.assemble(verbose=verbose)
            )
            if ric.atom_coords:
                ric.ak_set = set(ric.atom_coords.keys())

    def coords_to_structure(self) -> None:
        """Promote all ic atom_coords to Biopython Residue/Atom coords.

        IC atom_coords are homogeneous [4], Biopython atom coords are XYZ [3].
        """
        # rtm TODO: improve with numpy views on biopython Atom numpy array?
        # rtm:BpAtmVw
        # rtm not used any more

        # rounding here is faster than in assemble_residues, but get drift
        # along chain as process structure
        # np.round(self.atomArray, decimals=3, out=self.atomArray)

        self.ndx = 0
        for res in self.chain.get_residues():
            if 2 == res.is_disordered():
                for r in res.child_dict.values():
                    if r.internal_coord:
                        if r.internal_coord.atom_coords:
                            r.internal_coord.coords_to_residue()
                        elif (
                            r.internal_coord.rprev
                            and r.internal_coord.rprev[0].atom_coords
                        ):
                            r.internal_coord.rprev[0].coords_to_residue(rnext=True)
            elif res.internal_coord:
                if res.internal_coord.atom_coords:
                    res.internal_coord.coords_to_residue()
                elif (
                    res.internal_coord.rprev and res.internal_coord.rprev[0].atom_coords
                ):
                    res.internal_coord.rprev[0].coords_to_residue(rnext=True)

    def init_edra(self) -> None:
        """Create chain and residue di/hedra structures, init atomArray if needed."""
        """
        print("INIT_EDRA")
        if self.hedra == {}:
            # rtm this branch not used by read_pic now
            print("foo foo foo foo foo")
            if self.dihedra != {} or self.atomArrayIndex != {}:
                raise RuntimeError("di/hedra sets inconsistent")

            # loaded objects from PIC file, so no chain-level di/hedra
            # hLAL = {}
            hL12 = {}
            hAngle = {}
            hL23 = {}
            dic = {}
            ak_set = set()
            for ric in self.ordered_aa_ic_list:
                for k, h in ric.hedra.items():
                    self.hedra[k] = h
                    hL12[k] = h.L12  # h.lal[0]
                    hAngle[k] = h.Angle  # h.lal[1]
                    hL23[k] = h.L23  # h.lal[2]
                for k, d in ric.dihedra.items():
                    self.dihedra[k] = d
                    dic[k] = d.angle
                ak_set.update(ric.ak_set)

            # hedra
            self.hedraLen = len(self.hedra)
            # self.hedraIC = np.array(tuple(hLAL.values()))
            self.hedraL12 = np.fromiter(hL12.values(), dtype=np.float64)
            self.hedraAngle = np.fromiter(hAngle.values(), dtype=np.float64)
            self.hedraL23 = np.fromiter(hL23.values(), dtype=np.float64)

            # dihedra
            self.dihedraAngle = np.array(tuple(dic.values()))
            self.dihedraAngleRads = np.deg2rad(self.dihedraAngle)
            self.dihedraLen = len(self.dihedra)

            # homogeneous atoms to be in PDB coordinate space
            siz = len(ak_set)
            self.atomArrayIndex = dict(zip(ak_set, range(siz)))
            self.atomArrayValid = np.zeros(siz, dtype=bool)
            self.atomArray = np.zeros((siz, 4), dtype=np.float64)
            self.atomArray[:, 3] = 1

            # for atm in self.initNCaC.values():
            #    for ak, coords in atm.items():
            #        ndx = self.atomArrayIndex[ak]
            #        self.atomArray[ndx] = coords
            #        self.atomArrayValid[ndx] = True

        else:
            # atom_to_internal_coords() populates self.hedra via _gen_edra()
            # a_to_ic will set ic so create empty

            # rtm this branch not used either

            # hedra
            self.hedraLen = len(self.hedra)
            # self.hedraIC = np.empty((self.hedraLen, 3), dtype=np.float64)
            self.hedraL12 = np.empty((self.hedraLen), dtype=np.float64)
            self.hedraAngle = np.empty((self.hedraLen), dtype=np.float64)
            self.hedraL23 = np.empty((self.hedraLen), dtype=np.float64)

            # dihedra
            self.dihedraLen = len(self.dihedra)
            self.dihedraAngle = np.empty(self.dihedraLen)
            self.dihedraAngleRads = np.empty(self.dihedraLen)

            # atomArray built by IC_Chain.set_residues()

        # hedra
        self.hedraNdx = dict(zip(self.hedra.keys(), range(len(self.hedra))))

        self.hAtoms: np.ndarray = np.zeros((self.hedraLen, 3, 4), dtype=np.float64)
        self.hAtoms[:, :, 3] = 1.0  # homogeneous
        self.hAtomsR: np.ndarray = np.copy(self.hAtoms)
        self.hAtoms_needs_update = np.full(self.hedraLen, True)

        for ric in self.ordered_aa_ic_list:
            for k, h in ric.hedra.items():
                hndx = self.hedraNdx[k]
                # all h.lal become views on hedraIC
                # h.lal = self.hedraIC[hndx]
                h.L12 = self.hedraL12[hndx]
                h.Angle = self.hedraAngle[hndx]
                h.L23 = self.hedraL23[hndx]
                h.ndx = hndx

        # dihedra
        self.dihedraNdx = dict(zip(self.dihedra.keys(), range(self.dihedraLen)))

        self.dAtoms: np.ndarray = np.empty((self.dihedraLen, 4, 4), dtype=np.float64)
        self.dAtoms[:, :, 3] = 1.0  # homogeneous

        self.dCoordSpace: np.ndarray = np.empty(
            (2, self.dihedraLen, 4, 4), dtype=np.float64
        )

        self.dcsValid: np.ndarray = np.zeros((self.dihedraLen), dtype=np.bool)

        self.a4_pre_rotation = np.empty((self.dihedraLen, 4))
        a2da_map = {}  # np.empty(self.dihedraLen * 4, dtype=np.int)
        a2d_map = [[[], []] for _ in range(self.atomArrayValid.size)]

        for k, d in self.dihedra.items():
            dndx = self.dihedraNdx[k]
            d.ndx = dndx
            d.initial_coords = self.dAtoms[dndx]
            d.a4_pre_rotation = self.a4_pre_rotation[dndx]
            d.cst = self.dCoordSpace[0][dndx]
            d.rcst = self.dCoordSpace[1][dndx]
            # build map between atomArray and dAtoms
            dstep = dndx * 4
            for i in range(4):
                ndx = self.atomArrayIndex[k[i]]
                a2da_map[dstep + i] = ndx
                a2d_map[ndx][0].append(dndx)
                a2d_map[ndx][1].append(i)

        self.a2da_map = np.array(tuple(a2da_map.values()))
        self.d2a_map = self.a2da_map.reshape(-1, 4)

        # manually create np.where(atom in dihedral)
        self.a2d_map = [(np.array(xi[0]), np.array(xi[1])) for xi in a2d_map]

        self.dAtoms_needs_update = np.full(self.dihedraLen, True)

        self.dRev = np.array(tuple(d.reverse for d in self.dihedra.values()))
        self.dFwd = self.dRev != True  # noqa: E712
        self.dH1ndx = np.array(
            tuple(self.hedraNdx[d.h1key] for d in self.dihedra.values())
        )
        self.dH2ndx = np.array(
            tuple(self.hedraNdx[d.h2key] for d in self.dihedra.values())
        )
    """

    # @profile
    def init_atom_coords(self) -> None:
        """Set chain level di/hedra initial coords from angles and distances."""
        if not np.all(self.dAtoms_needs_update):
            self.dAtoms_needs_update |= (self.hAtoms_needs_update[self.dH1ndx]) | (
                self.hAtoms_needs_update[self.dH2ndx]
            )
            self.dcsValid &= np.logical_not(self.dAtoms_needs_update)

        # dihedra full size masks:
        mdFwd = self.dFwd & self.dAtoms_needs_update
        mdRev = self.dRev & self.dAtoms_needs_update

        # update size masks
        udFwd = self.dFwd[self.dAtoms_needs_update]
        udRev = self.dRev[self.dAtoms_needs_update]

        if np.any(self.hAtoms_needs_update):
            # hedra inital coords

            # supplementary angle radian: angles which add to 180 are supplementary
            sar = np.deg2rad(
                # 180.0 - self.hedraIC[:, 1][self.hAtoms_needs_update]
                180.0
                - self.hedraAngle[self.hAtoms_needs_update]
            )  # angle
            sinSar = np.sin(sar)
            cosSarN = np.cos(sar) * -1

            # a2 is len3 up from a2 on Z axis, X=Y=0
            self.hAtoms[:, 2, 2][self.hAtoms_needs_update] = self.hedraL23[
                self.hAtoms_needs_update
            ]

            # a0 X is sin( sar ) * len12
            self.hAtoms[:, 0, 0][self.hAtoms_needs_update] = (
                sinSar * self.hedraL12[self.hAtoms_needs_update]
            )

            # a0 Z is -(cos( sar ) * len12)
            # (assume angle always obtuse, so a0 is in -Z)
            self.hAtoms[:, 0, 2][self.hAtoms_needs_update] = (
                cosSarN * self.hedraL12[self.hAtoms_needs_update]
            )

            # same again but 'reversed' : a0 on Z axis, a1 at origin, a2 in -Z

            # a0r is len12 up from a1 on Z axis, X=Y=0
            self.hAtomsR[:, 0, 2][self.hAtoms_needs_update] = self.hedraL12[
                self.hAtoms_needs_update
            ]
            # a2r X is sin( sar ) * len23
            self.hAtomsR[:, 2, 0][self.hAtoms_needs_update] = (
                sinSar * self.hedraL23[self.hAtoms_needs_update]
            )
            # a2r Z is -(cos( sar ) * len23)
            self.hAtomsR[:, 2, 2][self.hAtoms_needs_update] = (
                cosSarN * self.hedraL23[self.hAtoms_needs_update]
            )

            self.hAtoms_needs_update[...] = False

            # dihedra parts other than dihedral angle

            dhlen = np.sum(self.dAtoms_needs_update)  # self.dihedraLen

            # only 4th atom takes work:
            # pick 4th atom based on rev flag
            self.a4_pre_rotation[mdRev] = self.hAtoms[self.dH2ndx, 0][mdRev]
            self.a4_pre_rotation[mdFwd] = self.hAtomsR[self.dH2ndx, 2][mdFwd]

            # numpy multiply, add operations below intermediate array but out= not
            # working with masking:
            self.a4_pre_rotation[:, 2][self.dAtoms_needs_update] = np.multiply(
                self.a4_pre_rotation[:, 2][self.dAtoms_needs_update], -1
            )  # a4 to +Z

            a4shift = np.empty(dhlen)
            a4shift[udRev] = self.hedraL23[self.dH2ndx][mdRev]  # len23
            a4shift[udFwd] = self.hedraL12[self.dH2ndx][mdFwd]  # len12

            self.a4_pre_rotation[:, 2][self.dAtoms_needs_update] = np.add(
                self.a4_pre_rotation[:, 2][self.dAtoms_needs_update], a4shift,
            )  # so a2 at origin

            # now build dihedra initial coords

            dH1atoms = self.hAtoms[self.dH1ndx]  # fancy indexing so
            dH1atomsR = self.hAtomsR[self.dH1ndx]  # these copy not view

            self.dAtoms[:, :3][mdFwd] = dH1atoms[mdFwd]
            # self.dAtoms[:, 3][mdFwd] = a4rot[udFwd]  # [self.dFwd]

            self.dAtoms[:, :3][mdRev] = dH1atomsR[:, 2::-1][mdRev]
            # self.dAtoms[:, 3][mdRev] = a4rot[udRev]  # [self.dRev]

            # self.dAtoms_needs_update[...] = False

        # build rz rotation matrix for dihedral angle
        rz = multi_rot_Z(self.dihedraAngleRads[self.dAtoms_needs_update])

        a4rot = np.matmul(
            rz, self.a4_pre_rotation[self.dAtoms_needs_update][:].reshape(-1, 4, 1)
        ).reshape(-1, 4)

        self.dAtoms[:, 3][mdFwd] = a4rot[udFwd]  # [self.dFwd]
        # self.dAtoms[:, :3][mdRev] = dH1atomsR[:, 2::-1][mdRev]
        self.dAtoms[:, 3][mdRev] = a4rot[udRev]  # [self.dRev]

        self.dAtoms_needs_update[...] = False

        # can't start assembly if initial NCaC is not valid, so copy from
        # hAtoms if needed
        # rtm would be better to use coordspace if oiginal coords are non-zero
        for iNCaC in self.initNCaCs:
            invalid = True
            if np.all(self.atomArrayValid[[self.atomArrayIndex[ak] for ak in iNCaC]]):
                invalid = False

            if invalid:
                hatoms = self.hAtoms[self.hedraNdx[iNCaC]]
                for i in range(3):
                    andx = self.atomArrayIndex[iNCaC[i]]
                    self.atomArray[andx] = hatoms[i]
                    self.atomArrayValid[andx] = True

                # print(self.hedraNdx[iNCaC])

    def update_dCoordSpace(self, workSelector: Optional[np.ndarray] = None) -> None:
        """Compute/update multiple coordinate space transforms for chain dihedra.

        Requires all atoms updated so calls ar2().

        :param workSelector numpy bool array: default None = update as needed
            mask to select dihedra for update
        """
        # rtm axes wrong
        if workSelector is None:
            self.ar2()  # ensure atoms updated, fast if nothing to do
            workSelector = np.logical_not(self.dcsValid)
        workSet = self.dSet[workSelector]
        self.dCoordSpace[:][workSelector] = multi_coord_space(
            workSet, np.sum(workSelector), True
        )
        # cspace = multi_coord_space(workSet, np.sum(workSelector), True)
        # self.dCoordSpace[workSelector] = np.swapaxes(cspace, 0, 1)
        self.dcsValid[workSelector] = True

    def propagate_changes(self) -> None:
        """Track through di/hedra to invalidate dependent atoms."""
        # chain_starts = [self.atomArrayIndex[akt[0]] for akt in self.initNCaCs]
        # cs = chain_starts.pop(0)

        csNdx = 0
        csLen = len(self.initNCaCs)
        atmNdx = AtomKey.fields.atm
        posNdx = AtomKey.fields.respos
        done = set()

        while csNdx < csLen:  # iterate over chain starts
            startAK = self.initNCaCs[csNdx][0]
            csStart = self.atomArrayIndex[startAK]
            csnTry = csNdx + 1
            if csLen == csnTry:
                csNext = self.AAsiz  # last segment to endof atomArray
            else:  # this segment to next chain start
                finAK = self.initNCaCs[csnTry][0]
                csNext = self.atomArrayIndex[finAK]
            # invalid_atoms = np.logical_not(self.atomArrayValid)
            # invalid_atom_ndxs = np.where(invalid_atoms)[0]
            # fin = invalid_atom_ndxs[csNext - 1]
            # for andx in invalid_atom_ndxs[csStart:csNext]:
            for andx in range(csStart, csNext):
                if not self.atomArrayValid[andx]:
                    ak = self.aktuple[andx]
                    atm = ak.akl[atmNdx]
                    pos = ak.akl[posNdx]
                    if atm in ("N", "CA", "C"):
                        # backbone moved so all to next start moved
                        self.atomArrayValid[andx:csNext] = False
                        # and done with this invalid_atom_ndxs segment
                        break
                    elif pos not in done and atm not in ("O", "H"):
                        # O and H are termini so ignore, no effect on subsequent atoms
                        # atomArray is sorted so sidechain atoms follow backbone
                        for i in range(andx, csNext):
                            if self.aktuple[i].akl[posNdx] == pos:
                                self.atomArrayValid[i] = False
                            else:
                                # done with residue sidechain when find next seq pos
                                # so need not go to fin
                                break
                        done.add(pos)
            csNdx += 1

        """
        rtm
            affected_hedra.update(self.a2h_map[andx])
        ahl = len(affected_hedra)
        ahl = ahl - 1
        newah = affected_hedra
        while ahl != len(affected_hedra):
            newdh = set()
            for h in newah:
                newdh.update(self.h1d_map[h])
            newah = set()
            for dh in newdh:
                newah.add(self.dH2ndx[dh])
            ahl = len(affected_hedra)
            affected_hedra.update(newah)
        for h in affected_hedra:
            self.atomArrayValid[self.h2aa[h]] = False
        """

    # @profile
    def internal_to_atom_coordinates(
        self, verbose: bool = False, promote: Optional[bool] = True,
    ) -> None:
        """Process IC data to Residue/Atom coords.

        Not yet vectorized.  rtm

        :param verbose bool: default False
            describe runtime problems
        :param promote bool: default True
            If True (the default) copy result atom XYZ coordinates to
            Biopython Atom objects for access by other Biopython methods;
            otherwise, updated atom coordinates must be accessed through
            IC_Residue and hedron objects.
        """
        # rtm
        # if self.dihedra == {}:
        #    return  # escape if nothing to process

        if not hasattr(self, "dAtoms_needs_update"):
            return  # escape on no data to process

        # if verbose:
        #    for ric in self.ordered_aa_ic_list:
        #        if not hasattr(ric, "NCaCKey"):
        #            print(f"no assembly for {ric} due to missing N, Ca and/or C atoms")

        if IC_Chain.ParallelAssembleResidues:
            self.propagate_changes()

        self.init_atom_coords()  # compute initial di/hedra coords
        # rtm self.atomArrayValid[...] = False
        # self.atomArrayValid[0:3] = True
        if IC_Chain.ParallelAssembleResidues:
            self.ar2(verbose=verbose)  # transform init di/hedra to chain coord space

            if verbose and not np.all(self.atomArrayValid):
                dSetValid = self.atomArrayValid[self.a2da_map].reshape(-1, 4)
                for ric in self.ordered_aa_ic_list:

                    for k, d in ric.dihedra.items():
                        if not dSetValid[d.ndx].all():
                            print(
                                f"missing coordinates for chain {ric.cic.chain.id} {ric.pretty_str()} dihedral: {d.id}"
                            )
        else:
            self.assemble_residues(verbose=verbose)  # internal to XYZ coordinates

        pass
        # if promote:
        #    self.coords_to_structure()  # promote to BioPython Residue/Atom

    # @profile
    def atom_to_internal_coordinates(self, verbose: bool = False) -> None:
        """Calculate dihedrals, angles, bond lengths for Atom data.

        :param verbose bool: default False
            describe runtime problems
        """
        """
        atomArray
        atomArrayIndex  : dict of atomKeys -> numpy array indexes
        atomArrayValid

        atomArray partly built by set_residues() when creating IC_Chain from Chain
        probably most of that is ok because works out chain breaks
        but maybe still extract atomArray build
        build_atomArray() needs to create the numpy arrays too so needs ak list and count
        need to build chain atomKey cache .akc and use cak() to get
        read_PIC needs to build chain akc as well but will need to get atom str
         - and what about altloc???
        hedraDict2chain could build c.akc but perhaps akc not otherwise needed in ic2a?
        good if dcs index could be consistent between a2ic, ic2a
        """
        # for res in self.chain.get_residues():
        #    if 2 == res.is_disordered():

        """
        hedraLen (scalar)
        hedraNdx  : dict of atomKey sequences -> numpy array indexes
        hedraL12
        hedraAngle
        hedraL23
        """
        """
        dihedraLen (scalar)
        dihedraNdx  : dict of atomKey sequences -> numpy array indexes
        dihedraAngle
        dihedraAngleRad

        dCoordSpace
        dcsValid

        """
        if self.ordered_aa_ic_list == []:
            return  # escape on no data to process
        hedraAtomDict = {}
        dihedraAtomDict = {}
        hInDset = set()  # rtm
        hedraDict2 = {}
        gCBdihedra = set()

        if self.ordered_aa_ic_list[0].hedra == {}:
            for ric in self.ordered_aa_ic_list:
                ric.create_edra(verbose=verbose)  # builds di/hedra objects

        if not hasattr(self, "atomArrayValid"):
            self.build_atomArray()  # ric.a2ic added gly CBs to akset

        # if added Gly C-betas when building di/hedra residue objects, now need
        # to add additional atoms to the chain level numpy arrays
        # if IC_Residue.gly_Cbeta and hasattr(self, "gcb"):
        #    gcbaa = np.array(tuple(self.gcb.values()))
        #    siz = len(self.gcb)
        #    start = len(self.atomArray)
        #    gcbaaNdx = dict(zip(self.gcb.keys(), range(start, start + siz)))
        #    gcbaaValid = np.zeros(siz, dtype=bool)

        #    self.atomArray = np.append(self.atomArray, gcbaa, axis=0)
        #    self.atomArrayValid = np.append(self.atomArrayValid, gcbaaValid, axis=0)
        #    self.atomArrayIndex.update(gcbaaNdx)
        #    delattr(self, "gcb")

        # self.init_edra()  # rtm create empty chain hLAL, da arrays

        if not hasattr(self, "hedraLen"):
            # hedra
            self.hedraLen = len(self.hedra)
            self.hedraL12 = np.empty((self.hedraLen), dtype=np.float64)
            self.hedraAngle = np.empty((self.hedraLen), dtype=np.float64)
            self.hedraL23 = np.empty((self.hedraLen), dtype=np.float64)

            # self.hedraNdx = dict(zip(sorted(self.hedra.keys()), range(len(self.hedra))))
            self.hedraNdx = dict(zip(self.hedra.keys(), range(len(self.hedra))))

            # dihedra
            self.dihedraLen = len(self.dihedra)
            self.dihedraAngle = np.empty(self.dihedraLen)
            self.dihedraAngleRads = np.empty(self.dihedraLen)

            self.dihedraNdx = dict(
                zip(sorted(self.dihedra.keys()), range(self.dihedraLen))
            )
            # self.dihedraNdx = dict(zip(self.dihedra.keys(), range(self.dihedraLen)))

        if not hasattr(self, "hAtoms_needs_update"):
            self.build_edraArrays()
        # rtm think good to here

        # rtm skip this
        if False:
            # rtm here we build h and d atom arrays from ric's
            # but should be able to mke some map from ak strings like for i2a
            # excluding the gcb parts here
            for ric in self.ordered_aa_ic_list:
                for k, d in ric.dihedra.items():
                    hInDset.update((d.h1key, d.h2key))
                    try:
                        # get tuple of atom_coords from ric dict
                        dihedraAtomDict[k] = d.gen_acs(ric.atom_coords)
                    except KeyError:
                        gCBdihedra.add(d)  # no atom_coords yet for gly CB
                        # init to rough approximation, overwrite later
                        # dihedron = O-C-Ca-Cb
                        # h1 = Ca-C-O (reversed)
                        # h2 = Cb-Ca-C (reversed)
                        # need dihedron atom coords all forward
                        h1 = d.hedron1.gen_acs(ric.atom_coords)
                        h1 = np.flipud(h1)  # reverse h1 coords for building dihedron
                        xgcb = np.append(h1, [h1[2]], axis=0)
                        xgcb[3, 0] = xgcb[3, 0] + 1.0
                        dihedraAtomDict[k] = xgcb

                for k, h in ric.hedra.items():
                    if k not in hInDset:
                        # print("inaccessible hedron outside dihedron: ", h)
                        try:
                            hedraAtomDict[k] = h.gen_acs(ric.atom_coords)
                        except KeyError:  # gly CB
                            hedraAtomDict[k] = np.array(
                                [[1, 2, 3, 1], [2, 2, 3, 1], [3, 2, 3, 1]]
                            )

        if self.dihedra == {}:
            return  # escape if no hedra loaded for this chain

        # process all hedra
        ha = self.atomArray[self.a2ha_map].reshape(-1, 3, 4)
        self.hedraL12 = np.linalg.norm(ha[:, 0] - ha[:, 1], axis=1)
        self.hedraL23 = np.linalg.norm(ha[:, 1] - ha[:, 2], axis=1)
        h_a0a2 = np.linalg.norm(ha[:, 0] - ha[:, 2], axis=1)
        np.rad2deg(
            np.arccos(
                (
                    np.square(self.hedraL12)
                    + np.square(self.hedraL23)
                    - np.square(h_a0a2)
                )
                / (2 * self.hedraL12 * self.hedraL23)
            ),
            out=self.hedraAngle,
        )

        # rtm skip this for now
        if hedraAtomDict != {}:
            print("A2I UNEXPECTED")
            # some hedra not in dihedra to process
            # issue from alternate CB path, triggered by residue sidechain path not
            # including n-ca-cb-xg
            # not needed to build chain but include for consistency / statistics
            hedraDict2 = {k: h for k, h in self.hedra.items() if k not in hInDset}
            lh2a = len(hedraDict2)
            if lh2a > 0:
                h2a = np.array(tuple(hedraAtomDict.values()))
                h2ai = dict(zip(hedraDict2.keys(), range(lh2a)))

                # get dad for hedra
                h_a0a1 = np.linalg.norm(h2a[:, 0] - h2a[:, 1], axis=1)
                h_a1a2 = np.linalg.norm(h2a[:, 1] - h2a[:, 2], axis=1)
                h_a0a2 = np.linalg.norm(h2a[:, 0] - h2a[:, 2], axis=1)
                h_a0a1a2 = np.rad2deg(
                    np.arccos(
                        ((h_a0a1 * h_a0a1) + (h_a1a2 * h_a1a2) - (h_a0a2 * h_a0a2))
                        / (2 * h_a0a1 * h_a1a2)
                    )
                )

                for k, h in hedraDict2.items():
                    hndx = h2ai[k]
                    # h.lal[:] = (h_a0a1[hndx], h_a0a1a2[hndx], h_a1a2[hndx])
                    h.L12 = h_a0a1[hndx]
                    h.Angle = h_a0a1a2[hndx]
                    h.L23 = h_a1a2[hndx]

        # now process dihedra
        # dLen = self.dihedraLen
        # dha = np.array(tuple(dihedraAtomDict.values()))
        # dhai = dict(zip(self.dihedra.keys(), range(dLen)))

        # rtm
        # ok need a ha array and a2ha_map to run all hedra
        # can we split out bonds as well...

        dha = self.atomArray[self.a2da_map].reshape(-1, 4, 4)

        # get dadad dist-angle-dist-angle-dist for dihedra
        # a0a1 = np.linalg.norm(dha[:, 0] - dha[:, 1], axis=1)
        # a1a2 = np.linalg.norm(dha[:, 1] - dha[:, 2], axis=1)
        # a2a3 = np.linalg.norm(dha[:, 2] - dha[:, 3], axis=1)

        # a0a2 = np.linalg.norm(dha[:, 0] - dha[:, 2], axis=1)
        # a1a3 = np.linalg.norm(dha[:, 1] - dha[:, 3], axis=1)
        # sqr_a1a2 = np.multiply(a1a2, a1a2)

        # a0a1a2 = np.rad2deg(
        #    np.arccos(((a0a1 * a0a1) + sqr_a1a2 - (a0a2 * a0a2)) / (2 * a0a1 * a1a2))
        # )

        # a1a2a3 = np.rad2deg(
        #    np.arccos((sqr_a1a2 + (a2a3 * a2a3) - (a1a3 * a1a3)) / (2 * a1a2 * a2a3))
        # )

        # develop coord_space matrix for 1st 3 atoms of dihedra:
        # mt = multi_coord_space(dha, self.dihedraLen, False)
        self.dCoordSpace = multi_coord_space(dha, self.dihedraLen, True)
        self.dcsValid[:] = True

        # now put atom 4 into that coordinate space and read dihedral as azimuth
        do4 = np.matmul(self.dCoordSpace[0], dha[:, 3].reshape(-1, 4, 1)).reshape(-1, 4)

        np.arctan2(do4[:, 1], do4[:, 0], out=self.dihedraAngleRads)
        np.rad2deg(self.dihedraAngleRads, out=self.dihedraAngle)

        """
        # rtm HERE now need map from dha to h1, h2 to store LAL results
        self.hedraL12 =
        # rtm here copying everything back to dihedra objects, they should reference
        # chain hLAL, da if anything

        # build hedra arrays

        # hIC = self.hedraIC
        hL12 = self.hedraL12
        hAngle = self.hedraAngle
        hL23 = self.hedraL23
        hNdx = self.hedraNdx

        for k, d in self.dihedra.items():
            dndx = dhai[k]
            # d.angle = dh1d[dndx]
            rev, hed1, hed2 = (d.reverse, d.hedron1, d.hedron2)
            h1ndx, h2ndx = (hNdx[d.h1key], hNdx[d.h2key])
            if not rev:
                # hIC[h1ndx, :] = (a0a1[dndx], a0a1a2[dndx], a1a2[dndx])
                hL12[h1ndx] = a0a1[dndx]
                hAngle[h1ndx] = a0a1a2[dndx]
                hL23[h1ndx] = a1a2[dndx]
                # hIC[h2ndx, :] = (a1a2[dndx], a1a2a3[dndx], a2a3[dndx])
                hL12[h2ndx] = a1a2[dndx]
                hAngle[h2ndx] = a1a2a3[dndx]
                hL23[h2ndx] = a2a3[dndx]
                # hed1.len12 = a0a1[dndx]
                # hed1.len23 = hed2.len12 = a1a2[dndx]
                # hed2.len23 = a2a3[dndx]
            else:
                # hIC[h1ndx, :] = (a1a2[dndx], a0a1a2[dndx], a0a1[dndx])
                hL12[h1ndx] = a1a2[dndx]
                hAngle[h1ndx] = a0a1a2[dndx]
                hL23[h1ndx] = a0a1[dndx]
                # hIC[h2ndx, :] = (a2a3[dndx], a1a2a3[dndx], a1a2[dndx])
                hL12[h2ndx] = a2a3[dndx]
                hAngle[h2ndx] = a1a2a3[dndx]
                hL23[h2ndx] = a1a2[dndx]
                # hed1.len23 = a0a1[dndx]
                # hed1.len12 = hed2.len23 = a1a2[dndx]
                # hed2.len12 = a2a3[dndx]

            # hed1.lal = hIC[h1ndx]
            hed1.L12 = hL12[h1ndx]
            hed1.Angle = hAngle[h1ndx]
            hed1.L23 = hL23[h1ndx]
            # hed2.lal = hIC[h2ndx]
            hed2.L12 = hL12[h2ndx]
            hed2.Angle = hAngle[h2ndx]
            hed2.L23 = hL23[h2ndx]

            # hed1.angle = a0a1a2[dndx]
            # hed2.angle = a1a2a3[dndx]
        """
        # for gCBd in gCBdihedra:
        #    gCBd.ric.build_glyCB(gCBd)
        if hasattr(self, "gcb"):
            self.spec_glyCB()

    def spec_glyCB(self) -> None:
        """Populate values for Gly C-beta, rest of chain complete.

        Data averaged from Sep 2019 Dunbrack cullpdb_pc20_res2.2_R1.0
        restricted to structures with amide protons.

        Ala avg rotation of OCCACB from NCACO query:
        select avg(g.rslt) as avg_rslt, stddev(g.rslt) as sd_rslt, count(*)
        from
        (select f.d1d, f.d2d,
        (case when f.rslt > 0 then f.rslt-360.0 else f.rslt end) as rslt
        from (select d1.angle as d1d, d2.angle as d2d,
        (d2.angle - d1.angle) as rslt from dihedron d1,
        dihedron d2 where d1.rdh_class='AOACACAACB' and
        d2.rdh_class='ANACAACAO' and d1.pdb=d2.pdb and d1.chn=d2.chn
        and d1.res=d2.res) as f) as g

        | avg_rslt          | sd_rslt          | count   |
        | -122.682194862932 | 5.04403040513919 | 14098   |

        """
        Ca_Cb_Len = 1.53363
        if hasattr(self, "scale"):  # used for openscad output
            Ca_Cb_Len *= self.scale  # type: ignore

        for gcb, gcbd in self.gcb.items():
            cbak = gcbd[3]
            self.atomArrayValid[self.atomArrayIndex[cbak]] = False
            ric = cbak.ric
            rN, rCA, rC, rO = ric.rak("N"), ric.rak("CA"), ric.rak("C"), ric.rak("O")
            gCBd = self.dihedra[gcbd]
            dndx = gCBd.ndx
            # generated dihedron is O-Ca-C-Cb
            # hedron2 is reversed: Cb-Ca-C (also h1 reversed: C-Ca-O)
            h2ndx = gCBd.hedron2.ndx
            self.hedraL12[h2ndx] = Ca_Cb_Len
            self.hedraAngle[h2ndx] = 110.17513
            self.hedraL23[h2ndx] = self.hedraL12[self.hedraNdx[(rCA, rC, rO)]]
            # gCBd.hedron2._invalidate_atoms()
            self.hAtoms_needs_update[gCBd.hedron2.ndx] = True
            for ak in gCBd.hedron2.aks:
                self.atomArrayValid[self.atomArrayIndex[ak]] = False

            refval = self.dihedra.get((rN, rCA, rC, rO), None)
            if refval:
                angl = 122.68219 + self.dihedraAngle[refval.ndx]
                self.dihedraAngle[dndx] = angl if (angl <= 180.0) else angl - 360.0
            else:
                self.dihedraAngle[dndx] = 120

    @staticmethod
    def _write_mtx(fp: TextIO, mtx: np.array) -> None:
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
        fp: TextIO, d: "Dihedron", hedraNdx: Dict, hedraSet: Set[EKT]
    ) -> None:
        fp.write(
            "[ {:9.5f}, {}, {}, {}, ".format(
                d.angle, hedraNdx[d.h1key], hedraNdx[d.h2key], (1 if d.reverse else 0)
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
        IC_Chain._write_mtx(fp, d.rcst)
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
        fp.write(f'   "{self.chain.id}", // chain id\n')

        # generate dict for all hedra to eliminate redundant references
        hedra = {}
        for ric in self.ordered_aa_ic_list:
            respos, resicode = ric.residue.id[1:]
            for k, h in ric.hedra.items():
                hedra[k] = h
        atomSet: Set[AtomKey] = set()
        bondDict: Dict = {}  # set()
        hedraSet: Set[EKT] = set()
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

        self.dcsValid[:] = False
        for ric in self.ordered_aa_ic_list:
            if "O" not in ric.akc:
                if ric.lc != "G" and ric.lc != "A":
                    print(
                        f"Unable to generate complete sidechain for {ric} {ric.lc} missing O atom"
                    )
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
            # ric.clear_transforms()

            # update residue atom coords for no start position
            # this makes ric.atom_coords new copy, not view of chain atomArray
            ric.atom_coords = cast(
                Dict[AtomKey, np.array], ric.assemble(resetLocation=True)
            )

            ndx2 = 0
            started = False
            for i in range(1 if backboneOnly else 2):
                if i == 1:
                    cma = "," if started else ""
                    fp.write(
                        f"{cma}\n       // {str(ric.residue.id)} {ric.lc} sidechain\n"
                    )
                started = False
                for dk, d in sorted(ric.dihedra.items()):
                    if d.h2key in hedraNdx and (
                        (i == 0 and d.is_backbone()) or (i == 1 and not d.is_backbone())
                    ):
                        if d.cic.dcsValid[d.ndx]:  # d.rcst is not None:
                            if started:
                                fp.write(",\n")
                            else:
                                started = True
                            fp.write("      ")
                            IC_Chain._writeSCAD_dihed(fp, d, hedraNdx, hedraSet)
                            dihedraNdx[dk] = ndx2
                            hedraSet.add(d.h1key)
                            hedraSet.add(d.h2key)
                            ndx2 += 1
                        else:
                            print(
                                f"Atom missing for {d.id3}-{d.id32}, OpenSCAD chain may be discontiguous"
                            )
        fp.write("   ],")  # end of residue entry dihedra table
        fp.write("\n  ],\n")  # end of all dihedra table

        # write hedra table

        fp.write("   [  //hedra\n")
        for hk in sorted(hedra):
            hed = hedra[hk]
            fp.write("     [ ")
            fp.write(
                "{:9.5f}, {:9.5f}, {:9.5f}".format(
                    set_accuracy_95(hed.len12),  # (hed.lal[0]),  # len12
                    set_accuracy_95(hed.angle),  # (hed.lal[1]),  # angle
                    set_accuracy_95(hed.len23),  # (hed.lal[2]),  # len23
                )
            )
            atom_str = ""  # atom and bond state
            atom_done_str = ""  # create each only once
            akndx = 0
            for ak in hed.aks:
                atm = ak.akl[AtomKey.fields.atm]
                res = ak.akl[AtomKey.fields.resname]
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
                + akl[AtomKey.fields.resname]
                + '", '
                + akl[AtomKey.fields.respos]
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
            for NCaCKey in sorted(ric.NCaCKey):  # type: ignore
                if 0 < len(ric.rprev):
                    for rpr in ric.rprev:
                        acl = [rpr.atom_coords[ak] for ak in NCaCKey]
                        mt, mtr = coord_space(acl[0], acl[1], acl[2], True)
                else:
                    mtr = np.identity(4, dtype=np.float64)
                if chnStarted:
                    fp.write(",\n")
                else:
                    chnStarted = True
                fp.write("     [ " + str(resNdx[ric]) + ', "' + str(ric.residue.id[1]))
                fp.write(ric.lc + '", //' + str(NCaCKey) + "\n")
                IC_Chain._write_mtx(fp, mtr)
                fp.write(" ]")
        fp.write("\n   ]\n")

        # make residue atom_coords consistent with chain atomArray again
        # copy from IC_Chain.init_atom_coords()
        # set all ric.atom_coords to be views on chain atomArray
        for ric in self.ordered_aa_ic_list:
            for ak in ric.ak_set:
                ric.atom_coords[ak] = self.atomArray[self.atomArrayIndex[ak]]


class IC_Residue:
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
    atom_coords: AtomKey indexed dict of numpy [4] arrays
        Local copy of atom homogeneous coordinates [4] for work
        Actually a view into IC_Chain's atomArray
        distinct from Bopython Residue/Atom values
    #atom_coords_vw: AtomKey indexed dict of numpy [3] arrays
        #numpy view into Biopython Residue/Atom values
    alt_ids: list of char
        AltLoc IDs from PDB file
    bfactors: dict
        AtomKey indexed B-factors as read from PDB file
    NCaCKey: List of tuples of AtomKeys
        List of tuples of N, Ca, C backbone atom AtomKeys; usually only 1
        but more if backbone altlocs. Set by link_dihedra()
    is20AA: bool
        True if residue is one of 20 standard amino acids, based on
        Residue resname
    isAccept: bool
        True if is20AA or in accept_resnames below
    accept_atoms: tuple
        list of PDB atom names to use when generating internal coordinates.
        Default is:

        `accept_atoms = accept_mainchain + accept_hydrogens`

        to exclude hydrogens in internal coordinates and generated PDB files,
        override as:

        `IC_Residue.accept_atoms = IC_Residue.accept_mainchain`

        to get only mainchain atoms plus amide proton, use:

        `IC_Residue.accept_atoms = IC_Residue.accept_mainchain + ('H',)`

        to convert D atoms to H, set `AtomKey.d2h = True` and use:

        `IC_Residue.accept_atoms = accept_mainchain + accept_hydrogens + accept_deuteriums`

        Note that accept_mainchain = accept_backbone + accept_sidechain.  Thus
        to generate sequence-agnostic conformational data for e.g. structure
        alignment in dihedral angle space, use:
        `IC_Residue.accept_atoms = accept_backbone`

        or set gly_Cbeta = True and use:

        `IC_Residue.accept_atoms = accept_backbone + ('CB',)`

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
    allBonds: bool default False
        whereas a PDB file just specifies atoms, OpenSCAD output for 3D printing
        needs all bonds specified explicitly - otherwise, e.g. PHE rings will not
        be closed.  This variable is managed by the Write_SCAD() code and enables
        this.
    cic: IC_Chain default None
        parent chain IC_Chain object, set in IC_Chain link_residues() and
        add_residue()

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
        multiply all IC_Residue atom_coords by passed matrix
    assemble(atomCoordsIn, resetLocation, verbose)
        Compute atom coordinates for this residue from internal coordinates
    atm241(coord)
        Convert 1x3 cartesian coords to 1x4 homogeneous coords
        Naming: 4x1 array is correct, but numpy handles automatically
    coords_to_residue()
        Convert homogeneous atom_coords to Biopython cartesian Atom coords
    create_edra(verbose)
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

    AllBonds: bool = False  # For OpenSCAD, generate explicit hedra covering all bonds if True.

    def __init__(self, parent: "Residue", NO_ALTLOC: bool = False) -> None:
        """Initialize IC_Residue with parent Biopython Residue.

        :param parent: Biopython Residue object
            The Biopython Residue this object extends
        :param NO_ALTLOC: bool default False
            Option to disable processing altloc disordered atoms, use selected.
        """
        # NO_ALTLOC=True will turn off altloc positions and just use selected
        self.residue = parent
        self.cic: IC_Chain
        # dict of hedron objects indexed by hedron keys
        self.hedra: Dict[HKT, Hedron] = {}
        # dict of dihedron objects indexed by dihedron keys
        self.dihedra: Dict[DKT, Dihedron] = {}
        # map of dihedron key (first 3 atom keys) to dihedron
        # for all dihedra in Residue
        # built by link_dihedra()
        # rtm self.id3_dh_index: Dict[HKT, List[Dihedron]] = {}
        # cache of AtomKey results for rak()
        self.akc: Dict[Union[str, Atom], AtomKey] = {}
        # set of AtomKeys involved in dihedra, used by split_akl, build_rak_cache
        # built by __init__ for XYZ (PDB coord) input, link_dihedra for PIC input
        self.ak_set: Set[AtomKey] = set()
        # reference to adjacent residues in chain
        self.rprev: List[IC_Residue] = []
        self.rnext: List[IC_Residue] = []
        # local copy, homogeneous coordinates for atoms, numpy [4]
        # generated from dihedra include some i+1 atoms
        # or initialised here from parent residue if loaded from coordinates
        # self.atom_coords: Dict["AtomKey", np.array] = {}  # homog coords
        # self.atom_coords_vw: Dict["AtomKey", np.array] = {}  # view on Atoms
        # bfactors copied from PDB file
        self.bfactors: Dict[str, float] = {}
        self.alt_ids: Union[List[str], None] = None if NO_ALTLOC else []
        self.is20AA = True
        self.isAccept = True
        # rbase = position, insert code or none, resname (1 letter if in 20)
        rid = parent.id
        rbase = [rid[1], rid[2] if " " != rid[2] else None, parent.resname]
        try:
            rbase[2] = three_to_one(rbase[2]).upper()
        except KeyError:
            self.is20AA = False
            if rbase[2] not in self.accept_resnames:
                self.isAccept = False

        self.rbase = tuple(rbase)
        self.lc = rbase[2]

        if self.isAccept:
            for atom in parent.get_atoms():
                if hasattr(atom, "child_dict"):
                    if NO_ALTLOC:
                        self._add_atom(atom.selected_child)
                    else:
                        for atm in atom.child_dict.values():
                            self._add_atom(atm)
                else:
                    self._add_atom(atom)
            if self.ak_set:  # only for coordinate (pdb) input
                self.build_rak_cache()  # init cache ready for atom_to_internal_coords
                # self.NCaCKey = [(self.rak("N"), self.rak("CA"), self.rak("C"))]

            # print(self.atom_coords)

    def __deepcopy__(self, memo):
        existing = memo.get(id(self), False)
        if existing:
            return existing
        dup = type(self).__new__(self.__class__)
        memo[id(self)] = dup
        dup.__dict__.update(self.__dict__)  # later replace what is not static
        dup.cic = memo[id(self.cic)]
        dup.residue = memo[id(self.residue)]
        # still need to update: rnext, rprev, akc, ak_set, di/hedra
        # dup.dc = True  # rtm
        return dup

    def rak(self, atm: Union[str, Atom]) -> "AtomKey":
        """Cache calls to AtomKey for this residue."""
        try:
            ak = self.akc[atm]
        except (KeyError):
            ak = self.akc[atm] = AtomKey(self, atm)
            if isinstance(atm, str):
                # print(atm)  # debug code
                ak.missing = True
        return ak

    def build_rak_cache(self) -> None:
        """Create explicit entries for for atoms so don't miss altlocs.

        This ensures that akc has an entry for selected atom name (e.g. "CA")
        amongst any that have altlocs.  Without this, rak() on the other altloc
        atom first may result in the main atom being missed.
        """
        for ak in sorted(self.ak_set):
            atmName = ak.akl[3]
            if self.akc.get(atmName) is None:
                self.akc[atmName] = ak

    accept_backbone = (
        "N",
        "CA",
        "C",
        "O",
        "OXT",
    )
    accept_sidechain = (
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
    )

    accept_mainchain = accept_backbone + accept_sidechain

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
    def atm241(coord: np.array) -> np.array:
        """Convert 1x3 cartesian coordinates to 1x4 homogeneous coordinates."""
        print("ATM241")
        arr41 = np.empty(4)
        arr41[0:3] = coord
        arr41[3] = 1.0
        return arr41

    def _add_atom(self, atm: Atom) -> None:
        """Filter Biopython Atom with accept_atoms; set atom_coords, ak_set.

        Arbitrarily renames O' and O'' to O and OXT
        """
        if "O" == atm.name[0]:
            if "O'" == atm.name:
                atm.name = "O"
            elif "O''" == atm.name:
                atm.name = "OXT"

        if atm.name not in self.accept_atoms:
            # print('skip:', atm.name)
            return
        ak = self.rak(atm)  # passing Atom here not string
        # rtm self.atom_coords_vw[ak] = atm.coord
        self.ak_set.add(ak)

    def __repr__(self) -> str:
        """Print string is parent Residue ID."""
        return str(self.residue.full_id)

    def pretty_str(self) -> str:
        """Nice string for residue ID."""
        id = self.residue.id
        return f"{self.residue.resname} {id[0]}{str(id[1])}{id[2]}"

    def load_PIC(self, edron: Dict[str, str]) -> None:
        """Process parsed (di-/h-)edron data from PIC file.

        :param edron: parse dictionary from Edron.edron_re
        """
        # rtm not used
        if edron["a4"] is None:  # PIC line is Hedron
            ek = (AtomKey(edron["a1"]), AtomKey(edron["a2"]), AtomKey(edron["a3"]))
            self.hedra[ek] = Hedron(ek, **edron)
        else:  # PIC line is Dihedron
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
        - set NCaCKey to be available AtomKeys
        """
        # print("LINK_DIHEDRA")
        # rtm not called so far from new read_PIC
        # called for loading PDB / atom coords
        # id3i: Dict[HKT, List[Dihedron]] = {}
        for dh in self.dihedra.values():
            dh.ric = self  # each dihedron can find its IC_Residue
            dh.cic = self.cic  # each dihedron can update chain dihedral angles
            # id3 = dh.id3
            # if id3 not in id3i:
            #    id3i[id3] = []
            # id3i[id3].append(dh)
            self.ak_set.update(dh.aks)
            # now in build_edraArrays rtm : dh.set_hedra()
        for h in self.hedra.values():  # collect any atoms in orphan hedra
            self.ak_set.update(h.aks)  # e.g. alternate CB path with no O
            h.cic = self.cic  # each hedron can update chain hedra

        # map to find each dihedron from atom tokens 1-3
        # rtm self.id3_dh_index = id3i

        # if loaded PIC data, akc not initialised yet
        if not self.akc:
            self.build_rak_cache()

        # initialise NCaCKey here:
        # new version copy from hedraDict2Chain
        self.NCaCKey = []
        self.NCaCKey.extend(
            self._split_akl(
                (AtomKey(self, "N"), AtomKey(self, "CA"), AtomKey(self, "C"))
            )
        )

        # not rak here to avoid polluting akc cache with no-altloc keys
        # so starting with 'generic' key:
        """
        self.NCaCKey = [(AtomKey(self, "N"), AtomKey(self, "CA"), AtomKey(self, "C"))]

        newNCaCKey: List[Tuple["AtomKey", ...]] = []

        try:
            for tpl in sorted(self.NCaCKey):
                newNCaCKey.extend(self._split_akl(tpl))
            self.NCaCKey = cast(List[HKT], newNCaCKey)
            # if len(newNCaCKey) != 1 and len(self.rprev) == 0:
            #  debug code to find examples of chains starting with disordered residues
            #    print(f"chain start multiple NCaCKey  {newNCaCKey} : {self}")
        except AttributeError:
            if verbose:
                print(
                    f"Missing N, Ca and/or C atoms for residue {str(self.residue)} chain {self.residue.parent.id}"
                )
        """

    def set_flexible(self) -> None:
        """For OpenSCAD, mark N-CA and CA-C bonds to be flexible joints."""
        for h in self.hedra.values():
            if h.dh_class == "NCAC":
                h.flex_female_1 = True
                h.flex_female_2 = True
            elif h.dh_class.endswith("NCA"):
                h.flex_male_2 = True
            elif h.dh_class.startswith("CAC") and h.aks[1].akl[3] == "C":
                h.flex_male_1 = True
            elif h.dh_class == "CBCAC":
                h.skinny_1 = True  # CA-CB bond interferes with flex join

    def set_hbond(self) -> None:
        """For OpenSCAD, mark H-N and C-O bonds to be hbonds (magnets)."""
        for h in self.hedra.values():
            if h.dh_class == "HNCA":
                h.hbond_1 = True
            elif h.dh_class == "CACO":
                h.hbond_2 = True

    def default_startpos(self) -> Dict["AtomKey", np.array]:
        """Generate default N-Ca-C coordinates to build this residue from."""
        atomCoords = {}
        cic = self.cic
        dlist0 = [cic.id3_dh_index.get(akl, None) for akl in sorted(self.NCaCKey)]
        dlist1 = [d for d in dlist0 if d is not None]
        # https://stackoverflow.com/questions/11264684/flatten-list-of-lists
        dlist = [cic.dihedra[val] for sublist in dlist1 for val in sublist]
        # dlist = self.id3_dh_index[NCaCKey]
        for d in dlist:
            for i, a in enumerate(d.aks):
                # atomCoords[a] = d.initial_coords[i]
                atomCoords[a] = cic.dAtoms[d.ndx][i]
        # if "O" not in self.akc and "CB" in self.akc:
        #    # need CB coord if no O coord - handle alternate CB path
        #    # but not clear how to do this for default position
        #    pass
        return atomCoords

    def get_startpos(self) -> Dict["AtomKey", np.array]:
        """Find N-Ca-C coordinates to build this residue from."""
        # rtm only used by assemble()
        startPos = {}
        if 0 < len(self.rprev):
            # if there is a previous residue, build on from it
            # nb akl for this res n-ca-c in rp (prev res) dihedra
            akl: List[AtomKey] = []
            for tpl in self.NCaCKey:
                akl.extend(tpl)
            if self.rak("O").missing:
                # alternate CB path - only use if O is missing
                # else have problems modifying phi angle
                akl.append(AtomKey(self, "CB"))
            for ak in akl:
                for rp in self.rprev:
                    rpak = rp.atom_coords.get(ak, None)
                    if rpak is not None:
                        startPos[ak] = rpak
            if 3 > len(startPos):  # if don't have all 3, reset to have none
                startPos = {}
        else:
            cic = self.cic
            for ncac in self.NCaCKey:
                if np.all(cic.atomArrayValid[[cic.atomArrayIndex[ak] for ak in ncac]]):
                    for ak in ncac:
                        startPos[ak] = cic.atomArray[cic.atomArrayIndex[ak]]
            if startPos == {}:
                startPos = self.default_startpos()

            # rtm
            # # get atom posns already added by load_structure
            # sp = self.residue.parent.internal_coord.initNCaC.get(self.rbase, None)
            # if sp is None:
            #    startPos = {}
            # else:
            #    # need copy Here (shallow ok) else assemble() adds to this dict
            #    startPos = cast(Dict["AtomKey", np.array], sp.copy())

        return startPos

    def clear_transforms(self):
        """Set cst and rcst attributes to none before assemble()."""
        print("CLEAR_TRANSFORMS")
        for d in self.dihedra.values():
            self.cic.dcsValid[d.ndx] = False

    def assemble(
        self, resetLocation: bool = False, verbose: bool = False,
    ) -> Union[Dict["AtomKey", np.array], Dict[HKT, np.array], None]:
        """Compute atom coordinates for this residue from internal coordinates.

        Join dihedrons starting from N-CA-C and N-CA-CB hedrons, computing protein
        space coordinates for backbone and sidechain atoms

        Sets forward and reverse transforms on each Dihedron to convert from
        protein coordinates to dihedron space coordinates for first three
        atoms (see coord_space())

        Not vectorized (yet).

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

        :param resetLocation: bool default False
            - Option to ignore start location and orient so N-Ca-C hedron
            at origin.

        :returns:
            Dict of AtomKey -> homogeneous atom coords for residue in protein space
            relative to previous residue

        """
        # debug statements below still useful, commented for performance
        # dbg = False

        cic = self.cic
        dcsValid = cic.dcsValid
        aaValid = cic.atomArrayValid
        aaNdx = cic.atomArrayIndex
        aa = cic.atomArray

        if not self.ak_set:
            return None  # give up now if no atoms to work with

        NCaCKey = sorted(self.NCaCKey)
        rseqpos = self.rbase[0]

        # order of these startLst entries matters
        startLst = self._split_akl((self.rak("C"), self.rak("CA"), self.rak("N")))
        if "CB" in self.akc:
            startLst.extend(
                self._split_akl((self.rak("N"), self.rak("CA"), self.rak("CB")))
            )
        if "O" in self.akc:
            startLst.extend(
                self._split_akl((self.rak("O"), self.rak("C"), self.rak("CA")))
            )

        startLst.extend(NCaCKey)

        for akl in startLst:
            for ak in akl:
                aaValid[aaNdx[ak]] = True

        q = deque(startLst)
        # resnum = self.rbase[0]

        # get initial coords from previous residue or IC_Chain info
        # or default coords
        if resetLocation:
            # use N-CA-C initial coords from creating dihedral
            atomCoords = self.default_startpos()
        else:
            atomCoords = self.get_startpos()

        while q:  # deque is not empty
            # if dbg:
            #    print("assemble loop start q=", q)
            h1k = cast(HKT, q.pop())
            dihedraKeys = cic.id3_dh_index.get(h1k, None)
            # if dbg:
            #    print(
            #        "  h1k:",
            #        h1k,
            #        "len dihedra: ",
            #        len(dihedra) if dihedra is not None else "None",
            #    )
            if dihedraKeys is not None:
                for dk in dihedraKeys:
                    d = cic.dihedra[dk]
                    dseqpos = int(d.aks[0].akl[AtomKey.fields.respos])
                    if not hasattr(d, "initial_coords"):
                        d.initial_coords = cic.dAtoms[d.ndx]
                    if 4 == len(d.initial_coords) and d.initial_coords[3] is not None:
                        # rtm this check does not work
                        # skip incomplete dihedron if don't have 4th atom due
                        # to missing input data
                        d_h2key = d.hedron2.aks
                        ak = d.aks[3]
                        # if dbg:
                        #    print("    process", d, d_h2key, d.aks)

                        acount = len([a for a in d.aks if a in atomCoords])

                        if 4 == acount:
                            # dihedron already done, queue 2nd hedron key
                            if dseqpos == rseqpos:  # only this residue
                                q.appendleft(d_h2key)
                            # if dbg:
                            #    print("    4- already done, append left")
                            if not dcsValid[d.ndx]:  # missing transform
                                # can happen for altloc atoms
                                # only needed for write_SCAD output
                                acs = [atomCoords[a] for a in h1k]
                                d.cst, d.rcst = coord_space(
                                    acs[0], acs[1], acs[2], True
                                )
                                dcsValid[d.ndx] = True
                        elif 3 == acount:
                            # if dbg:
                            #    print("    3- call coord_space")

                            acs = [atomCoords[a] for a in h1k]
                            d.cst, d.rcst = coord_space(acs[0], acs[1], acs[2], True)
                            dcsValid[d.ndx] = True
                            # print(d.cst)
                            # print(d.rcst)
                            # if dbg:
                            #    print(
                            #        "        initial_coords[3]=",
                            #        d.initial_coords[3].transpose(),
                            #    )
                            acak3 = d.rcst.dot(d.initial_coords[3])
                            # if dbg:
                            #    print("        acak3=", acak3.transpose())

                            # atomCoords[ak] = np.round(
                            #     acak3, 3
                            # )  # round to PDB format 8.3
                            atomCoords[ak] = acak3
                            aa[aaNdx[ak]] = acak3
                            aaValid[aaNdx[ak]] = True

                            # if dbg:
                            #    print(
                            #        "        3- finished, ak:",
                            #        akl[3],
                            #        "coords:",
                            #        atomCoords[akl[3]].transpose(),
                            #    )
                            if dseqpos == rseqpos:  # only this residue
                                q.appendleft(d_h2key)
                        else:
                            if verbose:
                                print("no coords to start", d)
                                print(
                                    [
                                        a
                                        for a in d.aks
                                        if atomCoords.get(a, None) is not None
                                    ]
                                )
                    else:
                        if verbose:
                            print("no initial coords for", d)

        return atomCoords

    def _split_akl(
        self,
        lst: Union[Tuple["AtomKey", ...], List["AtomKey"]],
        missingOK: bool = False,
    ) -> List[Tuple["AtomKey", ...]]:
        """Get AtomKeys for this residue (ak_set) given generic list of AtomKeys.

        Given a list of AtomKeys (aks) for a Hedron or Dihedron,
          return:
                list of matching aks that have id3_dh in this residue
                (ak may change if occupancy != 1.00)

            or
                multiple lists of matching aks expanded for all atom altlocs

            or
                empty list if any of atom_coord(ak) missing and not missingOK

        :param lst: list[3] or [4] of AtomKeys
            non-altloc AtomKeys to match to specific AtomKeys for this residue
        """
        altloc_ndx = AtomKey.fields.altloc
        occ_ndx = AtomKey.fields.occ

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
            if (
                ak in self.ak_set
                and ak.akl[altloc_ndx] is None
                and ak.akl[occ_ndx] is None
            ):
                # simple case no altloc and exact match in set
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
            if 0 == lenAKL and not missingOK:
                return []  # atom missing in atom_coords, cannot form object
            elif maxc < lenAKL:
                maxc = lenAKL
        if 1 == maxc:  # simple case no altlocs for any ak in list
            newAKL = []
            for akl in edraLst:
                if akl:  # may have empty lists if missingOK, do not append
                    newAKL.append(akl[0])
            return [tuple(newAKL)]
        else:
            new_edraLst = []
            for al in altlocs:
                # form complete new list for each altloc
                alhl = []
                for akl in edraLst:
                    lenAKL = len(akl)
                    if 0 == lenAKL:
                        continue  # ignore empty list from missingOK
                    if 1 == lenAKL:
                        alhl.append(akl[0])  # not all atoms will have altloc
                    # elif (lenAKL < maxc
                    #      and al not in posnAltlocs[akMap[akl[0]]]):
                    elif al not in posnAltlocs[akMap[akl[0]]]:
                        # this position has fewer altlocs than other positions
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
        for ak in lst:
            if ak.missing:
                return  # give up if atoms actually missing

        lenLst = len(lst)
        if 4 > lenLst:
            cdct, dct, obj = self.cic.hedra, self.hedra, Hedron
        else:
            cdct, dct, obj = self.cic.dihedra, self.dihedra, Dihedron  # type: ignore

        if isinstance(lst, List):
            tlst = tuple(lst)
        else:
            tlst = lst

        hl = self._split_akl(tlst)  # expand tlst with any altlocs
        # returns list of tuples

        for tnlst in hl:
            # do not add edron if split_akl() made something shorter
            if len(tnlst) == lenLst:
                # if edron already exists, then update not replace with new
                if tnlst not in cdct:
                    cdct[tnlst] = obj(tnlst)  # type: ignore
                if tnlst not in dct:
                    dct[tnlst] = cdct[tnlst]  # type: ignore

                dct[tnlst].needs_update = True  # type: ignore

    # @profile
    def create_edra(self, verbose: bool = False) -> None:
        """Create hedra and dihedra for atom coordinates.

        :param verbose: bool default False
            warn about missing N, Ca, C backbone atoms.
        """
        # on entry we have all Biopython Atoms loaded
        if not self.ak_set:
            return  # so give up if no atoms loaded for this residue

        sN, sCA, sC = self.rak("N"), self.rak("CA"), self.rak("C")
        if self.lc != "G":
            sCB = self.rak("CB")

        # first init di/hedra, AtomKey objects and atom_coords for di/hedra
        # which extend into next residue.

        if 0 < len(self.rnext) and self.rnext[0].ak_set:
            # atom_coords, hedra and dihedra for backbone dihedra
            # which reach into next residue
            for rn in self.rnext:
                nN, nCA, nC = rn.rak("N"), rn.rak("CA"), rn.rak("C")

                nextNCaC = rn._split_akl((nN, nCA, nC), missingOK=True)

                for tpl in nextNCaC:
                    for ak in tpl:
                        if ak in rn.ak_set:  # rn.atom_coords:
                            # self.atom_coords[ak] = rn.atom_coords[ak]
                            self.ak_set.add(ak)
                        else:
                            for rn_ak in rn.ak_set:  # rn.atom_coords.keys():
                                if rn_ak.altloc_match(ak):
                                    # self.atom_coords[rn_ak] = rn.atom_coords[rn_ak]
                                    self.ak_set.add(rn_ak)

                self._gen_edra((sN, sCA, sC, nN))  # psi
                self._gen_edra((sCA, sC, nN, nCA))  # omega i+1
                self._gen_edra((sC, nN, nCA, nC))  # phi i+1
                self._gen_edra((sCA, sC, nN))
                self._gen_edra((sC, nN, nCA))
                self._gen_edra((nN, nCA, nC))  # tau i+1

                # redundant next residue C-beta locator (alternate CB path)
                # otherwise missing O will cause no sidechain
                # not rn.rak so don't trigger missing CB for Gly
                nCB = rn.akc.get("CB", None)
                if nCB is not None and nCB in rn.ak_set:  # rn.atom_coords:
                    # self.atom_coords[nCB] = rn.atom_coords[nCB]
                    self.ak_set.add(nCB)
                    self._gen_edra((nN, nCA, nCB))
                    self._gen_edra((sC, nN, nCA, nCB))

        # if start of chain then need to init NCaC hedron as not in previous residue
        if 0 == len(self.rprev):
            self._gen_edra((sN, sCA, sC))

        # now init di/hedra for standard backbone atoms independent of neighbours
        backbone = ic_data_backbone
        for edra in backbone:
            # only need to build if this residue has all the atoms in the edra
            if all(atm in self.akc for atm in edra):
                r_edra = [self.rak(atom) for atom in edra]
                self._gen_edra(r_edra)  # [4] is label on some table entries

        # next init sidechain di/hedra
        if self.lc is not None:
            sidechain = ic_data_sidechains.get(self.lc, [])
            for edraLong in sidechain:
                edra = edraLong[0:4]  # [4] is label on some sidechain table entries
                # lots of H di/hedra can be avoided if don't have those atoms
                if all(atm in self.akc for atm in edra):
                    r_edra = [self.rak(atom) for atom in edra]
                    self._gen_edra(r_edra)
            if IC_Residue.AllBonds:  # openscad output needs all bond cylinders explicit
                sidechain = ic_data_sidechain_extras.get(self.lc, [])
                for edra in sidechain:
                    # test less useful here but avoids populating rak cache if possible
                    if all(atm in self.akc for atm in edra):
                        r_edra = [self.rak(atom) for atom in edra]
                        self._gen_edra(r_edra)

        # final processing of all dihedra just generated
        self.link_dihedra(verbose)

        # now do the actual work computing di/hedra values from atom coordinates
        # -> updated to process at chain level  (vectorized)
        #
        # for d in self.dihedra.values():
        #    # populate values and hedra for dihedron objects
        #    d.dihedron_from_atoms()
        # for h in self.hedra.values():
        #    # miss redundant hedra above, needed for some chi1 angles
        #    # also miss if missing atoms means hedron not in dihedra
        #    if h.atoms_needs_update:
        #        h.hedron_from_atoms(self.atom_coords)

        # create di/hedra for gly Cbeta if needed, populate values later
        if self.gly_Cbeta and "G" == self.lc:  # and self.atom_coords[sCB] is None:
            # add C-beta for Gly

            self.ak_set.add(AtomKey(self, "CB"))
            sCB = self.rak("CB")
            sCB.missing = False  # was True because akc cache did not have entry
            self.cic.akset.add(sCB)
            # self.atom_coords[sCB] = None

            # main orientation comes from O-C-Ca-Cb so make Cb-Ca-C hedron
            sO = self.rak("O")
            htpl = (sCB, sCA, sC)
            self._gen_edra(htpl)
            # rtm - old, cleanup
            # values generated in build_glyCB
            # h = self.hedra[htpl]
            # h.lal[2] = self.hedra[(sCA, sC, sO)].lal[0]  # CaCO len12 -> len23
            # h.lal[1] = 110.17513  # angle
            # h.lal[0] = Ca_Cb_Len  # len12

            # generate dihedral based on N-Ca-C-O offset from db query above
            dtpl = (sO, sC, sCA, sCB)
            self._gen_edra(dtpl)
            d = self.dihedra[dtpl]
            d.ric = self
            d.set_hedra()

            # rtm is this necessary, and can't we do just once below if so?
            self.link_dihedra(verbose)  # re-run for new dihedra

            # prepare to add new Gly CB atom(s)
            # in IC_Chain.atom_to_internal_coordinates()
            if not hasattr(self.cic, "gcb"):
                self.cic.gcb: Dict[AtomKey, Tuple] = {}
                # self.cic.gcb: List[AtomKey] = []
                # self.cic.gcb: Dict[AtomKey, np.array] = {}
            self.cic.gcb[sCB] = dtpl
            # self.cic.gcb.append(sCB)
            # self.cic.gcb[sCB] = np.array((0, 0, 0, 1), dtype=np.float64)

        if verbose:
            # oAtom =
            self.rak("O")  # trigger missing flag if needed
            missing = []
            for akk, akv in self.akc.items():
                if isinstance(akk, str) and akv.missing:
                    missing.append(akv)
            if missing:
                chn = self.residue.parent
                chn_id = chn.id
                chn_len = len(chn.internal_coord.ordered_aa_ic_list)
                print(f"chain {chn_id} len {chn_len} missing atom(s): {missing}")

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
        s += " " + ak.id + " " + f"{atm.get_bfactor():6.2f}"
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

        if (
            0 == len(self.rprev)
            and hasattr(self, "NCaCKey")
            and self.NCaCKey is not None
        ):
            NCaChedron = self.pick_angle(self.NCaCKey[0])  # first tau
            if NCaChedron is not None:
                try:
                    ts = IC_Residue._pdb_atom_string(self.residue["N"])
                    ts += IC_Residue._pdb_atom_string(self.residue["CA"])
                    ts += IC_Residue._pdb_atom_string(self.residue["C"])
                    s += ts  # only if no exception: have all 3 atoms
                except KeyError:
                    pass

        base = pdbid + " " + chainid + " "

        cic = self.cic
        for h in sorted(self.hedra.values()):
            hndx = h.ndx
            try:
                s += (
                    base
                    + h.id
                    + " "
                    + "{:9.5f} {:9.5f} {:9.5f}".format(
                        set_accuracy_95(cic.hedraL12[hndx]),  # (h.lal[0]),  # len12
                        set_accuracy_95(cic.hedraAngle[hndx]),  # (h.lal[1]),  # angle
                        set_accuracy_95(cic.hedraL23[hndx]),  # (h.lal[2]),  # len23
                    )
                )
            except KeyError:
                pass
            s += "\n"
        for d in sorted(self.dihedra.values()):
            try:
                s += (
                    base
                    + d.id
                    + " "
                    + "{:9.5f}".format(set_accuracy_95(cic.dihedraAngle[d.ndx]))
                )
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

    def coords_to_residue(self, rnext: bool = False) -> None:
        """Convert self.atom_coords to biopython Residue Atom coords.

        Copy homogeneous IC_Residue atom_coords to self.residue cartesian
        Biopython Atom coords.

        :param rnext: bool default False
            next IC_Residue has no atom coords due to missing atoms, so try to
            populate with any available coords calculated for this residue
            di/hedra extending into next
        """
        if rnext:
            respos, icode = self.rnext[0].residue.id[1:3]
        else:
            respos, icode = self.residue.id[1:3]
        respos = str(respos)
        spNdx, icNdx, resnNdx, atmNdx, altlocNdx, occNdx = AtomKey.fields

        if rnext:
            Res = self.rnext[0].residue
        else:
            Res = self.residue
        ndx = Res.parent.internal_coord.ndx

        for ak in sorted(self.atom_coords):
            # print(ak)
            if (
                self.cic.atomArrayValid[self.cic.atomArrayIndex[ak]]
                and respos == ak.akl[spNdx]
                and ((icode == " " and ak.akl[icNdx] is None) or icode == ak.akl[icNdx])
            ):

                ac = self.atom_coords[ak]
                atm_coords = ac[:3]
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
                    sn = Atm.get_serial_number()
                    if sn is not None:
                        ndx = sn + 1

        Res.parent.internal_coord.ndx = ndx

    def _get_ak_tuple(self, ak_str: str) -> Optional[Tuple["AtomKey", ...]]:
        """Convert atom pair string to AtomKey tuple.

        :param ak_str: str
            Two atom names separated by ':', e.g. 'N:CA'
            Optional position specifier relative to self,
            e.g. '-1C:N' for preceding peptide bond.
        """
        AK = AtomKey
        S = self
        angle_key2 = []
        akstr_list = ak_str.split(":")
        lenInput = len(akstr_list)
        for a in akstr_list:
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
        if len(angle_key2) != lenInput:
            return None
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
        else:
            return None
        return rval

    # @profile
    def pick_angle(
        self, angle_key: Union[EKT, str]
    ) -> Optional[Union["Hedron", "Dihedron"]]:
        """Get Hedron or Dihedron for angle_key.

        :param angle_key:
            - tuple of 3 or 4 AtomKeys
            - string of atom names ('CA') separated by :'s
            - string of [-1, 0, 1]<atom name> separated by ':'s. -1 is
              previous residue, 0 is this residue, 1 is next residue
            - psi, phi, omg, omega, chi1, chi2, chi3, chi4, chi5
            - tau (N-CA-C angle) see Richardson1981
            - except for tuples of AtomKeys, no option to access alternate disordered atoms

        Observe that a residue's phi and omega dihedrals, as well as the hedra
        comprising them (including the N:Ca:C tau hedron), are stored in the
        n-1 di/hedra sets; this is handled here, but may be an issue if accessing
        directly.

        The following are equivalent (except for sidechains with non-carbon
        atoms for chi2)::

            ric = r.internal_coord
            print(
                r,
                ric.get_angle("psi"),
                ric.get_angle("phi"),
                ric.get_angle("omg"),
                ric.get_angle("tau"),
                ric.get_angle("chi2"),
            )
            print(
                r,
                ric.get_angle("N:CA:C:1N"),
                ric.get_angle("-1C:N:CA:C"),
                ric.get_angle("-1CA:-1C:N:CA"),
                ric.get_angle("N:CA:C"),
                ric.get_angle("CA:CB:CG:CD"),
            )

        See ic_data.py for detail of atoms in the enumerated sidechain angles
        and the backbone angles which do not span the peptide bond. Using 's'
        for current residue ('self') and 'n' for next residue, the spanning
        angles are::

                (sN, sCA, sC, nN)   # psi
                (sCA, sC, nN, nCA)  # omega i+1
                (sC, nN, nCA, nC)   # phi i+1
                (sCA, sC, nN)
                (sC, nN, nCA)
                (nN, nCA, nC)       # tau i+1

        :return: Matching Hedron, Dihedron, or None.
        """
        rval: Optional[Union["Hedron", "Dihedron"]] = None
        if isinstance(angle_key, tuple):
            rval = self._get_angle_for_tuple(angle_key)
            if rval is None and self.rprev:
                rval = self.rprev[0]._get_angle_for_tuple(angle_key)
        elif ":" in angle_key:
            angle_key = cast(EKT, self._get_ak_tuple(cast(str, angle_key)))
            if angle_key is None:
                return None
            rval = self._get_angle_for_tuple(angle_key)
            if rval is None and self.rprev:
                rval = self.rprev[0]._get_angle_for_tuple(angle_key)
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
            ndx = (2 * int(angle_key[-1])) - 1
            try:
                akl = sclist[ndx]
                if akl[4] == angle_key:
                    klst = [self.rak(a) for a in akl[0:4]]
                    tklst = cast(DKT, tuple(klst))
                    rval = self.dihedra.get(tklst, None)
                else:
                    return None
            except IndexError:
                return None

        return rval

    def get_angle(self, angle_key: Union[EKT, str]) -> Optional[float]:
        """Get dihedron or hedron angle for specified key.

        See pick_angle() for key specifications.
        """
        edron = self.pick_angle(angle_key)
        if edron:
            return edron.angle
        return None

    def set_angle(self, angle_key: Union[EKT, str], v: float):
        """Set dihedron or hedron angle for specified key.

        See pick_angle() for key specifications.
        """
        rval = self.pick_angle(angle_key)
        if rval is not None:
            rval.angle = v

    def pick_length(
        self, ak_spec: Union[str, BKT]
    ) -> Tuple[Optional[List["Hedron"]], Optional[BKT]]:
        """Get list of hedra containing specified atom pair.

        :param ak_spec:
            - tuple of two AtomKeys
            - string: two atom names separated by ':', e.g. 'N:CA' with
              optional position specifier relative to self, e.g. '-1C:N' for
              preceding peptide bond.

        The following are equivalent::

            ric = r.internal_coord
            print(
                r,
                ric.get_length("0C:1N"),
            )
            print(
                r,
                None
                if not ric.rnext
                else ric.get_length((ric.rak("C"), ric.rnext[0].rak("N"))),
            )

        :return: list of hedra containing specified atom pair, tuple of atom keys
        """
        rlst: List[Hedron] = []
        # if ":" in ak_spec:
        if isinstance(ak_spec, str):
            ak_spec = cast(BKT, self._get_ak_tuple(ak_spec))
        if ak_spec is None:
            return None, None
        for hed_key, hed_val in self.hedra.items():
            if all(ak in hed_key for ak in ak_spec):
                rlst.append(hed_val)
        return rlst, ak_spec

    def get_length(self, ak_spec: Union[str, BKT]) -> Optional[float]:
        """Get bond length for specified atom pair.

        See pick_length() for ak_spec.
        """
        hed_lst, ak_spec2 = self.pick_length(ak_spec)
        if hed_lst is None or ak_spec2 is None:
            return None

        for hed in hed_lst:
            val = hed.get_length(ak_spec2)
            if val is not None:
                return val
        return None

    def set_length(self, ak_spec: Union[str, BKT], val: float) -> None:
        """Set bond length for specified atom pair.

        See pick_length() for ak_spec.
        """
        hed_lst, ak_spec2 = self.pick_length(ak_spec)
        if hed_lst is not None and ak_spec2 is not None:
            for hed in hed_lst:
                hed.set_length(ak_spec2, val)

    def applyMtx(self, mtx: np.array) -> None:
        """Apply matrix to atom_coords for this IC_Residue."""
        for ak, ac in self.atom_coords.items():
            # self.atom_coords[ak] = mtx @ ac
            self.atom_coords[ak] = mtx.dot(ac)


class Edron:
    """Base class for Hedron and Dihedron classes.

    Supports rich comparison based on lists of AtomKeys.

    Attributes
    ----------
    aks: tuple
        3 (hedron) or 4 (dihedron) AtomKeys defining this di/hedron
    id: str
        ':'-joined string of AtomKeys for this di/hedron
    needs_update: bool
        indicates di/hedron local atom_coords do NOT reflect current di/hedron
        angle and length values in hedron local coordinate space
    dh_class: str
        sequence of atoms (no position or residue) comprising di/hedron
        for statistics
    rdh_class: str
        sequence of residue, atoms comprising di/hedron for statistics
    edron_re: compiled regex (Class Attribute)
        A compiled regular expression matching string IDs for Hedron
        and Dihedron objects
    cic: IC_Chain reference
        Chain internal coords object containing this hedron; set in
        IC_Residue link_dihedra()
    ndx: int
        index into IC_Chain level numpy data arrays for di/hedra.
        Set in IC_Chain init_edra()

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
        r"(((?P<len12>\S+)\s+(?P<angle>\S+)\s+(?P<len23>\S+)\s*$)|"
        # dihedral angle for dihedron
        r"((?P<dihedral>\S+)\s*$))"
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
            # lst = cast(List[AtomKey], lst)
            # return ":".join(ak.id for ak in lst)
            if 4 == len(lst):
                return f"{lst[0].id}:{lst[1].id}:{lst[2].id}:{lst[3].id}"
            else:
                return f"{lst[0].id}:{lst[1].id}:{lst[2].id}"

    # @profile
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
        # (do not need to be recalculated from angle and or length values)
        self.needs_update = True

        # IC_Chain which contains this di/hedron
        self.cic: IC_Chain

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

    def __deepcopy__(self, memo):
        existing = memo.get(id(self), False)
        if existing:
            return existing
        dup = type(self).__new__(self.__class__)
        memo[id(self)] = dup
        dup.__dict__.update(self.__dict__)  # mostly static attribs
        dup.cic = memo[id(self.cic)]
        dup.aks = copy.deepcopy(self.aks, memo)
        return dup

    def gen_acs(self, atom_coords: Dict["AtomKey", np.array]) -> Tuple[np.array, ...]:
        """Generate tuple of atom coord arrays for keys in self.aks.

        :param atom_coords: AtomKey dict of atom coords for residue
        :raises: MissingAtomError any atoms in self.aks missing coordinates
        """
        aks = self.aks
        acs: List[np.array] = []
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
        """Comparison function ranking self vs. other; False on equal."""
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
    lal: numpy array of len12, angle, len23
        len12 = distance between 1st and 2nd atom
        angle = angle (degrees) formed by 3 atoms
        len23 = distance between 2nd and 3rd atoms

    atoms: 3x4 numpy arrray (view on chain array)
        3 homogeneous atoms comprising hedron, 1st on XZ, 2nd at origin, 3rd on +Z
    atomsR: 3x4 numpy array (view on chain array)
        atoms reversed, 1st on +Z, 2nd at origin, 3rd on XZ plane

    Methods
    -------
    get_length()
        get bond length for specified atom pair
    set_length()
        set bond length for specified atom pair
    angle(), len12(), len23()
        gettters and setters for relevant attributes (angle in degrees)
    """

    def __init__(self, *args: Union[List["AtomKey"], HKT], **kwargs: str) -> None:
        """Initialize Hedron with sequence of AtomKeys, kwargs.

        Acceptable input:
            As for Edron, plus optional 'len12', 'angle', 'len23'
            keyworded values.
        """
        super().__init__(*args, **kwargs)

        # print('initialising', self.id)

        # 3 matrices specifying hedron space coordinates of constituent atoms,
        # initially atom3 on +Z axis
        # self.atoms: HACS
        # 3 matrices, hedron space coordinates, reversed order
        # initially atom1 on +Z axis
        # self.atomsR: HACS

        # if "len12" in kwargs:
        #    # self.lal = np.array(
        #    #    (
        #    #        float(kwargs["len12"]),
        #    #        float(kwargs["angle"]),
        #    #        float(kwargs["len23"]),
        #    #    )
        #    # )
        #    self.L12 = np.array(float(kwargs["len12"]), dtype=np.float64)
        #    self.Angle = np.array(float(kwargs["angle"]), dtype=np.float64)
        #    self.L23 = np.array(float(kwargs["len23"]), dtype=np.float64)
        # else:
        #    # self.lal = np.zeros(3)
        #    self.L12 = np.zeros(1)
        #    self.Angle = np.zeros(1)
        #    self.L23 = np.zeros(1)

        # else:
        #    self.len12 = None
        #    self.angle = None
        #    self.len23 = None

        # print(self)

    # __deepcopy__ covered by Edron superclass

    def __repr__(self) -> str:
        """Print string for Hedron object."""
        # return f"3-{self.id} {self.rdh_class} {str(self.lal[0])} {str(self.lal[1])} {str(self.lal[2])}"
        return f"3-{self.id} {self.rdh_class} {str(self.L12)} {str(self.Angle)} {str(self.L23)}"

    @property
    def angle(self) -> float:
        """Get this hedron angle."""
        try:
            # return self.lal[1]  # _angle
            return self.cic.hedraAngle[self.ndx]  # _angle
        except AttributeError:
            return 0.0

    def _invalidate_atoms(self):
        print("INVALIDATE_ATOMS")
        self.cic.hAtoms_needs_update[self.ndx] = True
        for ak in self.aks:
            self.cic.atomArrayValid[self.cic.atomArrayIndex[ak]] = False

    @angle.setter
    def angle(self, angle_deg) -> None:
        """Set this hedron angle; sets needs_update."""
        # self.lal[1] = angle_deg  # view on chain numpy arrays
        self.cic.hedraAngle[self.ndx] = angle_deg
        # self._invalidate_atoms()
        self.cic.hAtoms_needs_update[self.ndx] = True
        self.cic.atomArrayValid[self.cic.atomArrayIndex[self.aks[2]]] = False

    @property
    def len12(self):
        """Get first length for Hedron."""
        try:
            # return self.lal[0]  # _len12
            return self.cic.hedraL12[self.ndx]
        except AttributeError:
            return 0.0

    @len12.setter
    def len12(self, len):
        """Set first length for Hedron; sets needs_update."""
        # self.lal[0]  # _len12 = len  # rtm
        self.cic.hedraL12[self.ndx] = len
        # self._invalidate_atoms()
        self.cic.hAtoms_needs_update[self.ndx] = True
        self.cic.atomArrayValid[self.cic.atomArrayIndex[self.aks[1]]] = False
        self.cic.atomArrayValid[self.cic.atomArrayIndex[self.aks[2]]] = False

    @property
    def len23(self) -> float:
        """Get second length for Hedron."""
        try:
            # return self.lal[2]  # _len23
            return self.cic.hedraL23[self.ndx]
        except AttributeError:
            return 0.0

    @len23.setter
    def len23(self, len):
        """Set second length for Hedron; sets needs_update."""
        # self.lal[2] = len
        self.cic.hedraL23[self.ndx] = len
        # self._invalidate_atoms()
        self.cic.hAtoms_needs_update[self.ndx] = True
        self.cic.atomArrayValid[self.cic.atomArrayIndex[self.aks[2]]] = False

    def get_length(self, ak_tpl: BKT) -> Optional[float]:
        """Get bond length for specified atom pair.

        :param ak_tpl: tuple of AtomKeys
            pair of atoms in this Hedron
        """
        if 2 > len(ak_tpl):
            return None
        if all(ak in self.aks[:2] for ak in ak_tpl):
            # return self.lal[0]  # len12
            return self.cic.hedraL12[self.ndx]
        if all(ak in self.aks[1:] for ak in ak_tpl):
            # return self.lal[2]  # len23
            return self.cic.hedraL23[self.ndx]
        return None

    def set_length(self, ak_tpl: BKT, newLength: float):
        """Set bond length for specified atom pair; sets needs_update.

        :param ak_tpl: tuple of AtomKeys
            pair of atoms in this Hedron
        """
        if 2 > len(ak_tpl):
            raise TypeError("Require exactly 2 AtomKeys: %s" % (str(ak_tpl)))
        elif all(ak in self.aks[:2] for ak in ak_tpl):
            # self.lal[0] = newLength  # len12
            self.cic.hedraL12[self.ndx] = newLength
        elif all(ak in self.aks[1:] for ak in ak_tpl):
            # self.lal[2] = newLength  # len23
            self.cic.hedraL23[self.ndx] = newLength
        else:
            raise TypeError("%s not found in %s" % (str(ak_tpl), self))
        self._invalidate_atoms()


class Dihedron(Edron):
    """Class to represent four joined atoms forming a dihedral angle.

    Attributes
    ----------
    angle: float
        Measurement or specification of dihedral angle in degrees
    hedron1, hedron2: Hedron object references
        The two hedra which form the dihedral angle
    h1key, h2key: tuples of AtomKeys
        Hash keys for hedron1 and hedron2
    id3,id32: tuples of AtomKeys
        First 3 and second 3 atoms comprising dihedron; hxkey orders may differ
    initial_coords: tuple[4] of numpy arrays [4]
        Local atom coords for 4 atoms, [0] on XZ plane, [1] at origin,
        [2] on +Z, [3] rotated by dihedral
    a4_pre_rotation: numpy array [4]
        4th atom of dihedral aligned to XZ plane (angle not applied)
    ic_residue: IC_Residue object reference
        IC_Residue object containing this dihedral
    reverse: bool
        Indicates order of atoms in dihedron is reversed from order of atoms
        in hedra (configured by set_hedra())
    cst, rcst: numpy array [4][4]
        transforms to and from coordinate space defined by first hedron.
        set by IC_Residue.assemble().  defined by id3 order NOT h1key order
        (atoms may be reversed between these two).  View on IC_Chain
        dCoordSpace

    Methods
    -------
    set_hedra()
        work out hedra keys and orientation for this dihedron
    angle()
        getter/setter for dihdral angle in degrees

    """

    def __init__(self, *args: Union[List["AtomKey"], DKT], **kwargs: str) -> None:
        """Initialize Dihedron with sequence of AtomKeys and optional dihedral angle.

        Acceptable input:
            As for Edron, plus optional 'dihedral' keyworded angle value.
        """
        super().__init__(*args, **kwargs)

        # hedra making up this dihedron; set by self:set_hedra()
        self.hedron1: Hedron  # = None
        self.hedron2: Hedron  # = None

        self.h1key: HKT  # = None
        self.h2key: HKT  # = None

        # h1, h2key above may be reversed; id3,2 will not be

        self.id3: HKT = cast(HKT, tuple(self.aks[0:3]))
        self.id32: HKT = cast(HKT, tuple(self.aks[1:4]))

        # 4 matrices specifying hedron space coordinates of constituent atoms,
        # in this space atom 3 is on on +Z axis
        # see coord_space()
        # rtm self.initial_coords: DACS
        # rtm self.a4_pre_rotation: np.array

        # IC_Residue object which includes this dihedron;
        # set by Residue:linkDihedra()
        self.ric: IC_Residue
        # order of atoms in dihedron is reversed from order of atoms in hedra
        self.reverse = False

        # coordinate space transform matrices
        # defined by id3 order NOT h1key order (may be reversed)
        # self.cst = None  # protein coords to 1st hedron coord space
        # self.rcst = None  # reverse = 1st hedron coords back to protein coords

        # if "dihedral" in kwargs:
        #    self.angle = float(kwargs["dihedral"])

    """
    def __deepcopy__(self, memo):
        existing = memo.get(id(self), False)
        if existing:
            return existing
        dup = super(Dihedron, self).__deepcopy__(memo)
        memo[id(self)] = dup
        dup.__dict__.update(self.__dict__)  # update later
        dup.cic = memo[id(self.cic)]
        # dup.ric = memo[id(self.ric)]
        dup.aks = copy.deepcopy(self.aks, memo)
        if hasattr(self, "hedron1"):
            dup.hedron1 = memo[id(self.hedron1)]
            dup.hedron2 = memo[id(self.hedron2)]
            dup.h1key = copy.deepcopy(self.h1key, memo)
            dup.h2key = copy.deepcopy(self.h2key, memo)
        dup.id3 = copy.deepcopy(self.id3, memo)
        dup.id32 = copy.deepcopy(self.id32, memo)
        # still need to update: hedron1,2, h1,2key, id3,2
        return dup
    """

    def __repr__(self) -> str:
        """Print string for Dihedron object."""
        return f"4-{str(self.id)} {self.rdh_class} {str(self.angle)} {str(self.ric)}"

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

    def set_hedra(self) -> Tuple[bool, Hedron, Hedron]:
        """Work out hedra keys and set rev flag."""
        try:
            return self.rev, self.hedron1, self.hedron2
        except AttributeError:
            pass

        rev = False
        res = self.ric
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

    @property
    def angle(self) -> float:
        """Get dihedral angle."""
        try:
            return self.cic.dihedraAngle[self.ndx]
        except AttributeError:
            try:
                return self._dihedral
            except AttributeError:
                return 360.0  # error value without type hint hassles

    @angle.setter
    def angle(self, dangle_deg_in: float) -> None:
        """Save new dihedral angle; sets needs_update.

        faster to modify IC_Chain level arrays directly.

        N.B. dihedron (i-1)C-N-CA-CB is ignored if O exists.
        C-beta is by default placed using O-C-CA-CB, but O is missing
        in some PDB file residues, which means the sidechain cannot be
        placed.  The alternate CB path (i-1)C-N-CA-CB is provided to
        circumvent this, but if this is needed then it must be adjusted in
        conjunction with PHI ((i-1)C-N-CA-C) as they overlap. **So if you
        change one of these angles you need to change the other yourself.**

        :param dangle_deg: float new dihedral angle in degrees
        """
        if dangle_deg_in > 180.0:
            dangle_deg = dangle_deg_in - 360.0
        elif dangle_deg_in < -180.0:
            dangle_deg = dangle_deg_in + 360.0
        else:
            dangle_deg = dangle_deg_in

        self._dihedral = dangle_deg
        self.needs_update = True
        # rtm
        if True:  # try:
            cic = self.cic
            dndx = self.ndx
            cic.dihedraAngle[dndx] = dangle_deg
            cic.dihedraAngleRads[dndx] = np.deg2rad(dangle_deg)
            cic.dAtoms_needs_update[dndx] = True
            cic.atomArrayValid[cic.atomArrayIndex[self.aks[3]]] = False

        # except AttributeError:
        #    pass


class AtomKey:
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
    endnum_re: compiled regex (Class Attribute)
        A compiled regular expresion capturing digits at end of a string
    d2h: bool (Class Attribute)
        Convert D atoms to H on input; must also modify IC_Residue.accept_atoms
    missing: bool default False
        AtomKey __init__'d from string is probably missing, set this flag to
        note the issue (not set here)
    ric: IC_Residue default None
        *If* initialised with IC_Residue, this stores the IC_residue

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

    endnum_re = re.compile(r"\D+(\d+)$")

    # PDB altLoc = Character = [\w ] (any non-ctrl ASCII incl space)
    # PDB iCode = AChar = [A-Za-z]

    fieldNames = ("respos", "icode", "resname", "atm", "altloc", "occ")
    fieldsDef = namedtuple(
        "fieldsDef", ["respos", "icode", "resname", "atm", "altloc", "occ"]
    )
    fields = fieldsDef(0, 1, 2, 3, 4, 5)

    d2h = False  # convert D Deuterium to H Hydrogen on input
    # icd = {"icr": 0, "atm": 0, "lst": 0, "dct": 0, "_": 0, "else": 0}

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
        self.ric = None
        # self.id = None
        for arg in args:
            if isinstance(arg, str):
                if "_" in arg:
                    # AtomKey.icd["_"] += 1
                    # got atom key string, recurse with regex parse
                    m = self.atom_re.match(arg)
                    if m is not None:
                        if akl != []:  # [] != akl:
                            raise Exception(
                                "Atom Key init full key not first argument: " + arg
                            )
                        # for fn in AtomKey.fieldNames:
                        #    akl.append(m.group(fn))
                        # akl = [m.group(fn) for fn in AtomKey.fieldNames]
                        akl = list(map(m.group, AtomKey.fieldNames))
                else:
                    # AtomKey.icd["else"] += 1
                    akl.append(arg)

            elif isinstance(arg, IC_Residue):
                # AtomKey.icd["icr"] += 1
                if akl != []:
                    raise Exception("Atom Key init Residue not first argument")
                akl = list(arg.rbase)
                self.ric = arg
            elif isinstance(arg, Atom):
                # AtomKey.icd["atm"] += 1
                if 3 != len(akl):
                    raise Exception("Atom Key init Atom before Residue info")
                akl.append(arg.name)
                altloc = arg.altloc
                akl.append(altloc if altloc != " " else None)
                occ = float(arg.occupancy)
                akl.append(str(occ) if occ != 1.00 else None)
            elif isinstance(arg, list) or isinstance(arg, tuple):
                # AtomKey.icd["lst"] += 1
                akl += arg
            elif isinstance(arg, dict):
                # AtomKey.icd["dct"] += 1
                for k in AtomKey.fieldNames:
                    akl.append(arg.get(k, None))
            else:
                raise Exception("Atom Key init not recognised")

        # process kwargs, initialize occ and altloc to None
        # if not specified above
        # for i in range(6):
        #    if len(akl) <= i:
        #        fld = kwargs.get(AtomKey.fieldNames[i])
        #        if fld is not None:
        #            akl.append(fld)

        for i in range(len(akl), 6):
            if len(akl) <= i:
                fld = kwargs.get(AtomKey.fieldNames[i])
                if fld is not None:
                    akl.append(fld)

        # tweak local akl to generate id string
        if isinstance(akl[0], int):
            akl[0] = str(akl[0])  # numeric residue position to string

        # occNdx = AtomKey.fields.occ
        # if akl[occNdx] is not None:
        #    akl[occNdx] = str(akl[occNdx])  # numeric occupancy to string

        if self.d2h:
            atmNdx = AtomKey.fields.atm
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

        # while len(akl) < 6:
        #    akl.append(None)  # add no altloc, occ if not specified
        akl += [None] * (6 - len(akl))

        self.akl = tuple(akl)
        self._hash = hash(self.akl)
        self.missing = False

    def __deepcopy__(self, memo):
        # will fail if .ric not in memo
        existing = memo.get(id(self), False)
        if existing:
            return existing
        dup = type(self).__new__(self.__class__)
        memo[id(self)] = dup
        dup.__dict__.update(self.__dict__)  # all static attribs except .ric
        if self.ric is not None:
            dup.ric = memo[id(self.ric)]
        # deepcopy complete
        return dup

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

    # @profile
    def _cmp(self, other: "AtomKey") -> Tuple[int, int]:
        """Comparison function ranking self vs. other."""
        for i in range(6):
            s, o = self.akl[i], other.akl[i]
            if s != o:
                # insert_code, altloc can be None, deal with first
                if s is None and o is not None:
                    # no insert code before named insert code
                    return 0, 1
                elif o is None and s is not None:
                    return 1, 0
                # now we know s, o not None
                # s, o = cast(str, s), cast(str, o)  # performance critical code

                if AtomKey.fields.atm != i:
                    # only sorting complications at atom level, occ.
                    # otherwise respos, insertion code will trigger
                    # before residue name
                    if AtomKey.fields.occ == i:
                        oi = int(float(s) * 100)
                        si = int(float(o) * 100)
                        return si, oi  # swap so higher occupancy comes first
                    elif AtomKey.fields.respos == i:
                        return int(s), int(o)
                    else:  # resname or altloc
                        return ord(s), ord(o)

                # atom names from here
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

                # sidechain vs sidechain, sidechain vs H
                ss = self._sidechain_sort_keys.get(s, None)
                os = self._sidechain_sort_keys.get(o, None)
                if ss is not None and os is not None:
                    return ss, os
                elif ss is not None and os is None:
                    return 0, 1
                elif ss is None and os is not None:
                    return 1, 0

                # amide single 'H' captured above in sidechain sort
                # now 'complex'' hydrogens after sidechain
                s0, s1, o0, o1 = s[0], s[1], o[0], o[1]
                s1d, o1d = s1.isdigit(), o1.isdigit()
                # if "H" == s0 == o0: # breaks cython
                if ("H" == s0) and ("H" == o0):

                    if (s1 == o1) or (s1d and o1d):
                        enmS = self.endnum_re.findall(s)
                        enmO = self.endnum_re.findall(o)
                        if (enmS != []) and (enmO != []):
                            return int(enmS[0]), int(enmO[0])
                        elif enmS == []:
                            return 0, 1
                        else:
                            return 1, 0
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
            return self.akl == other.akl
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

    Used by Hedron and Dihedron classes writing PIC and SCAD files.
    :param float num: input number
    :returns: float with specified accuracy
    """
    # return round(num, 5)  # much slower
    return float(f"{num:9.5f}")


# only used for writing PDB atoms so inline in
# _pdb_atom_string(atm: Atom)
# def set_accuracy_83(num: float) -> float:
#    """Reduce floating point accuracy to 8.3 (xxxxx.xxx).
#
#    Used by IC_Residue class, matches PDB output format.
#    :param float num: input number
#    :returns: float with specified accuracy
#    """
#    return float("{:8.3f}".format(num))


# internal coordinates construction Exceptions
class HedronMatchError(Exception):
    """Cannot find hedron in residue for given key."""

    pass


class MissingAtomError(Exception):
    """Missing atom coordinates for hedron or dihedron."""

    pass
