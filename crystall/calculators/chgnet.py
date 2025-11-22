from crystall import BaseCalculator
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from pymatgen.core import Structure
from pymatgen.io.vasp import Poscar
from decimal import Decimal, ROUND_DOWN
import os
from typing import Dict, Any, List, Optional, Sequence
import io
import contextlib
import logging
import numpy as np
from itertools import repeat
import traceback

energy_reference_data = {
    "_reference_provenance": {
        "source": "matbench_discovery",
        "matbench_discovery_version": "1.3.1",
        "snapshot_file": "2023-02-07-mp-elemental-reference-entries.json.gz",
    },
    "Ac": {
        "mp_id": "mp-862690",
        "chgnet_static": -4.064313888549805,
        "chgnet_relaxed": -4.064481735229492,
    },
    "Ag": {
        "mp_id": "mp-989737",
        "chgnet_static": -2.8087916374206543,
        "chgnet_relaxed": -2.808791399002075,
    },
    "Al": {
        "mp_id": "mp-134",
        "chgnet_static": -3.664341449737549,
        "chgnet_relaxed": -3.664341449737549,
    },
    "Ar": {
        "mp_id": "mp-23155",
        "chgnet_static": -0.05527100712060928,
        "chgnet_relaxed": -0.06331204622983932,
    },
    "As": {
        "mp_id": "mp-11",
        "chgnet_static": -4.577264785766602,
        "chgnet_relaxed": -4.580099105834961,
    },
    "Au": {
        "mp_id": "mp-81",
        "chgnet_static": -3.2357232570648193,
        "chgnet_relaxed": -3.2357234954833984,
    },
    "B": {
        "mp_id": "mp-160",
        "chgnet_static": -6.567556381225586,
        "chgnet_relaxed": -6.568004608154297,
    },
    "Ba": {
        "mp_id": "mp-122",
        "chgnet_static": -1.8816522359848022,
        "chgnet_relaxed": -1.8821139335632324,
    },
    "Be": {
        "mp_id": "mp-87",
        "chgnet_static": -3.726207971572876,
        "chgnet_relaxed": -3.72886323928833,
    },
    "Bi": {
        "mp_id": "mp-567597",
        "chgnet_static": -3.819345712661743,
        "chgnet_relaxed": -3.8228907585144043,
    },
    "Br": {
        "mp_id": "mp-998864",
        "chgnet_static": -1.510204553604126,
        "chgnet_relaxed": -1.5416802167892456,
    },
    "C": {
        "mp_id": "mp-2516584",
        "chgnet_static": -9.17431640625,
        "chgnet_relaxed": -9.177828788757324,
    },
    "Ca": {
        "mp_id": "mp-45",
        "chgnet_static": -1.9687459468841553,
        "chgnet_relaxed": -1.969692587852478,
    },
    "Cd": {
        "mp_id": "mp-1096861",
        "chgnet_static": -0.9207510948181152,
        "chgnet_relaxed": -0.9222511649131775,
    },
    "Ce": {
        "mp_id": "mp-567332",
        "chgnet_static": -5.858234405517578,
        "chgnet_relaxed": -5.862658500671387,
    },
    "Cl": {
        "mp_id": "mp-1008394",
        "chgnet_static": -1.8403372764587402,
        "chgnet_relaxed": -1.8926973342895508,
    },
    "Co": {
        "mp_id": "mp-54",
        "chgnet_static": -7.027796268463135,
        "chgnet_relaxed": -7.028364181518555,
    },
    "Cr": {
        "mp_id": "mp-90",
        "chgnet_static": -9.448907852172852,
        "chgnet_relaxed": -9.540873527526855,
    },
    "Cs": {
        "mp_id": "mp-1184151",
        "chgnet_static": -0.8381786346435547,
        "chgnet_relaxed": -0.8386663198471069,
    },
    "Cu": {
        "mp_id": "mp-30",
        "chgnet_static": -4.077049255371094,
        "chgnet_relaxed": -4.083481788635254,
    },
    "Dy": {
        "mp_id": "mp-10658",
        "chgnet_static": -4.583230018615723,
        "chgnet_relaxed": -4.583526611328125,
    },
    "Er": {
        "mp_id": "mp-1184115",
        "chgnet_static": -4.5463361740112305,
        "chgnet_relaxed": -4.546407699584961,
    },
    "Eu": {
        "mp_id": "mp-623532",
        "chgnet_static": -10.253379821777344,
        "chgnet_relaxed": -10.253378868103027,
    },
    "F": {
        "mp_id": "mp-561203",
        "chgnet_static": -1.8959310054779053,
        "chgnet_relaxed": -1.8960890769958496,
    },
    "Fe": {
        "mp_id": "mp-13",
        "chgnet_static": -8.355401992797852,
        "chgnet_relaxed": -8.35660171508789,
    },
    "Ga": {
        "mp_id": "mp-142",
        "chgnet_static": -3.006042718887329,
        "chgnet_relaxed": -3.0169613361358643,
    },
    "Gd": {
        "mp_id": "mp-155",
        "chgnet_static": -14.04521369934082,
        "chgnet_relaxed": -14.046666145324707,
    },
    "Ge": {
        "mp_id": "mp-32",
        "chgnet_static": -4.44661283493042,
        "chgnet_relaxed": -4.456517219543457,
    },
    "H": {
        "mp_id": "mp-24504",
        "chgnet_static": -3.3606221675872803,
        "chgnet_relaxed": -3.361328125,
    },
    "He": {
        "mp_id": "mp-23158",
        "chgnet_static": -0.015290394425392151,
        "chgnet_relaxed": -0.01529046893119812,
    },
    "Hf": {
        "mp_id": "mp-103",
        "chgnet_static": -9.91093635559082,
        "chgnet_relaxed": -9.912409782409668,
    },
    "Hg": {
        "mp_id": "mp-1184554",
        "chgnet_static": -0.2687626779079437,
        "chgnet_relaxed": -0.2687630355358124,
    },
    "Ho": {
        "mp_id": "mp-10659",
        "chgnet_static": -4.562739849090576,
        "chgnet_relaxed": -4.562739849090576,
    },
    "I": {
        "mp_id": "mp-639751",
        "chgnet_static": -1.5589423179626465,
        "chgnet_relaxed": -1.6062884330749512,
    },
    "In": {
        "mp_id": "mp-1184502",
        "chgnet_static": -2.6948251724243164,
        "chgnet_relaxed": -2.701188325881958,
    },
    "Ir": {
        "mp_id": "mp-101",
        "chgnet_static": -8.837793350219727,
        "chgnet_relaxed": -8.84253215789795,
    },
    "K": {
        "mp_id": "mp-1184905",
        "chgnet_static": -1.0803413391113281,
        "chgnet_relaxed": -1.0803413391113281,
    },
    "Kr": {
        "mp_id": "mp-612118",
        "chgnet_static": -0.023926615715026855,
        "chgnet_relaxed": -0.05584895610809326,
    },
    "La": {
        "mp_id": "mp-26",
        "chgnet_static": -4.894484996795654,
        "chgnet_relaxed": -4.894484996795654,
    },
    "Li": {
        "mp_id": "mp-1018134",
        "chgnet_static": -1.882352590560913,
        "chgnet_relaxed": -1.8823524713516235,
    },
    "Lu": {
        "mp_id": "mp-145",
        "chgnet_static": -4.490588665008545,
        "chgnet_relaxed": -4.494189739227295,
    },
    "Mg": {
        "mp_id": "mp-1056702",
        "chgnet_static": -1.5942153930664062,
        "chgnet_relaxed": -1.5942156314849854,
    },
    "Mn": {
        "mp_id": "mp-35",
        "chgnet_static": -8.964238166809082,
        "chgnet_relaxed": -9.110967636108398,
    },
    "Mo": {
        "mp_id": "mp-129",
        "chgnet_static": -10.69620418548584,
        "chgnet_relaxed": -10.69620418548584,
    },
    "N": {
        "mp_id": "mp-25",
        "chgnet_static": -8.342089653015137,
        "chgnet_relaxed": -8.344156265258789,
    },
    "Na": {
        "mp_id": "mp-1186081",
        "chgnet_static": -1.2779837846755981,
        "chgnet_relaxed": -1.2779837846755981,
    },
    "Nb": {
        "mp_id": "mp-75",
        "chgnet_static": -10.02518081665039,
        "chgnet_relaxed": -10.025711059570312,
    },
    "Nd": {
        "mp_id": "mp-123",
        "chgnet_static": -4.729693412780762,
        "chgnet_relaxed": -4.729693412780762,
    },
    "Ne": {
        "mp_id": "mp-111",
        "chgnet_static": -0.022461412474513054,
        "chgnet_relaxed": -0.036446765065193176,
    },
    "Ni": {
        "mp_id": "mp-23",
        "chgnet_static": -5.742119789123535,
        "chgnet_relaxed": -5.746542930603027,
    },
    "Np": {
        "mp_id": "mp-11534",
        "chgnet_static": -12.841249465942383,
        "chgnet_relaxed": -12.871532440185547,
    },
    "O": {
        "mp_id": "mp-723285",
        "chgnet_static": -4.890257835388184,
        "chgnet_relaxed": -4.901292324066162,
    },
    "Os": {
        "mp_id": "mp-49",
        "chgnet_static": -11.168569564819336,
        "chgnet_relaxed": -11.172502517700195,
    },
    "P": {
        "mp_id": "mp-1198724",
        "chgnet_static": -5.383859157562256,
        "chgnet_relaxed": -5.38781213760376,
    },
    "Pa": {
        "mp_id": "mp-10740",
        "chgnet_static": -9.436363220214844,
        "chgnet_relaxed": -9.436362266540527,
    },
    "Pb": {
        "mp_id": "mp-20483",
        "chgnet_static": -3.6611990928649902,
        "chgnet_relaxed": -3.6631569862365723,
    },
    "Pd": {
        "mp_id": "mp-2",
        "chgnet_static": -5.1975908279418945,
        "chgnet_relaxed": -5.202797889709473,
    },
    "Pm": {
        "mp_id": "mp-867200",
        "chgnet_static": -4.729453086853027,
        "chgnet_relaxed": -4.729453086853027,
    },
    "Pr": {
        "mp_id": "mp-38",
        "chgnet_static": -4.7471394538879395,
        "chgnet_relaxed": -4.7471394538879395,
    },
    "Pt": {
        "mp_id": "mp-126",
        "chgnet_static": -6.085573673248291,
        "chgnet_relaxed": -6.092453479766846,
    },
    "Pu": {
        "mp_id": "mp-571261",
        "chgnet_static": -14.252351760864258,
        "chgnet_relaxed": -14.265556335449219,
    },
    "Rb": {
        "mp_id": "mp-1179656",
        "chgnet_static": -0.9263635277748108,
        "chgnet_relaxed": -0.93392014503479,
    },
    "Re": {
        "mp_id": "mp-8",
        "chgnet_static": -12.451730728149414,
        "chgnet_relaxed": -12.456867218017578,
    },
    "Rh": {
        "mp_id": "mp-74",
        "chgnet_static": -7.2886643409729,
        "chgnet_relaxed": -7.300443172454834,
    },
    "Ru": {
        "mp_id": "mp-33",
        "chgnet_static": -9.130279541015625,
        "chgnet_relaxed": -9.140661239624023,
    },
    "S": {
        "mp_id": "mp-96",
        "chgnet_static": -4.138021469116211,
        "chgnet_relaxed": -4.141663074493408,
    },
    "Sb": {
        "mp_id": "mp-104",
        "chgnet_static": -4.060749053955078,
        "chgnet_relaxed": -4.061436653137207,
    },
    "Sc": {
        "mp_id": "mp-67",
        "chgnet_static": -6.274556636810303,
        "chgnet_relaxed": -6.274685382843018,
    },
    "Se": {
        "mp_id": "mp-570481",
        "chgnet_static": -3.51871395111084,
        "chgnet_relaxed": -3.52719783782959,
    },
    "Si": {
        "mp_id": "mp-149",
        "chgnet_static": -5.313340663909912,
        "chgnet_relaxed": -5.3134846687316895,
    },
    "Sm": {
        "mp_id": "mp-69",
        "chgnet_static": -4.687231540679932,
        "chgnet_relaxed": -4.687626838684082,
    },
    "Sn": {
        "mp_id": "mp-117",
        "chgnet_static": -3.816476583480835,
        "chgnet_relaxed": -3.823385238647461,
    },
    "Sr": {
        "mp_id": "mp-1187073",
        "chgnet_static": -1.6626938581466675,
        "chgnet_relaxed": -1.6642879247665405,
    },
    "Ta": {
        "mp_id": "mp-50",
        "chgnet_static": -11.84970760345459,
        "chgnet_relaxed": -11.851482391357422,
    },
    "Tb": {
        "mp_id": "mp-11698",
        "chgnet_static": -4.5957136154174805,
        "chgnet_relaxed": -4.596389293670654,
    },
    "Tc": {
        "mp_id": "mp-113",
        "chgnet_static": -10.33245849609375,
        "chgnet_relaxed": -10.339179039001465,
    },
    "Te": {
        "mp_id": "mp-19",
        "chgnet_static": -3.1557819843292236,
        "chgnet_relaxed": -3.1622684001922607,
    },
    "Th": {
        "mp_id": "mp-37",
        "chgnet_static": -7.407485485076904,
        "chgnet_relaxed": -7.407485485076904,
    },
    "Ti": {
        "mp_id": "mp-72",
        "chgnet_static": -7.784756183624268,
        "chgnet_relaxed": -7.785751819610596,
    },
    "Tl": {
        "mp_id": "mp-972351",
        "chgnet_static": -2.3402538299560547,
        "chgnet_relaxed": -2.345637798309326,
    },
    "Tm": {
        "mp_id": "mp-1018122",
        "chgnet_static": -4.464661121368408,
        "chgnet_relaxed": -4.465600490570068,
    },
    "U": {
        "mp_id": "mp-44",
        "chgnet_static": -11.152817726135254,
        "chgnet_relaxed": -11.168636322021484,
    },
    "V": {
        "mp_id": "mp-146",
        "chgnet_static": -9.075878143310547,
        "chgnet_relaxed": -9.077430725097656,
    },
    "W": {
        "mp_id": "mp-91",
        "chgnet_static": -12.75978946685791,
        "chgnet_relaxed": -12.762614250183105,
    },
    "Xe": {
        "mp_id": "mp-972256",
        "chgnet_static": -0.03307008743286133,
        "chgnet_relaxed": -0.03307008743286133,
    },
    "Y": {
        "mp_id": "mp-1187739",
        "chgnet_static": -6.440634250640869,
        "chgnet_relaxed": -6.4434380531311035,
    },
    "Yb": {
        "mp_id": "mp-972364",
        "chgnet_static": -1.3258570432662964,
        "chgnet_relaxed": -1.5238240957260132,
    },
    "Zn": {
        "mp_id": "mp-79",
        "chgnet_static": -1.2498449087142944,
        "chgnet_relaxed": -1.2626861333847046,
    },
    "Zr": {
        "mp_id": "mp-131",
        "chgnet_static": -8.509099960327148,
        "chgnet_relaxed": -8.509099960327148,
    },
}

class CHGNetStaticAdapter(BaseCalculator):
    name = "CHGNetStatic"
    objectives = [
        "CHGNet Static Energy [eV/atom]",
        "CHGNet Static RMS Force [eV/A]",
        "CHGNet Static Von Mises Stress [GPa]",
        "CHGNet Static Total Magmom [mu_B]",
        "CHGNet Static Volume [A^3/atom]",
    ]

    @staticmethod
    def run_per_structure(path: str, name: str, requested: Sequence[str]) -> Optional[Dict[str, Any]]:
        """Compute CHGNet static properties for a single structure file.

        Implemented as staticmethod to allow multiprocessing with process_map.
        """
        try:
            fp = os.path.join(path, name)
            s = Structure.from_file(fp)

            # Lazy load model per process (cached globally to avoid reloading in same process)
            global _CHGNET_STATIC_MODEL
            try:
                _CHGNET_STATIC_MODEL
            except NameError:
                logger = logging.getLogger("chgnet")
                prev_level = logger.level
                logger.setLevel(logging.ERROR)
                try:
                    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                        from chgnet.model import CHGNet
                        _CHGNET_STATIC_MODEL = CHGNet.load()
                finally:
                    logger.setLevel(prev_level)
            chgnet = _CHGNET_STATIC_MODEL

            num_atoms = len(s)
            comp = s.composition.get_el_amt_dict()
            energy_reference = 0.0
            for el, amt in comp.items():
                if el not in energy_reference_data:
                    raise KeyError(f"Missing CHGNet elemental reference for element '{el}'")
                energy_reference += energy_reference_data[el]["chgnet_relaxed"] * float(amt)

            static = chgnet.predict_structure(s)
            props: Dict[str, Any] = {}

            if "CHGNet Static Energy [eV/atom]" in requested and static.get("e") is not None:
                static_energy = float(static.get("e"))
                formation_energy_static = (static_energy * num_atoms - energy_reference) / num_atoms
                props["CHGNet Static Energy [eV/atom]"] = float(Decimal(formation_energy_static).quantize(Decimal("0.0001"), rounding=ROUND_DOWN))

            if "CHGNet Static RMS Force [eV/A]" in requested and static.get("f") is not None:
                f = np.array(static.get("f"))
                rms_force = BaseCalculator.vector_to_scalar('rms', f) or 0.0
                props["CHGNet Static RMS Force [eV/A]"] = float(Decimal(rms_force).quantize(Decimal("0.0001"), rounding=ROUND_DOWN))

            if "CHGNet Static Von Mises Stress [GPa]" in requested and static.get("s") is not None:
                vm = BaseCalculator.vector_to_scalar('vonMises', static.get("s"))
                if vm is not None:
                    props["CHGNet Static Von Mises Stress [GPa]"] = float(Decimal(vm).quantize(Decimal("0.0001"), rounding=ROUND_DOWN))

            if "CHGNet Static Total Magmom [mu_B]" in requested and static.get("m") is not None:
                m = static.get("m")
                total_m = BaseCalculator.vector_to_scalar('sum', m)
                props["CHGNet Static Total Magmom [mu_B]"] = float(Decimal(total_m).quantize(Decimal("0.0001"), rounding=ROUND_DOWN))

            if "CHGNet Static Volume [A^3/atom]" in requested:
                v = s.volume / num_atoms
                props["CHGNet Static Volume [A^3/atom]"] = float(Decimal(v).quantize(Decimal("0.0001"), rounding=ROUND_DOWN))

            return {
                "name": name,
                "properties": props,
            }
        except Exception as e:
            # Return stub result with name so caller can still match, and print error for debugging
            try:
                print(f"[CHGNetStaticAdapter] Error processing {name}: {e}")
                traceback.print_exc()
            except Exception:
                pass
            return {"name": name, "properties": {}}

    def run_on_directory(self, path: str, mode: str = "serial", max_workers: Optional[int] = None, objectives: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
        requested = list(self.objectives if objectives is None else objectives)
        unknown = [o for o in requested if o not in self.objectives]
        if unknown:
            raise ValueError(f"Unknown CHGNet Static objectives requested: {unknown}. Available: {self.objectives}")

        file_names: List[str] = []
        for f in os.listdir(path):
            fp = os.path.join(path, f)
            if os.path.isfile(fp):
                file_names.append(f)

        if mode == 'parallel':
            if max_workers is None:
                max_workers = os.cpu_count()
            results = process_map(self.run_per_structure, repeat(path), file_names, repeat(requested), max_workers=max_workers, chunksize=1)
            out: List[Dict[str, Any]] = []
            for result in results:
                if result is not None:
                    out.append(result)
        elif mode == 'serial':
            out: List[Dict[str, Any]] = []
            for name in tqdm(file_names, total=len(file_names), disable=False):
                result = self.run_per_structure(path, name, requested)
                if result is not None:
                    out.append(result)
        
        return out

class CHGNetRelaxAdapter(BaseCalculator):
    name = "CHGNetRelax"
    objectives = [
        "CHGNet Relaxed Formation Energy [eV/atom]",
        "CHGNet Relaxed RMS Force [eV/A]",
        "CHGNet Relaxed Von Mises Stress [GPa]",
        "CHGNet Relaxed Total Magmom [mu_B]",
        "CHGNet Relaxed Volume [A^3/atom]",
    ]

    @staticmethod
    def run_per_structure(path: str, name: str, requested: Sequence[str]) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
        """Compute CHGNet relaxation for a single structure returning (result_dict, final_structure)."""
        try:
            fp = os.path.join(path, name)
            s = Structure.from_file(fp)

            global _CHGNET_RELAXER
            try:
                _CHGNET_RELAXER
            except NameError:
                logger = logging.getLogger("chgnet")
                prev_level = logger.level
                logger.setLevel(logging.ERROR)
                try:
                    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                        from chgnet.model import StructOptimizer
                        _CHGNET_RELAXER = StructOptimizer()
                finally:
                    logger.setLevel(prev_level)
            relaxer = _CHGNET_RELAXER

            num_atoms = len(s)
            comp = s.composition.get_el_amt_dict()
            energy_reference = 0.0
            for el, amt in comp.items():
                if el not in energy_reference_data:
                    raise KeyError(f"Missing CHGNet elemental reference for element '{el}'")
                energy_reference += energy_reference_data[el]["chgnet_relaxed"] * float(amt)

            relax = relaxer.relax(s, verbose=False)
            props: Dict[str, Any] = {}

            final_structure = None
            if isinstance(relax, dict):
                final_structure = relax.get("final_structure")

            # Formation energy
            if "CHGNet Relaxed Formation Energy [eV/atom]" in requested and relax["trajectory"].energies is not None:
                relaxed_energy = float(relax["trajectory"].energies[-1]) / num_atoms
                formation_energy_relaxed = (relaxed_energy * num_atoms - energy_reference) / num_atoms
                props["CHGNet Relaxed Formation Energy [eV/atom]"] = float(Decimal(formation_energy_relaxed).quantize(Decimal("0.0001"), rounding=ROUND_DOWN))

            # RMS force
            if "CHGNet Relaxed RMS Force [eV/A]" in requested and relax["trajectory"].forces is not None:
                f = np.array(relax["trajectory"].forces[-1])
                rms_force = BaseCalculator.vector_to_scalar('rms', f) or 0.0
                props["CHGNet Relaxed RMS Force [eV/A]"] = float(Decimal(rms_force).quantize(Decimal("0.0001"), rounding=ROUND_DOWN))

            # Von Mises stress
            if "CHGNet Relaxed Von Mises Stress [GPa]" in requested and relax["trajectory"].stresses is not None:
                vm = BaseCalculator.vector_to_scalar('vonMises', relax["trajectory"].stresses[-1])
                if vm is not None:
                    props["CHGNet Relaxed Von Mises Stress [GPa]"] = float(Decimal(vm).quantize(Decimal("0.0001"), rounding=ROUND_DOWN))

            # Total magmom
            if "CHGNet Relaxed Total Magmom [mu_B]" in requested and relax["trajectory"].magmoms is not None:
                m = relax["trajectory"].magmoms[-1]
                total_m = BaseCalculator.vector_to_scalar('sum', m)
                props["CHGNet Relaxed Total Magmom [mu_B]"] = float(Decimal(total_m).quantize(Decimal("0.0001"), rounding=ROUND_DOWN))

            # Volume from final_structure if available
            poscar_str: Optional[str] = None
            if final_structure is not None:
                if "CHGNet Relaxed Volume [A^3/atom]" in requested:
                    v = final_structure.volume / num_atoms
                    props["CHGNet Relaxed Volume [A^3/atom]"] = float(Decimal(v).quantize(Decimal("0.0001"), rounding=ROUND_DOWN))
                # Serialize final structure to POSCAR string to avoid pickling heavy objects across processes
                try:
                    poscar_str = Poscar(final_structure).get_string()
                except Exception:
                    poscar_str = None

            return ({
                "name": name,
                "properties": props,
            }, poscar_str)
        except Exception as e:
            # Return stub result with name so caller can still match, and print error for debugging
            try:
                print(f"[CHGNetRelaxAdapter] Error processing {name}: {e}")
                traceback.print_exc()
            except Exception:
                pass
            return ({"name": name, "properties": {}}, None)

    def run_on_directory(self, path: str, mode: str = "serial", max_workers: Optional[int] = None, objectives: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
        requested = list(self.objectives if objectives is None else objectives)
        unknown = [o for o in requested if o not in self.objectives]
        if unknown:
            raise ValueError(f"Unknown CHGNet Relax objectives requested: {unknown}. Available: {self.objectives}")

        file_names: List[str] = []
        for f in os.listdir(path):
            fp = os.path.join(path, f)
            if os.path.isfile(fp):
                file_names.append(f)

        final_structures: List[tuple] = []

        if mode == 'parallel':
            if max_workers is None:
                max_workers = os.cpu_count()
            results = process_map(self.run_per_structure, repeat(path), file_names, repeat(requested), max_workers=max_workers, chunksize=1)
            out: List[Dict[str, Any]] = []
            for result, poscar_str in results:
                if result is not None:
                    out.append(result)
                if result is not None and poscar_str is not None:
                    final_structures.append((result['name'], poscar_str))
        elif mode == 'serial':
            out: List[Dict[str, Any]] = []
            for name in tqdm(file_names, total=len(file_names), disable=False):
                result, poscar_str = self.run_per_structure(path, name, requested)
                if result is not None:
                    out.append(result)
                if poscar_str is not None:
                    final_structures.append((name, poscar_str))

        # Write relaxed structures if directory does not already exist
        save_dir = f"{path}_{self.name}"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
            saved = 0
            for fname, poscar_str in final_structures:
                try:
                    with open(os.path.join(save_dir, fname), 'w') as f:
                        f.write(poscar_str)
                    saved += 1
                except Exception:
                    continue
            print(f"Created relaxed structure directory: {save_dir} (saved {saved} structures)")
        else:
            print(f"Relaxed structure directory already exists: {save_dir}. Skipping write.")

        return out
