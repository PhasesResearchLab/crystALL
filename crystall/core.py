# Base Python imports
import sys
import os
import io
import csv
import json
import time
import importlib.util
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import warnings

# Common tool libraries
from tqdm import tqdm
from natsort import natsorted
from colorama import Fore, Style
from distinctipy import get_colors
from plotly.offline import plot, iplot
from urllib.error import HTTPError
from decimal import Decimal, ROUND_DOWN
from itertools import combinations
from typing import Any, Sequence, List, Dict, Union, Literal, Optional

# ML / Data
from pymongo.database import Database
from scipy.spatial import ConvexHull
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score, pairwise_distances
from sklearn.cluster import KMeans, DBSCAN
from pymatgen.core import Composition
from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram

# Internal modules
from crystall.calculators import ObjectiveRegistry, load_adapters
from crystall.prototypes import run_local, run_MPDD, run_MPDD_devs, run_optimade

# Disable undesired warnings from torch and pymatgen ******************************************
warnings.filterwarnings(
    "ignore", message="(?:.|\n)*datapipes.*dataloader2.*deprecated(?:.|\n)*"
)
warnings.filterwarnings("ignore", message="(?:.|\n)*electronegativity(?:.|\n)*")


# Default descriptor for pySIPFENN ************************************************************
_DEFAULT_PYSIPFENN_DESCRIPTOR = "KS2022"


# Top-Level Functions *************************************************************************
def _pysipfenn_defaults(descriptor: str) -> tuple[str, List[str], str]:
    spec = importlib.util.find_spec("pysipfenn")
    if spec is None or not spec.submodule_search_locations:
        raise FileNotFoundError(
            "pysipfenn package not found; cannot locate models metadata."
        )

    package_root = list(spec.submodule_search_locations)[0]
    metadata_path = os.path.join(package_root, "modelsSIPFENN", "models.json")
    if not os.path.isfile(metadata_path):
        metadata_path = os.path.join(package_root, "models.json")
    if not os.path.isfile(metadata_path):
        raise FileNotFoundError("pysipfenn models.json not found in package data.")

    with open(metadata_path, "r", encoding="utf-8") as handle:
        models = json.load(handle)

    compatible_models = [
        name
        for name, meta in models.items()
        if isinstance(meta, dict) and meta.get("descriptor") == descriptor
    ]
    if not compatible_models:
        raise ValueError(f"No pySIPFENN models found for descriptor '{descriptor}'.")

    default_model = compatible_models[0]
    return package_root, compatible_models, default_model


# Convenience High-Level API Classes **********************************************************
class CrystALLSearch:
    """
    A self-contained environment for (1) obtaining POSCAR files of ``crystALL`` prototypes from either MPDD or a local
    dataset of POSCAR files, (2) featurizing the prototypes with KS2022 or future ``pySIPFENN`` descriptor calculators,
    (3) getting property predictions using ``pySIPFENN`` or external calculators via adapters, (4) clustering the prototypes in ``KS2022``
    feature space, and (5) selecting prototypes based on desired property and cluster affiliation.
    """

    def __init__(
        self, target_formula: Union[str, Composition], calculator: Optional[Any] = None
    ) -> None:
        """
        Initialize the CrystALLSearch class.

        Args:
            target_formula: The target formula as a string or pymatgen Composition object.
            calculator: A pySIPFENN Calculator object. If not provided, a default Calculator is used.

        Raises:
            TypeError: If the target formula is not a string or pymatgen Composition object.
        """

        assert isinstance(target_formula, (str, Composition)), (
            "Target formula must be a string or a pymatgen Composition object"
        )

        # Defined at initialization
        if isinstance(target_formula, str):
            self.target_formula = Composition(target_formula)
        elif isinstance(target_formula, Composition):
            self.target_formula = target_formula
        else:
            raise TypeError(
                "Target formula must be a string or a pymatgen Composition object"
            )

        self.Calculator = calculator if calculator is not None else None

        self.MPDD = None
        self.prototype_path: Optional[str] = None
        self._optimade_provider_registry: Dict[str, Dict[str, str]] = {}
        self._optimade_active_base_urls: List[str] = []

        # Internal parameters to use during the processing steps
        self.available_calculators: Dict[str, List[str]] = {}
        self.predicted_objectives: List[str] = []
        self._objective_registry: Dict[
            str, Any
        ] = {}  # code -> ObjectiveEntry, built after adapters load
        self.ks2022: np.ndarray = None
        self.cluster_affiliation: Dict[int, Dict[str, float]] = {}
        self.prototypes: List[Dict[str, Union[str, float]]] = []
        self.top_candidates: List[Dict[str, Union[str, float]]] = []

        print(
            f"Initialized CrystALLSearch for target formula: {Fore.YELLOW}{Style.BRIGHT}{self.target_formula.formula}{Style.RESET_ALL}"
        )

        print("\nChecking for available calculators on the computer..")

        self.setup_pysipfenn()

        # Supported calculators include builtin pySIPFENN and external adapters via entry points
        self.supported_calculators: Dict[str, Any] = {"pySIPFENN": None}
        self.supported_calculators.update(load_adapters())

        # Pre-instantiate all adapter entries once so every subsequent step (registry build,
        # readiness check, objective extraction) works with live instances — never raw classes.
        for _name in list(self.supported_calculators):
            if _name == "pySIPFENN":
                continue
            _entry = self.supported_calculators[_name]
            if isinstance(_entry, type):
                try:
                    self.supported_calculators[_name] = _entry()
                except Exception as e:
                    warnings.warn(f"Could not instantiate adapter '{_name}': {e}")
                    self.supported_calculators.pop(_name)

        # Build the objective registry from all instantiated adapters.
        self._objective_registry = ObjectiveRegistry.from_adapters(
            {
                name: adp
                for name, adp in self.supported_calculators.items()
                if name != "pySIPFENN"
            }
        )

        # Check environment readiness for each adapter (without auto-installing).
        # All entries are already instances at this point — no isinstance guard needed.
        for calc_name, adapter in self.supported_calculators.items():
            if calc_name == "pySIPFENN":
                continue

            if hasattr(adapter, "is_env_ready") and adapter.is_env_ready():
                # Store objective codes for available_calculators so that
                # define_objective, predicted_objectives, and prototype keys
                # all share the same code-based namespace.
                self.available_calculators[calc_name] = adapter.objectives.codes()

        # Print the status of calculators (found if present in available_calculators)
        for calc_name in self.supported_calculators.keys():
            if calc_name in self.available_calculators:
                try:
                    print(f"{Fore.GREEN}✔ {calc_name} ready.{Style.RESET_ALL}")
                except UnicodeEncodeError:
                    print(f"{Fore.GREEN}v {calc_name} ready.{Style.RESET_ALL}")
            else:
                if calc_name == "pySIPFENN":
                    continue
                try:
                    print(
                        f"{Style.DIM}✘ {calc_name} (run setup_calculator('{calc_name}')){Style.RESET_ALL}"
                    )
                except UnicodeEncodeError:
                    print(
                        f"{Style.DIM}x {calc_name} (run setup_calculator('{calc_name}')){Style.RESET_ALL}"
                    )

        # Update objectives with calculators found
        if self.Calculator is not None:
            self._update_objectives(self.Calculator)

    def _update_objectives(self, calculator: Any, verbose: bool = True) -> None:
        """
        Update the available objectives with the calculators found.

        Args:
            calculator: A pySIPFENN Calculator object.
            verbose: If True, prints the available objectives. Defaults to True.

        Raises:
            AssertionError: If the calculator is not a pySIPFENN Calculator object.
        """

        from pysipfenn import Calculator

        assert isinstance(calculator, Calculator), (
            "Calculator must be a Calculator object"
        )

        # available_calculators is already populated in __init__ by the
        # readiness loop (only ready adapters are included).

        if len(self.available_calculators) >= 1 and verbose:
            print("\nObjectives available for calculation:")
            for calc_name, obj_list in self.available_calculators.items():
                if not obj_list:
                    continue
                try:
                    colored_calc = f"{calc_name}"
                    colored_objs = ", ".join(
                        (
                            (
                                f"{Fore.GREEN}{Style.BRIGHT}"
                                f"{code}: {self._objective_registry[code].description}"
                                f"{Style.RESET_ALL}"
                            )
                            if code in self._objective_registry
                            and self._objective_registry[code].description
                            else f"{Fore.GREEN}{Style.BRIGHT}{code}{Style.RESET_ALL}"
                        )
                        for code in sorted(obj_list)
                    )
                    print(f"- {colored_calc}: {colored_objs}")
                except UnicodeEncodeError:
                    # Fallback without colors if terminal cannot render them
                    labels = [
                        (
                            f"{code}: {self._objective_registry[code].description}"
                            if code in self._objective_registry
                            and self._objective_registry[code].description
                            else code
                        )
                        for code in sorted(obj_list)
                    ]
                    print(f"- {calc_name}: {', '.join(labels)}")

    def _resolve_to_code(self, objective: str) -> str:
        """Normalise an objective identifier to its stable code.

        Accepts a bare code, a full ``'CODE: description'`` key, or a pySIPFENN
        model name. Raises ``ValueError`` with a simple hint if the code cannot
        be matched against known objective sets.
        """
        assert isinstance(objective, str) and len(objective.strip()) > 0, (
            "Objective must be a non-empty string"
        )

        # Objectives are always bare codes; accept a legacy "CODE: description"
        # key for backward compatibility by stripping anything after the colon.
        code = objective.split(":")[0].strip()
        if code in self._objective_registry:
            return code
        if code in self.predicted_objectives:
            return code
        flat_available = sorted(
            {c for codes in self.available_calculators.values() for c in codes}
        )
        if code in flat_available:
            return code
        known = list(self._objective_registry.codes()) + self.predicted_objectives
        close = [c for c in known if code[:4] and c.startswith(code[:4])]
        hint = f" Did you mean one of: {close}?" if close else ""
        raise ValueError(
            f"Objective code '{code}' not found in available objectives.{hint}"
        )

    def _check_objectives(self) -> None:
        """
        Check available objectives with predictions.

        Raises:
            AssertionError: If no objectives are available.

        Returns:
            List of common objectives between available objectives and predicted objectives.
        """
        # Build a list of all available objectives across calculators
        flat_available = []
        for lst in self.available_calculators.values():
            flat_available.extend(lst)

        assert isinstance(flat_available, list) and len(flat_available) >= 1, (
            "At least one objective is required for classification. Run `download_pysipfenn` first."
        )

        if any(value not in self.predicted_objectives for value in flat_available):
            print(
                "\nSome objectives do not have predictions calculated and will not be considered for classification."
            )
            for obj in flat_available:
                entry = self._objective_registry.get(obj)
                description = entry.description if entry else ""
                label = f"{obj}: {description}" if description else obj
                if obj not in self.predicted_objectives:
                    try:
                        print(f"{Style.DIM}✘ {Style.BRIGHT}{label}{Style.RESET_ALL}")
                    except UnicodeEncodeError:
                        print(f"{Style.DIM}x {Style.BRIGHT}{label}{Style.RESET_ALL}")
                else:
                    try:
                        print(f"{Fore.GREEN}✔ {Style.BRIGHT}{label}{Style.RESET_ALL}")
                    except UnicodeEncodeError:
                        print(f"{Fore.GREEN}v {Style.BRIGHT}{label}{Style.RESET_ALL}")

    def download_pysipfenn(
        self, mode: Literal["serial", "parallel"] = "parallel", retries=3, delay=5
    ) -> None:
        """Download the default models required for pySIPFENN operation.

        Args:
            mode: Mode to download models: ``'serial'`` or ``'parallel'``. Defaults to 'parallel'.
            retries: Number of retry attempts in case of failure.
            delay: Delay between retry attempts in seconds.
        """

        package_root, compatible_models, default_model = _pysipfenn_defaults(
            _DEFAULT_PYSIPFENN_DESCRIPTOR
        )
        print(f"Downloading {default_model} model for pysipfenn..")
        if self.Calculator is None:
            from pysipfenn import Calculator

            stdout, sys.stdout = sys.stdout, io.StringIO()
            self.Calculator = Calculator(autoLoad=False, verbose=False)
            sys.stdout = stdout
        for attempt in range(retries):
            try:
                self.Calculator.downloadModels(default_model)
                local_models = [
                    model
                    for model in compatible_models
                    if os.path.isfile(
                        os.path.join(package_root, "modelsSIPFENN", f"{model}.onnx")
                    )
                ]
                self.available_calculators["pySIPFENN"] = local_models
                break  # Exit the loop if successful
            except HTTPError as e:
                print(f"HTTPError: {e.code} - {e.reason}")
                if attempt < retries - 1:
                    print(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    raise
        print("Done! Models are ready to use.")
        self._update_objectives(self.Calculator, False)

    def setup_pysipfenn(self) -> None:
        """Set up builtin pySIPFENN models without eager imports."""
        package_root, models, compatible_models, default_model = _pysipfenn_defaults(
            _DEFAULT_PYSIPFENN_DESCRIPTOR
        )
        has_default = os.path.isfile(
            os.path.join(package_root, "modelsSIPFENN", f"{default_model}.onnx")
        )

        if not has_default:
            if sys.stdin.isatty():
                _response = input(
                    "pySIPFENN models not found. Would you like to automatically run "
                    "`download_pysipfenn`? (Y/n): "
                )
                _proceed = _response.strip().lower() in ("y", "")
            else:
                print(
                    "pySIPFENN models not found — running `download_pysipfenn` automatically "
                    "(non-interactive session)."
                )
                _proceed = True
            if _proceed:
                self.download_pysipfenn()
            return

        if self.Calculator is None:
            from pysipfenn import Calculator

            stdout, sys.stdout = sys.stdout, io.StringIO()
            self.Calculator = Calculator(autoLoad=False, verbose=False)
            sys.stdout = stdout

        local_models = [
            model
            for model in compatible_models
            if os.path.isfile(
                os.path.join(package_root, "modelsSIPFENN", f"{model}.onnx")
            )
        ]
        self.available_calculators["pySIPFENN"] = local_models
        self._update_objectives(self.Calculator, False)

    def setup_calculator(self, calculator_name: str) -> None:
        """Stub to setup an external calculator adapter.

        For micro-env adapters, this calls ``adapter.setup_env()`` to create
        the isolated venv and install dependencies. On success, the adapter
        is registered in ``available_calculators``.

        Args:
            calculator_name: Name of the calculator (adapter key).
        """
        adapter = getattr(self, "supported_calculators", {}).get(calculator_name)
        if adapter is None:
            warnings.warn(
                f"Calculator '{calculator_name}' not found in supported calculators."
            )
            return

        # Micro-env adapters: create environment
        if hasattr(adapter, "setup_env"):
            try:
                adapter.setup_env()
            except Exception as e:
                warnings.warn(
                    f"Failed to set up micro-environment for '{calculator_name}': {e}"
                )
                return

        objectives = list(adapter.objectives) if hasattr(adapter, "objectives") else []
        if calculator_name not in self.available_calculators:
            self.available_calculators[calculator_name] = list(objectives)
            print(f"{Fore.GREEN}✔ {calculator_name} ready.{Style.RESET_ALL}")

    def setup_MPDD(
        self, MPDD: Database = None, collections: List[str] = ["curated"]
    ) -> None:
        """
        Establish a connection to the MPDD database and set the collections to search in.

        Args:
            MPDD: A pymongo Database object to connect to. If not provided, uses the existing connection.
            collections: List of collection names to search in. Defaults to ['curated'].

        Raises:
            AssertionError: If MPDD is not a pymongo Database object, if no collections are found in the database, if any specified collection is not in the database, or if no collections are specified.
        """

        self.MPDD = MPDD if MPDD is not None else self.MPDD
        assert isinstance(self.MPDD, Database), "MPDD must be a pymongo Database object"
        availableCollections = self.MPDD.list_collection_names()
        assert len(availableCollections) > 0, (
            "No collections found in the database or the database connection has not been established properly"
        )
        assert set(collections).issubset(availableCollections), (
            "At least one of the specified collections is not available in the database"
        )
        filtered_collections = [
            col
            for col in availableCollections
            if "test" not in col.lower() and "abnormal" not in col.lower()
        ]
        print(
            f"\nConnected to database {Fore.MAGENTA}{Style.BRIGHT}{self.MPDD.name}{Style.RESET_ALL} with collections: {Fore.MAGENTA}{Style.BRIGHT}{', '.join(filtered_collections)}{Style.RESET_ALL}"
        )
        assert len(collections) > 0, "At least one collection must be specified"
        self.collections = collections
        print(
            f"Using collections: {Fore.GREEN}{Style.BRIGHT}{', '.join(collections)}{Style.RESET_ALL}"
        )

    def setup_OPTIMADE(
        self,
        providers: List[str] = None,
        base_urls: List[str] = None,
        exclude_providers: List[str] = None,
        check_reachable: bool = False,
    ) -> List[Dict[str, Union[str, bool, None]]]:
        """
        Configure OPTIMADE providers for use in generate_optimade().

        On first call (or any subsequent call), fetches the full provider registry
        via optimade.utils.get_providers() and builds an internal registry. The
        selected providers are stored as self._optimade_active_base_urls and passed
        directly to OptimadeClient(base_urls=...) in run_optimade, bypassing the
        expensive per-provider HTTP cascade that get_all_databases() performs at
        query time.

        Args:
            providers: List of provider IDs/aliases to activate (e.g. ['mp', 'cod']).
                If None and base_urls is also None, defaults to ['mp'] (Materials
                Project) — the most widely-used OPTIMADE provider and the natural
                analogue of crystALL's existing MPDD integration.
            base_urls: Explicit provider base URLs to use directly (advanced use).
                If provided, bypasses registry lookup for those entries. These are
                appended to any resolved provider URLs from `providers`.
            exclude_providers: Provider IDs to explicitly deactivate, regardless of
                what `providers` specifies.
            check_reachable: If True, performs a lightweight GET /v1/info ping on
                each provider base_url and sets the `reachable` flag accordingly.
                Opt-in because it requires one HTTP call per provider.

        Returns:
            List of dicts, one per known provider in the registry, each containing:
                id          : provider identifier string (e.g. 'mp')
                name        : human-readable provider name
                base_url    : provider index base URL
                description : short description from the registry
                active      : True if currently selected for querying
                reachable   : True/False if check_reachable=True, else None
        """

        try:
            from crystall.optimade import list_optimade_providers
        except ImportError as e:
            raise ImportError(
                "setup_OPTIMADE requires the 'optimade[http_client]' package. "
                "Install it with: pip install 'optimade[http_client]'"
            ) from e

        try:
            provider_status = list_optimade_providers(check_reachable=check_reachable)
        except Exception as e:
            if self._optimade_provider_registry:
                provider_status = [
                    {
                        "id": provider_id,
                        "name": provider.get("name", provider_id),
                        "description": provider.get("description", ""),
                        "base_url": provider.get("base_url", ""),
                        "reachable": None,
                    }
                    for provider_id, provider in self._optimade_provider_registry.items()
                ]
                print(
                    f"{Style.DIM}[crystALL] Could not refresh OPTIMADE provider registry; using cached registry. ({e}){Style.RESET_ALL}"
                )
            else:
                raise RuntimeError(
                    "Failed to fetch the OPTIMADE provider registry and no cached registry is available."
                ) from e

        registry: Dict[str, Dict[str, str]] = {
            provider["id"]: {
                "name": provider["name"],
                "description": provider["description"],
                "base_url": provider["base_url"],
            }
            for provider in provider_status
        }

        if not registry:
            raise RuntimeError(
                "OPTIMADE provider registry is empty or could not be parsed."
            )

        self._optimade_provider_registry = registry

        exclude_ids = set(exclude_providers or [])
        if providers is None and base_urls is None:
            active_ids = {"mp"}
        elif providers is None:
            active_ids = set()
        else:
            active_ids = set(providers)

        unknown_ids = sorted(active_ids.difference(registry.keys()))
        if unknown_ids:
            known_ids = ", ".join(sorted(registry.keys()))
            raise ValueError(
                f"Unknown OPTIMADE provider ID(s): {', '.join(unknown_ids)}. Known provider IDs: {known_ids}"
            )

        active_ids.difference_update(exclude_ids)

        resolved_base_urls: List[str] = [
            registry[provider_id]["base_url"] for provider_id in sorted(active_ids)
        ]
        custom_base_urls = []
        for url in base_urls or []:
            normalized = str(url).rstrip("/")
            if (
                normalized
                and normalized not in resolved_base_urls
                and normalized not in custom_base_urls
            ):
                custom_base_urls.append(normalized)
        self._optimade_active_base_urls = resolved_base_urls + custom_base_urls

        reachability = {}
        if check_reachable:
            from urllib.request import Request, urlopen

            for provider_id, provider in registry.items():
                info_url = provider["base_url"].rstrip("/") + "/v1/info"
                try:
                    request = Request(info_url, headers={"User-Agent": "crystALL"})
                    with urlopen(request, timeout=3):
                        reachability[provider_id] = True
                except Exception:
                    reachability[provider_id] = False

        provider_status = []
        for provider_id in sorted(registry.keys()):
            provider = registry[provider_id]
            provider_status.append(
                {
                    "id": provider_id,
                    "name": provider["name"],
                    "base_url": provider["base_url"],
                    "description": provider["description"],
                    "active": provider_id in active_ids,
                    "reachable": (
                        reachability.get(provider_id) if check_reachable else None
                    ),
                }
            )

        for url in custom_base_urls:
            provider_status.append(
                {
                    "id": None,
                    "name": "custom",
                    "base_url": url,
                    "description": "Custom provider base URL",
                    "active": True,
                    "reachable": None,
                }
            )

        print("\nOPTIMADE provider configuration:")
        header = f"{'ID':<12} {'Name':<36} {'Status':<12}"
        if check_reachable:
            header += " Reachable"
        print(header)

        for provider in provider_status:
            provider_id = provider["id"] if provider["id"] is not None else "custom"
            name = provider["name"][:36]
            status = (
                f"{Fore.GREEN}{Style.BRIGHT}✓ active{Style.RESET_ALL}"
                if provider["active"]
                else f"{Style.DIM}— inactive{Style.RESET_ALL}"
            )
            line = f"{provider_id:<12} {name:<36} {status:<12}"
            if check_reachable:
                reachable = provider["reachable"]
                if reachable is True:
                    reachable_text = f"{Fore.GREEN}{Style.BRIGHT}yes{Style.RESET_ALL}"
                elif reachable is False:
                    reachable_text = f"{Fore.RED}{Style.BRIGHT}no{Style.RESET_ALL}"
                else:
                    reachable_text = f"{Style.DIM}n/a{Style.RESET_ALL}"
                line += f" {reachable_text}"
            print(line)

        return provider_status

    def generate_MPDD(
        self,
        target_prototypes: Literal["native", "non-native"] = "non-native",
        filter_prototypes: Literal[
            "material.structID", "material.volNormStructID"
        ] = "material.structID",
        limit: int = 1000,
        devs: Dict[str, int] = None,
    ) -> None:
        """
        Generate prototype materials from the MPDD database based on the target formula.

        Args:
            target_prototypes: Specifies whether to use 'native' or 'non-native' compositions. Defaults to 'non-native'.
            filter_prototypes: Specifies the prototype filter to use. Defaults to 'material.structID'.
            limit: The maximum number of prototypes to generate. Defaults to 1000.
            devs: Dictionary with element symbols as keys and their allowed deviations as values. Defaults to None.

        Raises:
            AssertionError: If no connection to the MPDD has been established, or if no collections have been specified.
        """

        assert self.MPDD is not None, (
            "No database connection identified. Please run `setup_MPDD` first."
        )
        assert self.collections, (
            "No collections identified. Please provide a list of collections to search in."
        )
        assert devs is None or isinstance(devs, dict), (
            "Deviations must be a dictionary with element symbols as keys and integer values as deviations"
        )

        if devs is None:
            self.prototype_path = run_MPDD(
                self.target_formula,
                self.MPDD,
                self.collections,
                target_prototypes,
                filter_prototypes,
                limit,
            )
        else:
            self.prototype_path = run_MPDD_devs(
                self.target_formula,
                devs,
                self.MPDD,
                self.collections,
                target_prototypes,
                filter_prototypes,
                limit,
            )
        inputFiles = natsorted(os.listdir(self.prototype_path))
        self.prototypes = [
            {"index": i + 1, "name": file} for i, file in enumerate(inputFiles)
        ]

    def generate_local(self, prototype_path: str = None) -> None:
        """
        Generate prototypes from a local directory containing POSCAR or CIF files.

        Args:
            prototype_path: Path to the directory with prototype files. If not provided, uses the existing path.

        Raises:
            AssertionError: If no prototype directory is specified.
        """

        self.prototype_path = (
            prototype_path if prototype_path is not None else self.prototype_path
        )
        assert self.prototype_path, (
            "No prototypes folder identified. Please provide a path to the prototypes folder."
        )
        self.prototype_path = run_local(self.target_formula, self.prototype_path)
        inputFiles = natsorted(os.listdir(self.prototype_path))
        self.prototypes = [
            {"index": i, "name": file} for i, file in enumerate(inputFiles)
        ]

    def generate_optimade(
        self,
        target_prototypes: Literal["native", "non-native"] = "non-native",
        limit: int = 1000,
        parallel: bool = True,
    ) -> None:
        """
        Generate prototype materials from OPTIMADE providers based on the target formula.

        Provider selection is exclusively the responsibility of setup_OPTIMADE(). This method
        accepts no provider-related arguments. If setup_OPTIMADE() has not been called,
        it is called automatically with no arguments, activating the Materials Project
        (mp) as the default provider.

        Args:
            target_prototypes: Specifies whether to use 'native' or 'non-native' compositions.
                Defaults to 'non-native'.
            limit: The maximum number of prototypes to retrieve per provider. Defaults to 1000.
            parallel: If True, process candidate structures in parallel. Defaults to True.

        Raises:
            ImportError: If the 'optimade[http_client]' package is not installed.
        """

        assert self.target_formula is not None, (
            "No target formula identified. Please provide a formula first."
        )

        if not self._optimade_active_base_urls:
            print(
                "[crystALL] OPTIMADE providers not configured — defaulting to Materials Project (mp).\n"
                "Run CrystALLSearch.setup_OPTIMADE() to customise provider selection."
            )
            self.setup_OPTIMADE()

        self.prototype_path = run_optimade(
            target_formula=self.target_formula,
            base_urls=self._optimade_active_base_urls,
            target_prototypes=target_prototypes,
            limit=limit,
            parallel=parallel,
        )
        inputFiles = natsorted(os.listdir(self.prototype_path))
        self.prototypes = [
            {"index": i + 1, "name": file} for i, file in enumerate(inputFiles)
        ]

    def run_pysipfenn(self, mode: Literal["serial", "parallel"] = "serial") -> None:
        """
        Generate pySIPFENN predictors (features) and predict properties for the prototype materials using pySIPFENN models and update the prototypes with the predictions.

        Args:
            mode: Mode to run pySIPFENN: ``'serial'`` or ``'parallel'``. Defaults to 'serial'.

        Raises:
            AssertionError: If no prototype directory is specified, if the prototype directory path is not a string, if no pySIPFENN models are found, or if the number of prototypes does not match the number of property predictions.
        """

        assert self.prototype_path is not None, (
            "No prototypes folder identified. Please run `generate_MPDD` or `generate_local` first."
        )
        assert isinstance(self.prototype_path, str), (
            "Prototypes folder path must be a string"
        )
        if not self.available_calculators.get("pySIPFENN"):
            self.setup_pysipfenn()
        assert (
            "pySIPFENN" in self.available_calculators
            and len(self.available_calculators.get("pySIPFENN", [])) > 0
        ), "pySIPFENN models not found. Please run `download_pysipfenn` first."

        # Track which objectives (models) will be predicted by pySIPFENN
        for model in self.available_calculators.get("pySIPFENN", []):
            stdout, sys.stdout = sys.stdout, io.StringIO()
            self.Calculator.loadModels(network=model)
            sys.stdout = stdout
            if model not in self.predicted_objectives:
                self.predicted_objectives.append(model)

        self.Calculator.runFromDirectory(
            self.prototype_path,
            _DEFAULT_PYSIPFENN_DESCRIPTOR,
            mode=mode,
            max_workers=8,
        )
        resultDicts, self.ks2022 = (
            self.Calculator.get_resultDictsWithNames(),
            np.array(self.Calculator.descriptorData),
        )
        assert len(self.prototypes) == len(resultDicts), (
            "Number of prototypes does not match number of property predictions"
        )

        for result in resultDicts:
            name = result["name"]
            for prototype in self.prototypes:
                if prototype["name"] == name:
                    prototype.update(result)
                    for key in result.keys():
                        if isinstance(prototype[key], float):
                            prototype[key] = float(
                                Decimal(prototype[key]).quantize(
                                    Decimal("0.0001"), rounding=ROUND_DOWN
                                )
                            )
                    break

        print(
            f"Predicted properties for {Fore.GREEN}{Style.BRIGHT}{len(self.ks2022)}{Style.RESET_ALL} prototypes using pySIPFENN"
        )

    def get_calculator_properties(
        self,
        calculator: str,
        mode: Literal["serial", "parallel"] = "serial",
        max_workers: int | None = None,
        objectives: Sequence[str] | None = None,
        isif: int = 8,
        optimizer: str = "FIRE",
        **kwargs,
    ) -> None:
        """Run a pluggable calculator adapter over the prototype directory and merge results.

        Args:
            calculator: The adapter name as declared in entry points.
            mode: 'serial' or 'parallel' as supported by the adapter.
            objectives: Optional subset of objective names to compute/return.
            isif: ISIF relaxation setting (default 8). Ignored by non-relaxation adapters.
            optimizer: ASE optimizer name ('FIRE' or 'LBFGS'). Ignored by non-relaxation adapters.
            **kwargs: Additional keyword arguments passed to the adapter.

        Raises:
            AssertionError: if prototypes folder missing or unknown calculator name.
        """
        assert self.prototype_path is not None, (
            "No prototypes folder identified. Please run `generate_MPDD` or `generate_local` first."
        )
        assert isinstance(self.prototype_path, str), (
            "Prototypes folder path must be a string"
        )
        assert hasattr(self, "supported_calculators"), (
            "Supported calculators not initialized."
        )
        assert calculator in self.supported_calculators and calculator != "pySIPFENN", (
            f"Unknown calculator '{calculator}'. Available: {list(self.supported_calculators.keys())}"
        )

        adapter = self.supported_calculators[calculator]

        # If the adapter is a class (legacy entry-point registration), instantiate it
        if isinstance(adapter, type):
            try:
                adapter = adapter()
            except Exception as e:
                raise RuntimeError(f"Failed to instantiate adapter '{calculator}': {e}")

        # Assert micro-env readiness for micro-env adapters
        if hasattr(adapter, "is_env_ready"):
            assert adapter.is_env_ready(), (
                f"Micro-environment for '{calculator}' is not ready. "
                f"Run setup_calculator('{calculator}') first."
            )

        results = adapter.run_on_directory(
            self.prototype_path,
            mode=mode,
            max_workers=max_workers,
            objectives=objectives,
            isif=isif,
            optimizer=optimizer,
            **kwargs,
        )
        # Be tolerant to partial results: merge what we have and warn on mismatch
        if len(results) != len(self.prototypes):
            warnings.warn(
                f"Calculator '{calculator}' returned {len(results)} results for {len(self.prototypes)} prototypes; merging overlapping names."
            )

        # Resolve the adapter's ObjectiveRegistry once for the whole merge.
        adapter_reg = getattr(adapter, "objectives", None)
        if not isinstance(adapter_reg, ObjectiveRegistry):
            adapter_reg = None

        # Build index for fast merge
        name_to_index = {p.get("name"): i for i, p in enumerate(self.prototypes)}
        matched = 0
        for r in results:
            name = r.get("name") if isinstance(r, dict) else None
            props = r.get("properties", {}) if isinstance(r, dict) else {}
            if name in name_to_index:
                # Recode dispatch keys → objective codes using the adapter registry so
                # that define_objective / filter_prototypes can look up values with the
                # same code-based namespace used in predicted_objectives.
                recoded = {}
                for dispatch_key, val in props.items():
                    code = (
                        adapter_reg.code_for_dispatch_key(dispatch_key)
                        if adapter_reg is not None
                        else None
                    )
                    if code is not None:
                        recoded[code] = val
                self.prototypes[name_to_index[name]].update(recoded)
                matched += 1

        if matched == 0:
            raise AssertionError(
                "No calculator predictions matched any prototype names. Ensure filenames are preserved."
            )

        # Track only objectives actually produced (respecting user selection).
        # Bug 4 fix: extract code from each key (not the full key) and include all
        # objective types — not just scalars — so that Structure / DOS objectives
        # are also registered in predicted_objectives.
        produced_objectives = set()
        for r in results:
            props = r.get("properties", {}) if isinstance(r, dict) else {}
            for dispatch_key in props:
                code = (
                    adapter_reg.code_for_dispatch_key(dispatch_key)
                    if adapter_reg is not None
                    else None
                )
                if code is not None:
                    produced_objectives.add(code)
        # If user passed a subset, intersect
        if objectives is not None:
            produced_objectives = produced_objectives.intersection(set(objectives))
        for obj in sorted(produced_objectives):
            if obj not in self.predicted_objectives:
                self.predicted_objectives.append(obj)

    # No automatic objective aggregation here; users can call `define_objective` to derive new ones.

    def get_mpdd_convexhull(self) -> None:
        """
        Get the MPDD convex hull stability for the prototypes.

        Raises:
            AssertionError: If no connection to the MPDD has been established, or if no prototypes have been identified.
        """

        assert self.MPDD is not None, (
            "No database connection identified. Please run `setup_MPDD` first."
        )
        assert len(self.prototypes) > 0, (
            "No prototypes identified. Please run `generate_MPDD` or `generate_local` first."
        )

        MPDD_curated = self.MPDD["curated"]

        eList = self.target_formula.chemical_system.split("-")
        sysList = []
        for nComponents in range(eList.__len__()):
            for c in combinations(eList, 1 + nComponents):
                t = list(c)
                t.sort()
                sysList.append("-".join(t))

        entriesForPD = []
        formulaSet = set()
        i = 0
        for e in MPDD_curated.find(
            {"material.system": {"$in": sysList}, "properties.STAB_SIPFENN_NN30": 0},
            {
                "material.reducedFormula": 1,
                "metadata.parentDatabaseID": 1,
                "properties.EF_SIPFENN_NN30": 1,
            },
        ):
            i += 1
            formula = e["material"]["reducedFormula"]
            comp = Composition(formula)
            dH = e["properties"]["EF_SIPFENN_NN30"]

            name = f"{formula}-{e['_id']}"
            if "parentDatabaseID" in e["metadata"]:
                name = name + "-" + e["metadata"]["parentDatabaseID"]

            if dH is not None:
                pdEntry = PDEntry(comp, dH, name)
                if pdEntry.composition not in formulaSet:
                    entriesForPD.append(pdEntry)
                    formulaSet.add(pdEntry.composition)

        for prototype in self.prototypes:
            pdEntry = PDEntry(
                self.target_formula,
                prototype["SIPFENN_Krajewski2022_NN30"],
                prototype["name"],
            )
            entriesForPD.append(pdEntry)
            pd = PhaseDiagram(entriesForPD)
            stability = float(
                Decimal(pd.get_e_above_hull(pdEntry)).quantize(
                    Decimal("0.0001"), rounding=ROUND_DOWN
                )
            )
            prototype.update({"MPDD_Stability": stability})
            entriesForPD.pop()

    def define_objective(
        self, name: str, objectives: List[str], definition: str
    ) -> None:
        """
        Define a new objective as an arithmetic combination of existing objectives.

        The expression in `definition` supports the following tokens:
        - +, -, *, /, **
        - Parentheses: (...)
        - Indexed objectives: #X where X is the 0-based index in `objectives`

        Examples:
            name = 'Avg_Formation_Energy', objectives = ['EF_SIPFENN_NN30', 'EF_SIPFENN_NN20'], definition = '(#0 + #1) / 2'
            name = 'Scaled_Energy', objectives = ['EF_SIPFENN_NN30'], definition = '#0*3.18'
            name = 'Energy_Diff', objectives = ['EF_SIPFENN_NN30', 'EF_SIPFENN_NN20'], definition = '#0-#1'

        Args:
            name: The name of the new objective to create/update on each prototype.
            objectives: Ordered list of objective names referenced by #0, #1, ... in definition.
            definition: Arithmetic expression combining the objectives. Whitespaces are ignored.

        Raises:
            AssertionError: If prerequisites are not met or inputs are invalid.
        """

        # Basic validations
        assert isinstance(name, str) and len(name) > 0, (
            "Objective name must be a non-empty string"
        )
        assert (
            isinstance(objectives, list)
            and len(objectives) >= 1
            and all(isinstance(o, str) for o in objectives)
        ), "`objectives` must be a non-empty list of strings"
        assert isinstance(definition, str) and len(definition) > 0, (
            "`definition` must be a non-empty string"
        )
        assert len(self.prototypes) > 0, (
            "No prototypes identified. Please run `generate_MPDD` or `generate_local` first."
        )

        # Resolve every input to its canonical code, then validate against predictions
        resolved = [self._resolve_to_code(o) for o in objectives]
        missing = [r for r in resolved if r not in self.predicted_objectives]
        assert not missing, (
            f"All objectives must exist in predicted objectives. Missing: {missing}"
        )

        import re  # Local import to avoid global pollution if not already imported

        expr_template = definition.replace(" ", "")

        # Validate and transform #X tokens -> values[X] (0-based)
        indices = re.findall(r"#(\d+)", expr_template)
        if not indices:
            # Expression doesn't reference any objective; allow constants-only expressions
            pass
        else:
            # Check bounds
            for idx_str in indices:
                idx = int(idx_str)
                assert 0 <= idx < len(objectives), (
                    f"Index #{idx} in definition out of range for provided objectives (0..{len(objectives) - 1})"
                )

        expr = re.sub(r"#(\d+)", lambda m: f"values[{int(m.group(1))}]", expr_template)

        # Safety check: only allow numbers, parentheses, operators, decimal points, 'values' and index brackets
        # Replace allowed token 'values[<num>]' with '0' and then validate remaining characters
        tmp = re.sub(r"values\[\d+\]", "0", expr)
        # Allow digits, decimal points, scientific notation (e/E), operators, parentheses
        allowed_pattern = r"^[0-9eE\.+\-*/()\s]*$"  # ** is covered by * twice
        assert re.match(allowed_pattern, tmp) is not None, (
            "Definition contains invalid characters. Allowed: +, -, *, /, **, (), and #X references."
        )

        computed_count = 0
        for i, prototype in enumerate(self.prototypes):
            # Gather values for this prototype
            values = [prototype.get(r) for r in resolved]
            # All referenced values must be numeric
            if not all(isinstance(v, (int, float)) for v in values):
                # Skip this prototype if any required value missing or non-numeric
                continue
            try:
                result = eval(expr, {"__builtins__": {}}, {"values": values})
            except Exception:
                # If evaluation fails for any reason, skip updating this prototype
                continue
            if isinstance(result, (int, float)) and not (
                isinstance(result, float) and (result != result)
            ):  # exclude NaN
                # Round to 4 decimal places for consistency with existing outputs
                prototype[name] = round(float(result), 4)
                computed_count += 1

        # Register the new objective name only if at least one value was computed
        if computed_count > 0 and name not in self.predicted_objectives:
            self.predicted_objectives.append(name)

    def filter_prototypes(
        self,
        max_prototypes: int,
        cutoff: float = 0.5,
        objective: str | None = None,
    ) -> None:
        """
        Filter the prototypes based on the Formation Energy cutoff value provided.

        Args:
            max_prototypes: The maximum number of prototypes to keep.
            cutoff: The Formation Energy cutoff value. Defaults to 0.5.
            objective: Name of the objective to use for filtering. If None,
                the first entry in self.predicted_objectives is used.

        Raises:
            AssertionError: If no prototypes are identified or if no features are identified.
        """

        assert len(self.prototypes) > 0, (
            "No prototypes identified. Please run `generate_MPDD` or `generate_local` first."
        )
        assert len(self.ks2022) > 0, (
            "No features identified. Please run `run_pysipfenn` first."
        )

        objective = objective if objective is not None else self.predicted_objectives[0]
        objective = self._resolve_to_code(objective)
        assert objective in self.predicted_objectives, (
            "Objective should be one of the calculated objectives and/or their average: "
            + ", ".join(self.predicted_objectives)
        )

        if max_prototypes:
            self.prototypes.sort(
                key=lambda prototype: (prototype[objective], prototype["index"])
            )
            self.prototypes = self.prototypes[:max_prototypes]
        else:
            min_energy = min(prototype[objective] for prototype in self.prototypes)
            self.prototypes = [
                prototype
                for prototype in self.prototypes
                if prototype[objective] < cutoff + min_energy
            ]

        # Delete files not in filtered self.prototypes
        for filename in os.listdir(self.prototype_path):
            if filename not in {prototype["name"] for prototype in self.prototypes}:
                os.remove(os.path.join(self.prototype_path, filename))

        # Update the 'index' key of each dictionary in self.prototypes to its position in the list
        for i, prototype in enumerate(self.prototypes):
            prototype["index"] = i + 1

        # Update self.ks2022 to only include the values at the positions specified by the "index" key in the filtered self.prototypes
        indices = [prototype["index"] - 1 for prototype in self.prototypes]
        self.ks2022 = self.ks2022[indices]

        if max_prototypes:
            print(
                f"Filtered prototypes to {Fore.GREEN}{Style.BRIGHT}{len(self.prototypes)}{Style.RESET_ALL} based on the maximum number of prototypes of {max_prototypes}"
            )

        else:
            print(
                f"Filtered prototypes to {Fore.GREEN}{Style.BRIGHT}{len(self.prototypes)}{Style.RESET_ALL} based on the cutoff value of {cutoff}"
            )

    def cluster_prototypes_KMeans(
        self, n_clusters: int = None, same_composition: bool = False
    ) -> None:
        """
        Cluster the prototypes using KMeans, assign each data point to its cluster, and select the best prototypes.

        Args:
            n_clusters: Number of clusters. If not provided, the optimal number is determined automatically using silhouette score.
            same_composition: If True, only prototypes with the same composition are considered. Defaults to False.

        Raises:
            AssertionError: If no features are identified, if the number of clusters is not an integer, or if any objective is not allowed.
        """

        assert self.ks2022 is not None, (
            "No features identified. Please run `run_pysipfenn` first."
        )
        assert isinstance(n_clusters, int) or n_clusters is None, (
            "Number of clusters must be an integer"
        )
        assert n_clusters is None or n_clusters <= len(self.prototypes), (
            "Number of clusters must be less or equal than the number of prototypes"
        )

        # Check if the number of prototypes is sufficient for clustering
        if len(self.prototypes) < 5:
            print(
                f"At least 5 prototypes are required to run `cluster_prototypes`. {len(self.prototypes)} found."
            )
            return

        self._check_objectives()

        # Select, Scale and Embed features in 2D and 3D t-SNE space
        self._embed_features(same_composition)

        (
            print(
                "\nNo number of clusters has been provided. Determining the optimal number through `silhouette score`. This may take a few minutes..\n"
            )
            if n_clusters is None
            else None
        )

        # Determine the optimal number of clusters using silhouette score
        distances = pairwise_distances(self.embeddedks2022)
        distance_threshold = 0.1

        mask = np.ones(len(self.embeddedks2022), dtype=bool)
        for i in range(len(self.embeddedks2022)):
            for j in range(i + 1, len(self.embeddedks2022)):
                if distances[i, j] < distance_threshold:
                    mask[j] = False

        filtered_data = self.embeddedks2022[mask]  # Filter out overlapping embeddings

        sil_dict = {}
        if n_clusters is not None:
            k_range = [n_clusters]
        else:
            stop = max(2, len(filtered_data) - 1)
            k_range = np.unique(np.geomspace(2, stop, 40).astype(int))
            k_range = k_range[k_range >= 2] if k_range.size else np.array([2])
        for k in tqdm(
            k_range, desc="Processing", ncols=100, bar_format="{l_bar}{bar}{r_bar}"
        ):
            kmeans = KMeans(n_clusters=k, n_init=500, max_iter=500, random_state=580)
            clusterData = kmeans.fit(self.embeddedks2022)
            sil_score = silhouette_score(self.embeddedks2022, kmeans.labels_) - k / len(
                self.embeddedks2022
            )
            sil_dict[sil_score] = clusterData

        clusterData = sil_dict[max(sil_dict.keys())]

        print(
            f"\nClustered prototypes into {clusterData.n_clusters} clusters using KMeans in 2D t-SNE embedded space.\n"
        )

        # Assign each data point to its cluster
        clusters = {}
        for i, label in enumerate(clusterData.labels_):
            self.prototypes[i].update({"cluster": label + 1})
            clusters.setdefault(label + 1, []).append(self.prototypes[i])

        # Select the best prototype per cluster and Create a list of top candidates
        self._select_prototypes(clusters)

    def cluster_prototypes_DBSCAN(
        self,
        eps: float = 0.3,
        min_samples: int = 1,
        p: int = 1,
        same_composition: bool = False,
    ) -> None:
        """
        Cluster the prototypes using DBSCAN, assign each data point to its cluster, and select the best prototypes.

        Args:
            eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other. Defaults to 0.3.
            min_samples: The number of samples in a neighborhood for a point to be considered as a core point. Defaults to 1.
            p: The power parameter for the Minkowski metric. Defaults to 1.
            same_composition: If True, only prototypes with the same composition are considered. Defaults to False.

        Raises:
            AssertionError: If no features are identified, if eps is not a float, if min_samples is not an integer, or if any objective is not allowed.
        """

        assert self.ks2022 is not None, (
            "No features identified. Please run `run_pysipfenn` first."
        )
        assert isinstance(eps, float), "eps must be a float"
        assert isinstance(min_samples, int), "min_samples must be an integer"

        # Check if the number of prototypes is sufficient for clustering
        if len(self.prototypes) < 5:
            print(
                f"At least 5 prototypes are required to run `cluster_prototypes`. {len(self.prototypes)} found."
            )
            return

        self._check_objectives()

        # Select, Scale and Embed features in 2D and 3D t-SNE space
        self._embed_features(same_composition)

        # Cluster the prototypes using DBSCAN
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, p=p)
        dbscan.fit(self.scaledks2022)

        # Determine the number of clusters identified by DBSCAN
        n_clusters = len(set(dbscan.labels_)) - (1 if -1 in dbscan.labels_ else 0)
        print(f"\nClustered prototypes into {n_clusters} clusters using DBSCAN.\n")

        # Assign each data point to its cluster
        clusters = {}
        for i, label in enumerate(dbscan.labels_):
            self.prototypes[i].update({"cluster": label + 1})
            clusters.setdefault(label + 1, []).append(self.prototypes[i])

        # Select the best prototype per cluster and Create a list of top candidates
        self._select_prototypes(clusters)

    def _embed_features(self, same_composition: bool) -> None:
        """
        Embed the features in 2D t-SNE space.

        Args:
            same_composition: If True, only structural features are used. Defaults to False.

        Raises:
            AssertionError: If no features are identified.
        """

        assert len(self.prototypes) >= 5, (
            "At least 5 prototypes are required to run `cluster_prototypes`."
        )
        assert self.ks2022 is not None, (
            "No features identified. Please run `run_pysipfenn` first."
        )

        if same_composition:
            descriptor_module = importlib.import_module(
                f"pysipfenn.descriptorDefinitions.{_DEFAULT_PYSIPFENN_DESCRIPTOR}"
            )
            onlyStructural = descriptor_module.onlyStructural
            features2use = np.array([onlyStructural(v) for v in self.ks2022])
        else:
            features2use = self.ks2022

        # Select, Scale and Embed features in 2D and 3D t-SNE space
        scaler = MinMaxScaler()
        self.scaledks2022 = scaler.fit_transform(features2use)
        tsne = TSNE(
            n_components=2,
            perplexity=4,
            init="pca",
            learning_rate=10,
            max_iter=2000,
            metric="correlation",
            angle=0.2,
            n_jobs=1,
            random_state=580,
        )
        self.embeddedks2022 = tsne.fit_transform(self.scaledks2022)
        tsne_3d = TSNE(
            n_components=3,
            perplexity=4,
            init="pca",
            learning_rate=10,
            max_iter=2000,
            metric="correlation",
            angle=0.2,
            n_jobs=1,
            random_state=580,
        )
        self.embeddedks2022_3d = tsne_3d.fit_transform(self.scaledks2022)

    def _select_prototypes(self, clusters: dict) -> None:
        """
        Select the best prototype per cluster for each objective and store them in self.top_candidates.

        Args:
            clusters: A dictionary mapping cluster labels to lists of prototype dictionaries.

        Returns:
            List of dictionaries representing the selected prototypes.
        """

        # Select the best prototype per cluster based on the minimum predicted property
        self.cluster_affiliation = {}
        cluster_energies = {}
        for objective in self.predicted_objectives:
            self.cluster_affiliation[objective] = {}
            for label, cluster in sorted(clusters.items()):
                prototype = min(cluster, key=lambda x: x[objective])
                self.cluster_affiliation[objective].update({label: prototype["index"]})
                cluster_energies[label] = prototype[objective]

        # Sort clusters based on the minimum formation energy of their candidates
        sorted_clusters = sorted(cluster_energies, key=cluster_energies.get)
        cluster_mapping = {
            old_label: new_label + 1
            for new_label, old_label in enumerate(sorted_clusters)
        }

        # Reassign clusters based on the new ordering
        for prototype in self.prototypes:
            old_label = prototype["cluster"]
            new_label = cluster_mapping[old_label]
            prototype["cluster"] = new_label

        # Update cluster affiliation with new labels
        for objective in self.predicted_objectives:
            self.cluster_affiliation[objective] = {
                cluster_mapping[old_label]: index
                for old_label, index in self.cluster_affiliation[objective].items()
            }

        # Print the updated cluster information
        print("List of prototypes selected per energy of each cluster:")
        for objective in self.predicted_objectives:
            print(f"\nObjective: {objective}")
            for new_label in sorted(self.cluster_affiliation[objective].keys()):
                index = self.cluster_affiliation[objective][new_label]
                prototype = next(p for p in self.prototypes if p["index"] == index)
                print(f"           Cluster {new_label:02}: {prototype['name']}")

        # Create a list of top candidates
        self.top_candidates, i = [], 0
        for objective in self.predicted_objectives:
            for index in self.cluster_affiliation[objective].values():
                self.top_candidates.append(self.prototypes[index - 1].copy())
                self.top_candidates[i].update({"objective": objective})
                i += 1

        return self.top_candidates

    def export_table(
        self,
        content: List[str] = ["all-prototypes", "top-candidates"],
        file_prefix: str = "",
    ) -> None:
        """
        Export a CSV file with the specified content.

        Args:
            content: Specifies the type of data to export. Can be 'all-prototypes' and/or 'top-candidates'. Defaults to ['all-prototypes', 'top-candidates'].
            file_prefix: A prefix that will be added to the file name. Defaults to ''.

        Raises:
            AssertionError: If no prototypes are identified or if an invalid content type is provided.

        Outputs:
            CSV files: Writes CSV files for 'all-prototypes' and/or 'top-candidates' depending on the content argument. The files are named based on the content type and the file_prefix (if provided).
        """

        assert len(self.prototypes) > 0, (
            "No prototypes identified. Please run `generate_MPDD` or `generate_local` first."
        )
        assert not any(
            value not in ["all-prototypes", "top-candidates"] for value in content
        ), "Content should contain only `all_prototypes` and/or `top_candidates`"

        if "top-candidates" in content and not self.top_candidates:
            print(
                "No clustered prototypes identified. Please run `cluster_prototypes` first. \nChanging content to 'all-prototypes' only."
            )
            content = ["all-prototypes"]

        if "all-prototypes" in content:
            file_name = (
                "all-prototypes.csv"
                if file_prefix == ""
                else file_prefix + "_all_prototypes.csv"
            )
            with open(file_name, "w", newline="") as output_file:
                dict_writer = csv.DictWriter(output_file, self.prototypes[0].keys())
                dict_writer.writeheader()
                dict_writer.writerows(self.prototypes)

            print(
                f"\nExported {Fore.GREEN}{Style.BRIGHT}{len(self.prototypes)}{Style.RESET_ALL} prototypes to {file_name}"
            )

        if "top-candidates" in content:
            file_name = (
                "top-candidates.csv"
                if file_prefix == ""
                else file_prefix + "_top_candidates.csv"
            )
            top_candidates_exported = [
                d
                for d in self.top_candidates
                if d["objective"] in self.predicted_objectives
            ]

            with open(file_name, "w", newline="") as output_file:
                dict_writer = csv.DictWriter(
                    output_file, top_candidates_exported[0].keys()
                )
                dict_writer.writeheader()
                dict_writer.writerows(top_candidates_exported)

            print(
                f"Exported {Fore.GREEN}{Style.BRIGHT}top{Style.RESET_ALL} candidates for {len(self.predicted_objectives)} Objective(s) to {file_name}"
            )

    def plot_barchart(
        self,
        objective: Union[str, List[str]] = None,
        save: bool = False,
        file_name: str = "barchart_clusterPrototypes",
        color_blind: bool = False,
    ) -> None:
        """
        Plot bar charts for the prototypes.

        Args:
            objective: The objective to plot. Defaults to the first available objective.
            save: If True, saves the plot as an HTML file. Defaults to False.
            file_name: The name of the HTML file to save the plot. Defaults to 'barchart_clusterPrototypes'.
            color_blind: If True, uses a colorblind-friendly color palette. Defaults to False.

        Raises:
            AssertionError: If the objective is not one of the calculated objectives or their average.
        """

        objective = objective if objective is not None else self.predicted_objectives[0]
        assert isinstance(objective, str) or isinstance(objective, list), (
            "Objective must be a single value specified as a string or a list with a single string"
        )
        assert not (isinstance(objective, list) and len(objective) != 1), (
            "If objective is given as a list, it must contain exactly one string"
        )
        if isinstance(objective, list):
            objective = objective[0]
        objective = self._resolve_to_code(objective)
        assert objective in self.predicted_objectives, (
            "Objective should be one of the calculated objectives and/or their average: "
            + ", ".join(self.predicted_objectives)
        )

        if not self.top_candidates:
            print(
                "No clustered prototypes identified. Please run `cluster_prototypes` first. Aborting 2D plot generation."
            )
            return

        # Create a DataFrame from the data
        df = pd.DataFrame(
            {
                "cluster": [c["cluster"] for c in self.prototypes],
                "name": [c["name"].split("-")[0] for c in self.prototypes],
                "label": [
                    c["name"].split("-")[-2]
                    + "-"
                    + c["name"].split("-")[-1].replace(".POSCAR", "")
                    for c in self.prototypes
                ],
                "objective": [c[objective] for c in self.prototypes],
                "index": [c["index"] for c in self.prototypes],
            }
        )

        df = df.sort_values("cluster")

        # Define color set based on clusters
        colorblind_type = "Protanopia" if color_blind else None
        colors = get_colors(
            df["cluster"].nunique(),
            [(1.0, 1.0, 1.0), (1.0, 1.0, 0.0), (1.0, 0.5, 0.0)],
            colorblind_type=colorblind_type,
        )
        cluster_colors = {
            cluster: "#{:02x}{:02x}{:02x}".format(
                int(colors[i][0] * 255),
                int(colors[i][1] * 255),
                int(colors[i][2] * 255),
            )
            for i, cluster in enumerate(df["cluster"].unique())
        }

        # Sort the dataframe by 'objective' in ascending order
        df_sorted = df.sort_values(by=["objective", "index"]).reset_index(drop=True)
        # Create a positional x-axis so bars appear in the sorted order
        df_sorted["xpos"] = df_sorted.index + 1

        # Segment the DataFrame into df_prototypes and df_top_candidates
        df_prototypes = df_sorted[
            ~df_sorted["index"].isin(self.cluster_affiliation[objective].values())
        ]
        df_top_candidates = df_sorted[
            df_sorted["index"].isin(self.cluster_affiliation[objective].values())
        ]

        # Get the minimum and maximum values for the y-axis range
        y_min = df_sorted["objective"].min() * 1.001
        y_max = df_sorted["objective"].max() * 0.95

        # Trace for the legend title
        trace_legend = go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(color="black", symbol="triangle-right", size=10),
            showlegend=True,
            name="Top Candidates:",
        )

        # Create bar plots for prototypes and top candidates
        trace_prototypes = go.Bar(
            x=df_prototypes["xpos"],
            y=df_prototypes["objective"],
            hovertext=df_prototypes["objective"],
            hoverinfo="text",
            marker=dict(color="#a6a6a6"),
            width=0.4,
            name="Prototypes",
            showlegend=False,
        )

        # Create bar plots for each top candidate cluster
        traces_top_candidates = []
        for cluster in df_top_candidates["cluster"].unique():
            df_cluster = df_top_candidates[df_top_candidates["cluster"] == cluster]
            trace = go.Bar(
                x=df_cluster["xpos"],
                y=df_cluster["objective"],
                hovertext=df_cluster["objective"],
                hoverinfo="text",
                marker=dict(color=cluster_colors[cluster]),
                width=0.4,
                name=f"{df_cluster.iloc[0]['name']} (Cluster {cluster})",
            )
            traces_top_candidates.append(trace)

        data = [trace_prototypes, trace_legend] + traces_top_candidates

        layout = go.Layout(
            xaxis=dict(
                side="top",
                tickvals=df_sorted["xpos"],
                ticktext=df_sorted["label"],
                tickangle=300,
                showline=True,
                linewidth=1,
                linecolor="black",
                mirror=True,
                tickfont=dict(size=10),
            ),
            yaxis=dict(
                title="Enthalpy of Formation (eV/atom)",
                range=[y_min, y_max],
                showline=True,
                linewidth=1,
                linecolor="black",
                mirror=True,
            ),
            template="simple_white",
            height=600,
        )

        figure = go.Figure(data=data, layout=layout)

        if save:
            plot(figure, filename=f"{file_name}.html")
            print(f"Plot saved as {file_name}.html")

        else:
            iplot(figure, filename=file_name)

    def plot_2D(
        self,
        objective: Union[str, List[str]] = None,
        save: bool = False,
        file_name: str = "2D_clusterPrototypes",
        marker_symbol: bool = False,
        color_blind: bool = False,
    ) -> None:
        """
        Plot 2D scatter plots for the prototypes.

        Args:
            objective: The objective to plot. Defaults to the first available objective.
            save: If True, saves the plot as an HTML file. Defaults to False.
            file_name: The name of the HTML file to save the plot. Defaults to '2D_clusterPrototypes'.
            marker_symbol: If True, uses different marker symbols for each cluster. Defaults to False.
            color_blind: If True, uses a colorblind-friendly color palette. Defaults to False.

        Raises:
            AssertionError: If the objective is not one of the calculated objectives or their average.
        """

        objective = objective if objective is not None else self.predicted_objectives[0]
        assert isinstance(objective, str) or isinstance(objective, list), (
            "Objective must be a single value specified as a string or a list with a single string"
        )
        assert not (isinstance(objective, list) and len(objective) != 1), (
            "If objective is given as a list, it must contain exactly one string"
        )
        if isinstance(objective, list):
            objective = objective[0]
        objective = self._resolve_to_code(objective)
        assert objective in self.predicted_objectives, (
            "Objective should be one of the calculated objectives and/or their average: "
            + ", ".join(self.predicted_objectives)
        )

        if not self.top_candidates:
            print(
                "No clustered prototypes identified. Please run `cluster_prototypes` first. Aborting 2D plot generation."
            )
            return

        # Create a DataFrame from the data
        df = pd.DataFrame(
            {
                "x": np.transpose(self.embeddedks2022)[0],
                "y": np.transpose(self.embeddedks2022)[1],
                "cluster": [c["cluster"] for c in self.prototypes],
                "name": [c["name"].replace(".POSCAR", "") for c in self.prototypes],
                "objective": [c[objective] for c in self.prototypes],
                "index": [c["index"] for c in self.prototypes],
            }
        )

        df = df.sort_values("cluster")

        # Define a color for each cluster
        colorblind_type = "Protanopia" if color_blind else None
        colors = get_colors(
            df["cluster"].nunique(),
            [(1.0, 1.0, 1.0), (1.0, 1.0, 0.0), (1.0, 0.5, 0.0)],
            colorblind_type=colorblind_type,
        )

        # Define a symbol for each cluster (up to 8)
        symbols = [
            "circle",
            "circle-open",
            "cross",
            "diamond",
            "diamond-open",
            "square",
            "square-open",
            "x",
        ]

        # Create a scatter plot for each cluster
        data = []
        for i, cluster in enumerate(df["cluster"].unique()):
            df_cluster = df[df["cluster"] == cluster]
            df_cluster = df_cluster.sort_values(by=["objective", "index"])

            # Separate the data into top candidates and other cluster members
            df_prototypes = df_cluster[
                ~df_cluster["index"].isin(self.cluster_affiliation[objective].values())
            ]
            df_top_candidates = df_cluster[
                df_cluster["index"].isin(self.cluster_affiliation[objective].values())
            ]

            hovertext_prototypes = [
                "<br>".join(
                    [
                        "<b>Prototype: </b>"
                        + "-".join(
                            c["name"].split("-")[0:2] + [c["name"].split("-")[-1]]
                        )
                    ]
                    + ["<b>Affiliation: </b>" + c["name"].split("-")[2]]
                    + ["<b>Formation Energy: </b> " + str(round(c["objective"], 4))]
                )
                for _, c in df_cluster.iterrows()
            ]

            hovertext_top_candidates = hovertext_prototypes.pop(0)

            color_discrete_map = {
                "Top Candidate": "#ff7f0e",
                "Other Cluster Entries": "#{:02x}{:02x}{:02x}".format(
                    int(colors[i][0] * 255),
                    int(colors[i][1] * 255),
                    int(colors[i][2] * 255),
                ),
            }

            if marker_symbol and len(df["cluster"].unique()) <= 8:
                symbol = symbols[i]
            else:
                symbol = symbols[0]

            # Create scatter plot for other cluster members
            scatter_prototypes = dict(
                mode="markers",
                name=f"Cluster {cluster}",
                type="scatter",
                x=df_prototypes["x"],
                y=df_prototypes["y"],
                hovertext=hovertext_prototypes,  # set the hover name
                hoverinfo="text+name",  # only show the hovertext
                marker=dict(
                    size=5,
                    color=color_discrete_map[
                        "Other Cluster Entries"
                    ],  # set the color for other cluster members
                    symbol=symbol,
                ),
            )
            data.append(scatter_prototypes)

            # Create scatter plot for top candidates
            scatter_top_candidates = dict(
                mode="markers",
                name=f"Cluster {cluster}",
                showlegend=False,  # don't show this in the legend
                type="scatter",
                x=df_top_candidates["x"],
                y=df_top_candidates["y"],
                hovertext=hovertext_top_candidates,  # set the hover name
                hoverinfo="text+name",  # only show the hovertext
                marker=dict(
                    size=5,
                    color=color_discrete_map[
                        "Top Candidate"
                    ],  # set the color for top candidates
                    symbol=symbol,
                ),
            )
            data.append(scatter_top_candidates)

        # Create a shape for each cluster
        shapes = []
        for i, cluster in enumerate(df["cluster"].unique()):
            df_cluster = df[df["cluster"] == cluster]
            points = df_cluster[["x", "y"]].values
            if (
                not np.linalg.matrix_rank(points - points[0], tol=1e-12) < 2
                and len(df_cluster) >= 3
            ):  # Ensures at least 3 points that are not collinear
                hull = ConvexHull(df_cluster[["x", "y"]])
                polygon = {
                    "type": "path",
                    "path": "M "
                    + " L ".join(
                        f"{p[0]} {p[1]}"
                        for p in df_cluster.iloc[hull.vertices][["x", "y"]].values
                    )
                    + " Z",
                    "line": {
                        "color": "rgb"
                        + str(colors[i]).replace(
                            " ", ""
                        ),  # use the color defined for this cluster
                        "width": 1,
                    },
                    "fillcolor": "rgb"
                    + str(colors[i]).replace(
                        " ", ""
                    ),  # use the color defined for this cluster
                    "opacity": 0.2,
                }
                shapes.append(polygon)

        layout = dict(
            shapes=shapes,
            template="plotly_white",  # use the plotly_white template
            height=500,  # set the height of the plot
            width=500,  # set the width of the plot
            title=f"Embedded Prototypes ({len(self.scaledks2022[0])}D ➡ 2D t-SNE space)<br><sub>🟠 Top Candidates ({objective})</sub>",  # add the title here
        )

        figure = dict(data=data, layout=layout)

        if save:
            plot(figure, filename=f"{file_name}.html")
            print(f"Plot saved as {file_name}.html")

        else:
            iplot(figure, filename=file_name)

    def plot_3D(
        self,
        objective: Union[str, List[str]] = None,
        save: bool = False,
        file_name: str = "3D_clusterPrototypes",
        marker_symbol: bool = False,
        color_blind: bool = False,
    ) -> None:
        """
        Plot 3D scatter plots for the prototypes.

        Args:
            objective: The objective to plot. Defaults to the first available objective.
            save: If True, saves the plot as an HTML file. Defaults to False.
            file_name: The name of the HTML file to save the plot. Defaults to '3D_clusterPrototypes'.
            marker_symbol: If True, uses different marker symbols for each cluster. Defaults to False.
            color_blind: If True, uses a colorblind-friendly color palette. Defaults to False.

        Raises:
            AssertionError: If the objective is not one of the calculated objectives or their average.
        """

        objective = objective if objective is not None else self.predicted_objectives[0]
        assert isinstance(objective, str) or isinstance(objective, list), (
            "Objective must be a single value specified as a string or a list with a single string"
        )
        assert not (isinstance(objective, list) and len(objective) != 1), (
            "If objective is given as a list, it must contain exactly one string"
        )
        if isinstance(objective, list):
            objective = objective[0]
        objective = self._resolve_to_code(objective)
        assert objective in self.predicted_objectives, (
            "Objective should be one of the calculated objectives and/or their average: "
            + ", ".join(self.predicted_objectives)
        )

        if not self.top_candidates:
            print(
                "No clustered prototypes identified. Please run `cluster_prototypes` first. Aborting 3D plot generation."
            )
            return

        # Create a DataFrame from the data
        df = pd.DataFrame(
            {
                "x": np.transpose(self.embeddedks2022_3d)[0],
                "y": np.transpose(self.embeddedks2022_3d)[1],
                "z": np.transpose(self.embeddedks2022_3d)[2],
                "cluster": [c["cluster"] for c in self.prototypes],
                "name": [c["name"].replace(".POSCAR", "") for c in self.prototypes],
                "objective": [c[objective] for c in self.prototypes],
                "index": [c["index"] for c in self.prototypes],
            }
        )

        df = df.sort_values("cluster")

        # Define a color for each cluster
        colorblind_type = "Protanopia" if color_blind else None
        colors = get_colors(
            df["cluster"].nunique(),
            [(1.0, 1.0, 1.0), (1.0, 1.0, 0.0), (1.0, 0.5, 0.0)],
            colorblind_type=colorblind_type,
        )

        # Define a symbol for each cluster (up to 8)
        symbols = [
            "circle",
            "circle-open",
            "cross",
            "diamond",
            "diamond-open",
            "square",
            "square-open",
            "x",
        ]

        # Create a scatter plot and a mesh for each cluster
        data = []
        for i, cluster in enumerate(df["cluster"].unique()):
            df_cluster = df[df["cluster"] == cluster]
            df_cluster = df_cluster.sort_values(by=["objective", "index"])

            # Separate the data into top candidates and other cluster members
            df_prototypes = df_cluster[
                ~df_cluster["index"].isin(self.cluster_affiliation[objective].values())
            ]
            df_top_candidates = df_cluster[
                df_cluster["index"].isin(self.cluster_affiliation[objective].values())
            ]

            hovertext_prototypes = [
                "<br>".join(
                    [
                        "<b>Prototype: </b>"
                        + "-".join(
                            c["name"].split("-")[0:2] + [c["name"].split("-")[-1]]
                        )
                    ]
                    + ["<b>Affiliation: </b>" + c["name"].split("-")[2]]
                    + ["<b>Formation Energy: </b> " + str(round(c["objective"], 4))]
                )
                for _, c in df_cluster.iterrows()
            ]

            hovertext_top_candidates = hovertext_prototypes.pop(0)

            color_discrete_map = {
                "Top Candidate": "#ff7f0e",
                "Other Cluster Entries": "#{:02x}{:02x}{:02x}".format(
                    int(colors[i][0] * 255),
                    int(colors[i][1] * 255),
                    int(colors[i][2] * 255),
                ),
            }

            if marker_symbol and len(df["cluster"].unique()) <= 8:
                symbol = symbols[i]
            else:
                symbol = symbols[0]

            # Create scatter plot for other cluster members
            scatter_prototypes = dict(
                mode="markers",
                name=f"Cluster {cluster}",
                type="scatter3d",
                x=df_prototypes["x"],
                y=df_prototypes["y"],
                z=df_prototypes["z"],
                hovertext=hovertext_prototypes,  # set the hover name
                hoverinfo="text+name",  # only show the hovertext
                marker=dict(
                    size=3,
                    color=color_discrete_map[
                        "Other Cluster Entries"
                    ],  # set the color for other cluster members
                    symbol=symbol,
                ),
            )
            data.append(scatter_prototypes)

            # Create scatter plot for top candidates
            scatter_top_candidates = dict(
                mode="markers",
                name=f"Cluster {cluster}",
                showlegend=False,  # don't show this in the legend
                type="scatter3d",
                x=df_top_candidates["x"],
                y=df_top_candidates["y"],
                z=df_top_candidates["z"],
                hovertext=hovertext_top_candidates,  # set the hover name
                hoverinfo="text+name",  # only show the hovertext
                marker=dict(
                    size=3,
                    color=color_discrete_map[
                        "Top Candidate"
                    ],  # set the color for top candidates
                    symbol=symbol,
                ),
            )
            data.append(scatter_top_candidates)

            # Create mesh for the cluster
            mesh = dict(
                alphahull=0,
                name=f"Cluster {cluster}",
                opacity=0.1,
                type="mesh3d",
                x=df_cluster["x"],
                y=df_cluster["y"],
                z=df_cluster["z"],
                color=color_discrete_map[
                    "Other Cluster Entries"
                ],  # set the color of the mesh
                hoverinfo="text+name",  # don't show any hover information for the mesh
            )
            data.append(mesh)

        layout = dict(
            template="plotly_white",  # use the plotly_white template
            scene=dict(xaxis_title="", yaxis_title="", zaxis_title=""),
            height=500,  # set the height of the plot
            width=500,  # set the width of the plot
            title=f"Embedded Prototypes ({len(self.scaledks2022[0])}D ➡ 3D t-SNE space)<br><sub>🟠 Top Candidates ({objective})</sub>",  # add the title here
        )

        figure = dict(data=data, layout=layout)

        if save:
            plot(figure, filename=f"{file_name}.html")
            print(f"Plot saved as {file_name}.html")

        else:
            iplot(figure, filename=file_name)


# Package Metadata ****************************************************************************
__name__ = "crystall"
__version__ = "X.X.X"
__release__ = "YYYY-MM"
__authors__ = "Amaral, Ricardo and Krajewski, Adam"
__doc__ = "crystALL.mpdd.org"
__rep__ = "github.com/PhasesResearchLab/crystALL"
__bibtex__ = "@software{amaral-krajewski2026crystall, title={crystALL: A data-driven materials discovery platform}, author={{Amaral, R.} and {Krajewski, A. M.}}, journal={npj Computational Materials}, volume={X}, number={1}, pages={1-10}, year={2026}, publisher={Nature Publishing Group}, doi={10.xxxx.xxxxx-xxxx}}"
