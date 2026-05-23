from crystall.core import *
from collections import defaultdict
from duecredit import BibTeX, due
from duecredit.io import TextOutput, get_text_rendering
from duecredit.collector import CollectorSummary
import os
import sys
from colorama import Back, Fore, Style

# ── DueCredit output formatter ───────────────────────────────────────────────
_DUECREDIT_PREFIX = (
    "\n[DueCredit-Report] If you use crystALL in your research, please do the due:"
)


class DuecreditTextOutput(TextOutput):
    """TextOutput variant that emits only the References section."""

    def dump(self, tags=None, path_prefix: str | None = None) -> None:
        packages, modules, objects = self._get_collated_citations(tags)
        pmo = {**packages, **modules, **objects}
        if path_prefix is not None:
            pmo = {
                path: citations
                for path, citations in pmo.items()
                if path == path_prefix or path.startswith(path_prefix + ".")
            }
        if not pmo:
            return

        self.fd.write(_DUECREDIT_PREFIX)
        printed_keys = []
        for path in sorted(pmo):
            for cit in pmo[path]:
                ek = cit.entry.key
                if ek in printed_keys:
                    continue
                if path == "crystall":
                    label = "crystALL:software"
                else:
                    label = (
                        path.split(".")[-1] + ":method"
                        if path.startswith("crystall.")
                        else path
                    )
                rendered = get_text_rendering(cit.entry, style=self.style).strip()
                doi = self._extract_doi(cit)
                doi_text = f" DOI: {doi}" if doi else ""
                self.fd.write(f"\n\n • {rendered}{doi_text} [{label}]")
                printed_keys.append(ek)
        self.fd.write("\n")

    def _extract_doi(self, cit) -> str | None:
        entry = cit.entry
        if hasattr(entry, "_rawentry") and isinstance(entry._rawentry, str):
            raw = entry._rawentry
            import re

            doi_match = re.search(r"doi=\{([^}]+)\}", raw, re.IGNORECASE | re.DOTALL)
            if doi_match:
                return doi_match.group(1).strip()
        return None


# ── DueCredit output configuration ───────────────────────────────────────────
def _configure_duecredit_output() -> None:
    """Patch DueSwitch.dump to emit only the crystALL references block.

    Called once at package import.  Replaces the verbose default DueCredit
    report with :class:`DuecreditTextOutput`, which prints only the
    ``References`` section prefixed by the crystALL courtesy banner.
    """
    os.environ.setdefault("DUECREDIT_OUTPUTS", "stdout")

    def _patched_dump(self, **kwargs):
        collector = self._DueSwitch__collectors[True]
        print(f"{Back.WHITE}{Fore.BLACK}")
        output = DuecreditTextOutput(fd=sys.stdout, collector=collector)
        output.dump(path_prefix="crystall")
        print(f"{Style.RESET_ALL}")

    due.__class__.dump = _patched_dump
    due.activate(True)


# ── Package-level citation ────────────────────────────────────────────────────
def _cite_crystall_package() -> None:
    """Emit the crystALL package-level DueCredit citation.

    Uses the DOI declared in pyproject.toml and exposed as ``__doi__``
    by ``crystall.core``.  Called once at package import so the citation
    is registered regardless of which sub-module the user accesses.
    """
    from crystall.core import __bibtex__

    # duecredit does not support @software yet, so we patch it to @misc
    patched_bibtex = __bibtex__.replace("@software", "@misc", 1)

    due.cite(
        BibTeX(patched_bibtex),
        description=(
            "crystALL: Data-driven candidate structure ensemble generation "
            "beyond the convex hull"
        ),
        path="crystall",
        tags=["reference-implementation"],
        cite_module=False,
    )


_configure_duecredit_output()
_cite_crystall_package()
