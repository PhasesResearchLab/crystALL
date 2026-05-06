from crystall.core import *
from collections import defaultdict
from duecredit import Doi, due
from duecredit.io import TextOutput, get_text_rendering
from duecredit.collector import CollectorSummary
import os
import sys
from colorama import Style

# ── DueCredit output formatter ───────────────────────────────────────────────
_DUECREDIT_PREFIX = (
    "\nDueCredit Report:\n" + "-" * 17 + "\n\n"
    f"{Style.BRIGHT}"
    "If you use crystALL or any associated community models in your research, "
    "please cite the references listed below. The DueCredit report reflects "
    "the specific components and models used in your workflow.\n"
    f"{Style.RESET_ALL}"
)


class DuecreditTextOutput(TextOutput):
    """TextOutput variant that emits only the References section."""

    def dump(self, tags=None) -> None:
        packages, modules, objects = self._get_collated_citations(tags)
        pmo = {**packages, **modules, **objects}
        if not pmo:
            return

        paths = sorted(pmo)
        entry_keys = [c.entry.key for p in paths for c in pmo[p]]
        citation_nr = defaultdict(int)
        refnr = 1
        for ek in entry_keys:
            if ek not in citation_nr:
                citation_nr[ek] = refnr
                refnr += 1

        self.fd.write(_DUECREDIT_PREFIX)
        printed_keys = []
        for path in paths:
            for cit in pmo[path]:
                ek = cit.entry.key
                if ek not in printed_keys:
                    self.fd.write(f"\n[{citation_nr[ek]}] ")
                    self.fd.write(get_text_rendering(cit.entry, style=self.style))
                    printed_keys.append(ek)
        self.fd.write("\n")


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
        output = DuecreditTextOutput(fd=sys.stdout, collector=collector)
        summary = CollectorSummary.__new__(CollectorSummary)
        summary._outputs = [output]
        summary.dump()

    due.__class__.dump = _patched_dump
    due.activate(True)


# ── Package-level citation ────────────────────────────────────────────────────
def _cite_crystall_package() -> None:
    """Emit the crystALL package-level DueCredit citation.

    Uses the DOI declared in pyproject.toml and exposed as ``__doi__``
    by ``crystall.core``.  Called once at package import so the citation
    is registered regardless of which sub-module the user accesses.
    """
    from crystall.core import __doi__

    due.cite(
        Doi(__doi__),
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
