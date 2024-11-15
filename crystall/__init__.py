from crystall.crystALL import *

# Disable undesired warnings from torch and pymatgen
import warnings
warnings.filterwarnings("ignore", message="(?:.|\n)*datapipes.*dataloader2.*deprecated(?:.|\n)*")
warnings.filterwarnings("ignore", message="(?:.|\n)*electronegativity(?:.|\n)*")

# Fix Windows console color issues
from colorama import just_fix_windows_console
just_fix_windows_console()