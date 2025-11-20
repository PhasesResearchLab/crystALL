# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

from crystall.core import __name__, __version__, __authors__, __file__
from datetime import datetime

project = __name__
release = __version__
author = __authors__
copyright = f'{datetime.now().year}, {author}'
html_logo = "_static/crystALL.png"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.linkcode',
              'sphinx.ext.duration',
              'sphinx.ext.coverage',
              'sphinx.ext.napoleon',
              'sphinx.ext.autodoc',
              'sphinx_autodoc_typehints',
              'sphinx_github_changelog',
              'sphinx_rtd_size',
              'myst_nb'
              ]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for napoleon ----------------------------------------------------
napoleon_use_param = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ['_static']
sphinx_rtd_size_width = "70%"
html_css_files = [
    'custom.css',
]

html_context = {
    "display_github": True,
    "github_user": "PhasesResearchLab",
    "github_repo": "crystALL",
    "github_version": "main",
    "conf_py_path": "/docs/",
}

html_theme_options = {
    'logo_only': True,
    'flyout_display': 'attached'
}

# Resolve function for the linkcode extension.
def linkcode_resolve(domain, info):
    def find_source():
        import inspect
        import os, sys
        obj = sys.modules[info["module"]]
        for part in info["fullname"].split("."):
            obj = getattr(obj, part)

        fn = inspect.getsourcefile(obj)
        fn = os.path.relpath(fn, start=os.path.dirname(__file__))
        source, lineno = inspect.getsourcelines(obj)
        return fn, lineno, lineno + len(source) - 1

    if domain != "py" or not info["module"]:
        return None

    try:
        rel_path, line_start, line_end = find_source()
        # __file__ is imported from crystall
        filename = f"crystall/{rel_path}#L{line_start}-L{line_end}"
    except Exception:
        # no need to be relative to core here as module includes full path.
        filename = info["module"].replace(".", "/") + ".py"

    tag = "v" + __version__
    return f"https://github.com/PhasesResearchLab/crystALL/blob/{tag}/{filename}"