# Titli Documentation

This directory contains the Sphinx-based documentation for the Titli project.

## Building the Documentation

### Prerequisites

Install Sphinx and required extensions:

```bash
pip install sphinx sphinx-rtd-theme sphinx-autodoc-typehints
```

### Build HTML Documentation

To build the HTML documentation:

```bash
cd docs
make html
```

The generated HTML files will be in `build/html/`. Open `build/html/index.html` in a web browser to view the documentation.

### Build Other Formats

Sphinx supports multiple output formats:

```bash
# PDF (requires LaTeX)
make latexpdf

# EPUB
make epub

# Plain text
make text

# View all available formats
make help
```

### Clean Build Artifacts

To remove all generated files:

```bash
make clean
```

## Documentation Structure

```
docs/
├── source/
│   ├── conf.py              # Sphinx configuration
│   ├── index.rst            # Main documentation page
│   ├── installation.rst     # Installation guide
│   ├── quickstart.rst       # Quick start guide
│   ├── usage.rst            # Detailed usage guide
│   ├── changelog.rst        # Version history
│   ├── license.rst          # License information
│   └── api/                 # API reference documentation
│       ├── fe.rst           # Feature extractors API
│       ├── ids.rst          # IDS models API
│       └── utils.rst        # Utilities API
├── build/                   # Generated documentation (not in git)
├── Makefile                 # Unix/Linux build commands
└── make.bat                 # Windows build commands
```

## Updating Documentation

### Modifying Existing Pages

Edit the `.rst` files in the `source/` directory. After making changes, rebuild the documentation:

```bash
make html
```

### Adding New Pages

1. Create a new `.rst` file in `source/` or appropriate subdirectory
2. Add the file to the `toctree` in `source/index.rst` or another appropriate parent page
3. Rebuild the documentation

### Updating API Documentation

API documentation is automatically generated from docstrings in the source code. To update:

1. Modify docstrings in the Python source files
2. Rebuild the documentation

## Documentation Style

- Use reStructuredText (RST) format
- Follow Google or NumPy docstring conventions
- Include code examples where appropriate
- Add cross-references using Sphinx roles (`:doc:`, `:ref:`, `:class:`, etc.)

## Viewing Documentation Online

The documentation can be hosted on:

- [Read the Docs](https://readthedocs.org/)
- GitHub Pages
- Your own web server

## Troubleshooting

### Import Errors During Build

If you see import errors when building:

```bash
# Install project dependencies
pip install -e ..
```

### Missing Dependencies

```bash
# Install all documentation dependencies
pip install -r requirements-docs.txt  # If such file exists
# Or install individually:
pip install sphinx sphinx-rtd-theme sphinx-autodoc-typehints
```

### Warnings During Build

Some warnings are expected (e.g., network timeout for intersphinx). Critical errors will stop the build.

## Contributing

When contributing documentation:

1. Build locally and verify changes
2. Check for broken links and references
3. Ensure code examples are accurate and working
4. Follow the existing documentation structure and style

For more information, see the [Sphinx documentation](https://www.sphinx-doc.org/).
