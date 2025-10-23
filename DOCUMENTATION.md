# Documentation Setup Guide

This guide explains how to build and deploy the ExpMate documentation.

## Local Development

### 1. Install Documentation Dependencies

```bash
pip install -r docs-requirements.txt
```

This installs:
- MkDocs: Static site generator
- Material for MkDocs: Beautiful theme
- mkdocstrings: API documentation from docstrings
- pymdown-extensions: Markdown extensions

### 2. Serve Documentation Locally

```bash
mkdocs serve
```

This starts a local development server at http://localhost:8000 with live reloading.

### 3. Build Documentation

```bash
mkdocs build
```

This generates static HTML in the `site/` directory.

## Deployment Options

### Option 1: GitHub Pages (Recommended)

The repository includes a GitHub Actions workflow that automatically builds and deploys documentation to GitHub Pages.

#### Setup Steps:

1. **Enable GitHub Pages:**
   - Go to repository Settings → Pages
   - Set Source to "Deploy from a branch"
   - Select `gh-pages` branch
   - Click Save

2. **Push to Main Branch:**
   ```bash
   git push origin main
   ```

3. **Documentation will be available at:**
   ```
   https://kunheek.github.io/expmate/
   ```

The workflow (`.github/workflows/docs.yml`) automatically:
- Builds documentation on every push to main
- Deploys to GitHub Pages
- Runs on pull requests (build only, no deploy)

### Option 2: ReadTheDocs

1. **Sign up at ReadTheDocs:**
   Visit https://readthedocs.org/ and sign in with GitHub

2. **Import Project:**
   - Click "Import a Project"
   - Select `kunheek/expmate`
   - Configure project settings

3. **Add .readthedocs.yml:**
   ```yaml
   version: 2
   
   build:
     os: ubuntu-22.04
     tools:
       python: "3.11"
   
   mkdocs:
     configuration: mkdocs.yml
   
   python:
     install:
       - requirements: docs-requirements.txt
       - method: pip
         path: .
   ```

4. **Documentation will be available at:**
   ```
   https://expmate.readthedocs.io/
   ```

### Option 3: Manual Deployment

Build and deploy manually:

```bash
# Build documentation
mkdocs build

# Deploy to GitHub Pages manually
mkdocs gh-deploy
```

## Documentation Structure

```
docs/
├── index.md                    # Home page
├── getting-started/
│   ├── installation.md        # Installation guide
│   ├── quickstart.md          # Quick start tutorial
│   └── concepts.md            # Core concepts
├── guide/
│   ├── configuration.md       # Config management guide
│   ├── logging.md             # Logging guide
│   ├── checkpoints.md         # Checkpoint guide
│   ├── distributed.md         # DDP guide
│   ├── tracking.md            # Tracking guide
│   └── cli.md                 # CLI tools guide
├── examples/
│   ├── minimal.md             # Minimal example
│   ├── training.md            # Full training example
│   ├── ddp.md                 # DDP example
│   └── sweeps.md              # Hyperparameter sweeps
├── api/
│   ├── config.md              # Config API reference
│   ├── logger.md              # Logger API reference
│   ├── parser.md              # Parser API reference
│   ├── checkpoint.md          # Checkpoint API reference
│   ├── tracking.md            # Tracking API reference
│   └── utils.md               # Utils API reference
├── contributing.md            # Contributing guide
└── changelog.md               # Changelog
```

## Writing Documentation

### Markdown Files

Documentation is written in Markdown with some extensions:

#### Code Blocks

````markdown
```python
from expmate import Config

config = Config({'key': 'value'})
```
````

#### Admonitions

```markdown
!!! note
    This is a note.

!!! warning
    This is a warning.

!!! tip
    This is a tip.
```

#### Tabs

```markdown
=== "Python"
    ```python
    print("Hello")
    ```

=== "Bash"
    ```bash
    echo "Hello"
    ```
```

### API Documentation

API documentation is generated from docstrings using mkdocstrings:

```markdown
::: expmate.config.Config
    options:
      show_root_heading: true
      show_source: true
```

This automatically extracts and formats the docstring from `Config` class.

### Docstring Format

Use Google-style docstrings:

```python
def function(arg1: str, arg2: int = 0) -> bool:
    """Short description.

    Longer description with more details.

    Args:
        arg1: Description of arg1
        arg2: Description of arg2

    Returns:
        Description of return value

    Raises:
        ValueError: When something goes wrong

    Examples:
        >>> function("test", 42)
        True
    """
    pass
```

## Updating Documentation

### 1. Edit Markdown Files

Make changes to files in the `docs/` directory.

### 2. Preview Changes

```bash
mkdocs serve
```

Visit http://localhost:8000 to see your changes.

### 3. Commit and Push

```bash
git add docs/
git commit -m "docs: update documentation"
git push origin main
```

Documentation will automatically deploy if using GitHub Pages workflow.

## Troubleshooting

### Build Fails

Check that all dependencies are installed:
```bash
pip install -r docs-requirements.txt
pip install -e .
```

### Links Don't Work

Use relative links in markdown:
```markdown
[Link to quickstart](getting-started/quickstart.md)
```

### API Docs Don't Generate

Ensure docstrings are properly formatted and the module is importable:
```bash
python -c "from expmate import Config; print(Config.__doc__)"
```

## Resources

- [MkDocs Documentation](https://www.mkdocs.org/)
- [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/)
- [mkdocstrings](https://mkdocstrings.github.io/)
- [GitHub Pages](https://pages.github.com/)
- [ReadTheDocs](https://docs.readthedocs.io/)
