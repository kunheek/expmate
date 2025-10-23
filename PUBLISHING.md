# PyPI Publishing Checklist

This checklist will help you publish ExpMate to PyPI.

## Pre-Publishing Checklist

### 1. Documentation ✅
- [x] README.md created with comprehensive guide
- [x] MkDocs documentation site created
- [x] API documentation configured
- [x] Examples documented
- [x] Contributing guide added
- [x] Changelog created

### 2. Package Configuration

- [ ] Verify version in `pyproject.toml` and `src/expmate/__init__.py`
- [ ] Check package metadata (name, description, author, etc.)
- [ ] Verify dependencies and optional dependencies
- [ ] Update classifiers if needed
- [ ] Ensure LICENSE file exists

### 3. Testing

```bash
# Run all tests
pytest

# Check test coverage
pytest --cov=expmate --cov-report=html

# Run linting
ruff check src/expmate tests

# Format code
black src/expmate tests

# Type checking
mypy src/expmate
```

### 4. Documentation Build

```bash
# Install documentation dependencies
pip install -r docs-requirements.txt

# Build documentation locally
mkdocs build

# Serve and verify
mkdocs serve
```

Visit http://localhost:8000 to check documentation.

### 5. Package Build

```bash
# Install build tools
pip install build twine

# Build package
python -m build

# Verify package contents
tar -tzf dist/expmate-0.1.0.tar.gz
unzip -l dist/expmate-0.1.0-py3-none-any.whl
```

## Publishing Steps

### 1. Create Test PyPI Account

Sign up at https://test.pypi.org/account/register/

### 2. Test Upload (Recommended)

```bash
# Upload to Test PyPI
python -m twine upload --repository testpypi dist/*

# Test installation
pip install --index-url https://test.pypi.org/simple/ expmate
```

### 3. Create PyPI Account

Sign up at https://pypi.org/account/register/

### 4. Upload to PyPI

```bash
# Upload to PyPI
python -m twine upload dist/*
```

### 5. Verify Installation

```bash
# Install from PyPI
pip install expmate

# Verify
python -c "import expmate; print(expmate.__version__)"
expmate --version
```

## Post-Publishing Steps

### 1. Create Git Tag

```bash
git tag v0.1.0
git push origin v0.1.0
```

### 2. Create GitHub Release

1. Go to https://github.com/kunheek/expmate/releases
2. Click "Create a new release"
3. Select tag `v0.1.0`
4. Add release notes from CHANGELOG.md
5. Publish release

### 3. Deploy Documentation

#### GitHub Pages (Automatic)

Documentation will deploy automatically via GitHub Actions when you push to main.

To enable:
1. Go to Settings → Pages
2. Set Source to "Deploy from a branch"
3. Select `gh-pages` branch
4. Documentation will be at: https://kunheek.github.io/expmate/

#### ReadTheDocs (Optional)

1. Go to https://readthedocs.org/
2. Import project: kunheek/expmate
3. Documentation will be at: https://expmate.readthedocs.io/

### 4. Announce Release

Share on:
- GitHub Discussions
- Twitter/X
- Reddit (r/MachineLearning, r/learnmachinelearning)
- ML community forums

## Troubleshooting

### Build Fails

```bash
# Clean build artifacts
rm -rf dist/ build/ *.egg-info

# Rebuild
python -m build
```

### Upload Fails

Common issues:
- **Version already exists**: Increment version number
- **Invalid credentials**: Check PyPI API token
- **File size too large**: Check that test data isn't included

### Documentation Not Building

```bash
# Check for errors
mkdocs build --strict

# Verify imports work
python -c "from expmate import Config"
```

## Using PyPI API Tokens (Recommended)

Instead of username/password, use API tokens:

1. Go to https://pypi.org/manage/account/token/
2. Create a new API token
3. Create `~/.pypirc`:

```ini
[pypi]
  username = __token__
  password = pypi-AgEIcHlwaS5vcmc...  # Your token

[testpypi]
  username = __token__
  password = pypi-AgENdGVzdC5weXBp...  # Your test token
```

Then upload with:

```bash
python -m twine upload dist/*
```

## Version Numbering

Follow Semantic Versioning (https://semver.org/):

- **MAJOR** (1.0.0): Breaking changes
- **MINOR** (0.1.0): New features, backwards compatible
- **PATCH** (0.0.1): Bug fixes, backwards compatible

Examples:
- `0.1.0`: Initial release
- `0.1.1`: Bug fix
- `0.2.0`: New features
- `1.0.0`: Stable release

## Continuous Deployment (Future)

Consider setting up GitHub Actions for automatic publishing:

```yaml
# .github/workflows/publish.yml
name: Publish to PyPI

on:
  release:
    types: [published]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: pip install build twine
      - name: Build package
        run: python -m build
      - name: Publish to PyPI
        env:
          TWINE_USERNAME: __token__
          TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
        run: twine upload dist/*
```

## Resources

- [Python Packaging Guide](https://packaging.python.org/)
- [PyPI](https://pypi.org/)
- [Test PyPI](https://test.pypi.org/)
- [Semantic Versioning](https://semver.org/)
- [GitHub Releases](https://docs.github.com/en/repositories/releasing-projects-on-github)
