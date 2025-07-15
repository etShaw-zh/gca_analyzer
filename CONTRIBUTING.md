# Contributing to GCA Analyzer

Thank you for your interest in contributing to GCA Analyzer! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contributing Process](#contributing-process)
- [Code Style](#code-style)
- [Testing](#testing)
- [Documentation](#documentation)
- [Issue Guidelines](#issue-guidelines)
- [Pull Request Guidelines](#pull-request-guidelines)
- [Release Process](#release-process)

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct:

- Be respectful and inclusive
- Welcome newcomers and help them get started
- Focus on constructive feedback
- Be patient with questions and different skill levels
- Respect different viewpoints and experiences

## Getting Started

### Prerequisites

- Python 3.9 or higher
- Git
- Basic knowledge of group communication analysis concepts

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/your-username/gca_analyzer.git
   cd gca_analyzer
   ```

## Development Setup

### 1. Create Virtual Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
# Install development dependencies
pip install -e ".[dev]"

# Or install from requirements
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### 3. Install Pre-commit Hooks

```bash
pre-commit install
```

### 4. Verify Installation

```bash
# Run tests to verify setup
pytest

# Run linting
flake8 gca_analyzer/
mypy gca_analyzer/
```

## Contributing Process

### 1. Choose or Create an Issue

- Look for issues labeled `good first issue` or `help wanted`
- Create a new issue if you find a bug or have a feature request
- Discuss your approach before starting work on large changes

### 2. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-number-description
```

### 3. Make Changes

- Follow the code style guidelines
- Write tests for new functionality
- Update documentation as needed
- Ensure all tests pass

### 4. Commit Changes

```bash
git add .
git commit -m "type: brief description

Detailed description of changes made.

Fixes #issue-number"
```

Commit message format:
- `feat`: new feature
- `fix`: bug fix
- `docs`: documentation changes
- `style`: code style changes
- `refactor`: code refactoring
- `test`: adding or updating tests
- `chore`: maintenance tasks

### 5. Push and Create Pull Request

```bash
git push origin your-branch-name
```

Then create a pull request on GitHub.

## Code Style

### Python Code Style

We follow PEP 8 with some modifications:

- Maximum line length: 100 characters
- Use double quotes for strings
- Use type hints for function signatures
- Follow Google docstring style

### Linting Tools

- **flake8**: Code style checking
- **mypy**: Type checking
- **black**: Code formatting (optional but recommended)

Run linting:
```bash
flake8 gca_analyzer/
mypy gca_analyzer/
```

### Code Organization

- Keep functions small and focused
- Use descriptive variable names
- Add docstrings to all public functions and classes
- Group related functionality into modules

## Testing

### Test Structure

```
tests/
├── test_analyzer.py          # Core analyzer tests
├── test_analyzer_edge_cases.py  # Edge case tests
├── test_llm_processor.py     # LLM processor tests
├── test_main.py             # CLI tests
├── test_utils.py            # Utility function tests
├── test_visualizer.py       # Visualization tests
└── conftest.py              # Test configuration
```

### Writing Tests

- Use pytest for testing
- Write unit tests for individual functions
- Write integration tests for workflows
- Use fixtures for common test data
- Mock external dependencies

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_analyzer.py

# Run with coverage
pytest --cov=gca_analyzer --cov-report=html

# Run specific test function
pytest tests/test_analyzer.py::test_analyze_conversation
```

### Test Guidelines

- Aim for 100% code coverage
- Test both happy paths and error conditions
- Use descriptive test names
- Keep tests independent and isolated

## Documentation

### Documentation Structure

```
docs/
├── source/
│   ├── api_reference.rst    # API documentation
│   ├── examples/           # Usage examples
│   ├── faq.rst            # Frequently asked questions
│   ├── troubleshooting.rst # Common issues and solutions
│   └── index.rst          # Main documentation
├── build/                 # Generated documentation
└── Makefile              # Build scripts
```

### Building Documentation

```bash
cd docs
make html
```

### Documentation Guidelines

- Use reStructuredText (.rst) format
- Include code examples
- Keep explanations clear and concise
- Update documentation when changing APIs

## Issue Guidelines

### Creating Issues

When creating an issue, please:

1. Use a clear, descriptive title
2. Provide detailed description
3. Include steps to reproduce (for bugs)
4. Add relevant labels
5. Reference related issues or PRs

### Issue Templates

We provide templates for:
- Bug reports
- Feature requests
- Documentation improvements
- Performance issues

### Issue Labels

- `bug`: Something isn't working
- `enhancement`: New feature or improvement
- `documentation`: Documentation related
- `good first issue`: Good for newcomers
- `help wanted`: Looking for contributors
- `priority/high`: High priority issues
- `priority/medium`: Medium priority issues
- `priority/low`: Low priority issues

## Pull Request Guidelines

### Before Submitting

- [ ] Tests pass locally
- [ ] Code follows style guidelines
- [ ] Documentation is updated
- [ ] Commit messages are clear
- [ ] Branch is up to date with main

### PR Description

Include:
- Summary of changes
- Related issue number
- Testing notes
- Breaking changes (if any)
- Screenshots (if applicable)

### Review Process

1. Automated checks must pass
2. Code review by maintainers
3. Address feedback
4. Final approval and merge

### CI/CD Checks

All PRs must pass:
- Unit tests (Python 3.9, 3.10, 3.11, 3.12, 3.13)
- Linting (flake8, mypy)
- Documentation build
- Security scanning

## Release Process

### Version Numbering

We use Semantic Versioning (SemVer):
- `MAJOR.MINOR.PATCH`
- Major: Breaking changes
- Minor: New features (backward compatible)
- Patch: Bug fixes

### Release Steps

1. Update version in `__version__.py`
2. Update CHANGELOG.md
3. Create release PR
4. Tag release after merge
5. GitHub Actions handles PyPI publication

## Development Tips

### Debugging

- Use `pytest -s` to see print statements
- Use `pytest --pdb` to drop into debugger
- Add logging statements for complex logic

### Performance

- Profile code using `cProfile`
- Use appropriate data structures
- Consider memory usage for large datasets

### IDE Setup

Recommended settings for VS Code:
- Python extension
- Pylint or flake8 extension
- MyPy extension
- Git Lens extension

## Getting Help

### Communication Channels

- GitHub Issues: Bug reports and feature requests
- GitHub Discussions: General questions and ideas
- Email: For security issues or private matters

### Resources

- [Project Documentation](https://gca-analyzer.readthedocs.io)
- [Python Documentation](https://docs.python.org/3/)
- [Pytest Documentation](https://docs.pytest.org/)
- [Sphinx Documentation](https://www.sphinx-doc.org/)

## Recognition

Contributors are recognized in:
- CONTRIBUTORS.md file
- GitHub contributors page
- Release notes
- Documentation acknowledgments

Thank you for contributing to GCA Analyzer! Your help makes this project better for everyone.