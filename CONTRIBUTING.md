# Contributing to Neurogebra

Thank you for your interest in contributing to Neurogebra! This document provides guidelines and information about contributing to this project.

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or higher
- Git
- A GitHub account

### Setting Up Development Environment

1. **Fork the repository** on GitHub

2. **Clone your fork:**
   ```bash
   git clone https://github.com/fahiiim/NeuroGebra.git
   cd NeuroGebra
   ```

3. **Create a virtual environment:**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   # or
   source venv/bin/activate  # macOS/Linux
   ```

4. **Install in development mode:**
   ```bash
   pip install -e ".[dev]"
   ```

5. **Install pre-commit hooks:**
   ```bash
   pre-commit install
   ```

## 📐 Development Workflow

### Branch Naming

- `feature/description` - New features
- `fix/description` - Bug fixes
- `docs/description` - Documentation changes
- `refactor/description` - Code refactoring

### Making Changes

1. Create a new branch:
   ```bash
   git checkout -b feature/my-new-feature
   ```

2. Make your changes

3. Run tests:
   ```bash
   pytest tests/ -v
   ```

4. Check code quality:
   ```bash
   black src/ tests/
   ruff check src/ tests/
   mypy src/neurogebra
   ```

5. Commit your changes:
   ```bash
   git add .
   git commit -m "Add: descriptive message"
   ```

6. Push and create a Pull Request:
   ```bash
   git push origin feature/my-new-feature
   ```

## 📝 Code Style

- **Formatting**: We use [Black](https://black.readthedocs.io/) with line length 88
- **Linting**: We use [Ruff](https://docs.astral.sh/ruff/)
- **Type Hints**: Use type hints for function signatures
- **Docstrings**: Google style docstrings for all public functions/classes

### Example

```python
def my_function(param1: str, param2: int = 10) -> bool:
    """
    Brief description.

    Longer description if needed.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value

    Raises:
        ValueError: When something goes wrong

    Examples:
        >>> my_function("hello", 5)
        True
    """
    ...
```

## 🧪 Testing

- Write tests for all new features
- Maintain >80% code coverage
- Use pytest fixtures for shared setup
- Test edge cases

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=neurogebra --cov-report=term-missing

# Run specific test file
pytest tests/test_core/test_expression.py -v
```

## 📦 Adding New Expressions

To add a new expression to the repository:

1. Choose the appropriate module (e.g., `src/neurogebra/repository/activations.py`)

2. Add the expression:
   ```python
   expressions["my_expression"] = Expression(
       name="my_expression",
       symbolic_expr="x**2 + 1",
       metadata={
           "category": "activation",
           "description": "Clear description",
           "usage": "When to use this",
           "pros": ["Advantage 1"],
           "cons": ["Disadvantage 1"],
       },
   )
   ```

3. Write tests in the corresponding test file

4. Update the CHANGELOG

## 🐛 Bug Reports

Please use the [GitHub issue tracker](https://github.com/fahiiim/NeuroGebra/issues) with the bug report template.

Include:
- Steps to reproduce
- Expected behavior
- Actual behavior
- Python version and OS
- Neurogebra version

## 💡 Feature Requests

Use the [GitHub issue tracker](https://github.com/fahiiim/NeuroGebra/issues) with the feature request template.

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

## 🙏 Thank You!

Every contribution makes Neurogebra better for everyone. Thank you!
