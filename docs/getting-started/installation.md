# Installation

## Prerequisites

Before installing Neurogebra, make sure you have:

- **Python 3.8 or higher** installed on your computer
- **pip** (Python's package manager — comes with Python)

!!! info "Don't have Python?"
    Download Python from [python.org](https://www.python.org/downloads/). During installation on Windows, check **"Add Python to PATH"**.

---

## Step 1: Install Neurogebra

Open your terminal (Command Prompt on Windows, Terminal on Mac/Linux) and type:

```bash
pip install neurogebra
```

That's it! Neurogebra is now installed.

---

## Step 2: Verify Installation

Let's make sure it worked. Open a Python shell:

```bash
python
```

Then type:

```python
import neurogebra
print(neurogebra.__version__)
```

You should see the version number printed (e.g., `0.1.1`). 

---

## Optional Extras

Neurogebra has optional features you can install:

### Visualization Tools
```bash
pip install neurogebra[viz]
```
Adds plotting and visualization (requires matplotlib and plotly).

### Performance Boost
```bash
pip install neurogebra[fast]
```
Adds Numba JIT compilation for faster numerical evaluation.

### Framework Bridges
```bash
pip install neurogebra[frameworks]
```
Adds PyTorch, TensorFlow, and JAX integration.

### Everything
```bash
pip install neurogebra[all]
```
Installs all optional dependencies at once.

---

## Troubleshooting

!!! warning "Common Issues"

    **`pip` not found?**
    
    Try `pip3 install neurogebra` or `python -m pip install neurogebra`.

    **Permission denied?**
    
    Try `pip install --user neurogebra` or use a virtual environment.

    **Old Python version?**
    
    Neurogebra requires Python 3.8+. Check with `python --version`.

---

## Using a Virtual Environment (Recommended)

A virtual environment keeps your project dependencies isolated:

=== "Windows"

    ```bash
    python -m venv myenv
    myenv\Scripts\activate
    pip install neurogebra
    ```

=== "Mac/Linux"

    ```bash
    python3 -m venv myenv
    source myenv/bin/activate
    pip install neurogebra
    ```

---

**Next:** [Your First Program →](first-program.md)
