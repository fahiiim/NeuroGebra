# Project 1: Linear Regression — Neurogebra vs PyTorch

Build a complete linear regression model to predict house prices. We'll implement the **exact same thing** in both Neurogebra and PyTorch so you can see the differences.

---

## 🎯 Goal

Given house sizes (sq ft), predict the price ($).

```
Input: House size (1000, 1500, 2000, ...)
Output: Predicted price ($200k, $300k, $400k, ...)
```

---

## Step 1: Create the Dataset

=== "Neurogebra"
    ```python
    import numpy as np

    # Generate synthetic house data
    np.random.seed(42)
    
    # Features: house size in sq ft (scaled to 0-1)
    X = np.random.uniform(500, 3500, 100)
    X_normalized = (X - X.mean()) / X.std()
    
    # Target: price in $1000s (true relationship: price = 200 * size + 50 + noise)
    y_true = 200 * X_normalized + 50 + np.random.normal(0, 10, 100)
    
    # Train/test split
    X_train, X_test = X_normalized[:80], X_normalized[80:]
    y_train, y_test = y_true[:80], y_true[80:]
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples:     {len(X_test)}")
    print(f"X range: [{X_normalized.min():.2f}, {X_normalized.max():.2f}]")
    print(f"y range: [{y_true.min():.1f}, {y_true.max():.1f}]")
    ```

=== "PyTorch"
    ```python
    import numpy as np
    import torch
    
    # Generate synthetic house data
    np.random.seed(42)
    
    X = np.random.uniform(500, 3500, 100)
    X_normalized = (X - X.mean()) / X.std()
    y_true = 200 * X_normalized + 50 + np.random.normal(0, 10, 100)
    
    X_train, X_test = X_normalized[:80], X_normalized[80:]
    y_train, y_test = y_true[:80], y_true[80:]
    
    # PyTorch needs tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples:     {len(X_test)}")
    ```

!!! note "Key Difference #1"
    Neurogebra works directly with **NumPy arrays** — no conversion needed.
    PyTorch requires converting to **torch.Tensor** objects first.

---

## Step 2: Define the Model

=== "Neurogebra"
    ```python
    from neurogebra import Expression
    
    # Define model: y = w * x + b
    model = Expression(
        "house_price",
        "w * x + b",
        params={"w": 0.0, "b": 0.0},
        trainable_params=["w", "b"]
    )
    
    print(f"Model: {model.symbolic_expr}")
    print(f"Parameters: w={model.params['w']}, b={model.params['b']}")
    ```

=== "PyTorch"
    ```python
    import torch.nn as nn
    
    # Define model
    model = nn.Linear(1, 1)  # 1 input → 1 output
    
    # Initialize weights to 0 (to match Neurogebra)
    nn.init.zeros_(model.weight)
    nn.init.zeros_(model.bias)
    
    print(f"Model: {model}")
    print(f"Parameters: w={model.weight.item():.1f}, b={model.bias.item():.1f}")
    ```

!!! note "Key Difference #2"
    Neurogebra: You write the **actual math formula** `w * x + b`. You can read and understand it.
    PyTorch: You specify `nn.Linear(1, 1)` — the math is hidden inside the module.

---

## Step 3: Set Up Training

=== "Neurogebra"
    ```python
    from neurogebra.core.trainer import Trainer
    
    # Create trainer with Adam optimizer
    trainer = Trainer(
        model,
        learning_rate=0.1,
        optimizer="adam"
    )
    
    print("Optimizer: Adam")
    print(f"Learning rate: {trainer.learning_rate}")
    ```

=== "PyTorch"
    ```python
    import torch.optim as optim
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.1)
    
    print("Loss: MSE")
    print("Optimizer: Adam")
    print(f"Learning rate: 0.1")
    ```

!!! note "Key Difference #3"
    Neurogebra: One `Trainer` object handles everything.
    PyTorch: You need separate `criterion` (loss) and `optimizer` objects.

---

## Step 4: Train the Model

=== "Neurogebra"
    ```python
    # Train!
    history = trainer.fit(
        X_train,
        y_train,
        epochs=200,
        loss_fn="mse",
        verbose=True
    )
    
    # Check learned parameters
    print(f"\nLearned: w = {model.params['w']:.4f} (true: 200)")
    print(f"Learned: b = {model.params['b']:.4f} (true: 50)")
    print(f"Final loss: {history['loss'][-1]:.4f}")
    ```

=== "PyTorch"
    ```python
    # Training loop
    history = {"loss": []}
    
    for epoch in range(200):
        # Forward pass
        predictions = model(X_train_t)
        loss = criterion(predictions, y_train_t)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Record history
        history["loss"].append(loss.item())
        
        # Print progress
        if epoch % 20 == 0 or epoch == 199:
            print(f"Epoch {epoch:>4d}/200: Loss = {loss.item():.6f}")
    
    print(f"\nLearned: w = {model.weight.item():.4f} (true: 200)")
    print(f"Learned: b = {model.bias.item():.4f} (true: 50)")
    print(f"Final loss: {history['loss'][-1]:.4f}")
    ```

!!! note "Key Difference #4"
    **Neurogebra:** One line — `trainer.fit(X, y, epochs=200)`. Everything is handled for you.
    
    **PyTorch:** You write the full training loop manually:
    
    1. Forward pass
    2. Compute loss
    3. Zero gradients
    4. Backward pass
    5. Optimizer step
    6. Record metrics
    
    This gives PyTorch more flexibility, but Neurogebra is **much simpler to learn**.

---

## Step 5: Evaluate on Test Data

=== "Neurogebra"
    ```python
    # Predict on test data
    y_pred = np.array([model.eval(x=float(xi)) for xi in X_test])
    
    # Calculate test MSE
    test_mse = np.mean((y_pred - y_test) ** 2)
    
    # Calculate R² score
    ss_res = np.sum((y_test - y_pred) ** 2)
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    r2 = 1 - ss_res / ss_tot
    
    print(f"Test MSE:  {test_mse:.4f}")
    print(f"R² Score:  {r2:.4f}")
    print(f"\nSample predictions:")
    for i in range(5):
        print(f"  x={X_test[i]:.2f} → predicted={y_pred[i]:.1f}, actual={y_test[i]:.1f}")
    ```

=== "PyTorch"
    ```python
    # Predict on test data
    with torch.no_grad():
        y_pred_t = model(X_test_t)
    y_pred = y_pred_t.numpy().flatten()
    
    # Calculate test MSE
    test_mse = np.mean((y_pred - y_test) ** 2)
    
    # Calculate R² score
    ss_res = np.sum((y_test - y_pred) ** 2)
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    r2 = 1 - ss_res / ss_tot
    
    print(f"Test MSE:  {test_mse:.4f}")
    print(f"R² Score:  {r2:.4f}")
    print(f"\nSample predictions:")
    for i in range(5):
        print(f"  x={X_test[i]:.2f} → predicted={y_pred[i]:.1f}, actual={y_test[i]:.1f}")
    ```

!!! note "Key Difference #5"
    PyTorch requires `torch.no_grad()` context and converting back to NumPy.
    Neurogebra evaluates directly — no special context needed.

---

## Step 6: Visualize Results

=== "Neurogebra"
    ```python
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Training Loss
    axes[0].plot(history["loss"], linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss (MSE)")
    axes[0].set_title("Training Loss — Neurogebra")
    axes[0].set_yscale("log")
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Predictions vs Actual
    x_line = np.linspace(X_test.min(), X_test.max(), 100)
    y_line = np.array([model.eval(x=float(xi)) for xi in x_line])
    
    axes[1].scatter(X_test, y_test, alpha=0.7, label="Actual", color="blue")
    axes[1].plot(x_line, y_line, color="red", linewidth=2, label="Predicted")
    axes[1].set_xlabel("House Size (normalized)")
    axes[1].set_ylabel("Price ($1000s)")
    axes[1].set_title("Predictions — Neurogebra")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    ```

=== "PyTorch"
    ```python
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Training Loss
    axes[0].plot(history["loss"], linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss (MSE)")
    axes[0].set_title("Training Loss — PyTorch")
    axes[0].set_yscale("log")
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Predictions vs Actual
    x_line = np.linspace(X_test.min(), X_test.max(), 100)
    x_line_t = torch.tensor(x_line, dtype=torch.float32).unsqueeze(1)
    with torch.no_grad():
        y_line = model(x_line_t).numpy().flatten()
    
    axes[1].scatter(X_test, y_test, alpha=0.7, label="Actual", color="blue")
    axes[1].plot(x_line, y_line, color="red", linewidth=2, label="Predicted")
    axes[1].set_xlabel("House Size (normalized)")
    axes[1].set_ylabel("Price ($1000s)")
    axes[1].set_title("Predictions — PyTorch")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    ```

---

## Step 7: Understand What You Built (Neurogebra Bonus)

This is where Neurogebra really shines — **understanding and introspection**:

```python
from neurogebra import MathForge

forge = MathForge()

# Explain the model
print("=== Your Model ===")
print(f"Formula: y = {model.params['w']:.2f} * x + {model.params['b']:.2f}")
print(f"Symbolic: {model.symbolic_expr}")
print()

# Examine the gradient
grad = model.gradient("x")
print(f"Gradient dy/dx = {grad.symbolic_expr}")
print(f"This means: for every 1 unit increase in x, y increases by {model.params['w']:.2f}")
print()

# Examine the loss function
mse = forge.get("mse")
print(f"Loss function: {mse.symbolic_expr}")
print(f"Loss gradient: {mse.gradient('y_pred').symbolic_expr}")
print()

# Understand activations used
print("=== Available Activations You Could Add ===")
for name in ["relu", "sigmoid", "tanh"]:
    act = forge.get(name)
    print(f"  {name}: {act.symbolic_expr}")
```

!!! tip "Educational Value"
    In PyTorch, you can't easily inspect formulas or see gradients symbolically.
    Neurogebra shows you **exactly what's happening** at every step.

---

## Full Side-by-Side Comparison

| Aspect | Neurogebra | PyTorch |
|--------|------------|---------|
| **Data format** | NumPy arrays | torch.Tensor |
| **Model definition** | Math formula: `"w*x + b"` | `nn.Linear(1, 1)` |
| **Lines for training** | 1 (`trainer.fit(...)`) | ~10 (manual loop) |
| **See the formula** | ✅ `model.symbolic_expr` | ❌ Hidden in module |
| **See gradients** | ✅ `model.gradient("x")` | ❌ Only numerical values |
| **Total lines of code** | ~15 | ~35 |
| **Learning curve** | Gentle | Steep |
| **Production ready** | Educational | Production |
| **GPU support** | Via bridges | Native |

---

## What You Learned

1. **Linear regression** fits `y = wx + b` to data
2. **Training** = adjusting w and b to minimize loss
3. **MSE loss** measures average squared error
4. **Adam optimizer** efficiently updates parameters
5. Neurogebra lets you **see and understand** every step
6. PyTorch gives **more control** but requires more code

---

**Next Project:** [Image Classifier →](project2-image-classifier.md)
