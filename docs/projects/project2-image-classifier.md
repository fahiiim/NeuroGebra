# Project 2: Image Classifier — Neurogebra vs PyTorch

Build a digit classifier using the **MNIST-style dataset**. We'll train a model to recognize handwritten digits (0-9) comparing both frameworks side by side.

---

## 🎯 Goal

```
Input:  8x8 pixel image of a handwritten digit
Output: Which digit (0-9) it represents
```

---

## Step 1: Load the Dataset

=== "Neurogebra"
    ```python
    import numpy as np
    from sklearn.datasets import load_digits
    
    # Load sklearn's built-in digits dataset (8x8 images)
    digits = load_digits()
    X = digits.data          # (1797, 64) — flattened 8x8 images
    y = digits.target        # (1797,) — labels 0-9
    
    # Normalize pixel values to 0-1
    X = X / 16.0  # Max pixel value in this dataset is 16
    
    # Train/test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training: {X_train.shape[0]} images")
    print(f"Testing:  {X_test.shape[0]} images")
    print(f"Image shape: 8×8 = {X_train.shape[1]} features")
    print(f"Classes: {np.unique(y_train)}")
    ```

=== "PyTorch"
    ```python
    import numpy as np
    import torch
    import torch.nn as nn
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    
    # Load sklearn's built-in digits dataset (8x8 images)
    digits = load_digits()
    X = digits.data / 16.0
    y = digits.target
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Convert to PyTorch tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t  = torch.tensor(X_test, dtype=torch.float32)
    y_test_t  = torch.tensor(y_test, dtype=torch.long)
    
    print(f"Training: {X_train_t.shape[0]} images")
    print(f"Testing:  {X_test_t.shape[0]} images")
    print(f"Image shape: {X_train_t.shape[1]} features")
    ```

---

## Step 2: Visualize Sample Images

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_train[i].reshape(8, 8), cmap='gray')
    ax.set_title(f"Label: {y_train[i]}", fontsize=12)
    ax.axis('off')
plt.suptitle("Sample Training Images", fontsize=14)
plt.tight_layout()
plt.show()
```

---

## Step 3: Build the Neural Network

=== "Neurogebra"
    ```python
    from neurogebra.builders.model_builder import ModelBuilder
    
    builder = ModelBuilder()
    
    # Build a simple classifier
    model = builder.sequential([
        {"type": "dense", "units": 64, "activation": "relu", "input_shape": (64,)},
        {"type": "dense", "units": 32, "activation": "relu"},
        {"type": "dense", "units": 10, "activation": "softmax"}
    ])
    
    print(model.summary())
    ```
    
    **What this means (in plain English):**
    
    - **Layer 1**: Takes 64 inputs (pixels), outputs 64 values, applies ReLU
    - **Layer 2**: Takes 64 values, compresses to 32, applies ReLU
    - **Layer 3**: Takes 32 values, outputs 10 (one per digit), applies Softmax (probabilities)

=== "PyTorch"
    ```python
    import torch.nn as nn
    
    class DigitClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 10)
            )
        
        def forward(self, x):
            return self.network(x)
    
    model = nn.Module()
    model = DigitClassifier()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print(model)
    ```

!!! note "Key Difference"
    **Neurogebra**: Declarative style — describe what you want with a dictionary.
    **PyTorch**: Class-based — define a class with `__init__` and `forward` methods.
    
    For beginners, Neurogebra's approach is much more intuitive.

---

## Step 4: Implement Training from Scratch with Autograd

Since image classification requires a more hands-on approach, let's build a **complete neural network from scratch** using Neurogebra's autograd engine — and compare with PyTorch.

=== "Neurogebra"
    ```python
    from neurogebra.core.autograd import Value
    import random
    
    class Neuron:
        """Single neuron with weights, bias, and activation."""
        def __init__(self, n_inputs):
            self.w = [Value(random.uniform(-1, 1) * (2/n_inputs)**0.5) for _ in range(n_inputs)]
            self.b = Value(0.0)
        
        def __call__(self, x):
            # w · x + b
            activation = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
            return activation.relu()
        
        def parameters(self):
            return self.w + [self.b]
    
    class Layer:
        """Layer of neurons."""
        def __init__(self, n_inputs, n_outputs):
            self.neurons = [Neuron(n_inputs) for _ in range(n_outputs)]
        
        def __call__(self, x):
            return [n(x) for n in self.neurons]
        
        def parameters(self):
            return [p for n in self.neurons for p in n.parameters()]
    
    class MLP:
        """Multi-Layer Perceptron."""
        def __init__(self, sizes):
            self.layers = [Layer(sizes[i], sizes[i+1]) for i in range(len(sizes)-1)]
        
        def __call__(self, x):
            for layer in self.layers:
                x = layer(x)
            return x
        
        def parameters(self):
            return [p for layer in self.layers for p in layer.parameters()]
    
    # Create network: 64 → 32 → 10
    mlp = MLP([64, 32, 10])
    
    print(f"Total parameters: {len(mlp.parameters()):,}")
    print(f"Architecture: 64 → 32 → 10")
    ```

=== "PyTorch"
    ```python
    import torch
    import torch.nn as nn
    import torch.optim as optim
    
    model = DigitClassifier()  # Defined in Step 3
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"Architecture: 64 → 64 → 32 → 10")
    ```

---

## Step 5: Train the Network

=== "Neurogebra"
    ```python
    # Training loop with Neurogebra autograd
    learning_rate = 0.01
    batch_size = 32
    epochs = 20
    history = {"loss": [], "accuracy": []}
    
    for epoch in range(epochs):
        # Shuffle data
        indices = list(range(len(X_train)))
        random.shuffle(indices)
        
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        # Mini-batch training
        for start in range(0, min(len(X_train), 320), batch_size):  # Use subset for speed
            batch_idx = indices[start:start + batch_size]
            batch_loss = Value(0.0)
            
            for idx in batch_idx:
                x = [Value(float(v)) for v in X_train[idx]]
                
                # Forward pass
                scores = mlp(x)
                
                # Simple cross-entropy-like loss
                # Softmax + negative log likelihood
                target = int(y_train[idx])
                
                # Compute max for numerical stability
                max_score = max(s.data for s in scores)
                exp_scores = [(s - Value(max_score)).exp() for s in scores]
                sum_exp = sum(exp_scores)
                
                # Loss = -log(probability of correct class)
                prob_correct = exp_scores[target] / sum_exp
                sample_loss = -(prob_correct.log())
                batch_loss = batch_loss + sample_loss
                
                # Track accuracy
                predicted = max(range(10), key=lambda i: scores[i].data)
                correct += (predicted == target)
                total += 1
            
            # Average loss
            batch_loss = batch_loss / len(batch_idx)
            
            # Backward pass
            for p in mlp.parameters():
                p.grad = 0.0
            batch_loss.backward()
            
            # Update weights (SGD)
            for p in mlp.parameters():
                p.data -= learning_rate * p.grad
            
            epoch_loss += batch_loss.data
        
        accuracy = correct / total if total > 0 else 0
        history["loss"].append(epoch_loss)
        history["accuracy"].append(accuracy)
        
        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:>3d}/{epochs}: Loss = {epoch_loss:.4f}, Accuracy = {accuracy:.1%}")
    ```

=== "PyTorch"
    ```python
    # Training loop with PyTorch
    epochs = 20
    batch_size = 32
    history = {"loss": [], "accuracy": []}
    
    for epoch in range(epochs):
        # Shuffle data
        perm = torch.randperm(len(X_train_t))
        X_shuffled = X_train_t[perm]
        y_shuffled = y_train_t[perm]
        
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        # Mini-batch training
        for start in range(0, len(X_train_t), batch_size):
            X_batch = X_shuffled[start:start + batch_size]
            y_batch = y_shuffled[start:start + batch_size]
            
            # Forward pass
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # Track accuracy
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)
        
        accuracy = correct / total
        history["loss"].append(epoch_loss)
        history["accuracy"].append(accuracy)
        
        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:>3d}/{epochs}: Loss = {epoch_loss:.4f}, Accuracy = {accuracy:.1%}")
    ```

!!! note "Key Difference"
    **Neurogebra's autograd**: You see every computation step — the forward pass, softmax, 
    cross-entropy, and gradient updates are all explicit. This is **extremely educational** 
    because you understand exactly what backpropagation does.
    
    **PyTorch**: `loss.backward()` and `optimizer.step()` handle everything — faster but opaque.

---

## Step 6: Evaluate on Test Data

=== "Neurogebra"
    ```python
    # Test evaluation
    correct = 0
    total = 0
    predictions = []
    
    for i in range(len(X_test)):
        x = [Value(float(v)) for v in X_test[i]]
        scores = mlp(x)
        predicted = max(range(10), key=lambda j: scores[j].data)
        predictions.append(predicted)
        
        if predicted == y_test[i]:
            correct += 1
        total += 1
    
    test_accuracy = correct / total
    print(f"\n=== Test Results (Neurogebra) ===")
    print(f"Test Accuracy: {test_accuracy:.1%}")
    print(f"Correct: {correct}/{total}")
    ```

=== "PyTorch"
    ```python
    # Test evaluation
    with torch.no_grad():
        outputs = model(X_test_t)
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == y_test_t).sum().item()
        total = y_test_t.size(0)
    
    test_accuracy = correct / total
    print(f"\n=== Test Results (PyTorch) ===")
    print(f"Test Accuracy: {test_accuracy:.1%}")
    print(f"Correct: {correct}/{total}")
    ```

---

## Step 7: Confusion Matrix & Analysis

```python
import matplotlib.pyplot as plt
import numpy as np

# Using Neurogebra predictions (or PyTorch — same visualization code)
predictions = np.array(predictions)  # from Neurogebra evaluation above

# Confusion matrix
confusion = np.zeros((10, 10), dtype=int)
for true, pred in zip(y_test, predictions):
    confusion[true][pred] += 1

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot confusion matrix
im = axes[0].imshow(confusion, cmap='Blues')
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")
axes[0].set_title("Confusion Matrix")
axes[0].set_xticks(range(10))
axes[0].set_yticks(range(10))
for i in range(10):
    for j in range(10):
        axes[0].text(j, i, str(confusion[i][j]), 
                    ha='center', va='center', fontsize=8)
plt.colorbar(im, ax=axes[0])

# Plot per-class accuracy
class_acc = confusion.diagonal() / confusion.sum(axis=1)
axes[1].bar(range(10), class_acc, color='steelblue')
axes[1].set_xlabel("Digit")
axes[1].set_ylabel("Accuracy")
axes[1].set_title("Per-Digit Accuracy")
axes[1].set_xticks(range(10))
axes[1].set_ylim(0, 1.1)
for i, acc in enumerate(class_acc):
    axes[1].text(i, acc + 0.02, f"{acc:.0%}", ha='center', fontsize=9)

plt.tight_layout()
plt.show()
```

---

## Step 8: Visualize Predictions

```python
# Show some predictions
fig, axes = plt.subplots(2, 5, figsize=(15, 6))

for i, ax in enumerate(axes.flat):
    ax.imshow(X_test[i].reshape(8, 8), cmap='gray')
    true_label = y_test[i]
    pred_label = predictions[i]
    
    color = 'green' if true_label == pred_label else 'red'
    ax.set_title(f"True: {true_label}, Pred: {pred_label}", 
                color=color, fontsize=11, fontweight='bold')
    ax.axis('off')

plt.suptitle("Test Predictions (Green=Correct, Red=Wrong)", fontsize=14)
plt.tight_layout()
plt.show()
```

---

## Understanding What the Network Learned (Neurogebra Bonus)

```python
from neurogebra import MathForge

forge = MathForge()

# Understand the math behind each component
print("=== What You Built ===\n")

# 1. ReLU activation explanation
relu = forge.get("relu")
print(f"Activation: ReLU = {relu.symbolic_expr}")
print(f"  Gradient: {relu.gradient('x').symbolic_expr}")
print(f"  Purpose: Adds non-linearity, kills negative values\n")

# 2. Understanding the loss
print("Cross-Entropy Loss:")
print("  L = -log(P(correct class))")
print("  If model is confident and correct → small loss")
print("  If model is wrong → large loss\n")

# 3. Understanding softmax
print("Softmax converts scores to probabilities:")
print("  P(class_i) = exp(score_i) / sum(exp(all_scores))")
print("  All probabilities sum to 1.0\n")

# 4. Network architecture
print("Your network architecture:")
print("  Input (64 pixels) → Dense(32) + ReLU → Dense(10) + Softmax → Prediction")
print(f"  Total parameters: {len(mlp.parameters()):,}")
```

---

## Side-by-Side Summary

| Aspect | Neurogebra | PyTorch |
|--------|------------|---------|
| **Data prep** | NumPy only | NumPy → Tensor conversion |
| **Model definition** | Dict-based or class-based | Class-based (nn.Module) |
| **Training loop** | Explicit computation visible | Abstracted behind `.backward()` |
| **Evaluation** | Manual iteration | Vectorized with `torch.no_grad()` |
| **Speed** | Slower (educational) | Fast (optimized C++) |
| **Understanding** | ✅ See every gradient | ❌ Gradients are hidden |
| **Best for** | Learning ML concepts | Production ML |

---

## What You Learned

1. How **image classification** works (pixels → features → class)
2. How **neural networks** process information through layers
3. How **softmax** converts scores to probabilities
4. How **cross-entropy loss** measures classification error
5. How **backpropagation** flows through a multi-layer network
6. The trade-off between **educational clarity** and **production speed**

---

**Next Project:** [Neural Network from Scratch →](project3-neural-network.md)
