# Project 3: Neural Network from Scratch — Neurogebra vs PyTorch

Build a **complete neural network from scratch** to solve a real classification problem — understanding every single component. This is the ultimate learning project.

---

## 🎯 Goal

Create a neural network that classifies points into spiral categories — a problem that **cannot** be solved with linear models.

```
Input:  (x, y) coordinate on a 2D plane
Output: Which spiral (0, 1, or 2) the point belongs to
```

This is a classic non-linear classification problem that truly tests neural network capability.

---

## Step 1: Generate the Spiral Dataset

```python
import numpy as np
import matplotlib.pyplot as plt

def generate_spirals(n_points=100, n_classes=3, noise=0.1):
    """Generate spiral dataset — a classic non-linear classification problem."""
    X = np.zeros((n_points * n_classes, 2))
    y = np.zeros(n_points * n_classes, dtype=int)
    
    for class_idx in range(n_classes):
        start = n_points * class_idx
        end = n_points * (class_idx + 1)
        
        r = np.linspace(0.0, 1.0, n_points)
        theta = np.linspace(
            class_idx * 4, (class_idx + 1) * 4, n_points
        ) + np.random.randn(n_points) * noise
        
        X[start:end, 0] = r * np.sin(theta)
        X[start:end, 1] = r * np.cos(theta)
        y[start:end] = class_idx
    
    return X, y

# Generate data
np.random.seed(42)
X, y = generate_spirals(n_points=100, n_classes=3, noise=0.15)

# Train/test split
indices = np.random.permutation(len(X))
split = int(0.8 * len(X))
X_train, X_test = X[indices[:split]], X[indices[split:]]
y_train, y_test = y[indices[:split]], y[indices[split:]]

# Visualize
plt.figure(figsize=(8, 8))
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
for i in range(3):
    mask = y_train == i
    plt.scatter(X_train[mask, 0], X_train[mask, 1], 
               c=colors[i], label=f'Class {i}', alpha=0.7, s=30)
plt.title("Spiral Dataset — Can Your Neural Network Solve This?", fontsize=14)
plt.xlabel("x₁")
plt.ylabel("x₂")
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.show()

print(f"Training: {len(X_train)} samples")
print(f"Testing:  {len(X_test)} samples")
print(f"Classes:  3 spirals")
print(f"Features: 2 (x, y coordinates)")
```

!!! info "Why Spirals?"
    A linear model draws straight lines to separate classes. Spirals are **intertwined** — 
    you need a neural network with non-linear activations to separate them. This proves 
    your network actually works!

---

## Step 2: Build the Neural Network

### Neurogebra — Every Component Visible

```python
from neurogebra.core.autograd import Value
import random

random.seed(42)

class Neuron:
    """A single neuron: computes w·x + b, then applies activation."""
    
    def __init__(self, n_inputs, activation='relu'):
        # Xavier initialization for better training
        limit = (6 / (n_inputs + 1)) ** 0.5
        self.w = [Value(random.uniform(-limit, limit)) for _ in range(n_inputs)]
        self.b = Value(0.0)
        self.activation = activation
    
    def __call__(self, x):
        # Linear: w·x + b
        raw = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        
        # Activation
        if self.activation == 'relu':
            return raw.relu()
        elif self.activation == 'tanh':
            return raw.tanh()
        elif self.activation == 'linear':
            return raw
        return raw.relu()
    
    def parameters(self):
        return self.w + [self.b]


class Layer:
    """A layer of neurons."""
    
    def __init__(self, n_inputs, n_outputs, activation='relu'):
        self.neurons = [Neuron(n_inputs, activation) for _ in range(n_outputs)]
    
    def __call__(self, x):
        return [neuron(x) for neuron in self.neurons]
    
    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]


class NeuralNetwork:
    """Complete neural network with multiple layers."""
    
    def __init__(self, layer_sizes, activations=None):
        """
        Args:
            layer_sizes: [input_size, hidden1, hidden2, ..., output_size]
            activations: activation for each layer (default: relu, linear for last)
        """
        if activations is None:
            activations = ['relu'] * (len(layer_sizes) - 2) + ['linear']
        
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            self.layers.append(
                Layer(layer_sizes[i], layer_sizes[i+1], activations[i])
            )
        
        n_params = len(self.parameters())
        print(f"Neural Network created:")
        print(f"  Architecture: {' → '.join(map(str, layer_sizes))}")
        print(f"  Activations:  {activations}")
        print(f"  Parameters:   {n_params:,}")
    
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


# Create neural network: 2 inputs → 16 hidden → 16 hidden → 3 outputs
nn_neuro = NeuralNetwork(
    layer_sizes=[2, 16, 16, 3],
    activations=['relu', 'relu', 'linear']
)
```

### PyTorch — The Standard Way

```python
import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)

class SpiralNet(nn.Module):
    """Neural network for spiral classification."""
    
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 3)
        )
    
    def forward(self, x):
        return self.network(x)

nn_torch = SpiralNet()
total_params = sum(p.numel() for p in nn_torch.parameters())
print(f"\nPyTorch Neural Network:")
print(f"  Architecture: 2 → 16 → 16 → 3")
print(f"  Parameters:   {total_params:,}")
```

!!! note "Architecture Comparison"
    Both networks have the same structure: **2 → 16 → 16 → 3**.  
    Neurogebra: ~35 lines to define (you understand every line).  
    PyTorch: ~15 lines (uses pre-built components).

---

## Step 3: Implement Softmax & Cross-Entropy Loss

=== "Neurogebra"
    ```python
    def softmax_cross_entropy(scores, target):
        """
        Compute softmax probabilities and cross-entropy loss.
        
        This is EXACTLY what PyTorch's CrossEntropyLoss does internally,
        but here you can see every step!
        
        Steps:
        1. Find max score (for numerical stability)
        2. Compute exp(score - max) for each class
        3. Normalize to get probabilities (softmax)
        4. Loss = -log(probability of correct class)
        """
        # Step 1: Numerical stability — subtract max
        max_score = max(s.data for s in scores)
        
        # Step 2: Compute exponentials
        exp_scores = [(s - Value(max_score)).exp() for s in scores]
        
        # Step 3: Softmax — normalize to probabilities
        sum_exp = sum(exp_scores)
        probabilities = [e / sum_exp for e in exp_scores]
        
        # Step 4: Cross-entropy loss
        # -log(P(correct class))
        loss = -(probabilities[target].log())
        
        return loss, probabilities
    
    # Example: verify it works
    dummy_scores = [Value(2.0), Value(1.0), Value(0.1)]  # Raw scores for 3 classes
    loss, probs = softmax_cross_entropy(dummy_scores, target=0)  # Target = class 0
    
    print("=== Softmax + Cross-Entropy Demo ===")
    print(f"Raw scores: [{', '.join(f'{s.data:.1f}' for s in dummy_scores)}]")
    print(f"Probabilities: [{', '.join(f'{p.data:.3f}' for p in probs)}]")
    print(f"Sum of probs: {sum(p.data for p in probs):.3f} (should be 1.0)")
    print(f"Loss (target=class 0): {loss.data:.4f}")
    print(f"(Lower loss = higher confidence in correct class)")
    ```

=== "PyTorch"
    ```python
    # PyTorch does this in ONE line:
    criterion = nn.CrossEntropyLoss()
    
    # That single line contains all the softmax + cross-entropy math!
    # Convenient, but you don't see the internals.
    
    # Demo:
    dummy_scores = torch.tensor([[2.0, 1.0, 0.1]])
    dummy_target = torch.tensor([0])
    loss = criterion(dummy_scores, dummy_target)
    
    probs = torch.softmax(dummy_scores, dim=1)
    print(f"Raw scores:    {dummy_scores.numpy()}")
    print(f"Probabilities: {probs.numpy()}")
    print(f"Loss:          {loss.item():.4f}")
    ```

---

## Step 4: Train Both Networks

### Neurogebra Training

```python
# ═══════════════════════════════════════════════
# NEUROGEBRA TRAINING — See every gradient flow!
# ═══════════════════════════════════════════════

learning_rate = 0.05
epochs = 30
batch_size = 16
neuro_history = {"loss": [], "accuracy": []}

print("Training Neurogebra Neural Network...")
print("=" * 50)

for epoch in range(epochs):
    # Shuffle training data
    perm = np.random.permutation(len(X_train))
    
    epoch_loss = 0.0
    correct = 0
    total = 0
    
    # Mini-batch training
    for start in range(0, len(X_train), batch_size):
        batch_idx = perm[start:start + batch_size]
        batch_loss = Value(0.0)
        
        for idx in batch_idx:
            # Convert input to Value objects
            x_input = [Value(float(X_train[idx, 0])),
                       Value(float(X_train[idx, 1]))]
            target = int(y_train[idx])
            
            # Forward pass — compute scores
            scores = nn_neuro(x_input)
            
            # Compute loss
            loss, probs = softmax_cross_entropy(scores, target)
            batch_loss = batch_loss + loss
            
            # Track accuracy
            predicted = max(range(3), key=lambda i: scores[i].data)
            correct += (predicted == target)
            total += 1
        
        # Average batch loss
        batch_loss = batch_loss / len(batch_idx)
        
        # ==============================
        # BACKWARD PASS — The Magic Part
        # ==============================
        
        # 1. Zero all gradients
        for p in nn_neuro.parameters():
            p.grad = 0.0
        
        # 2. Backpropagate — compute dLoss/dParam for every parameter
        batch_loss.backward()
        
        # 3. Update every parameter: param = param - lr * gradient
        for p in nn_neuro.parameters():
            p.data -= learning_rate * p.grad
        
        epoch_loss += batch_loss.data
    
    accuracy = correct / total
    neuro_history["loss"].append(epoch_loss)
    neuro_history["accuracy"].append(accuracy)
    
    if epoch % 5 == 0 or epoch == epochs - 1:
        print(f"  Epoch {epoch:>3d}/{epochs}: Loss = {epoch_loss:.4f}, Accuracy = {accuracy:.1%}")

print(f"\nFinal Training Accuracy: {neuro_history['accuracy'][-1]:.1%}")
```

### PyTorch Training

```python
# ═══════════════════════════════════════════════
# PYTORCH TRAINING — Fast and optimized
# ═══════════════════════════════════════════════

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)

optimizer = optim.Adam(nn_torch.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

torch_history = {"loss": [], "accuracy": []}

print("Training PyTorch Neural Network...")
print("=" * 50)

for epoch in range(epochs):
    perm = torch.randperm(len(X_train_t))
    
    epoch_loss = 0.0
    correct = 0
    total = 0
    
    for start in range(0, len(X_train_t), batch_size):
        X_batch = X_train_t[perm[start:start + batch_size]]
        y_batch = y_train_t[perm[start:start + batch_size]]
        
        # Forward pass
        outputs = nn_torch(X_batch)
        loss = criterion(outputs, y_batch)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == y_batch).sum().item()
        total += y_batch.size(0)
    
    accuracy = correct / total
    torch_history["loss"].append(epoch_loss)
    torch_history["accuracy"].append(accuracy)
    
    if epoch % 5 == 0 or epoch == epochs - 1:
        print(f"  Epoch {epoch:>3d}/{epochs}: Loss = {epoch_loss:.4f}, Accuracy = {accuracy:.1%}")

print(f"\nFinal Training Accuracy: {torch_history['accuracy'][-1]:.1%}")
```

---

## Step 5: Compare Training Progress

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss comparison
axes[0].plot(neuro_history["loss"], label="Neurogebra", linewidth=2, color='#FF6B6B')
axes[0].plot(torch_history["loss"], label="PyTorch", linewidth=2, color='#4ECDC4')
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].set_title("Training Loss Comparison")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Accuracy comparison
axes[1].plot(neuro_history["accuracy"], label="Neurogebra", linewidth=2, color='#FF6B6B')
axes[1].plot(torch_history["accuracy"], label="PyTorch", linewidth=2, color='#4ECDC4')
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].set_title("Training Accuracy Comparison")
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].set_ylim(0, 1.05)

plt.tight_layout()
plt.show()
```

---

## Step 6: Evaluate on Test Data

=== "Neurogebra"
    ```python
    # Neurogebra test evaluation
    neuro_correct = 0
    neuro_predictions = []
    
    for i in range(len(X_test)):
        x_input = [Value(float(X_test[i, 0])), Value(float(X_test[i, 1]))]
        scores = nn_neuro(x_input)
        predicted = max(range(3), key=lambda j: scores[j].data)
        neuro_predictions.append(predicted)
        neuro_correct += (predicted == y_test[i])
    
    neuro_test_acc = neuro_correct / len(X_test)
    print(f"Neurogebra Test Accuracy: {neuro_test_acc:.1%}")
    ```

=== "PyTorch"
    ```python
    # PyTorch test evaluation
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)
    
    with torch.no_grad():
        outputs = nn_torch(X_test_t)
        _, torch_predictions = torch.max(outputs, 1)
    
    torch_test_acc = (torch_predictions == y_test_t).float().mean().item()
    torch_predictions = torch_predictions.numpy()
    print(f"PyTorch Test Accuracy: {torch_test_acc:.1%}")
    ```

---

## Step 7: Visualize Decision Boundaries

This is the most satisfying visualization — see how the network **learned to separate spirals**:

```python
def plot_decision_boundary(predict_fn, X, y, title, ax):
    """Plot the decision boundary of a classifier."""
    h = 0.02  # Step size
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h),
        np.arange(y_min, y_max, h)
    )
    
    # Predict on grid
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = np.array([predict_fn(p) for p in grid_points])
    Z = Z.reshape(xx.shape)
    
    # Plot
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    for i in range(3):
        mask = y == i
        ax.scatter(X[mask, 0], X[mask, 1], c=colors[i], 
                  label=f'Class {i}', edgecolors='k', linewidth=0.5, s=30)
    
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(loc='upper right')
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")

# Prediction functions
def neuro_predict(point):
    x = [Value(float(point[0])), Value(float(point[1]))]
    scores = nn_neuro(x)
    return max(range(3), key=lambda i: scores[i].data)

def torch_predict(point):
    x = torch.tensor(point, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        scores = nn_torch(x)
    return torch.argmax(scores).item()

# Plot both decision boundaries
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

plot_decision_boundary(neuro_predict, X_test, y_test, 
                       f"Neurogebra (Acc: {neuro_test_acc:.0%})", axes[0])
plot_decision_boundary(torch_predict, X_test, y_test,
                       f"PyTorch (Acc: {torch_test_acc:.0%})", axes[1])

plt.suptitle("Decision Boundaries — Neural Network Learned to Separate Spirals!", 
             fontsize=15, fontweight='bold')
plt.tight_layout()
plt.show()
```

---

## Step 8: Deep Dive — What Did the Network Learn?

### Inspect Neurogebra's Parameters

```python
from neurogebra import MathForge

forge = MathForge()

print("=" * 60)
print("INSPECTING WHAT THE NETWORK LEARNED")
print("=" * 60)

# 1. Layer-by-layer parameter statistics
for i, layer in enumerate(nn_neuro.layers):
    weights = [p.data for neuron in layer.neurons for p in neuron.w]
    biases = [neuron.b.data for neuron in layer.neurons]
    
    print(f"\nLayer {i+1}:")
    print(f"  Neurons:      {len(layer.neurons)}")
    print(f"  Weights:      mean={np.mean(weights):.4f}, std={np.std(weights):.4f}")
    print(f"  Weight range: [{min(weights):.4f}, {max(weights):.4f}]")
    print(f"  Biases:       mean={np.mean(biases):.4f}, std={np.std(biases):.4f}")

# 2. Understanding with MathForge
print("\n" + "=" * 60)
print("THE MATH BEHIND YOUR NETWORK")
print("=" * 60)

relu = forge.get("relu")
print(f"\nReLU activation:     {relu.symbolic_expr}")
print(f"ReLU gradient:       {relu.gradient('x').symbolic_expr}")
print(f"ReLU at x=2:         {relu.eval(x=2.0)}")
print(f"ReLU at x=-2:        {relu.eval(x=-2.0)}")
print(f"ReLU grad at x=2:    {relu.gradient('x').eval(x=2.0)}")
print(f"ReLU grad at x=-2:   {relu.gradient('x').eval(x=-2.0)}")

print(f"\nEach neuron computes:")
print(f"  output = ReLU(w₁·x₁ + w₂·x₂ + b)")
print(f"  = ReLU(weighted_sum + bias)")
print(f"  = max(0, weighted_sum + bias)")

print(f"\nThe final layer computes 3 scores (one per class).")
print(f"Softmax converts scores to probabilities.")
print(f"We predict the class with highest probability.")
```

### Visualize Individual Neuron Activations

```python
# See what each neuron in the first layer "looks at"
fig, axes = plt.subplots(2, 4, figsize=(16, 8))

h = 0.05
x_range = np.arange(-1.5, 1.5, h)
y_range = np.arange(-1.5, 1.5, h)
xx, yy = np.meshgrid(x_range, y_range)

for idx, ax in enumerate(axes.flat[:min(8, len(nn_neuro.layers[0].neurons))]):
    neuron = nn_neuro.layers[0].neurons[idx]
    
    # Compute activation for each point
    Z = np.zeros_like(xx)
    for i in range(len(x_range)):
        for j in range(len(y_range)):
            x_val = Value(float(xx[j, i]))
            y_val = Value(float(yy[j, i]))
            result = neuron([x_val, y_val])
            Z[j, i] = result.data
    
    im = ax.contourf(xx, yy, Z, levels=20, cmap='viridis')
    ax.set_title(f"Neuron {idx+1}")
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")

plt.suptitle("What Each Neuron in Layer 1 Responds To", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
```

!!! tip "Educational Insight"
    Each neuron learns a **different linear boundary** (a line in 2D space). The ReLU 
    activation makes it respond only on one side of that line. By combining many such 
    neurons across multiple layers, the network can carve out **complex curved boundaries** 
    — which is how it separates the spirals!

---

## Final Comparison

### Code Complexity

| Component | Neurogebra (lines) | PyTorch (lines) |
|-----------|-------------------|----------------|
| Neural network definition | 60 | 15 |
| Softmax + cross-entropy | 15 | 1 |
| Training loop | 35 | 20 |
| Evaluation | 8 | 5 |
| **Total** | **~120** | **~40** |

### Understanding Gained

| What You Understand | Neurogebra | PyTorch |
|---------------------|------------|---------|
| How neurons compute outputs | ✅ You wrote it | ⚠️ Hidden in `nn.Linear` |
| How gradients flow backward | ✅ You see `.backward()` flow | ⚠️ Happens inside autograd |
| How softmax works | ✅ You implemented it | ⚠️ Inside `CrossEntropyLoss` |
| How weights get updated | ✅ `p.data -= lr * p.grad` | ⚠️ Inside `optimizer.step()` |
| How decision boundaries form | ✅ You can inspect neurons | ⚠️ Requires extra tools |

### When to Use Each

| Scenario | Use Neurogebra | Use PyTorch |
|----------|---------------|-------------|
| Learning ML concepts | ✅ | |
| Understanding backprop | ✅ | |
| Course assignments | ✅ | |
| Research prototyping | ✅ | ✅ |
| Production ML systems | | ✅ |
| Large-scale training | | ✅ |
| GPU acceleration | | ✅ |
| Pre-trained models | | ✅ |

---

## What You Learned in This Project

1. **Neural networks** are layers of simple neurons stacked together
2. **Each neuron** computes: output = activation(w·x + b)
3. **Backpropagation** computes gradients by following the chain rule backward
4. **Softmax** converts raw scores to probabilities
5. **Cross-entropy** measures how wrong the predicted probabilities are
6. **Non-linear activations** (ReLU) allow networks to learn curved boundaries
7. **Multiple layers** allow increasingly complex decision boundaries
8. Neurogebra makes every step **visible and educational**
9. PyTorch provides **speed and convenience** for production

---

## Congratulations! 🎉

You've completed all three projects! You now understand:

- **Linear Regression** — the foundation of ML
- **Image Classification** — how networks see images
- **Neural Networks from Scratch** — how every component works

### Your Learning Path from Here

```
You are here ──────────────────────────────►
                                             
Neurogebra (understanding)      PyTorch (production)
├── ✅ Expressions              ├── torchvision
├── ✅ Autograd                 ├── DataLoader
├── ✅ Training                 ├── GPU training
├── ✅ Neural Networks          ├── Pre-trained models
└── ✅ Loss & Optimization      └── Deployment
```

You have a **solid foundation**. Whether you continue with Neurogebra for deeper understanding 
or move to PyTorch for production work, you now know **what's actually happening** inside the 
black box.

---

**Back to:** [Home](../index.md) | [API Reference](../api/reference.md)
