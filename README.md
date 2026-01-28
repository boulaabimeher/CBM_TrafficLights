# Concept Bottleneck Models (CBMs)

## Overview

Concept Bottleneck Models (CBMs) are interpretable machine learning models that make predictions **through human-understandable concepts** instead of directly from raw inputs.  
Rather than learning a black-box mapping from data to labels, CBMs explicitly model *how* decisions are made by reasoning via predefined concepts.

This design improves interpretability, transparency, and controllability, which is critical in high-stakes domains such as medical imaging, autonomous driving, and decision support systems.

---

## Motivation

Traditional deep learning models learn a direct mapping:

X → Y

where:
- X is the raw input (e.g., image pixels)
- Y is the target label (e.g., stop / continue)

Although effective, this approach provides little insight into *why* a prediction was made.

Humans reason differently:
> “The traffic light is green, but an ambulance is approaching, so I should stop.”

CBMs aim to replicate this structured reasoning process.

---

## Core Idea

A Concept Bottleneck Model decomposes prediction into two explicit steps:

1. **Concept prediction**: infer interpretable concepts from the input  
2. **Decision prediction**: infer the final task label using only those concepts  

All information relevant to the decision must pass through the **concept bottleneck**, enforcing interpretability by design.

---

## Probabilistic Formulation

### Standard Prediction

A standard model estimates:

p(Y | X)

meaning the decision is made directly from the input.

---

### Concept-Based Reformulation

CBMs introduce a set of interpretable concepts:

C = {c1, c2, ..., ck}

The prediction is factorized as:

p(Y | X) = Σ_C p(Y | C) · p(C | X)

This assumes that once the concepts C are known, the input X provides no additional information about Y.

Interpretation:
- p(C | X): concept prediction from the input
- p(Y | C): task prediction based only on concepts

This factorization defines the **concept bottleneck**.

---

## Model Architecture

A CBM consists of two main components:

### 1. Concept Encoder

g(X) → Ĉ

- Input: raw data X
- Output: predicted concept vector

Ĉ = [p(c1 | X), p(c2 | X), ..., p(ck | X)]

This component is typically implemented using a neural network (CNN, ViT, Transformer, etc.).

---

### 2. Task Predictor

f(Ĉ) → Ŷ

- Input: predicted concepts
- Output: final prediction

This component is usually a lightweight classifier such as a linear layer or MLP.

---

## Training Objective

Given a dataset:

D = {(x, c, y)}

where:
- x is the input
- c are ground-truth concepts
- y is the target label

CBMs are trained using a joint loss:

L = L_task + L_concept

Where:
- L_concept measures how well the model predicts concepts
- L_task measures how well the model predicts the final label

This ensures both accurate concept learning and correct task performance.

---

## Training Strategies

CBMs can be trained using different strategies:

### Independent Training
- Train the concept encoder and task predictor separately
- The task predictor sees true concepts during training
- At inference, predicted concepts are used (possible error propagation)

### Sequential Training
- Train the concept encoder first
- Freeze it, then train the task predictor on predicted concepts

### Joint Training
- Train both components end-to-end
- Balances concept accuracy and task performance

---

## Example: Driving Decision Scenario

Consider a driving task with the following concepts:

- Green light on the selected lane
- Car present in the intersection
- Ambulance visible
- Ambulance approaching from another lane

Pipeline:
1. Input image of the intersection
2. Concept encoder predicts each concept
3. Task predictor outputs a stop / continue decision

The reasoning becomes explicit:
> “I stopped because an ambulance was approaching.”

---

## Interpretability and Intervention

CBMs offer two key advantages:

### Interpretability
Predicted concepts can be inspected to explain decisions.

### Concept Intervention
Concept values can be manually modified to test alternative scenarios:
- Setting `Ambulance = True` forces the model to reconsider its decision

This enables debugging, safety validation, and human-in-the-loop control.

---

## Comparison

| Aspect | Standard Deep Learning | Concept Bottleneck Models |
|------|------------------------|---------------------------|
| Interpretability | ❌ | ✅ |
| Human-aligned reasoning | ❌ | ✅ |
| Concept-level intervention | ❌ | ✅ |
| Decision transparency | ❌ | ✅ |

---

## Key Takeaway

Concept Bottleneck Models bridge the gap between **performance and interpretability** by enforcing structured, human-understandable reasoning inside machine learning systems.  
They provide a principled framework for explainable and controllable AI.
