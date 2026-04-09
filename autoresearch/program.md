# Route Choice AutoResearch - Program

## Goal

Maximize **test rho_sq (McFadden's pseudo R-squared)** for transit route choice model.

## Background

- Seoul metropolitan smart card data: 37M trips, 530K ODs
- 9 features: IVT, Access/Egress walk, Transfer walk, Transfers, Fare, Has bus/train/GTX
- OD-level stratified split (80/20, seed=42)
- Current baseline: TasteNet rho_sq=0.542 (target: beat ASU-DNN 0.686)
- TasteNet: context(OD distance, choice set size) -> beta(z) -> V = X \* beta(z)
- Focus on improving TasteNet architecture only. Do NOT switch to other model types.

## File Structure

```
program.md   <- This file (agent SOP, DO NOT modify)
prepare.py   <- Data loading + evaluation (DO NOT modify)
train.py     <- Model + training (THE ONLY file agent modifies)
results.tsv  <- Experiment log (auto-recorded)
baseline.json <- Current best performance (auto-updated)
best_model.pt <- Best model weights (auto-saved)
```

## Rules

1. **Only modify train.py.** Never touch prepare.py or program.md.
2. Each experiment must complete within **5 minutes (300 seconds)**.
3. Optimization target: **test rho_sq** (single metric, higher is better).
4. Change EXPERIMENT_NAME and MODEL_NAME before each experiment.
5. Results are auto-compared with baseline and auto-logged.

## Experiment Loop

```
1. Read train.py to understand current model/hyperparameters.
2. Think of an improvement:
   - Architecture change (hidden dims, layers, activation, skip connections)
   - Hyperparameter tuning (lr, dropout, batch size)
   - New model structure
   - Training strategy (optimizer, scheduler, loss variant)
3. Modify train.py.
4. Run: python train.py
5. Check results:
   - val_rho_sq > baseline -> SUCCESS (auto-updated)
   - val_rho_sq <= baseline -> FAIL (auto-logged, discard changes)
6. Review results.tsv for experiment history.
7. Plan next experiment, go to step 3.
```

## What You Can Modify (in train.py)

### Model Architecture

- RouteChoiceModel class structure
- Hidden dimensions, number of layers
- Activation functions (ReLU, GELU, SiLU, etc.)
- Skip connections, residual connections
- Attention mechanisms
- Alternative-specific vs shared networks
- Normalization (BatchNorm, LayerNorm)

### Hyperparameters

- LEARNING_RATE (current: 1e-3)
- BATCH_SIZE (current: 2048)
- MAX_EPOCHS (current: 100)
- PATIENCE (current: 15)
- dropout rate (current: 0.1)

### Training Strategy

- Optimizer (Adam, AdamW, SGD, etc.)
- Scheduler changes
- Loss function variants (focal loss, label smoothing)
- Gradient clipping strategy
- Learning rate warmup

## Constraints (DO NOT change)

- 9 input features (MODEL_FEATURES) are fixed
- 2 context features (CONTEXT_FEATURES) are fixed
- MAX_ALTS = 5
- Data split is fixed (OD-level stratified, seed=42)
- Evaluation uses prepare.py's evaluate() function
- forward(X, z, mask) signature must be preserved:
  - X: (batch, MAX_ALTS, n_features)
  - z: (batch, n_context)
  - mask: (batch, MAX_ALTS)
  - return: (batch, MAX_ALTS) probabilities

## Experiment Ideas (TasteNet focused)

1. Increase taste_net hidden_dim: 32 -> 64 / 128 / 256
2. Deeper taste_net: 2-layer -> 3-layer or 4-layer
3. Change activation: ReLU -> GELU / SiLU / Tanh
4. Add more context features (derived from X, e.g., mean IVT per OD)
5. Multi-head TasteNet: separate taste_nets for different feature groups
6. Add bias/intercept term: V = X @ beta(z) + alpha(z)
7. Alternative-specific TasteNet: different beta(z) per alternative
8. Residual connection: beta(z) = beta_base + delta(z)
9. AdamW + cosine annealing scheduler
10. Label smoothing or focal loss
11. Larger context: add num_transfers_mean, fare_mean as context
12. Attention over alternatives before utility calculation

## Start

```bash
# Check data
python prepare.py

# Run baseline
python train.py

# Then iterate: modify train.py -> run -> check results
```
