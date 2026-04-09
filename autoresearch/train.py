"""
Route Choice AutoResearch - Training Script (SEARCH LAYER).

This is the ONLY file the AI agent modifies.
Optimize: test rho_sq (McFadden's pseudo R-squared)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

from prepare import (set_seed, get_device, load_data, load_mnl_beta, evaluate,
                     train_model, log_result, load_baseline, save_baseline,
                     print_comparison, masked_softmax, N_FEATURES, N_CONTEXT, MAX_ALTS)

# === HYPERPARAMETERS (modify freely) ===
EXPERIMENT_NAME = 'exp30_prenorm_resnet'
MODEL_NAME = 'TasteNet-PreNorm'
LEARNING_RATE = 1e-3
BATCH_SIZE = 2048
MAX_EPOCHS = 200
PATIENCE = 25
TIME_BUDGET = 300  # seconds
SEED = 42


# === MODEL (modify freely) ===
class ResBlock(nn.Module):
    """Pre-norm residual block: LN -> Linear -> GELU -> Dropout."""
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return x + self.net(self.ln(x))


class RouteChoiceModel(nn.Module):
    """TasteNet-Res5B288: single deep residual network, 5 blocks, 288 dim.

    Slightly wider and one more block than exp20 (best at 0.5764).
    """
    def __init__(self, n_features=N_FEATURES, n_context=N_CONTEXT,
                 hidden_dim=256, n_blocks=4, dropout=0.1):
        super().__init__()
        self.n_features = n_features
        enriched_ctx = n_context + n_features * 2

        n_expanded = n_features + 6 + 8  # 23
        input_dim = n_expanded + enriched_ctx

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.input_ln = nn.LayerNorm(hidden_dim)
        self.blocks = nn.ModuleList([ResBlock(hidden_dim, dropout) for _ in range(n_blocks)])
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, X, z, mask):
        B, A, n_f = X.shape
        mask_count = mask.sum(1, keepdim=True).clamp(min=1)
        x_masked = X * mask.unsqueeze(-1)
        x_mean = x_masked.sum(1) / mask_count
        x_var = (x_masked ** 2).sum(1) / mask_count - x_mean ** 2
        x_std = x_var.clamp(min=0).sqrt()
        z_enriched = torch.cat([z, x_mean, x_std], dim=-1)

        # Feature expansion
        X_sq = X[:, :, :6] ** 2
        X_int = torch.stack([
            X[:, :, 0] * X[:, :, 4],
            X[:, :, 0] * X[:, :, 5],
            X[:, :, 5] * X[:, :, 4],
            X[:, :, 0] * X[:, :, 6],
            X[:, :, 0] * X[:, :, 7],
            X[:, :, 1] * X[:, :, 2],
            X[:, :, 3] * X[:, :, 4],
            X[:, :, 5] * X[:, :, 6],
        ], dim=-1)
        X_expanded = torch.cat([X, X_sq, X_int], dim=-1)  # (B, A, 23)

        # Residual network
        z_exp = z_enriched.unsqueeze(1).expand(-1, A, -1)
        alt_input = torch.cat([X_expanded, z_exp], dim=-1)
        h = self.input_ln(self.input_proj(alt_input))
        for block in self.blocks:
            h = block(h)
        V = self.output(h).squeeze(-1)

        return masked_softmax(V, mask)


# === MAIN ===
def main():
    print(f'{"="*60}')
    print(f'Experiment: {EXPERIMENT_NAME} | Model: {MODEL_NAME}')
    print(f'{"="*60}')

    set_seed(SEED)
    device = get_device()
    print(f'Device: {device}')

    train_loader, test_loader, train_ds, test_ds = load_data(batch_size=BATCH_SIZE)
    model = RouteChoiceModel()
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Parameters: {n_params:,}')

    print(f'\nTraining (budget: {TIME_BUDGET}s, max epochs: {MAX_EPOCHS})...')
    t0 = time.time()
    best_state, best_loss, n_epochs = train_model(
        model, train_loader, test_loader, train_ds, test_ds,
        lr=LEARNING_RATE, epochs=MAX_EPOCHS, patience=PATIENCE,
        device=device, time_budget=TIME_BUDGET, verbose=True)
    train_time = time.time() - t0
    print(f'\nDone: {n_epochs} epochs in {train_time:.1f}s')

    print('\nEvaluating...')
    metrics = evaluate(model, test_loader, test_ds, device=device)

    print(f'\n{"="*40}')
    print(f'  RESULTS: {MODEL_NAME}')
    print(f'{"="*40}')
    print(f'  rho_sq:    {metrics["rho_sq"]:.6f}')
    print(f'  FPR Top-1: {metrics["fpr_top1"]:.4f} ({metrics["fpr_top1"]*100:.2f}%)')
    print(f'  FPR Top-3: {metrics["fpr_top3"]:.4f} ({metrics["fpr_top3"]*100:.2f}%)')
    print(f'  RMSE:      {metrics["rmse"]:.4f}')
    print(f'  Params:    {n_params:,}')

    baseline = load_baseline()
    print_comparison(baseline, metrics)

    if baseline is None:
        kept = True; notes = 'initial baseline'
    elif metrics['rho_sq'] > baseline['rho_sq']:
        kept = True; notes = f'improved +{metrics["rho_sq"]-baseline["rho_sq"]:.6f}'
        print(f'\n  >>> IMPROVEMENT! Updating baseline. <<<')
    else:
        kept = False; notes = f'no improvement (baseline={baseline["rho_sq"]:.6f})'
        print(f'\n  >>> No improvement. Discarding. <<<')

    log_result(EXPERIMENT_NAME, MODEL_NAME, metrics, kept, notes)
    if kept:
        save_baseline(metrics, MODEL_NAME, EXPERIMENT_NAME)
        torch.save(best_state, 'best_model.pt')
        print('  Baseline updated & model saved.')

    print(f'\nval_rho_sq: {metrics["rho_sq"]:.6f}')
    return metrics

if __name__ == '__main__':
    main()
