import torch

def R2Metrics(y_true, y_pred):
    """
    R2 score
    """
    with torch.no_grad():
        SS_residual = ((y_true - y_pred) ** 2).sum()
        SS_total = ((y_true - y_true.mean()) ** 2).sum() + 1e-8
        return 1 - SS_residual / SS_total