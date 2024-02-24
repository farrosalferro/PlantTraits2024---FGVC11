def R2Loss(y_pred, y_true):
    SS_residual = ((y_true - y_pred) ** 2).sum()
    SS_total = ((y_true - y_true.mean()) ** 2).sum() + 1e-8
    return SS_residual / SS_total