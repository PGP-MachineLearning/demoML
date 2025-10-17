import torch
import torch.nn as nn
from analysis.data_processer import DataProcesser

def metrics(y_true, y_pred): 
    mse = nn.MSELoss()(y_true, y_pred).item()
    rmse = mse ** 0.5
    mae = nn.L1Loss()(y_true, y_pred).item()

    numerator = torch.abs(y_pred - y_true)
    denominator = (torch.abs(y_true) + torch.abs(y_pred)) / 2
    smape = torch.where(denominator == 0, torch.zeros_like(numerator), numerator / denominator)
    smape = 100 * torch.mean(smape).item() 
    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "smape": smape
    }

def evaluar_modelo(model, test_loader, normalize_output):
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
                
            all_predictions.append(outputs)
            all_labels.append(labels)

    all_predictions = torch.cat(all_predictions, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    if normalize_output:
        all_predictions = all_predictions * DataProcesser.MAX_SCORE
        all_labels = all_labels * DataProcesser.MAX_SCORE
    
    return metrics(all_labels, all_predictions)
