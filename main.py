
import pandas as pd
from model.loader import CustomDataset
from model.model import Model
from model.trainer import train_model
from torch.utils.data import DataLoader
from analysis.logger import Logger
from analysis.data_split import stratified_split
from model.LinearClamp import LinearClamp
from model.metrics import evaluar_modelo

import torch
import torch.optim as optim
import torch.nn as nn
import os 



df = pd.read_csv("datos/Resilience_CleanOnly_v1_PREPROCESSED_v2.csv", encoding="latin1")
sizes = [len(df.columns)-1, 32, 16, 1]
output_activation = nn.Sigmoid
intermediate_activation = nn.Tanh
normalize_output = True
loss = nn.MSELoss
name = f"model_{sizes}_output_{output_activation.__name__}_intermediate_{intermediate_activation.__name__}_normalized_data_{normalize_output}_loss_{loss.__name__}"

comparison_table = pd.read_csv("results/comparison_table.csv")


weight_path = f"results/{name}.pth" if os.path.exists(f"results/{name}.pth") else None
logger = Logger("results/logs")

train_df, val_df, test_df = stratified_split(df)

train_data = CustomDataset(train_df, normalize_output)
val_data = CustomDataset(val_df, normalize_output)
test_data = CustomDataset(test_df, normalize_output)
train_loader = DataLoader(train_data, batch_size=5, shuffle=True)
val_loader = DataLoader(val_data, batch_size=5, shuffle=False)
test_loader = DataLoader(test_data, batch_size=5, shuffle=False)


model = Model(weight_path,description=name, hidden_sizes=sizes, output_activation=output_activation)  
criterion = loss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

trained_model, val_loss = train_model(model, train_loader, val_loader, criterion, optimizer, 100, logger)

torch.save(trained_model.state_dict(), f"results/{name}.pth")


#Metricas

metricas_train = evaluar_modelo(trained_model, train_loader , normalize_output)
metricas_val = evaluar_modelo(trained_model, val_loader , normalize_output)

#logger.log(f"MSE: {metricas['mse']:.6f}")
#logger.log(f"RMSE: {metricas['rmse']:.6f}")
#logger.log(f"MAE: {metricas['mae']:.6f}")
#logger.log(f"SMAPE: {metricas['smape']:.2f}%")

new_row = {
    "name": name,
    "loss": val_loss,

    "train_mse": metricas_train['mse'],
    "train_rmse": metricas_train['rmse'],
    "train_mae": metricas_train['mae'],
    "train_smape": metricas_train['smape'],

    "val_mse": metricas_val['mse'],
    "val_rmse": metricas_val['rmse'],
    "val_mae": metricas_val['mae'],
    "val_smape": metricas_val['smape'],
}   

comparison_table = pd.concat([comparison_table, pd.DataFrame([new_row])], ignore_index=True)
comparison_table.to_csv("results/comparison_table.csv", index=False)