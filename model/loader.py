import pandas as pd
import torch
from analysis.data_processer import DataProcesser

class CustomDataset(torch.utils.data.Dataset):
    
    data : pd.DataFrame
    probabilities : pd.Series
    column_count : int
    
    def __init__(self, df : pd.DataFrame, normalize_output: bool) -> None:
        self.data =  df
        self.column_count = len(self.data.columns) - 1
        if normalize_output:
            self.data["target"] = self.data["target"] / DataProcesser.MAX_SCORE
        
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        df = self.data[idx:idx+1]
        sample_label = df["target"].values[0]
        sample_data = df.drop(columns=["target"]).values[0]
        return torch.tensor(sample_data, dtype=torch.float32), torch.tensor(sample_label, dtype=torch.float32)