import pandas as pd
import numpy as np
from sympy import true

class DataProcesser:
    
    column_renames =  {
        'r8.2_who5_num': 'target',
        'postweight_unscaled': 'weight',
        'age_fct6': 'age_group',
        'gender_fct2': 'gender',
        'eth_fct4': 'ethnicity',
        'r4_previnworkforce_fct3': 'employment_pre_lockdown',
        'r4.5_fct3': 'work_type',
        'r7.1_fct3': 'smoking_status',
        'r5.2_fct2': 'self_rated_health',
        'r8.17_fct2': 'mental_health_diagnosis',
        'r5.10': 'physical_disability',
        'r9.1_fct2': 'trauma_exposure',
        'r3.2_fct4': 'bubble_type',
        'r3.3_num': 'bubble_size_num',
        'r3.7_fct3': 'social_connection_freq',
        'r3.8': 'contact_change',
        'r4_lesswork_fct2': 'less_work',
        'r4_lostwork_fct2': 'lost_work',
        'r5.6_fct3': 'covid_exposure',
        'r3.4': 'bubble_satisfaction',
        'r3.10': 'bubble_relationships',
        'r3.11': 'loneliness',
        'r3.12': 'time_covid_info',
        'income_band': 'income_band',
        'education_qual': 'education_qual',
        'r8.16_1_fct2': 'stress_self_health',
        'r8.16_2_fct2': 'stress_family_health',
        'r8.16_3_fct2': 'stress_finances',
        'r8.16_6_fct2': 'stress_employment',
        'r8.16_4_fct2': 'stress_covid_conseq',
        'r12.1_11_fct2': 'positive_lockdown_personal',
        'r12.1_13_fct2': 'positive_lockdown_society',
        'r6.4_fct2': 'alcohol_pre',
        'r6.5_fct2': 'alcohol_during',
        'r6_change_fct3': 'alcohol_change',
    }

    
    MAX_SCORE : int = 25
    dataset : pd.DataFrame
    population : int
    score_key : str = "target"
    
    def __init__(self,dataset : pd.DataFrame, ignoring_threshold: float = 0.40) -> None:
        self.data = dataset

        self.data.rename(columns=self.column_renames, inplace=True)
        #self.population = self.data["weight"].sum()
        self.population = len(self.data)
        
    def clean_dataset(self) -> pd.DataFrame:

        self.data = self.data.drop(columns = [c for c in self.data.columns if c not in self.column_renames.values()])
        
        keys_drop_na = ["employment_pre_lockdown","bubble_type"]
        for key in keys_drop_na:
            self.data[key].dropna(inplace=True)
            
            
        self.data['lost_work'] = np.where(self.data['lost_work'].isnull() & (self.data['employment_pre_lockdown']!='Employed'), 'Without work', self.data['lost_work'])
        
        self.data['loneliness'] = np.where(self.data['loneliness'].isnull(), 'None', self.data['loneliness'])
        
        self.data['less_work'] = np.where(self.data['less_work'].isnull() & (self.data['employment_pre_lockdown']!='Employed'), 'Without work', self.data['less_work'])
        
        self.data['work_type'] = np.where(self.data['work_type'].isnull() & (self.data['employment_pre_lockdown']!='Employed'), 'Without work', self.data['work_type'])
        self.data['work_type'] = np.where(self.data['work_type'].isnull(), 'Not essential worker', self.data['work_type'])
        
        self.data['bubble_relationships'] = np.where(self.data['bubble_relationships'].isnull() & (self.data['bubble_type']=='Live by myself'), 'Without bubble', self.data['bubble_relationships'])
        self.data['bubble_relationships'] = np.where(self.data['bubble_relationships'].isnull(), 'Very well', self.data['bubble_relationships'])
        
        self.data['mental_health_diagnosis'] = np.where(self.data['mental_health_diagnosis'].isnull(), 'No', self.data['mental_health_diagnosis'])
        
        self.data['social_connection_freq'] = np.where(self.data['social_connection_freq'].isnull(), 'High', self.data['social_connection_freq'])
        
        self.data['gender'] = np.where(self.data['gender'].isnull(), 'Female', self.data['gender'])
        
        self.data['contact_change'] = np.where(self.data['contact_change'].isnull(), 'It has stayed the same', self.data['contact_change'])
        
        self.data['positive_lockdown_personal'] = np.where(self.data['positive_lockdown_personal'].isnull(), 'No', self.data['positive_lockdown_personal'])
        
        self.data['positive_lockdown_society'] = np.where(self.data['positive_lockdown_society'].isnull(), 'No', self.data['positive_lockdown_society'])
        
        self.data['alcohol_change'] = np.where(self.data['alcohol_change'].isnull(), 'No change', self.data['alcohol_change'])
        
        self.data['alcohol_during'] = np.where(self.data['alcohol_during'].isnull(), 'Low level', self.data['alcohol_during'])
        
        replace_with_most_common = ["income_band","time_covid_info","smoking_status","covid_exposure"]
        
        for table in replace_with_most_common:
            most_common = self.data[table].value_counts(dropna=False).idxmax()
            self.data[table] = np.where(self.data[table].isnull(), most_common, self.data[table])
        
        for col in self.data.columns:
            if self.data[col].dtype == 'object':
                self.data[col] = pd.factorize(self.data[col])[0]
        
        
        
    
if __name__ == "__main__":
    df = pd.read_csv("datos/Resilience_CleanOnly_v1.csv", encoding="latin1")
    
    data_processer = DataProcesser(df)
    
    result_df = data_processer.process_dataset()
    
    result_df.to_csv("datos/Resilience_CleanOnly_v1_PREPROCESSED.csv", index=False)