import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

for model in ['eth']:    
    print(f"Results for: {model}")
    for ph in [12]:
        print(f"-----------------PH: {ph} -------------------")

############################# eth/ucy #######################################
        perf_df = pd.DataFrame()
        for f in glob.glob(f"results/{model}*_{ph}_fde_most_likely.csv"):
            dataset_df = pd.read_csv(f)
            dataset_df['model'] = model
            perf_df = perf_df.append(dataset_df, ignore_index=True)
            del perf_df['Unnamed: 0']
        print(f"FDE Most Likely @{ph*0.5}s: {perf_df['value'][perf_df['type'] == 'ml'].mean()}")      
        del perf_df
        
        perf_df = pd.DataFrame()
        for f in glob.glob(f"results/{model}*_{ph}_ade_most_likely.csv"):
            dataset_df = pd.read_csv(f)
            dataset_df['model'] = model
            perf_df = perf_df.append(dataset_df, ignore_index=True)
            del perf_df['Unnamed: 0']
        print(f"ADE Most Likely @{ph*0.5}s: {perf_df['value'][perf_df['type'] == 'ml'].mean()}")      
        del perf_df
        
        perf_df = pd.DataFrame()
        for f in glob.glob(f"results/{model}*_{ph}_fde_best_of.csv"):
            dataset_df = pd.read_csv(f)
            dataset_df['model'] = model
            perf_df = perf_df.append(dataset_df, ignore_index=True)
            del perf_df['Unnamed: 0']
        print(f"FDE best_of @{ph*0.5}s: {perf_df['value'][perf_df['type'] == 'best_of'].mean()}")      
        del perf_df
        
        perf_df = pd.DataFrame()
        for f in glob.glob(f"results/{model}*_{ph}_ade_best_of.csv"):
            dataset_df = pd.read_csv(f)
            dataset_df['model'] = model
            perf_df = perf_df.append(dataset_df, ignore_index=True)
            del perf_df['Unnamed: 0']
        print(f"ADE best_of @{ph*0.5}s: {perf_df['value'][perf_df['type'] == 'best_of'].mean()}")      
        del perf_df
        
        perf_df = pd.DataFrame()
        for f in glob.glob(f"results/{model}*_{ph}_kde_best_of.csv"):
            dataset_df = pd.read_csv(f)
            dataset_df['model'] = model
            perf_df = perf_df.append(dataset_df, ignore_index=True)
            del perf_df['Unnamed: 0']
        print(f"KDE best_of @{ph*0.5}s: {perf_df['value'][perf_df['type'] == 'best_of'].mean()}")      
        print("----------------------------------------------")
        del perf_df

