import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

for model in ['hmr']:    
    print(f"Results for: {model}")
    for ph in [4]:#2,4,6,8,12        print(f"-----------------PH: {ph} -------------------")

############################# nuScenes #######################################   
        perf_df = pd.DataFrame()
        for f in glob.glob(f"results/nuscene/{model}*_{ph}_fde_top1.csv"):
            dataset_df = pd.read_csv(f)
            dataset_df['model'] = model
            perf_df = perf_df.append(dataset_df, ignore_index=True)
            del perf_df['Unnamed: 0']
        print(f"FDE TOP1 @{ph*0.5}s: {perf_df['value'][perf_df['type'] == 'top1'].mean():.2f}")      
        del perf_df
        
        perf_df = pd.DataFrame()
        for f in glob.glob(f"results/nuscene/{model}*_{ph}_ade_top5.csv"):
            dataset_df = pd.read_csv(f)
            dataset_df['model'] = model
            perf_df = perf_df.append(dataset_df, ignore_index=True)
            del perf_df['Unnamed: 0']
        print(f"ADE TOP5 @{ph*0.5}s: {perf_df['value'][perf_df['type'] == 'top5'].mean():.2f}")      
        del perf_df

        perf_df = pd.DataFrame()
        for f in glob.glob(f"results/nuscene/{model}*_{ph}_miss_rate_top5.csv"):
            dataset_df = pd.read_csv(f)
            dataset_df['model'] = model
            perf_df = perf_df.append(dataset_df, ignore_index=True)
            del perf_df['Unnamed: 0']
        print(f"MissRate TOP5 @{ph*0.5}s: {perf_df['value'][perf_df['type'] == 'top5'].mean():.2f}")
        del perf_df
        
        perf_df = pd.DataFrame()
        for f in glob.glob(f"results/nuscene/{model}*_{ph}_ade_top10.csv"):
            dataset_df = pd.read_csv(f)
            dataset_df['model'] = model
            perf_df = perf_df.append(dataset_df, ignore_index=True)
            del perf_df['Unnamed: 0']
        print(f"ADE TOP10 @{ph*0.5}s: {perf_df['value'][perf_df['type'] == 'top10'].mean():.2f}")      
        del perf_df
        
        perf_df = pd.DataFrame()
        for f in glob.glob(f"results/nuscene/{model}*_{ph}_miss_rate_top10.csv"):
            dataset_df = pd.read_csv(f)
            dataset_df['model'] = model
            perf_df = perf_df.append(dataset_df, ignore_index=True)
            del perf_df['Unnamed: 0']
        print(f"MissRate TOP10 @{ph*0.5}s: {perf_df['value'][perf_df['type'] == 'top10'].mean():.2f}")
        del perf_df     
                
        perf_df = pd.DataFrame()
        for f in glob.glob(f"results/nuscene/{model}*_{ph}_rv.csv"):
            dataset_df = pd.read_csv(f)
            dataset_df['model'] = model
            perf_df = perf_df.append(dataset_df, ignore_index=True)
            del perf_df['Unnamed: 0']
        print(f"OR @{ph*0.5}s: {perf_df['value'][perf_df['type'] == 'viols'].sum() / (len(perf_df['value'][perf_df['type'] == 'viols'].index)):.2f}")
        del perf_df
        
        

              
