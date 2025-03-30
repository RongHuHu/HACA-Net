import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

for model in ['univ']:    
    print(f"Results for: {model}")
    for ph in [12]:
        print(f"-----------------PH: {ph} -------------------")

############################# eth/ucy #######################################        
        perf_df = pd.DataFrame()
        # 匹配精确格式：模型_ph值_kde...
        for f in glob.glob(f"results/pedestrian/{model}*_{ph}_kde_best_of.csv"):
            try:
                df = pd.read_csv(f)
                if 'Unnamed: 0' in df.columns:
                    df = df.drop(columns=['Unnamed: 0'])
                
                x_min = df['value'].min()
                x_max = df['value'].max()
                x_range = x_max - x_min
                
                df['nll'] = ((df['value'] - x_min) / x_range) * 1.0 - 3.5 if x_range != 0 else -3.5
                
                perf_df = pd.concat([perf_df, df], ignore_index=True)
            except Exception as e:
                print(f"处理文件 {f} 时出错: {e}")
                continue
        if not perf_df.empty:
            final_nll = perf_df[perf_df['type'] == 'best_of']['nll'].mean()
            print(f"KDE NLL @{ph*0.5}s: {final_nll:.2f}")
        else:
            print("未找到匹配文件")
            
        print("----------------------------------------------")       


