import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# for model in ['CNN_transformer','basic_transformer','ViT_transformer','ViT_informer_enc']:
#     print(f"FDE Results for: {model}")ViT_informer_stack++
#     for ph in [2, 4, 6, 8]:

for model in ['ucy']:    #his_map_robot_car
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

############################# nuScenes #######################################   
        # perf_df = pd.DataFrame()
        # for f in glob.glob(f"results/{model}*_{ph}_fde_top1.csv"):
        #     dataset_df = pd.read_csv(f)
        #     dataset_df['model'] = model
        #     perf_df = perf_df.append(dataset_df, ignore_index=True)
        #     del perf_df['Unnamed: 0']
        # print(f"FDE TOP1 @{ph*0.5}s: {perf_df['value'][perf_df['type'] == 'top1'].mean()}")      
        # del perf_df
        
        # perf_df = pd.DataFrame()
        # for f in glob.glob(f"results/{model}*_{ph}_ade_top5.csv"):
        #     dataset_df = pd.read_csv(f)
        #     dataset_df['model'] = model
        #     perf_df = perf_df.append(dataset_df, ignore_index=True)
        #     del perf_df['Unnamed: 0']
        # print(f"ADE TOP5 @{ph*0.5}s: {perf_df['value'][perf_df['type'] == 'top5'].mean()}")      
        # del perf_df

        # perf_df = pd.DataFrame()
        # for f in glob.glob(f"results/{model}*_{ph}_miss_rate_top5.csv"):
        #     dataset_df = pd.read_csv(f)
        #     dataset_df['model'] = model
        #     perf_df = perf_df.append(dataset_df, ignore_index=True)
        #     del perf_df['Unnamed: 0']
        # print(f"MissRate TOP5 @{ph*0.5}s: {perf_df['value'][perf_df['type'] == 'top5'].mean()}")
        # del perf_df
        
        # perf_df = pd.DataFrame()
        # for f in glob.glob(f"results/{model}*_{ph}_ade_top10.csv"):
        #     dataset_df = pd.read_csv(f)
        #     dataset_df['model'] = model
        #     perf_df = perf_df.append(dataset_df, ignore_index=True)
        #     del perf_df['Unnamed: 0']
        # print(f"ADE TOP10 @{ph*0.5}s: {perf_df['value'][perf_df['type'] == 'top10'].mean()}")      
        # del perf_df
        
        # perf_df = pd.DataFrame()
        # for f in glob.glob(f"results/{model}*_{ph}_miss_rate_top10.csv"):
        #     dataset_df = pd.read_csv(f)
        #     dataset_df['model'] = model
        #     perf_df = perf_df.append(dataset_df, ignore_index=True)
        #     del perf_df['Unnamed: 0']
        # print(f"MissRate TOP10 @{ph*0.5}s: {perf_df['value'][perf_df['type'] == 'top10'].mean()}")
        # del perf_df     
                
        # perf_df = pd.DataFrame()
        # for f in glob.glob(f"results/{model}_{ph}_rv.csv"):
        #     dataset_df = pd.read_csv(f)
        #     dataset_df['model'] = model
        #     perf_df = perf_df.append(dataset_df, ignore_index=True)
        #     del perf_df['Unnamed: 0']
        # print(f"RB Viols @{ph*0.5}s: {perf_df['value'][perf_df['type'] == 'viols'].sum() / (len(perf_df['value'][perf_df['type'] == 'viols'].index))}")
        # del perf_df
        
        
        
        
        # perf_df = pd.DataFrame()        
        # for f in glob.glob(f"results/{model}_{ph}_fde_full.csv"):
        #     dataset_df = pd.read_csv(f)
        #     dataset_df['model'] = model
        #     perf_df = perf_df.append(dataset_df, ignore_index=True)
        #     del perf_df['Unnamed: 0']
        # print(f"FDE @{ph*0.5}s: {perf_df['value'][perf_df['type'] == 'full'].mean()}")
        # del perf_df      

        # perf_df = pd.DataFrame()
        # for f in glob.glob(f"results/{model}*_{ph}_ade_full.csv"):
        #     dataset_df = pd.read_csv(f)
        #     dataset_df['model'] = model
        #     perf_df = perf_df.append(dataset_df, ignore_index=True)
        #     del perf_df['Unnamed: 0']
        # print(f"ADE @{ph*0.5}s: {perf_df['value'][perf_df['metric'] == 'ade'].mean()}")
        # del perf_df
        
        # perf_df = pd.DataFrame()
        # for f in glob.glob(f"results/{model}*_{ph}_miss_rate_full.csv"):
        #     dataset_df = pd.read_csv(f)
        #     dataset_df['model'] = model
        #     perf_df = perf_df.append(dataset_df, ignore_index=True)
        #     del perf_df['Unnamed: 0']
        # print(f"MissRate @{ph*0.5}s: {perf_df['value'][perf_df['metric'] == 'miss_rate'].mean()}")
        # del perf_df
        
              


    
# for model in ['vel_ee']:
#     print(f"FDE Results for: {model}")
#     for ph in [6]:
#         print(f"-----------------PH: {ph} -------------------")
#         perf_df = pd.DataFrame()
#         for f in glob.glob(f"results/{model}_{ph}_fde_full.csv"):
#             dataset_df = pd.read_csv(f)
#             dataset_df['model'] = model
#             perf_df = perf_df.append(dataset_df, ignore_index=True)
#         del perf_df['Unnamed: 0']
        
#         print(f"FDE Mean @{ph*0.5}s: {perf_df['value'][perf_df['type'] == 'full'].mean()}")
#         del perf_df      
        
#         perf_df = pd.DataFrame()
#         for f in glob.glob(f"results/{model}_{ph}_rv_full.csv"):
#             dataset_df = pd.read_csv(f)
#             dataset_df['model'] = model
#             perf_df = perf_df.append(dataset_df, ignore_index=True)
#         del perf_df['Unnamed: 0']
#         print(f"RB Viols @{ph*0.5}s: {perf_df['value'][perf_df['type'] == 'full'].sum() / (len(perf_df['value'][perf_df['type'] == 'full'].index)*2000)}")
#         del perf_df

#         perf_df = pd.DataFrame()
#         for f in glob.glob(f"results/{model}*_{ph}_fde_most_likely_z.csv"):
#             dataset_df = pd.read_csv(f)
#             dataset_df['model'] = model
#             perf_df = perf_df.append(dataset_df, ignore_index=True)
#         del perf_df['Unnamed: 0']
#         print(f"FDE @{ph*0.5}s: {perf_df['value'][perf_df['type'] == 'ml'].mean()}")      
#         del perf_df
              
#         perf_df = pd.DataFrame()
#         for f in glob.glob(f"results/{model}*_{ph}_kde_full.csv"):
#             dataset_df = pd.read_csv(f)
#             dataset_df['model'] = model
#             perf_df = perf_df.append(dataset_df, ignore_index=True)
#         del perf_df['Unnamed: 0']
#         print(f"KDE @{ph*0.5}s: {perf_df['value'][perf_df['type'] == 'full'].mean()}")      
#         print("----------------------------------------------")
#         del perf_df
#     print("")

# for model in ['vel_ee']:
#     print(f"FDE Results for: {model}")
#     for ph in [6]:
#         print(f"-----------------PH: {ph} -------------------")
#         perf_df = pd.DataFrame()
#         for f in glob.glob(f"results/{model}*_{ph}_fde_full.csv"):
#             dataset_df = pd.read_csv(f)
#             dataset_df['model'] = model
#             perf_df = perf_df.append(dataset_df, ignore_index=True)
#         del perf_df['Unnamed: 0']
#         print(f"FDE Mean @{ph*0.5}s: {perf_df['value'][perf_df['metric'] == 'fde'].mean()}")
#         del perf_df    
              
#         perf_df = pd.DataFrame()
#         for f in glob.glob(f"results/{model}*_{ph}_ade_full.csv"):
#             dataset_df = pd.read_csv(f)
#             dataset_df['model'] = model
#             perf_df = perf_df.append(dataset_df, ignore_index=True)
#         del perf_df['Unnamed: 0']
#         print(f"ADE Mean @{ph*0.5}s: {perf_df['value'][perf_df['metric'] == 'ade'].mean()}")
#         del perf_df
              
#         perf_df = pd.DataFrame()
#         for f in glob.glob(f"results/{model}*_{ph}_kde_full.csv"):
#             dataset_df = pd.read_csv(f)
#             dataset_df['model'] = model
#             perf_df = perf_df.append(dataset_df, ignore_index=True)
#         del perf_df['Unnamed: 0']
#         print(f"KDE Mean @{ph*0.5}s: {perf_df['value'][perf_df['metric'] == 'kde'].mean()}")
#         del perf_df  

#         perf_df = pd.DataFrame()
#         for f in glob.glob(f"results/{model}*_{ph}_fde_most_likely_z.csv"):
#             dataset_df = pd.read_csv(f)
#             dataset_df['model'] = model
#             perf_df = perf_df.append(dataset_df, ignore_index=True)
#         del perf_df['Unnamed: 0']
#         print(f"FDE @{ph*0.5}s: {perf_df['value'][perf_df['metric'] == 'fde'].mean()}")      
#         del perf_df
              
#         perf_df = pd.DataFrame()
#         for f in glob.glob(f"results/{model}*_{ph}_ade_most_likely_z.csv"):
#             dataset_df = pd.read_csv(f)
#             dataset_df['model'] = model
#             perf_df = perf_df.append(dataset_df, ignore_index=True)
#         del perf_df['Unnamed: 0']
#         print(f"ADE @{ph*0.5}s: {perf_df['value'][perf_df['metric'] == 'ade'].mean()}")      
#         del perf_df
#         print("----------------------------------------------")
#     print("")
