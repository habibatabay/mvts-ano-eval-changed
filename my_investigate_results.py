from operator import index
from sys import prefix
from unittest import result
import pandas as pd
import glob
import numpy as np
       
def extract_results(prefix, model_name):
    base_path = '.\\my_results\\intermediate\\'
    files = glob.glob(base_path+prefix+f'_combined_{model_name}_*')
    accs = []
    for csv_file in files:
        data = pd.read_csv(csv_file)
        acc = data.iloc[:,1].to_numpy()
        accs.append(acc)
    avg_acc  = np.mean(np.array(accs),axis=0)
    return avg_acc


if __name__ == "__main__":
    model_names = ['UnivarAutoEncoder','AutoEncoder', 'LSTMED', 'TcnED', 'VAE_LSTM','MSCRED', 'OmniAnoAlgo']
    model_names = model_names[:-2]
    prefix = '2022_08_18_18_06_03'
    results = pd.DataFrame(data=np.zeros((len(model_names),38)),columns=list(range(1,39)), index=model_names)
    for model in model_names:
        results.loc[model] = extract_results(prefix, model)
    result_path = f'./my_results/intermediate/{prefix}_summary.csv'
    results = results.transpose()
    results.to_csv(result_path)
    print('results saved to ',result_path)
