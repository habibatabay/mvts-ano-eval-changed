from my_data_functions import load_data_partial, get_results, load_data_all, get_seqs_events, classify_scores
from src.algorithms import AutoEncoder, LSTMED, UnivarAutoEncoder,VAE_LSTM, OmniAnoAlgo, MSCRED, TcnED
from src.algorithms.algorithm_utils import get_sub_seqs
import numpy as np
import pandas as pd

from src.algorithms.algorithm_utils import fit_univar_distr
from src.evaluation.evaluation_utils import get_scores_channelwise, threshold_and_predict
from datetime import datetime
import os
from sklearn.metrics import recall_score
import logging
from src.datasets.dataset import get_events
import torch
import glob

# Global configs
sequence_length = 15
# num_epochs = 50
num_epochs = 5
hidden_size = 10
n_layers_ed = (10,10)
batch_size = 16
learning_rate = 0.0001
seed = 0
n_data_folds = 3


def get_normalized_scores(train_scores, test_scores):
    mean_scores = np.mean(train_scores, axis=0)
    scores = test_scores - mean_scores
    scores = np.sqrt(np.mean(scores**2, axis=1))
    return scores

def get_fitted_scores(error_tc_train, error_tc_test, distr_name='univar_gaussian'):
    distr_params = [fit_univar_distr(error_tc_train[:, i], distr=distr_name) for i in range(error_tc_train.shape[1])]
    score_t_train, _, score_t_test, score_tc_train, _, score_tc_test = get_scores_channelwise(distr_params, train_raw_scores=error_tc_train,
                                       val_raw_scores=None, test_raw_scores=error_tc_test,
                                       drop_set=set([]), logcdf=True)
    return score_t_train, score_t_test

def collect_results(y_true, y_pred):
    aps, auroc, _ = get_results(y_true , y_pred, top_k= 0, print_results=False)
    result = classify_scores(y_pred, y_true, method='pr')
    p_aps, r_aps = result['1']['precision'], result['1']['recall']
    result = classify_scores(y_pred, y_true, method='roc')
    p_roc, r_roc = result['1']['precision'], result['1']['recall']
    print(f'\nfinal test : APS={aps:0.3f}, AUROC={auroc:0.3f}, Pre APS={p_aps:0.3f}, Rec APS={r_aps:0.3f}, Pre ROC={p_roc:0.3f}, Rec ROC={r_roc:0.3f}')
    return [aps, auroc, p_aps, r_aps, p_roc, r_roc]

def setup_out_dir(dataset_name, model_name, feature_type, folder_idx='all'):
    path = f'my_trained_models/{dataset_name}/{model_name}/{feature_type}/{folder_idx}/'
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def get_model(model_name,features_dim, out_dir=None):

    if model_name == 'AutoEncoder':
        model = AutoEncoder(sequence_length=sequence_length, num_epochs=num_epochs, hidden_size=hidden_size, lr=learning_rate,batch_size=batch_size, seed=seed, gpu=0, out_dir=out_dir)
    elif model_name == 'UnivarAutoEncoder':
        model = UnivarAutoEncoder(sequence_length=sequence_length, num_epochs=num_epochs, hidden_size=hidden_size, lr=learning_rate,batch_size=batch_size, seed=seed, gpu=0, out_dir=out_dir)
    elif model_name == 'LSTMED':
        model = LSTMED(sequence_length=sequence_length,hidden_size=hidden_size,num_epochs=num_epochs,batch_size=batch_size,lr=learning_rate,n_layers=n_layers_ed,seed=seed, gpu=0, out_dir=out_dir)
    elif model_name == 'VAE_LSTM':
        model = VAE_LSTM(sequence_length=sequence_length, num_epochs= num_epochs,n_dim=features_dim, intermediate_dim=2*hidden_size, z_dim=hidden_size, lr=learning_rate,batch_size=batch_size, seed=seed, gpu=0, out_dir=out_dir)
    elif model_name == 'TcnED':
        model = TcnED(sequence_length=sequence_length,num_epochs= num_epochs, num_channels=[features_dim],kernel_size=hidden_size, lr=learning_rate,batch_size=batch_size, seed=seed, gpu=0, out_dir=out_dir)
    elif model_name == 'MSCRED':
        model = MSCRED(sequence_length=sequence_length, num_epochs=num_epochs, lr=learning_rate, batch_size=batch_size, seed=seed, gpu=0, out_dir=out_dir)
    elif model_name == 'OmniAnoAlgo':
        model = OmniAnoAlgo(sequence_length=sequence_length, num_epochs=num_epochs,z_dim=hidden_size, batch_size=batch_size, seed=seed, gpu=0, out_dir=out_dir)
    # elif model_name == 'PcaRecons':
    #     model = PcaRecons(seed=seed, out_dir=out_dir)    
    # elif model_name == 'RawSignalBaseline':
    #     model = RawSignalBaseline(seed=seed, out_dir=out_dir)
    return model

def apply_threshold(train_scores, test_scores):
    thresh = np.mean(train_scores) + np.std(train_scores)
    train_labels = np.where(train_scores < thresh, 0, 1)
    test_labels = np.where(test_scores < thresh, 0, 1)
    return thresh, train_labels, test_labels

def partition_label_indecies(labels, seq_len):
    zero_idxs = set([])
    one_idxs = set([])
    i = 0 
    while i <= len(labels)-seq_len:
        if sum(labels[i:i+seq_len]) > 0:
            one_idxs.add(i)
        else:
            zero_idxs.add(i)

        i += 1
    return np.array(sorted(zero_idxs)), np.array(sorted(one_idxs))
# partition_label_indecies([0,0,0,1,0,1,0,0,0],3)

def test_model(model, x_train,  x_test, score_distr_name):
    test_preds = model.predict_sequences(x_test)
    train_preds = model.predict_sequences(x_train)
    if score_distr_name == 'normalized_error':
        test_scores = get_normalized_scores(train_preds['error_tc'], test_preds['error_tc'])
    else:
        if test_preds['score_t'] is None:
            train_scores, test_scores = get_fitted_scores(train_preds['error_tc'], test_preds['error_tc'])  
        else:
            train_scores, test_scores = train_preds['score_t'], test_preds['score_t']
    test_scores = test_scores[sequence_length-1:]
    return test_scores
 
def experiment_on_folder(dataset_name, model_name, folder_idx, feature_type, body_part='upper',
                        training_modes=['video-specific'],pretrained_model=None, load_saved=True):

    print(f'\n\nprocessing folder {folder_idx}...')

    if pretrained_model is None and 'video-specific' in training_modes and len(training_modes) == 1:
        assert False


    x_data, y_data = load_data_partial(dataset_name, folder_idx, feature_type, body_part, train_ratio=0.0)
    y_data = y_data.values
    y_data.shape = (-1,)


    features_dim = x_data.shape[1]
    out_dir=setup_out_dir(dataset_name, model_name, feature_type, folder_idx)
    model = get_model(model_name,features_dim, out_dir=out_dir)


    x_seqs = get_sub_seqs(x_data.values, seq_len=sequence_length)
    y_seqs = np.array([1 if sum(y_data[i:i + sequence_length])>0 else 0 for i in range(len(x_seqs))])
    e_seqs = get_seqs_events(y_data, sequence_length)
    train_ratio = 0.3
    # top_ratio = 0.1
    # top_k = int(len(x_seqs) * top_ratio)
    # top_k = np.sum(y_seqs)
    val_ratio = 0.2
    # i = 1
    # x_train = None    

    n_train = int(len(x_seqs) * train_ratio)
    x_train = x_seqs[:n_train]
    y_train = y_seqs[:n_train]
    x_test = x_seqs[n_train:]
    y_test = y_seqs[n_train:]
    # e_test = e_seqs[n_train:]

    n_train = int(len(x_train) * (1-val_ratio))
    x_val = x_train[n_train:]
    # y_val = y_train[n_train:]
    x_train = x_train[:n_train]
    y_train = y_train[:n_train]

    results_g = None
    results_vs = None
    results_c = None

    if 'global' in training_modes:
        pre_test_preds = pretrained_model.predict_sequences(x_test )
        pre_train_preds = pretrained_model.predict_sequences(x_train)
        if pre_test_preds['score_t'] is None:
            pre_train_scores, pre_test_scores = get_fitted_scores(pre_train_preds['error_tc'], pre_test_preds['error_tc'])  
        else:
            pre_train_scores, pre_test_scores = pre_train_preds['score_t'], pre_test_preds['score_t']
        
        pre_test_scores = pre_test_scores[sequence_length-1:]

        results_g = collect_results(y_test, pre_test_scores)
        if len(training_modes) == 1:
            return [results_g]

    if model.model is None:
        loaded = False
        if load_saved==True and hasattr(model, 'torch_save') and model.torch_save == True:
            saved_models = glob.glob(out_dir + 'trained_model*.pt')
            if len(saved_models)>0:
                # input('loading')
                model.model = torch.load(saved_models[0], model.device)
                loaded = True
        if loaded == False:
            model.fit_sequences(x_train, x_val)
            if hasattr(model,'torch_save') and  model.torch_save == True:
                torch.save(model.model, out_dir+f'trained_model.pt')
    
    
    test_preds = model.predict_sequences(x_test )
    train_preds = model.predict_sequences(x_train )
    # if score_distr_name == 'normalized_error':
    #     test_scores = get_normalized_scores(train_preds['error_tc'], test_preds['error_tc'])
    # else: univar_gaussian
    if test_preds['score_t'] is None:
        train_scores, test_scores = get_fitted_scores(train_preds['error_tc'], test_preds['error_tc'])  
    else:
        train_scores, test_scores = train_preds['score_t'], test_preds['score_t']
    test_scores = test_scores[sequence_length-1:]

    results_vs = collect_results(y_test, test_scores)
    results = []
    if 'video-specific' in training_modes:
        results.append(results_vs)

    if 'combined' in training_modes:
        test_scores = (test_scores + pre_test_scores) / 2   
        results_c = collect_results(y_test, test_scores)
        results.append(results_c)
    
    if 'global' in training_modes:
        results.append(results_g)
    
    return results
    # aps, auroc, _ = get_results(y_test , test_scores, top_k= 0, print_results=False)
    # result = classify_scores(test_scores, y_test, method='pr')
    # p_aps, r_aps = result['1']['precision'], result['1']['recall']
    # result = classify_scores(test_scores, y_test, method='roc')
    # p_roc, r_roc = result['1']['precision'], result['1']['recall']
    
    # e_acc = get_events_accuracy(e_test, test_scores, top_k)

    # return aps, auroc, p_aps, r_aps, p_roc, r_roc

def experiments_on_dataset(dataset_name:str, model_name:str, feature_type:str, training_modes:list, exp_time=''):
    pretrained_model = None
    if 'global' in training_modes or 'combined' in training_modes:
        x_train, _ = load_data_all(feature_type, body_part)
        features_dim = x_train.shape[1]
        out_dir = setup_out_dir(dataset_name,model_name, feature_type)
        pretrained_model = get_model(model_name, features_dim, out_dir)
        saved_model_dir= out_dir + 'trained_model.pt'
        if  os.path.exists(saved_model_dir):
            print('loading pre-trained model ...')
            pretrained_model.model = torch.load(saved_model_dir, pretrained_model.device)
        else:
            print('pre-training model ...')
            pretrained_model.fit(x_train)
            if hasattr(pretrained_model,'torch_save') and  pretrained_model.torch_save == True:
                print('saving pre-trained model...')
                torch.save(pretrained_model.model, saved_model_dir)
    n=38
    if dataset_name == 'CombinedDataset':
        n = 91
    # time_now = exp_time if exp_time != '' else datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S')

    dataset_stats = [] 
    for folder_idx in range (1, n+1):
 
        folder_stats = experiment_on_folder(dataset_name, model_name, folder_idx, feature_type, training_modes=training_modes, pretrained_model=pretrained_model)

        dataset_stats.append(folder_stats)
    
    dataset_stats = np.array(dataset_stats)
    avg_stats = []

    for i in range(len(training_modes)):
        stats = dataset_stats[:,i]
        avg_stats.append(np.average(stats, axis=0))
        print(f'\nAverage results on {model_name} by {feature_type} features:')
        print(f"APS: {avg_stats[i][0]:0.3f}, AUC: {avg_stats[i][1]:0.3f}, Pre APS: {avg_stats[i][2]:0.3f}, Rec APS: {avg_stats[i][3]:0.3f}, Pre AUC: {avg_stats[i][4]:0.3f}, Rec AUC: {avg_stats[i][5]:0.3f}\n")
    return avg_stats


def run_all_experiments(dataset_name, model_names, feature_types, training_modes):
    metrics = ['APS', 'AUC','PreAPS','RecAPS','PreAUC','RecAUC']
    results = []
    for i in range(len(training_modes)):
        results.append(pd.DataFrame(data=np.zeros((len(model_names),len(feature_types)*len(metrics))), 
                            columns=pd.MultiIndex.from_product([feature_types, metrics]), index=model_names))

    time_now = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S') 


    for  model_name in model_names:
        for feature_name in feature_types:
            stats = experiments_on_dataset(dataset_name, model_name, feature_name, training_modes, exp_time=time_now)
            for i in range(len(training_modes)):
                results[i].loc[model_name, feature_name] = stats[i]
                results[i].to_csv(f'./my_results/{time_now}_{training_modes[i]}.csv')
                print(f'results are saved to "{time_now}_{training_modes[i]}.csv"')

if __name__ == '__main__':
    datasets = ['edBB', 'MyDataset','CombinedDataset']
    feature_types = ['original','combined','array', 'angle_distance', 'angle', 'distance']
    body_parts = ['full', 'upper']
    body_part = body_parts[1]
    model_names = ['UnivarAutoEncoder','AutoEncoder', 'LSTMED', 'TcnED', 'VAE_LSTM','MSCRED', 'OmniAnoAlgo']#, 'PcaRecons', 'RawSignalBaseline']
    # distr_names = ['univar_gaussian', 'univar_lognormal', 'univar_lognorm_add1_loc0', 'chi']
    # thresh_methods = ['top_k_time']#, 'best_f1_test', 'tail_prob']
    train_modes = ['video-specific', 'global','combined']
    np.random.seed(0)
    # experiment_on_folder(datasets[-1], model_names[0], folder_idx=1, feature_type=feature_types[0] ,mode=train_modes[0], load_saved=False)


    # experiments_on_dataset(datasets[-1], model_names[3], feature_types[1], train_modes[0])

    run_all_experiments(datasets[-1], model_names[-2:], feature_types, train_modes)
