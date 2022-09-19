from my_data_functions import load_data_partial, get_results, load_edBB_all, get_seqs_events, get_events_accuracy
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
body_parts = ['full', 'upper']
body_part = body_parts[1]
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

def collect_results(score_t_train, score_t_test, y_test, thres_method='tail_prob'):
    test_anom_frac = (np.sum(y_test)) / len(y_test)
    true_events = get_events(y_test)
    logger = None
    composite_best_f1 = True
    thres_config_dict = {'tail_prob':{"tail_prob": 4}}
    logger = logging.getLogger('test')
    opt_thres, pred_labels, avg_prec, auroc = threshold_and_predict(score_t_test, y_test, true_events=true_events,
                                                                        logger=logger,
                                                                        test_anom_frac=test_anom_frac,
                                                                        thres_method=thres_method,
                                                                        point_adjust=False,
                                                                        score_t_train=score_t_train,
                                                                        thres_config_dict=thres_config_dict,
                                                                        return_auc=True,
                                                                        composite_best_f1=composite_best_f1)
    acc = recall_score(y_test, pred_labels,labels=[1])
    print(f"APS: {avg_prec:0.3f}, AUROC: {auroc:0.3f}, {thres_method} ACC: {acc:0.3f}")
    return avg_prec, auroc, acc

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
 
def experiment_on_folder(dataset_name, model_name, folder_idx, feature_type, 
                        score_distr_name='univar_gaussian', mode='singlepass',pretrained_model=None):

    print(f'\n\nprocessing folder {folder_idx}...')

    x_data, y_data = load_data_partial(dataset_name, folder_idx, feature_type, body_part, train_ratio=0.0)

    features_dim = x_data.shape[1]
    out_dir=setup_out_dir(dataset_name, model_name, feature_type, folder_idx)
    model = get_model(model_name,features_dim, out_dir=out_dir)
    best_loss = None
    if mode == 'pretrain':
        model = pretrained_model

    else:
        saved_models = glob.glob(out_dir + 'trained_model_*_.pt')
        if len(saved_models)>0:
            if hasattr(model, 'torch_save') and model.torch_save == True:
                model.model = torch.load(saved_models[0], model.device)
                best_loss = np.float(os.path.basename (saved_models[0]).split('_')[2])
                

    x_seqs = get_sub_seqs(x_data.values, seq_len=sequence_length)
    y_seqs = np.array([1 if sum(y_data.iloc[i:i + sequence_length])>0 else 0 for i in range(len(x_seqs))])
    e_seqs = get_seqs_events(y_data.values, sequence_length)
    train_ratio = 0.3
    select_ratio = 0.3
    top_ratio = 0.1
    top_k = int(len(x_seqs) * top_ratio)
    # top_k = np.sum(y_seqs)
    val_ratio = 0.2
    i = 1
    x_train = None
    results = []
    
    while  True:
        if x_train is None:
            n_train = int(len(x_seqs) * train_ratio)
            x_train = x_seqs[:n_train]
            y_train = y_seqs[:n_train]
            x_test = x_seqs[n_train:]
            y_test = y_seqs[n_train:]
            e_test = e_seqs[n_train:]

            n_train = int(len(x_train) * (1-val_ratio))
            x_val = x_train[n_train:]
            y_val = y_train[n_train:]
            x_train = x_train[:n_train]
            y_train = y_train[:n_train]

            x_train_best = x_train.copy()
            y_train_best = y_train.copy()
            x_test_best = x_test.copy()
            y_test_best = y_test.copy()
            e_test_best = e_test.copy()
            
            if model.model == None:
                best_loss = model.fit_sequences(x_train, x_val)
                if hasattr(model,'torch_save') and  model.torch_save == True:
                    torch.save(model.model, out_dir+f'trained_model_{best_loss}_.pt')
            best_val_loss = best_loss

        else:
            n_train = int(len(x_seqs) * select_ratio)
                
            x_train = np.concatenate((x_train, x_seqs[:n_train]), axis=0)
            y_train = np.concatenate((y_train, y_seqs[:n_train]), axis=0)
            # x_train = x_seqs[:n_train]
            # y_train = y_seqs[:n_train]
            x_test = x_seqs[n_train:]
            y_test = y_seqs[n_train:]
            e_test = e_seqs[n_train:]
            
            best_val_loss = model.fit_sequences(x_train, x_val)
       
        if mode != 'multipass':
            break
        
        print('step',i)
        if i>1:
            if best_loss > best_val_loss:
                print(f'model improved to loss: {best_val_loss:0.5f}')
                best_loss = best_val_loss
                x_test_best = x_test.copy()
                y_test_best = y_test.copy()
                e_test_best = e_test.copy()
                x_train_best = x_train.copy()
                y_train_best = y_train.copy()
            else:
                print(f'Exit with no model improvement')
                break

        test_scores = test_model(model, x_train, x_test, score_distr_name)

        
        if len(x_test_best) < top_k:
            print(f'Exit with top_k limitation')
            break

        test_idx = np.argsort(test_scores)
        x_seqs = x_test[test_idx]
        y_seqs = y_test[test_idx]
        try:
            e_seqs = np.array(e_test)[test_idx]
        except:
            e_seqs = e_test[test_idx]

        i += 1
           
            
    test_preds = model.predict_sequences(x_test_best )
    train_preds = model.predict_sequences(x_train_best )
    if score_distr_name == 'normalized_error':
        test_scores = get_normalized_scores(train_preds['error_tc'], test_preds['error_tc'])
    else:
        if test_preds['score_t'] is None:
            train_scores, test_scores = get_fitted_scores(train_preds['error_tc'], test_preds['error_tc'])  
        else:
            train_scores, test_scores = train_preds['score_t'], test_preds['score_t']
    test_scores = test_scores[sequence_length-1:]

    if mode == 'combined':
        pre_test_preds = pretrained_model.predict_sequences(x_test_best )
        pre_train_preds = pretrained_model.predict_sequences(x_train_best )
        if pre_test_preds['score_t'] is None:
            pre_train_scores, pre_test_scores = get_fitted_scores(pre_train_preds['error_tc'], pre_test_preds['error_tc'])  
        else:
            pre_train_scores, pre_test_scores = pre_train_preds['score_t'], pre_test_preds['score_t']
        
        pre_test_scores = pre_test_scores[sequence_length-1:]

        test_scores = (test_scores + pre_test_scores) / 2   
        # test_scores = np.max((test_scores,pre_test_scores),axis=0)

        

        # sort_idx = np.argsort(test_scores)
        # pre_sort_idx = np.argsort(pre_test_scores)
        # i = 1
        # shared_idx = list(set(sort_idx[-top_k:]).intersection(pre_sort_idx[-top_k:]))
        # test_scores = test_scores[shared_idx]
        # y_test = y_test[shared_idx]

    aps, auroc, acc = get_results(y_test_best , test_scores, top_k= top_k, print_results=False)
    e_acc = get_events_accuracy(e_test_best, test_scores, top_k)
    results.append(f'\nfinal test : APS={aps:0.3f}, AUROC={auroc:0.3f}, ACC={acc:0.3f}, Event ACC={e_acc:0.3f}')
    with open('readme.txt', 'a') as f:
        f.write('readme')

    print(*results)
    return aps, auroc, acc, e_acc

def experiments_on_dataset(dataset_name, model_name, feature_type, distr_name='normalized_error', mode='singlepass', save_results=False, exp_time=''):
    pretrained_model = None
    if mode == 'pretrain' or mode == 'combined':
        x_train, _ = load_edBB_all(feature_type, body_part)
        features_dim = x_train.shape[1]
        out_dir = setup_out_dir('edBB',model_name, feature_type)
        pretrained_model = get_model(model_name, features_dim, out_dir)
        saved_model_dir= out_dir + 'trained_model.pt'
        if  os.path.exists(saved_model_dir):
            print('loading pre-trained model on edBB...')
            pretrained_model.model = torch.load(saved_model_dir, pretrained_model.device)
        else:
            print('pre-training model on edBB...')
            pretrained_model.fit(x_train)
            if hasattr(pretrained_model,'torch_save') and  pretrained_model.torch_save == True:
                print('saving pre-trained model...')
                torch.save(pretrained_model.model, saved_model_dir)
    n=38
    time_now = exp_time if exp_time != '' else datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S')
    metrics = ['Acc','APS', 'AUROC', 'EACC']
    results_df = pd.DataFrame(data=np.zeros((n,len(metrics))), 
                            columns=metrics, index=list(range(1,n+1)))

    aps_avg=[]
    auroc_avg = []
    acc_avg = []
    e_acc_avg = []
    for folder_idx in range (1, n+1):
 
        aps,auroc, acc, e_acc = experiment_on_folder(dataset_name, model_name, folder_idx, feature_type, distr_name, mode, pretrained_model)
        if save_results:
            results_df.loc[folder_idx] = [acc, aps, auroc, e_acc]

            results_df.to_csv(f'./my_results/intermediate/{time_now}_{mode}_{model_name}_{feature_type}.csv')
            print(f'intermediate results are saved to "{time_now}_{mode}_{model_name}_{feature_type}.csv"')

        aps_avg.append(aps)
        auroc_avg.append(auroc)
        acc_avg.append(acc)
        e_acc_avg.append(e_acc)

    aps_avg = np.mean(aps_avg)
    auroc_avg = np.mean(auroc_avg)
    acc_avg = np.mean(acc_avg)
    e_acc_avg = np.mean(e_acc_avg)

    print(f'\nAverage results on {dataset_name} of {model_name} by {feature_type} features:')
    print(f"APS: {aps_avg:0.3f}, AUROC: {auroc_avg:0.3f}, ACC: {acc_avg:0.3f}, EACC: {e_acc_avg:0.3f}\n")
    return aps_avg, auroc_avg, acc_avg, e_acc_avg


def run_all_experiments(dataset_name, model_names, feature_types, distr_name, mode, time_label=''):
    metrics = ['Acc','APS', 'AUROC','EACC']
    results = pd.DataFrame(data=np.zeros((len(model_names),len(feature_types)*len(metrics))), 
                            columns=pd.MultiIndex.from_product([feature_types, metrics]), index=model_names)
    if time_label=='':
        time_now = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S') 
    else:
        time_now = time_label 

    for  model_name in model_names:
        for feature_name in feature_types:
            aps, roc, acc, e_acc = experiments_on_dataset(dataset_name, model_name, feature_name, distr_name, mode, save_results=True, exp_time=time_now)
            results.loc[model_name, feature_name] = [acc, aps, roc, e_acc ]
            results.to_csv(f'./my_results/{time_now}_{mode}.csv')
            print(f'results are saved to "{time_now}_{mode}.csv"')

if __name__ == '__main__':
    datasets = ['edBB', 'MyDataset']
    feature_types = ['original','array', 'angle_distance', 'angle', 'distance']
    model_names = ['UnivarAutoEncoder','AutoEncoder', 'LSTMED', 'TcnED', 'VAE_LSTM','MSCRED', 'OmniAnoAlgo']#, 'PcaRecons', 'RawSignalBaseline']
    distr_names = ['normalized_error', 'univar_gaussian']#, 'univar_lognormal', 'univar_lognorm_add1_loc0', 'chi']
    thresh_methods = ['top_k_time']#, 'best_f1_test', 'tail_prob']
    train_modes = ['singlepass', 'multipass', 'pretrain', 'combined']
    np.random.seed(0)
    time_label=''
    # time_label='2022_08_29_10_33_01'
    # experiment_on_folder(datasets[1], model_names[1], folder_idx=17, feature_type=feature_types[2] ,mode=train_modes[0])
    # experiments_on_dataset(datasets[1], model_names[1], feature_types[3], distr_names[1], train_modes[0],True)
    run_all_experiments(datasets[1], model_names[-1:], feature_types[-1:], distr_names[1], train_modes[2], time_label)
