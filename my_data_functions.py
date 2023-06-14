import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
import numpy as np
from sklearn.metrics import roc_curve, classification_report, precision_recall_curve 


def load_edBB_all(feature_type, body_part, normals_only=True):
    n = 38
    x_all, y_all = [], []
    for folder_idx in range(1, n+1):
        x_folder, y_folder = load_data_partial('edBB', folder_idx, feature_type, body_part, train_ratio=1.0)
        x_all.append(x_folder.to_numpy())
        y_folder = y_folder.to_numpy().reshape((-1,1))
        y_all.append(y_folder)
    x_all = np.vstack(x_all)
    y_all = np.vstack(y_all)
    if normals_only:
        idxs = np.where(y_all == 0)[0]
        x_all = x_all[idxs]
        y_all = y_all[idxs]
    return pd.DataFrame(x_all), pd.DataFrame(y_all)

def load_data_all(feature_type, body_part):
    n = 91
    x_all, y_all = [], []
    for folder_idx in range(1, n+1):
        x_folder, y_folder,_ ,_ = load_data_partial('CombinedDataset', folder_idx, feature_type, body_part, train_ratio=0.3)
        x_all.append(x_folder.to_numpy())
        y_folder = y_folder.to_numpy().reshape((-1,1))
        y_all.append(y_folder)
    x_all = np.vstack(x_all)
    y_all = np.vstack(y_all)
    
    return pd.DataFrame(x_all), pd.DataFrame(y_all)

def load_data_partial(dataset, folder_idx, feature_type, body_part, train_ratio=0.3):

    base_path = f'./data/{dataset}/{folder_idx:02d}'
    if dataset == 'edBB':
        base_path = f'./data/{dataset}'

    if feature_type == 'original' or feature_type == 'combined':
        #load skeletons
        coordinates_path = base_path + '/coordinates_movnet.csv'
        if dataset == 'edBB':
            coordinates_path = base_path + f'/coordinates_movnet/{folder_idx:02d}.csv'
        time_series = pd.read_csv(coordinates_path, header=None, index_col=0)

        if body_part == 'upper':
            time_series = time_series.iloc[:,:22]
            if feature_type == 'combined':
                rw = time_series.iloc[:, [20, 21]]
                rw.set_axis([2,3],axis=1,inplace=True)
                lw = time_series.iloc[:, [18, 19]]
                lw.set_axis([4,5],axis=1,inplace=True)
                le = time_series.iloc[:, [6, 7]]
                le.set_axis([0,1],axis=1,inplace=True)
                re = time_series.iloc[:, [8, 9]]
                re.set_axis([0,1],axis=1,inplace=True)
                n = time_series.iloc[:, [0, 1]]
                n.set_axis([0,1],axis=1,inplace=True)
                h = (le+re) / 2
                n = n - h
                time_series = pd.concat([n,rw,lw], axis=1)
        elif body_part == 'head':
            time_series = time_series.iloc[:, :10]
        elif body_part == 'left_hand':
            time_series = time_series.iloc[:, [10, 11, 14, 15, 18, 19]]
        elif body_part == 'right_hand':
            time_series = time_series.iloc[:, [12, 13, 16, 17, 20, 21]]
        elif body_part == 'left_wrist':
            time_series = time_series.iloc[:, [18, 19]]
        elif body_part == 'right_wrist':
            time_series = time_series.iloc[:, [20, 21]]
        

    elif feature_type == 'array':
        features_path = base_path + f'/array_features.csv'
        if dataset == 'edBB':
            features_path = base_path + f'/array_features/{folder_idx:02d}.csv'
        time_series = pd.read_csv(features_path, header=None, index_col=0)
        if body_part == 'upper':
            time_series = time_series.iloc[:, :22]
    else:
        features_path = base_path + f'/angle_distance_features.csv'
        if dataset == 'edBB':
            features_path = base_path + f'/angle_distance_features/{folder_idx:02d}.csv'
        data = pd.read_csv(features_path, header=None, index_col=0)
        
        if feature_type == 'angle_distance':
            time_series = data
            if body_part == 'upper':
                time_series = time_series.iloc[:, :22]
        elif feature_type == 'angle_plus_distance':
            data = data.to_numpy()
            ts1 = data[:,list(range(0, data.shape[1], 2))]
            ts2 = data[:,list(range(1, data.shape[1], 2))]
            time_series =pd.DataFrame(ts1+ts2)
            if body_part == 'upper':
                time_series = time_series.iloc[:, :11]
        else:
            if feature_type == 'angle':
                rem = 0
            else:
                rem = 1
            time_series = data.iloc[:,list(range(rem, data.shape[1], 2))]
            if body_part == 'upper':
                time_series = time_series.iloc[:, :11]
            elif body_part == 'head':
                time_series = time_series.iloc[:, :4]
    
    #load labels
    labels_path = base_path + '/labels.csv'
    if dataset == 'edBB':
        labels_path=base_path+f'/labels_movnet/{folder_idx:02d}.csv'

    labels = pd.read_csv(labels_path, header=None, index_col=0)

    if train_ratio == 1.0 or train_ratio == 0.0:
        return time_series, labels

    n_data = len(labels)
    n_train = int(n_data * train_ratio)
    
    x_train = time_series.iloc[:n_train]    
    y_train = labels.iloc[:n_train]
    x_test = time_series.iloc[n_train:]
    y_test = labels.iloc[n_train:]
 

    if np.sum(y_train.values) > 0:
        idxs = np.where(y_train.to_numpy() == 1)[0]
        idxs2 = np.where(y_test.to_numpy() == 0)[0][:len(idxs)]
        x_train2 = x_train.to_numpy()
        x_test2 = x_test.to_numpy()
        x_train2[idxs] = x_test2[idxs2]
        x_test2[idxs2] = x_train.iloc[idxs].values
        x_train = pd.DataFrame(x_train2)
        x_test = pd.DataFrame(x_test2)
        # xs = x_train.iloc[idxs,:]
        
        # x_train.iloc[idxs,:] = x_test.iloc[idxs2,:].values
        # x_test.iloc[idxs2,:] = xs.values
        y_train.iloc[idxs] = 0
        y_test.iloc[idxs2] = 1


        print('some records of training data exchanged.')

    return x_train, y_train, x_test, y_test

def get_results(y_real, pred_scores, top_k, print_results=True):
    aps = average_precision_score(y_real, pred_scores, pos_label=1)
    roc = roc_auc_score(y_real, pred_scores)
    idxs = np.argsort(pred_scores)
    acc=0
    if top_k>0:
        idxs = idxs[-top_k:]
        y_pred = y_real[idxs]
        acc = np.sum(y_pred) / top_k
    if print_results:
        print(f"APS: {aps:0.3f}, AUROC: {roc:0.3f}, Top-{top_k} Accuracy: {acc:0.3f}")
    return aps, roc, acc

def get_events_accuracy(events, scores, top_k):
    true_events = set(events[0])
    for i in range(1,len(events)):
        for x in events[i]:
            if x!=0:
                true_events.add(x)

    idxs = np.argsort(scores)
    idxs = idxs[-top_k:]
    e_pred = []
    for i in range(len(idxs)):
        e_pred.append(events[idxs[i]])

    pred_events = set(e_pred[0])
    for i in range(1,len(e_pred)):
        for x in e_pred[i]:
            if x!=0:
                pred_events.add(x)
    acc = len(pred_events) / len(true_events)
    print('total events:',len(true_events), 'detected:',len(pred_events))
    return acc

def test(dataset, feature_type):
    x_train, y_train, x_test, y_test = load_data_partial(dataset,folder_idx=1,feature_type=feature_type,body_part="upper")
    print('loaded from',dataset,', features:',feature_type)        
    print(f'x_train: {x_train.shape}, y_train: {y_train.shape}, x_test: {x_test.shape}, y_test: {y_test.shape}')

def get_seqs_events(y_data,sequence_length):
    labels = []
    new_label = 0
    i = 0
    while i < len(y_data):
        while i < len(y_data) and y_data[i] == 0:
            i += 1
            labels.append(0)
        if i >= len(y_data):
            break
        new_label += 1
        while i < len(y_data) and y_data[i] == 1:
            i += 1
            labels.append(new_label)
    # x_seqs = [y_data[i:i + sequence_length] for i in range(len(y_data)-sequence_length)]
    y_seqs = [list(set(labels[i:i+sequence_length])) for i in range(len(labels)-sequence_length+1)]
    for i in range(len(y_seqs)):
        if  len(y_seqs[i]) > 1:
            y_seqs[i].remove(0)
    return y_seqs #, x_seqs

def classify_scores(scores, labels, method='pr'):

    if method == 'roc':
        fpr, tpr, thresholds = roc_curve(labels, scores)
        # Find the threshold that maximizes performance
        optimal_idx = np.argmax(tpr - fpr)
    else:
        precision, recall, thresholds = precision_recall_curve(labels, scores)
        f1_scores = 2 * (precision * recall) / (precision + recall)
        f1_scores[np.isnan(f1_scores)] = 0
        optimal_idx = np.argmax(f1_scores)

    optimal_threshold = thresholds[optimal_idx]

    # Classify test samples based on selected threshold
    y_pred = (scores >= optimal_threshold).astype(int)
    print('Results by ', method,':')
    print(classification_report(labels,y_pred))
    return classification_report(labels,y_pred, output_dict=True)

if __name__ == '__main__':
    # test('MyDataset','distance')
    x, y = load_edBB_all('original','upper')  
    print('x:',x.shape)
    x, y = load_data_all('original','upper')  
    print('x:',x.shape)

    # for i in range(1,92):
    #     print('loading folder',i)
    #     x, y,xt,yt = load_data_partial('CombinedDataset',i,'original','combined') 
        # print('test size:',len(xt),'anomaly labels:',sum(y))
        # print('X:',x.shape,'y:',y.shape)

    # y_data = np.array([0,0,0,1,1,1,0,1,1,1,0,0,1,1,0,0])
    # seq_len = 8
    # y, x = get_seqs_lbls(y_data, seq_len)
    # print('y=\n',y)
    # print('x=\n',x)

 