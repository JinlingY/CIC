from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.decomposition import FactorAnalysis
from tqdm.auto import tqdm
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
import numpy
import torch.nn as nn
import pickle
import other_functions as OF
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.decomposition import FactorAnalysis
from tqdm.auto import tqdm
import torch.nn.functional as F
from sklearn.neighbors import KernelDensity

from scipy.stats import pearsonr
from scipy.stats import kendalltau  
from scipy.stats import spearmanr
from scipy.signal import correlate
from sklearn.metrics import normalized_mutual_info_score
import seaborn as sns
import pandas as pd
from sklearn.metrics import roc_curve, auc,precision_recall_curve

def Normalize(x,y):
    x0=x*x/(x*x+y*y)
    y0=y*y/(x*x+y*y)
    return x0, y0

def Ortho_loss(x, y):
    cosine_similarity = torch.nn.functional.cosine_similarity(x, y, dim=1)
    orthogonality_loss = torch.mean((cosine_similarity)**2)
    return orthogonality_loss

def cal_kld_loss(mean, logvar):
    return -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=1).mean()

class MinMaxNormalize:
    """
    Minmax normalization for a torch tensor to (-1, 1)

    Parameters
    ----------
    tensor: torch.Tensor
        The tensor to be normalized
    dim: int
        The dimension to be normalized
    """

    def __init__(self, x, dim=0, dtype="float64"):
        self.dim = dim
        self.dtype = dtype
        x = x.astype(self.dtype)
        self.min = np.min(x, axis=dim, keepdims=True)
        self.max = np.max(x, axis=dim, keepdims=True)

    def normalize(self, x):
        x = x.astype(self.dtype)
        return 2 * (x - self.min) / (self.max - self.min) - 1

    def denormalize(self, x):
        x = x.astype(self.dtype)
        return (x + 1) / 2 * (self.max - self.min) + self.min
def CCA(A, B):
    #correlations = []
    A=np.transpose(A)
    min_samples = min(A.shape[0], B.shape[0])    
    A = A[:min_samples]  
    B = B[:min_samples]

    cca = CCA(n_components=1)
    
    cca.fit(A, B)
   
    A_train_r, B_train_r = cca.transform(A, B)
    #canonical_correlation = cca.score(A, B)
    canonical_correlation=(np.corrcoef(A_train_r[:, 0], B_train_r[:, 0])[0, 1])

    canonical_correlation = np.clip(canonical_correlation, -1, 1)
    return canonical_correlation
def CCA2(A, B):
    A=np.transpose(A)
    min_samples = min(A.shape[0], B.shape[0])    
    A = A[:min_samples]
    B = B[:min_samples]    

    ica_A = FastICA(n_components=1)
    A_ica = ica_A.fit_transform(A)

    ica_B = FastICA(n_components=1)
    B_ica = ica_B.fit_transform(B)


    ica_corr = np.corrcoef(A_ica.squeeze(), B_ica.squeeze())[0, 1]
    return ica_corr
def CCA4(A, B):
    A=np.transpose(A)
    min_samples = min(A.shape[0], B.shape[0])    
    A = A[:min_samples]
    B = B[:min_samples]    
 
    corr_matrix = np.corrcoef(A.T, B.T)

    eigenvalues, eigenvectors = np.linalg.eig(corr_matrix)

    max_eigenvalue_index = np.argmax(eigenvalues)
    max_eigenvector = eigenvectors[:, max_eigenvalue_index]
    overall_correlation = np.max(np.abs(max_eigenvector))
    return overall_correlation
def plot_score_matrix(mat, labels=None, annot=True, fontsize=8, fmt='.3f', linewidths=1, tight_layout=True,
                      cmap='YlGnBu', tick_bin=1, ticklabel_rotation=0, ax=None, figsize=(6, 5),
                      diag_line=False, **kwargs):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(mat, annot=annot,annot_kws={'size': fontsize}, fmt=fmt, linewidths=linewidths, cmap=cmap, ax=ax, **kwargs)

    if labels is None:
        labels = np.arange(0, len(mat), tick_bin)

    ticks = np.arange(0+.5, len(mat)+.5, tick_bin)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    if len(ticks) == len(labels):
        ax.set_xticklabels(labels, rotation=ticklabel_rotation,fontsize=fontsize)
        ax.set_yticklabels(labels, rotation=ticklabel_rotation,fontsize=fontsize)

    if diag_line:
        ax.plot([0, len(mat)], [0, len(mat)], c='gray')

    if tight_layout:
        plt.tight_layout()

    return ax
def plot_annot_square(idx, ax=None, **kwargs):
    ys, xs = idx
    if ax is None:
        fig, ax = plt.subplots()

    for x, y in zip(xs, ys):
        ax.plot([x, x, x+1, x+1, x], [y, y+1, y+1, y, y], **kwargs)
    return ax
def find_differences(A, B):
    differences = (A != B)
    rows, cols = np.where(differences)
    pos=[rows, cols]
    return pos


def delay_embedding2(data, embedding_dim, lag):

    data = np.array(data)
    n = data.shape[0]
    m = n - (embedding_dim - 1) * lag

    embedded_data = np.zeros((m, embedding_dim))
  
    for i in range(m):
        for j in range(embedding_dim):
            embedded_data[i, j] = data[i+embedding_dim - 1 - (j) * lag]
    embedded_data = embedded_data[:, ::-1]
    return embedded_data

def generate_embedd_data(data,embedding_dim, time_delay):
    X_dict0 = {};X_dict1 = {}
    n_dataset0 = TensorDataset();n_dataset1 = TensorDataset()
    for i in range(data.shape[1]):
        #X{i} =data[:,i]
        
        x_delay=delay_embedding2(data[:,i], embedding_dim, time_delay)
        x_delay0 = x_delay[:-2, :]; x_delay1 = x_delay[2:, :]#t-1和t
        normalizer_x0 = MinMaxNormalize(x_delay0); normalizer_x1 = MinMaxNormalize(x_delay1)
        n_x0 = normalizer_x0.normalize(x_delay0); n_x1 = normalizer_x1.normalize(x_delay1)
        n_x0 =torch.from_numpy(n_x0); n_x1 =torch.from_numpy(n_x1)
        X_dict0[f"X{i}"] =n_x0.float(); X_dict1[f"X{i}"] =n_x1.float()
        n_dataset0.tensors += (X_dict0[f"X{i}"],)
        n_dataset1.tensors += (X_dict1[f"X{i}"],)

    return X_dict0, X_dict1, n_dataset0, n_dataset1

def logistic_3_system(rx, ry, rw, noise, betaxy, betaxz, betayx, betayz, num_steps):
   
    Y = np.empty(num_steps)
    X = np.empty(num_steps)
    Z = np.empty(num_steps)

    X[0] = 0.4
    Y[0] = 0.4
    Z[0] = 0.4
    data0=np.zeros((num_steps, 3))
    for j in range(1, num_steps):
       
        X[j] = X[j-1] * (rx - rx * X[j-1]- betaxy * Y[j-1]- betaxz * Z[j-1]) + np.random.normal(0, noise)
        Y[j] = Y[j-1] * (ry - ry * Y[j-1]- betayx * X[j-1]- betayz * Z[j-1]) + np.random.normal(0, noise)
        Z[j] = Z[j-1] * (rw - rw * Z[j-1]) + np.random.normal(0, noise)
    
    data0[:,0]=X;data0[:,1]=Y;data0[:,2]=Z
    return data0


def logistic_8_system( noise, beta, num_steps):
    r=np.array([3.9, 3.5, 3.62, 3.75, 3.65, 3.72, 3.57, 3.68])
    data0=np.zeros((num_steps, 8))
   
    X1 = np.empty(num_steps);X2 = np.empty(num_steps);X3 = np.empty(num_steps);X4 = np.empty(num_steps)
    X5 = np.empty(num_steps);X6 = np.empty(num_steps);X7 = np.empty(num_steps);X8 = np.empty(num_steps)

    X1[0] = 0.4;X2[0] = 0.4;X3[0] = 0.4;X4[0] = 0.4;X5[0] = 0.4;X6[0] = 0.4;X7[0] = 0.4;X8[0] = 0.4
    for j in range(1, num_steps):
        
        X1[j] = X1[j-1] * (r[0] - r[0] * X1[j-1]) + np.random.normal(0, noise)
        X2[j] = X2[j-1] * (r[1] - r[1] * X2[j-1]) + np.random.normal(0, noise)
        X3[j] = X3[j-1] * (r[2] - r[2] * X3[j-1]- beta * X1[j-1]- beta * X2[j-1]) + np.random.normal(0, noise)
        X4[j] = X4[j-1] * (r[3] - r[3] * X4[j-1]- beta * X2[j-1]) + np.random.normal(0, noise)
        X5[j] = X5[j-1] * (r[4] - r[4] * X5[j-1]- beta * X3[j-1]) + np.random.normal(0, noise)
        X6[j] = X6[j-1] * (r[5] - r[5] * X6[j-1]- beta * X3[j-1]) + np.random.normal(0, noise)
        X7[j] = X7[j-1] * (r[6] - r[6] * X7[j-1]- beta * X6[j-1]) + np.random.normal(0, noise)
        X8[j] = X8[j-1] * (r[7] - r[7] * X8[j-1]- beta * X6[j-1]) + np.random.normal(0, noise)
    data0[:,0]=X1;data0[:,1]=X2;data0[:,2]=X3;data0[:,3]=X4
    data0[:,4]=X5;data0[:,5]=X6;data0[:,6]=X7;data0[:,7]=X8
    return data0


def GRN_Dream4_data(n_nold = 10, Net_num=10):
    GRN_Net = {};GRN_data = {}
    #n_nold = 10
    #Net_num=10
    for j in range(Net_num):
        Net_posi = pd.read_csv(f'/.../data/gene/net{j+1}_truth.tsv', delimiter='\s+', header=None)
        data_posi = np.load(f'/.../data/gene/net{j+1}_expression.npy')
        #Network
        Net_posi.iloc[:, :2] = Net_posi.iloc[:, :2].apply(lambda x: [int(v.split('G')[1]) for v in x])
        edges_truth = Net_posi[[0, 1]].values.astype('int')
        gold_mat = OF.edges_to_mat(edges_truth - 1, n_nold)
        #truths = skip_diag_tri(gold_mat).ravel()
        GRN_Net[f"Net{j}"] =torch.from_numpy(gold_mat)
        #data
        GRN_data[f"Net{j}"]=data_posi
    return GRN_Net, GRN_data


def Confounder(Net):
    N=Net.shape[1]
    Updata_Net=Net.clone()
    Net_confd=torch.zeros(N,N)
    for i in range(N):
        for j in range(N):
            if i!=j:
                for k in range(N):
                    if k!=j:
                        if Net[i,j]==1:
                            if Net[i,k] == 1: 
                                Net_confd[j,k]=i
                                if Net[j,k]!=1:
                                    Updata_Net[j,k]=2   
                                if Net[k,j]==0:
                                    Updata_Net[k,j]=2                               
    return Updata_Net,Net_confd 

def confound_CCA(data,Net_ground,Net_confd,out_s,Z0):
    N=Net_ground.shape[1]
    CCA=torch.zeros(N,N)
    for i in range(N):
        for j in range(N):
            if i != j:   
                if Net_ground[i,j]==2:
                    index = torch.round(Net_confd[i, j]).long()
                    A=data[:,int(torch.round(Net_confd[i,j]))].reshape(1, -1)
                    B=out_s[f"s{i},s{j}"].detach().cpu().numpy()
                    B[np.isnan(B)]=0
                    #B=torch.where(torch.isnan(B), torch.tensor(0.0), B).numpy()
                    CCA[i,j] = CCA(A, B)
                if Net_ground[i,j]!=2:
                    B=out_s[f"s{i},s{j}"].detach().cpu().numpy()
                    B[np.isnan(B)]=0
                    #B=torch.where(torch.isnan(B), torch.tensor(0.0), B).numpy()
                    CCA[i,j] = CCA(data[:,Z0].reshape(1, -1), B)
    CCA_label2= CCA[Net_ground == 2].tolist()
    CCA_label0= CCA[Net_ground != 2].tolist()
    CCA_data=CCA_label2+CCA_label0
    df = pd.DataFrame({'CCA': CCA_data, 'label': [2] * len(CCA_label2) + [0] * len(CCA_label0)})
    df['label'] = df['label'].replace({2: 'Confounder', 0: 'Non-confounder'})
    return CCA,df

def confound_CCA1(data,Net_ground,Net_confd,out_s,Z0):
    N=Net_ground.shape[1]
    CCA=torch.zeros(N,N)
    for i in range(N):
        for j in range(N):
            if i != j:   
                if Net_ground[i,j]==2:
                    index = torch.round(Net_confd[i, j]).long()
                    A=data[:,int(torch.round(Net_confd[i,j]))].reshape(1, -1)
                    B=out_s[f"s{i},s{j}"].detach().cpu().numpy()
                    B[np.isnan(B)]=0
                    #B=torch.where(torch.isnan(B), torch.tensor(0.0), B).numpy()
                    CCA[i,j] = CCA4(A, B)
                if Net_ground[i,j]!=2:
                    B=out_s[f"s{i},s{j}"].detach().cpu().numpy()
                    B[np.isnan(B)]=0
                    #B=torch.where(torch.isnan(B), torch.tensor(0.0), B).numpy()
                    CCA[i,j] = CCA4(data[:,Z0].reshape(1, -1), B)
    CCA_label2= CCA[Net_ground == 2].tolist()
    CCA_label0= CCA[Net_ground != 2].tolist()
    CCA_data=CCA_label2+CCA_label0
    df = pd.DataFrame({'CCA': CCA_data, 'label': [2] * len(CCA_label2) + [0] * len(CCA_label0)})
    df['label'] = df['label'].replace({2: 'Confounder', 0: 'Non-confounder'})
    return CCA,df
def confounder_index(Net,Net_ground):
    #y_pred=Net
    TP = np.count_nonzero((Net.reshape(-1) == 1) & (Net_ground.reshape(-1) == 1))#
    FN = np.count_nonzero((Net.reshape(-1) == 0) & (Net_ground.reshape(-1) == 1))#
    TN = np.count_nonzero((Net.reshape(-1) == 0) & (Net_ground.reshape(-1) == 0))#
    FP = np.count_nonzero((Net.reshape(-1) == 1) & (Net_ground.reshape(-1) == 0))#
    precision1, recall1, thresholds = precision_recall_curve(Net_ground[~torch.eye(Net_ground.shape[0], dtype=bool)].reshape(-1), Net[~torch.eye(Net.shape[0], dtype=bool)].reshape(-1))
    precision0=(TN / (TN + FP+0.000001))

    recall0=(TN / (TN + FN+0.000001))

    accuracy=(TP+TN) / (TP + TN + FP+ FN)
    f1=(2*precision0+recall1)/(precision1+recall1+0.000001)  

    return precision0, precision1, recall0, recall1, accuracy, f1