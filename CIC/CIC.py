
#Dynamical causality under invisible confounders(CIC)
import torch
from utils import generate_embedd_data
from model import DualVAE,train_dualvae,evaluate



def CIC(data,weights,xy_dim,z_dim,hid_dim,embedding_dim,time_delay,T,num_epochs,device):
    N=data.shape[1]
    X_dict0_T, X_dict1_T, n_dataset0_T, n_dataset1_T=generate_embedd_data(data,embedding_dim, time_delay)
    X_dict0_V, X_dict1_V, n_dataset0_V, n_dataset1_V=generate_embedd_data(data,embedding_dim, time_delay)
    count = 0
    MSE_s0=torch.zeros(N,N)
    MSE_zx0=torch.zeros(N,N)
    MSE_s=torch.zeros(N,N);  MSE_zx=torch.zeros(N,N) 
    av_MSE_s=torch.zeros(N,N)
    av_MSE_zx=torch.zeros(N,N)
    causal_index=torch.zeros(N,N);Net_causal=torch.zeros(N,N)
    out_s = {}
    for i in range(N):
        for j in range(N):
            if i != j: 
                count = 0       
                for t in range(T):
                    model = DualVAE(x_dim=xy_dim, y_dim=xy_dim, zx_dim=z_dim, zy_dim=z_dim, s_dim=z_dim, hidden_dims=[hid_dim, hid_dim, hid_dim]).to(device)
                    model_T = train_dualvae(model,weights, n_dataset0_T.tensors[i],n_dataset1_T.tensors[j],device,num_epochs, batch_size=128, learning_rate=1e-3)
                    out1,MSE_s0[i,j],MSE_zx0[i,j] = evaluate(model_T,weights,n_dataset0_V.tensors[i],n_dataset1_V.tensors[j],device)

                    MSE_s[i,j] += MSE_s0[i,j].item()
                    MSE_zx[i,j] += MSE_zx0[i,j].item()
                    out_s[f"s{i},s{j}"] = out1.sample["s_x"]
                    count += 1
                av_MSE_s[i,j] = MSE_s[i,j] / count
                av_MSE_zx[i,j] = MSE_zx[i,j] / count

                causal_index[i,j]=av_MSE_zx[i,j]
          
    for i in range(N):
        for j in range(N):
            if i != j:         
                if causal_index[i,j]<=0.25:
                    Net_causal[i,j]=0
                elif causal_index[i,j]>0.25 and causal_index[i,j]<0.75:
                    if causal_index[j,i]>=0.75:
                        Net_causal[i,j]=0 
                    elif causal_index[j,i]<0.75:
                        Net_causal[i,j]=2   
                elif causal_index[i,j]>=0.75:
                    Net_causal[i,j]=1
                #print("i",i,"j",j,"Net_groud:", Net_groud[i,j])
                #print("Net_causal:", Net_causal[i,j])
    return out_s, causal_index, Net_causal

#