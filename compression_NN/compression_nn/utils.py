import time, sys, os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder


def plot_test_error(model, test_loader, device='cuda', output=False, out_name='test', Y_min=0, Y_max=1, color='steelblue'):

    g=[0, 1]

    test_loss1, test_loss2 = torch.zeros(len(g)).to(device), torch.zeros(len(g)).to(device)
    test_loss, points = 0.0, 0

    ## Model performance metrics on test set
    num_maps=test_loader.dataset.tensors[0].shape[0]

    # define the arrays containing the value of the parameters
    params_true = np.zeros((num_maps,len(g)), dtype=np.float32)
    params_NN   = np.zeros((num_maps,len(g)), dtype=np.float32)
    errors_NN   = np.zeros((num_maps,len(g)), dtype=np.float32)

    # model.eval()
    for x, y in test_loader:
        with torch.no_grad():
            bs    = x.shape[0]         #batch size
            if x.dtype == torch.float64:
                x = x.float()
            x     = x.to(device)       #send data to device
            y     = y.to(device)  #send data to device
            p     = model.predict(x)           #prediction for mean and variance
            y_NN  = p           #prediction for mean

            # save results to their corresponding arrays
            
            params_true[points:points+x.shape[0]] = y.cpu().numpy() 
            if isinstance(y_NN, torch.Tensor):
                y_NN = y_NN.cpu().numpy()
            params_NN[points:points+x.shape[0]]   = y_NN

            points    += x.shape[0]
            
    # normalization if needed
    params_true = params_true * (Y_max - Y_min) + Y_min
    params_NN   = params_NN   * (Y_max - Y_min) + Y_min
    
    test_error = 100*np.mean(np.sqrt((params_true - params_NN)**2)/params_true,axis=0)
    
    RMSE = np.sqrt(np.mean((params_true - params_NN)**2,axis=0))
    RMSE_P = RMSE*100
    params_true_mean = np.mean(params_true)
    tmp = np.mean((params_true - params_true_mean)**2, axis=0)
    R2 = 1 - (RMSE)**2 / tmp
    # print('Error Omega_m = %.3f'%test_error[0])
    print(r' RMSE = %.3f'%RMSE[0])
    print(r' $R^2$ = %.3f'%R2[0])
    print('Error S_8 = %.3f'%test_error[0])


    f, axarr = plt.subplots(1, 2, figsize=(15,10))
    axarr[0].plot(np.linspace(min(params_true[:,0]),max(params_true[:,0]),100),np.linspace(min(params_true[:,0]),max(params_true[:,0]),100),color="black")

    axarr[0].plot(params_true[:,0],params_NN[:,0],marker="o",ls="none",markersize=2, color=color)
    axarr[0].set_xlabel(r"True $\Omega_m$")
    axarr[0].set_ylabel(r"Predicted $\Omega_m$")
    # axarr.text(0.1,0.9,"%.3f %% error" % test_error[0],fontsize=12,transform=axarr.transAxes)
    
    axarr[0].text(0.08,0.9,r"RMSE = %.3f %% " % RMSE_P[0],fontsize=12,transform=axarr[0].transAxes)
    axarr[0].text(0.08,0.82,r"$R^2$ = %.3f" % R2[0],fontsize=12,transform=axarr[0].transAxes)
    
    
    axarr[1].plot(np.linspace(min(params_true[:,1]),max(params_true[:,1]),100),np.linspace(min(params_true[:,1]),max(params_true[:,1]),100),color="black")
    axarr[1].plot(params_true[:,1],params_NN[:,1],marker="o",ls="none",markersize=2)
    axarr[1].set_xlabel(r"True $S_8$")
    axarr[1].set_ylabel(r"Predicted $S_8$")
    axarr[1].text(0.1,0.9,"%.3f %% error" % test_error[1],fontsize=12,transform=axarr[1].transAxes)

    if output:
        f.savefig('./output/'+out_name+'.pdf', dpi=300, format='pdf')

        # Also save for LFI later
        info = dict()
        info['params'] = params_true
        info['compressed_DV'] = params_NN
        np.save('./output/'+out_name+'_compressed_dv',info)

        
def plot_test_error_all_param_3param(model, test_loader, n_params, label_normalizer=None, device='cuda', output=False, out_name='test', Y_min=0, Y_max=1, color='steelblue'):
    assert n_params==3, 'this function is for plotting and saving 3 param training'
    g=range(n_params)

    test_loss1, test_loss2 = torch.zeros(len(g)).to(device), torch.zeros(len(g)).to(device)
    points = 0

    ## Model performance metrics on test set
    num_maps=test_loader.dataset.tensors[0].shape[0]

    # define the arrays containing the value of the parameters
    params_true = np.zeros((num_maps,len(g)), dtype=np.float32)
    params_NN   = np.zeros((num_maps,len(g)), dtype=np.float32)
    errors_NN   = np.zeros((num_maps,len(g)), dtype=np.float32)

    # model.eval()
    for x, y in test_loader:
        with torch.no_grad():
            bs    = x.shape[0]         #batch size
            if x.dtype == torch.float64:
                x = x.float()
            x     = x.to(device)       #send data to device
            y     = y.to(device)  #send data to device
            p     = model.predict(x)           #prediction for mean and variance
            y_NN  = p           #prediction for mean

            # save results to their corresponding arrays
            params_true[points:points+x.shape[0]] = y.cpu().numpy()
            if isinstance(y_NN, torch.Tensor):
                y_NN = y_NN.cpu().numpy()
            params_NN[points:points+x.shape[0]]   = y_NN

            points    += x.shape[0]
            
    # inverse-normalization if needed
    if label_normalizer is not None:
        params_NN   = label_normalizer.inverse_transform(params_NN)

    test_error = 100*np.mean(np.sqrt((params_true - params_NN)**2)/params_true,axis=0)
    
    RMSE = np.sqrt(np.mean((params_true - params_NN)**2,axis=0))
    RMSE_P = RMSE*100
    params_true_mean = np.mean(params_true)
    tmp = np.mean((params_true - params_true_mean)**2, axis=0)
    R2 = 1 - (RMSE)**2 / tmp
    # print('Error Omega_m = %.3f'%test_error[0])
    print(r' RMSE = %.3f'%RMSE[0])
    print(r' $R^2$ = %.3f'%R2[0])
    print('Error S_8 = %.3f'%test_error[0])


    # KZ: let me first assume n_params is even 
    f, axarr = plt.subplots(1, 3, figsize=(20,10))
    
    for i in range(n_params):
        # if i%2==0:
        row_idx = i
        print('test', i)
        axarr[row_idx].plot(np.linspace(min(params_true[:,i]),max(params_true[:,i]),100),np.linspace(min(params_true[:,i]),max(params_true[:,i]),100),color="black")

        axarr[row_idx].plot(params_true[:,i],params_NN[:,i],marker="o",ls="none",markersize=2, color=color)
        axarr[row_idx].set_xlabel(r"True param "+str(i))
        axarr[row_idx].set_ylabel(r"Predicted param "+ str(i))

        axarr[row_idx].text(0.08,0.9,r"RMSE = %.3f %% " % RMSE_P[i],fontsize=12,transform=axarr[row_idx].transAxes)
        axarr[row_idx].text(0.08,0.82,r"$R^2$ = %.3f" % R2[i],fontsize=12,transform=axarr[row_idx].transAxes)
 
    if output:
        f.savefig('./output/'+out_name+'.pdf', dpi=300, format='pdf')

        # Also save for LFI later
        info = dict()
        info['params'] = params_true
        info['compressed_DV'] = params_NN
        np.save('./output/'+out_name+'_compressed_dv',info)
        
        

def create_label_mapping(labels, return_mapping=False):
    """
    Convert float labels to integer labels based on unique combinations of first two columns.
    
    Parameters:
    labels: numpy array of shape (N, 3) containing float labels
    return_mapping: bool, whether to return the mapping dictionary
    
    Returns:
    - integer_labels: numpy array of shape (N,) containing integer labels from 1 to num_unique
    - mapping: (optional) dictionary containing the mapping from float pairs to integers
    """
    # Extract first two columns and convert to strings for hashing
    # Round to handle floating point precision issues
    label_pairs = [f"{row[0]:.6f},{row[1]:.6f}" for row in labels]
    
    # Create label encoder
    le = LabelEncoder()
    integer_labels = le.fit_transform(label_pairs)
    
    # Add 1 to make labels start from 1 instead of 0
    integer_labels = integer_labels + 1
    
    if return_mapping:
        # Create mapping dictionary
        unique_pairs = le.classes_
        mapping = {pair: (idx + 1) for idx, pair in enumerate(unique_pairs)}
        
        # Convert string pairs back to float tuples in mapping
        float_mapping = {}
        for pair_str, idx in mapping.items():
            float1, float2 = map(float, pair_str.split(','))
            float_mapping[(float1, float2)] = idx
            
        return integer_labels, float_mapping
    
    return integer_labels