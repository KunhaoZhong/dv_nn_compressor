import matplotlib.pyplot as plt
import math
import frogress
import numpy as np
import copy
import os
try:
    import jax
    import jax.numpy as jnp

    #Haiku is a simple neural network library for JAX 
    import haiku as hk

    #Optax is a gradient processing and optimization library for JAX. 
    import optax

    import tensorflow as tf
    from tensorflow_probability.substrates import jax as tfp
    
    tfd = tfp.distributions
    tfb = tfp.bijectors

except:
    pass
import sys

from numpy.random import default_rng

                
import sys
sys.path.insert(0, "/global/cfs/cdirs/des/mgatti/tensorflow_115/")
import tensorflow as tf
import emcee as mc
import numpy as np

import optuna

from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU, ReLU
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.layers import Input, Add
from tensorflow.keras.callbacks import EarlyStopping



import glob

outp = '/pscratch/sd/m/minsu98/GLASS_dv/compr/nbody_'
    
    
    
def doit(uu):

    
    key,p = uu
    if 1==1:
    #if not os.path.exists(outp+'{0}_{1}.npy'.format(key,p)):
        def create_model(input_size, layer_sizes, lr):
            model = Sequential()
            
           # reg = None if (l1_reg == 0 and l2_reg == 0) else l1_l2(l1=l1_reg, l2=l2_reg)
    
    
            model.add(Dense(layer_sizes[0], input_dim=input_size, kernel_initializer='normal'))#,kernel_regularizer=reg))
            model.add(LeakyReLU(alpha=0.01))

            for size in layer_sizes[1:]:
                model.add(Dense(size, kernel_initializer='normal'))#,kernel_regularizer=reg))
                model.add(LeakyReLU(alpha=0.01))

            model.add(Dense(1, kernel_initializer='normal'))  # Output layer
            model.compile(optimizer=Adam(learning_rate=lr), loss='mse')
            return model

        
        
        def objective(trial):
            input_size = compressed_DV_NN[key]['DV_train'].shape[1]  # Example to get input_size
            num_layers = trial.suggest_int('num_layers', 1, 6)  # Let Optuna decide between 1 and 10 layers

            
            # Start with a large possible range for the first layer
            layer_sizes = [trial.suggest_int('layer_0_size', 20, 120)]
            # Subsequent layers have a max size smaller than the previous layer
            for i in range(1, num_layers):
                max_size = layer_sizes[i-1] - 1  # Ensure strictly smaller size
                layer_sizes.append(trial.suggest_int(f'layer_{i}_size', 5, max(max_size, 20)))  # Minimum size 5   
                
            # activation_types = [trial.suggest_categorical(f'activation_{i}', ['relu', 'leaky_relu']) for i in range(num_layers)]
            lr = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
            
            #l2_reg = trial.suggest_float('l2_reg', 0.0, 0.01) if apply_regularization else 0

            model = create_model(input_size, layer_sizes, lr)#, l1_reg, l2_reg)

            # Dummy data for the example
            X_test,X_train, X_val =  np.array(compressed_DV_NN[key]['DV_test']),np.array(compressed_DV_NN[key]['DV_train']), np.array(compressed_DV_NN[key]['DV_val'])
            y_test,y_train, y_val =  np.array(compressed_DV_NN[key]['params_test'][:,p]),np.array(compressed_DV_NN[key]['params_train'][:,p]), np.array(compressed_DV_NN[key]['params_val'][:,p])

            history = model.fit(
                X_train, y_train, epochs=150, batch_size=32, verbose=0,
                validation_data=(X_val, y_val),
                callbacks=[EarlyStopping(monitor='val_loss', patience=8, verbose=1, mode='min', restore_best_weights=True)]
            )
            predictions = model.predict(X_val)
            print ('training loop loss ',np.mean((y_val-predictions)**2))
            return copy.deepcopy(np.mean((y_val-predictions)**2))




        # RUN THE OPTIMISATION ------------

        study = optuna.create_study(storage='sqlite://///pscratch/sd/m/minsu98/optuna/{0}_{1}.db'.format(key,p), load_if_exists=True, direction='minimize',study_name ='{0}_{1}'.format(key,p))

        #'''

        total_trials = 120
        batch_size = 50
        for _ in range(0, total_trials, batch_size):
            study.optimize(objective, n_trials=batch_size, n_jobs=-1)
    
       # study.opt

        print('Best parameters:', study.best_params)
        print('Best validation loss:', study.best_value)

        #'''
        # Rebuild and rerun the model with the best parameters
        input_size = compressed_DV_NN[key]['DV_train'].shape[1]
        best_params = study.best_params
 
        best_model = create_model(
            input_size,  # You need to specify this based on your actual data setup
            [best_params[f'layer_{i}_size'] for i in range(best_params['num_layers'])],
            best_params['learning_rate'])

        
        # we're using some extra 2000 sims to test the model to check the performance. ideally these are the sims that should be used for LFI ----
        X_test,X_train, X_val =  np.array(compressed_DV_NN[key]['DV_test']),np.array(compressed_DV_NN[key]['DV_train']), np.array(compressed_DV_NN[key]['DV_val'])
        y_test,y_train, y_val =  np.array(compressed_DV_NN[key]['params_test'][:,p]),np.array(compressed_DV_NN[key]['params_train'][:,p]), np.array(compressed_DV_NN[key]['params_val'][:,p])

        
        # use the same used for the training just to check everything is OK ------------------------------------------------------
        
        history_final = best_model.fit(
                X_train, y_train, epochs=150, batch_size=32, 
                validation_data=(X_val, y_val),
                callbacks=[EarlyStopping(monitor='val_loss', patience=8, mode='min', restore_best_weights=True)]
            )


        predictions = best_model.predict(X_val)

        MSE = np.mean((y_val-predictions)**2)

        results = dict()
        results['predictions_val'] = predictions
        results['p_val'] = y_val
        results['MSE_10k_val'] = np.mean((y_val-predictions)**2)

        
        predictions = best_model.predict(compressed_DV_NN[key]['DV_cov'])

        results['cov'] = predictions

        
        predictions = best_model.predict(X_test)

        results['predictions_test'] = predictions
        results['p_test'] = y_test
        results['MSE_test'] = np.mean((y_test-predictions)**2)
        results['best_params'] = study.best_params
        np.save(outp+'{0}_{1}.npy'.format(key,p),results)

        
        
        
#module load python
#source activate bfd_env3
#cd /global/homes/m/mgatti/CMB_lensing_SBI/code
#srun --nodes=4 --tasks-per-node=3   python run_optuna_glass.py 



if __name__ == '__main__':
    from mpi4py import MPI 
    import sys

    
    derivatives_dict = np.load('/global/cfs/cdirs/des/mgatti/CMB_lensing/DV/SBI_forecast/MOPED_derivatives.npy',allow_pickle=True).item()

    # Dirac + Dark Grid simulations and target datavector:
    DD = np.load('/global/cfs/cdirs/des/mgatti/CMB_lensing/DV/SBI_forecast/compression/compression_data_combined.npy',allow_pickle=True).item()
    stat = DD['stat']
    mask = DD['mask']
    target = DD['data']

    parms = stat['WL_2']['params']
    
    statistic = 'WL_2'
    extra =   (stat[statistic]['params'][:,3]<0.8)  & (stat[statistic]['params'][:,3]>0.2) & (stat[statistic]['params'][:,4]>0.1) &   (stat[statistic]['params'][:,4]<0.9) 

    additional_mask = (stat[statistic]['params'][:,2]>0.1) & extra

    
    mask_l = np.array(16*[ True,  True,  True,  True, False, False,  True,  True,  True,
            True,  True,  True, False, False, False, False,  True,  True,
            True,  True, False, False, False, False,  True,  True, False,
           False, False, False, False, False, False, False, False, False])

    mask_nbody_wph = np.hstack([np.array([False]*60),np.array([False]*120),mask_l])

    indict = dict()
    indict['WL_23_WPH_short'] = np.concatenate( ( list( range(320) ), np.array( range(320, 1076))[mask_nbody_wph]) )
    indict['WL_23_WPH_short_CMBWL'] = np.concatenate( ( list( range(320) ), np.array( range(320, 1076))[mask_nbody_wph], list(range(1076, 1108) )) )
    indict['WPH'] = np.array( range(160, 916))[mask_nbody_wph]
    indict['CMBWL'] = range(160, 192)


    odict = dict()
    odict['WL_23_WPH_short'] = 'WL_23_WPH'
    odict['WL_23_WPH_short_CMBWL'] = 'WL_23_WPH_WCMBL'
    odict['WPH'] = 'WL_2_WPH'
    odict['CMBWL'] = 'WL_2_WCMBL'

    for key in odict.keys():
        stat[key] = stat[odict[key]]
        stat[key]['dv'] = stat[key]['dv'][:,indict[key]]
        
        derivatives_dict[key] = derivatives_dict[odict[key]]
        derivatives_dict[key]['cov'] = derivatives_dict[key]['cov'][:,indict[key]]


    compressed_DV_NN = dict()
    
    for key in ['CMBWL']:#['WL_2', 'WL_3', 'WL_23', 'WL_23_WPH_short', 'WL_23_WPH_short_CMBWL', 'WPH']:
        
        dv_cov = derivatives_dict[key]['cov']
        
            # these are the pars/DV that will be used for the compression.
        pars = jnp.array(stat[key]['params'][mask&additional_mask,:16])
        dv = jnp.array(stat[key]['dv'][mask&additional_mask,:])

        # these are the pars/DV that will be used for the LFI step later on
        pars_LFI = jnp.array(stat[key]['params'][(~mask)&additional_mask,:16])
        dv_LFI = jnp.array(stat[key]['dv'][(~mask)&additional_mask,:])

        len_dv = dv.shape[1]  
        samples = dv.shape[0]

        # split into training & validation for the compression -------
        rng = default_rng()
        numbers = rng.choice(samples , size=3000, replace=False)
        special = np.in1d(np.arange(samples ),numbers)


        pars_train = jnp.array(pars[~special])
        pars_val =  jnp.array(pars[special])
        dv_train = jnp.array(dv[~special])
        dv_val = jnp.array(dv[special])
        
        lo  = np.percentile(dv_train, q = 1 ,axis=0)
        hi  = np.percentile(dv_train, q = 99,axis=0)

        min_ = -3#-1.5
        max_ = 4#2.5
        # compressed_DV_PCA[statistic]['DV_train']     =  np.clip((compressed_DV_PCA[statistic]['DV_train']-lo)*2/(hi-lo)-0.5,min_,max_)
        dv_val   =  np.clip((dv_val  -lo)*2/(hi-lo)-0.5,min_,max_)#*0.4
        dv_train =  np.clip((dv_train-lo)*2/(hi-lo)-0.5,min_,max_)#*0.4
        dv_LFI   =  np.clip((dv_LFI  -lo)*2/(hi-lo)-0.5,min_,max_)#*0.4
        dv_cov   =  np.clip((dv_cov  -lo)*2/(hi-lo)-0.5,min_,max_)#*0.4
        
#         dv_val =    np.clip(0.5+0.2 * ( dv_val-np.median( dv,axis=0)) / np.std( dv, axis=0) ,-0.5,1.5)
#         dv_train =  np.clip(0.5+0.2 * ( dv_train-np.median( dv,axis=0)) / np.std( dv, axis=0) ,-0.5,1.5)
#         dv_LFI =    np.clip(0.5+0.2 * ( dv_LFI-np.median( dv,axis=0)) / np.std( dv, axis=0) ,-0.5,1.5)

        # if key == 'WL_2_WCMBL_CMBL': lock = 'all_gauss'
        # elif key == 'WL_23_WPH_WCMBL_CMBL': lock = 'all'
        # else: lock = key
        
        compressed_DV_NN[key] = dict()
        compressed_DV_NN[key]['DV_train'] = dv_train
        compressed_DV_NN[key]['DV_val'] = dv_val
        compressed_DV_NN[key]['DV_test'] = dv_LFI
        compressed_DV_NN[key]['DV_cov'] = dv_cov
        compressed_DV_NN[key]['params_train'] = pars_train
        compressed_DV_NN[key]['params_val'] = pars_val
        compressed_DV_NN[key]['params_test'] = pars_LFI
        
#     for k in compressed_DV_NN.keys():
#         for word in ['DV_train', 'DV_val', 'DV_test']:
#             compressed_DV_NN[k][word] = (compressed_DV_NN[k][word]-np.median(compressed_DV_NN[k][word],axis=0))/np.std(compressed_DV_NN[k][word],axis=0)


    
    
    run_count = 0
    
    
    # par_ind = int(sys.argv[1])

    jobs = []
    for key in compressed_DV_NN.keys():
        for i in range(16):
            if not os.path.exists(outp+'{0}_{1}.npy'.format(key,i)):
                jobs.append([key,i])
        # if not os.path.exists(outp+'{0}_{1}.npy'.format(key, par_ind)):
            # jobs.append([key,par_ind])
        
        
    while run_count<len(jobs):
        comm = MPI.COMM_WORLD
        print("Hello! I'm rank %d from %d running in total..." % (comm.rank, comm.size))
        if (run_count+comm.rank)<len(jobs):
            #try:
            print(jobs[run_count+comm.rank])
            doit(jobs[run_count+comm.rank])
            #except:
            #    pass
        run_count+=comm.size
        comm.bcast(run_count,root = 0)
        comm.Barrier() 

        
# source activate bfd_env3
#srun --nodes=4 --tasks-per-node=3   python run_optuna_glass.py 