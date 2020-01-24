import numpy as np
import matplotlib.pyplot as plt
import glob
plt.style.use('seaborn')


##### LOAD FORWARD MODEL RETURNS #####
fwd_file_paths = glob.glob('./results/*forward_*.npy')
fwd_return = []
for i, fwd_file_path in enumerate(fwd_file_paths):
    fwd_return.append(np.load(fwd_file_path))
    print('fwd return length : ', fwd_return[i].shape[0])


##### LOAD BACKWARD MODEL RETURNS #####
bwd_file_paths = glob.glob('./results/*backward_*.npy')
bwd_return = []
for i, bwd_file_path in enumerate(bwd_file_paths):
    bwd_return.append(np.load(bwd_file_path))
    print('bwd return length : ', bwd_return[i].shape[0])

##### LOAD MODEL FREE RETURNS #####
mf_file_paths = glob.glob('./results/*None_*.npy')
mf_return = []
for i, mf_file_path in enumerate(mf_file_paths):
    mf_return.append(np.load(mf_file_path))
    print('mf return length : ', mf_return[i].shape[0])


plt.plot(np.arange(fwd_return[0].shape[0])*5e3, np.mean(fwd_return,axis=0), '.-b', label='Forward Return')
plt.plot(np.arange(bwd_return[0].shape[0])*5e3, np.mean(bwd_return,axis=0), '.-g', label='Backward Return')
plt.plot(np.arange(mf_return[0].shape[0])*5e3, np.mean(mf_return,axis=0), '.-r', label='Model free Return')
plt.show()