#!/usr/bin/env python
# coding: utf-8

# <div style="text-align:center;"><h1> Analyze the short channel data collected from Spectrapro+EMDCCD </h1></div>
# 
# Below is <span style="color:red">the process structure</span> of this jupyter notebook.  
# The data you get containing three informations such as time, wavelength and intensity, represented as: $ I(t,\lambda) $
# 
# **Note that when you do *in-vivo* experiment, if you use <span style="color:red">binning</span>, no need to do extra moving average on spectrum. Like this example notebook.**  
# **However, if you <span style="color:red">don't</span> use <span style="color:red">binning</span>, you need to adjust the <span style="color:red">moving average</span> on spectrum decreaing the noise on the spectrum**
# - Format Setting
# 
# - Process Spectrapro+EMCCD raw data (image to csv file) <span style="color:red">(on image mode)</span>
#     * Get Background Data
#             $$ \bar{I}_{bg}(\lambda) = \frac{\sum_{t=0}^{N} I_{bg}(t,\lambda)}{N} $$
#     * Subtract background
#             $$ \hat{I}(t, \lambda) = I_{NoSpike}(t,\lambda) - \bar{I}_{bg}(\lambda)$$
# 
# - Sync Spectrapro image data to QEpro format csv file (time*wavelength)
# 
# - Initialize Processer Instance
# 
# - Analyze *in-vivo* Data
#     * Process Raw *in-vivo* Data
#         * Load <span style="color:red">sync</span> *in-vivo* data 
#         * Remove spike
#         * EMD
#         * Get diastolic and systolic peaks
#         * Plot Raw Data
#         * Plot all results
# 
# - Analyze Phantom Data for Calibration
#     * Process spectrapro+EMCCD raw phantom data
#         * Sync spectraprofile to QEpro file
#     * Load <span style="color:red">sync</span> measured phantom data  
#     * Load simulated phantom data
#     * fit measured phantom and simulated phantom
#     * Save fitting result as csv file
#     * Plot all measured phantom together

# In[1]:


import numpy as np
import pandas as pd
from PyEMD import EMD 
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib
from utils import process_raw_data, process_phantom
from glob import glob
from tqdm import tqdm
import matplotlib.ticker as mtick
import json
from scipy.interpolate import interp1d
from natsort import natsorted 
# Default settings
mpl.rcParams.update(mpl.rcParamsDefault)
plt.style.use("seaborn-darkgrid")


# ## Format Setting

# In[11]:


subject_name = 'Julie'
exp = 'VM5s'
date = '20240124'
pname = ['VM5s']

time_resolution = 0.1 # [sec] # 0.10404, 0.10129
using_SDS = 10 # [mm]
phantom_measured_ID = ['22', '33', '44', '55']
phantom_simulated_ID = ['2', '3', '4', '5']
SDS_idx = 3  # SDS=10 mm, phantom simulated idx

## exp setting
baseline_start = 0 # [sec]
baseline_end = 61 # [sec]
exp_start = 61 # [sec]
exp_end = 66 # [sec]
recovery_start = 66 # [sec]
recovery_end = 366 # [sec] 

# setting
moving_window = 3 # [nm]
time_interval = 30 # [sec]

## EMCCD setting 
ifolder = os.path.join('dataset', subject_name, 'SDS1', f'{date}')
# used short channel
used_ch = 'ch3'
SDS_LIST = [4.5, 7.5, 10.5]
ofolder = os.path.join('dataset', subject_name, 'SDS1', f'm_out_{date}_{exp}')
os.makedirs(ofolder, exist_ok=True)

# Wavelength calibration factor defined : y = ax + b (x: pixel, y: nm)
a = 0.6217
b = 297.08

# Choose wavelength
start_wl = 647 # --> x1 (pixel)
stop_wl = 889 # --> x2 (pixel)

# Choose fiber output range
row_choose = ((0, 0), (0, 0), (2, 4))

mother_folder_name = os.path.join(subject_name, "SDS1", date, exp)
background_filenpath = os.path.join("dataset", subject_name, "SDS1", date,'background.csv')
data_filepath = os.path.join(ofolder, f'{subject_name}_SDS1_sync.csv')


# ## Process Spectrapro+EMCCD raw data (image to csv file)

# In[3]:


# %% Load target spectrum
folder_list = glob(os.path.join(ifolder, '*'))
stdname = 'standard0.1'
if any(stdname in folder_list[i] for i in range(len(folder_list))):
    path_det_bg = glob(os.path.join(ifolder, stdname, 'background*'))
    path_det_bg = natsorted(path_det_bg)
    
    path_det = glob(os.path.join(ifolder, stdname, 'std*')) 
    path_det = natsorted(path_det)
    
    bg_arr, df_det_mean, df_det_ch1, df_det_ch2, df_det_ch3 = process_raw_data.get_spec(path_det_bg, path_det, row_choose, a, b, img_size=(20, 320))
    df_det_mean.loc[start_wl:stop_wl, :].to_csv(os.path.join(ofolder, stdname+'_det_mean.csv'), index = False)
    df_det_ch1.loc[start_wl:stop_wl, :].to_csv(os.path.join(ofolder, stdname+'_det_ch1.csv'), index = False)
    df_det_ch2.loc[start_wl:stop_wl, :].to_csv(os.path.join(ofolder, stdname+'_det_ch2.csv'), index = False)
    df_det_ch3.loc[start_wl:stop_wl, :].to_csv(os.path.join(ofolder, stdname+'_det_ch3.csv'), index = False)
    
for tar_name in tqdm(pname):
    path_det_bg = glob(os.path.join(ifolder, tar_name, 'background*'))
    path_det_bg = natsorted(path_det_bg)
    
    # path_det = glob(os.path.join(ifolder, tar_name, subject_name + '*')) 
    path_det = glob(os.path.join(ifolder, tar_name, exp + '*')) 
    path_det = natsorted(path_det)
    
    bg_arr, df_det_mean, df_det_ch1, df_det_ch2, df_det_ch3 = process_raw_data.get_spec(path_det_bg, path_det, row_choose, a, b, img_size=(6, 1600))
    df_det_mean.loc[start_wl:stop_wl, :].to_csv(os.path.join(ofolder, tar_name+'_det_mean.csv'), index = False)
    df_det_ch1.loc[start_wl:stop_wl, :].to_csv(os.path.join(ofolder, tar_name+'_det_ch1.csv'), index = False)
    df_det_ch2.loc[start_wl:stop_wl, :].to_csv(os.path.join(ofolder, tar_name+'_det_ch2.csv'), index = False)
    df_det_ch3.loc[start_wl:stop_wl, :].to_csv(os.path.join(ofolder, tar_name+'_det_ch3.csv'), index = False)
    det_list = [df_det_ch1, df_det_ch2, df_det_ch3]
    stat_list = process_raw_data.get_stat(det_list, start_wl, stop_wl)
    fig, ax = plt.subplots(1, 3, figsize=(16, 8), dpi=300)
    for i in range(3):
        for j in range(df_det_ch1.shape[1]-1):
            ax[i].plot(det_list[i].loc[start_wl:stop_wl, 'wl'], det_list[i].loc[start_wl:stop_wl, f'shot_{j}'])
        ax[i].set_title(f'SDS = {SDS_LIST[i]} mm')
        ax[i].set_xticks(np.arange(700, 881, 30))
        ax[i].set_xlabel('Wavelength (nm)')
        ax[i].set_ylabel('Intensity (counts)')
        ax2 = ax[i].twinx()
        ax2.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f%%'))
        ax2.plot(stat_list[i]['wl'], 100 * stat_list[i]['cv'], color='black', linestyle='--')
        ax2.set_ylabel('CV')
        ax2.set_yticks(np.linspace(0, 10, 6))
        ax2.tick_params(axis='y')
        
    fig.suptitle(f'Phantom {tar_name}')
    fig.tight_layout()
    fig.savefig(os.path.join(ofolder, f'{tar_name}.png'))
    fig.show()

# ## Sync Spectrapro image data to QEpro format csv file

# In[12]:


data = pd.read_csv(os.path.join(ofolder, f'{pname[0]}_det_{used_ch}.csv'))
total_wl = data['wl'].to_numpy(dtype=np.float64)
data = data.to_numpy().T
time = [i*time_resolution for i in range(1,data.shape[0])]
df = {'Wavelength (nm)': time}
for idx, wl in enumerate(total_wl):
    df[wl] = data[1:,idx]
df = pd.DataFrame(df)
df.to_csv(os.path.join(ofolder, f'{subject_name}_SDS1_sync.csv'), index=False)
df


# ## Initialize Processer Instance

# In[7]:


process_raw_data.create_folder(mother_folder_name)

Processer = process_raw_data(baseline_start=baseline_start,
                             baseline_end=baseline_end,
                             exp_start=exp_start,
                             exp_end=exp_end,
                             recovery_start=recovery_start,
                             recovery_end=recovery_end,
                             time_resolution=time_resolution,
                             time_interval=time_interval,
                             mother_folder_name=mother_folder_name,
                             using_SDS=using_SDS)


# ## Process Raw in-vivo Data

# In[26]:


# load raw data
raw_data, total_wl = Processer.read_file(data_filepath)

# select range 700nm~850nm
# idx_700nm = np.argmin(np.abs(total_wl-700))
# idx_850nm = np.argmin(np.abs(total_wl-850))
# raw_data, total_wl = raw_data[:, idx_700nm:idx_850nm], total_wl[idx_700nm:idx_850nm]

# remove spike
data_no_spike = raw_data.copy()
for ts in range(0, recovery_end, time_interval):
    td = ts + time_interval
    if ((ts>=exp_start) & (ts<=exp_end)) or ((td>=exp_start) & (td<=exp_end)): # while in the experiment period, don't remove spike
        pass # do nothing
    else:
        data_no_spike[round(ts/time_resolution):round(td/time_resolution)] = Processer.remove_spike(total_wl, 
                                                                                                    raw_data[round(ts/time_resolution):round(td/time_resolution)], 
                                                                                                    normalStdTimes=5, 
                                                                                                    ts=ts)
        
# moving average
for i in range(data_no_spike.shape[0]):
    if i == 0:
        moving_avg_I_data, moving_avg_wl_data = process_raw_data.moving_avg(moving_window, total_wl, data_no_spike[i,:])
        moving_avg_I_data = moving_avg_I_data.reshape(1,-1)
        data_moving_avg = moving_avg_I_data
    else:
        moving_avg_I_data, moving_avg_wl_data = process_raw_data.moving_avg(moving_window, total_wl, data_no_spike[i,:])
        moving_avg_I_data = moving_avg_I_data.reshape(1,-1) 
        data_moving_avg = np.concatenate((data_moving_avg,moving_avg_I_data))


## EMD
data_EMD = data_moving_avg.copy()
# remove all-time signal based at first
imfs = EMD().emd(data_moving_avg.mean(axis=1))
imfs[-1] -= imfs[-1].mean()
artifact = [imfs[0], imfs[-4], imfs[-3], imfs[-2], imfs[-1]]
#artifact = imfs[-4:] 
# remove artifact
for art in artifact:
    data_EMD -= art.reshape(-1, 1)
    
## manually remove artifact(?) at index: 565~590 %2024/02/01
time = np.array([i for i in range(555)])
f_interpolate = interp1d(time, data_EMD[0:555].transpose(), kind='linear', bounds_error=False, fill_value='extrapolate')
inter_time = np.array([i for i in range(565,590)])
data_EMD_int = f_interpolate(inter_time)
data_EMD[565:590] = data_EMD_int.transpose()



plot_data_After_EMD, start_time, end_time, X, plot_data_Before_EMD = data_EMD, 0, data_EMD.shape[0]*time_resolution, 50,data_moving_avg
plt.rcParams.update({'font.size': 30})
#fig, ax = plt.subplots(imfs.shape[0]+1, 1, figsize=(25, 10))
time = np.linspace(start_time, X, plot_data_After_EMD[round(start_time/time_resolution):round(X/time_resolution)].shape[0])
plt.figure(figsize=(35,10))
plt.title(f'SDS:{using_SDS}mm ' + f",{start_time}s~{X}s {subject_name} After EMD filtering")
plt.plot(time, plot_data_After_EMD.mean(axis=1)[round(start_time/time_resolution):round(X/time_resolution)], 'b')
plt.xlabel('time (sec)')
plt.ylabel('meaured intensity (gray level)')
tick_positions = np.arange(start_time, X + 1, step=5)
plt.xticks(tick_positions)
plt.show()

plt.figure(figsize=(35,10))
plt.title(f'SDS:{using_SDS}mm ' + f",{start_time}s~{X}s {subject_name} Before EMD filtering")
plt.plot(time, plot_data_Before_EMD.mean(axis=1)[round(start_time/time_resolution):round(X/time_resolution)], 'r')
plt.xlabel('time (sec)')
plt.ylabel('meaured intensity (gray level)')
tick_positions = np.arange(start_time, X + 1, step=5)
plt.xticks(tick_positions)
plt.show()

fig, ax = plt.subplots(imfs.shape[0]+2, 1, figsize=(30, 30))
ax[0].set_title(f'SDS:{using_SDS}mm ' + f",{start_time}s~{X}s {subject_name} baseline Before EMD filtering")
ax[0].plot(time, plot_data_Before_EMD.mean(axis=1)[round(start_time/time_resolution):round(X/time_resolution)], 'r')
ax[0].set_xticks([])
new_fig_width = 60  # Set your desired width
new_fig_height = 60  # Set your desired height
ax[0].get_figure().set_size_inches(new_fig_width, new_fig_height)
#tick_positions = np.arange(start_time, X + 1, step=5)
#ax[0].set_xticks(tick_positions)
# plt.xlabel('time (sec)')
# plt.ylabel('meaured intensity (gray level)')
# plt.show()
# fig, ax = plt.subplots(imfs.shape[0]+2, 1, figsize=(30, 30))
# ax[0].set_title(f'SDS:{using_SDS}mm ' + f",{start_time}s~{X}s {subject_name} baseline Before EMD filtering")

########
fig, ax = plt.subplots(imfs.shape[0]+2, 1, figsize=(30, 30))
tick_positions = np.arange(start_time, X + 1, step=5)
ax[10].set_xticks(tick_positions)
ax0 = plt.subplot2grid((imfs.shape[0]+2, 1), (0, 0), rowspan=1, colspan=2)
ax0.set_title(f'SDS:{using_SDS}mm ' + f",{start_time}s~{X}s {subject_name} baseline Before EMD filtering")

# Plot and adjust y-axis size for baseline Before EMD filtering
plot_data = plot_data_Before_EMD.mean(axis=1)[round(start_time/time_resolution):round(X/time_resolution)]
ax0.plot(time, plot_data, 'r')
y_axis_size = 5  # Adjust as needed
ax0.set_ylim(min(plot_data) - y_axis_size, max(plot_data) + y_axis_size)

# Hide X-axis ticks for the first subplot
ax0.set_xticks([])




#########
fig, ax = plt.subplots(imfs.shape[0]+2, 1, figsize=(30, 30))
ax0 = plt.subplot2grid((imfs.shape[0]+2, 1), (1, 0), rowspan=1, colspan=1)
ax0.set_title(f'SDS:{using_SDS}mm ' + f",{start_time}s~{X}s {subject_name} baseline Before EMD filtering")

#EMD baseline NEW
fig, ax = plt.subplots(imfs.shape[0]+2, 1, figsize=(30, 60))
ax[0].set_title(f'SDS:{using_SDS}mm ' + f",{start_time}s~{X}s {subject_name} baseline After EMD filtering")
ax[0].plot(time, plot_data_After_EMD.mean(axis=1)[round(start_time/time_resolution):round(X/time_resolution)], 'b')
ax[0].set_xticks([])
plt.xlabel('time (sec)')
# tick_positions = np.arange(start_time, X + 1, step=5)
# plt.xticks(tick_positions)
# plt.ylabel('meaured intensity (gray level)')

ax[1].set_title(f'SDS:{using_SDS}mm ' + f",{start_time}s~{X}s {subject_name} baseline Before EMD filtering")
ax[1].plot(time, plot_data_Before_EMD.mean(axis=1)[round(start_time/time_resolution):round(X/time_resolution)], 'r')
ax[1].set_ylim(920, 970)
ax[1].set_xticks([])

# Plot and adjust y-axis size for baseline Before EMD filtering
plot_data = plot_data_Before_EMD.mean(axis=1)[round(start_time/time_resolution):round(X/time_resolution)]
ax0.plot(time, plot_data, 'r')
y_axis_size = 5  # Adjust as needed
ax0.set_ylim(min(plot_data) - y_axis_size, max(plot_data) + y_axis_size)

# Hide X-axis ticks for the first subplot
ax0.set_xticks([])

for n, imf in enumerate(imfs):
    # plt.figure(figsize=(25,10))
    #ax[n+1].set_title("imf " + str(n+1))  
    time = np.linspace(start_time,X, imf[round(start_time/time_resolution):round(X/time_resolution)].shape[0])
    ax[n+2].plot(time, imf[round(start_time/time_resolution):round(X/time_resolution)], 'g')
    ax[n+2].set_xticks([])
    #tick_positions = np.arange(start_time, X + 1, step=5)
    #ax[n+1].set_xticks(tick_positions)
    # plt.xlabel('time (sec)')
    # plt.ylabel('meaured intensity (gray level)')
    # plt.show()
tick_positions = np.arange(start_time, X + 1, step=5)
ax[-7].set_xticks(tick_positions)
ax[-1].set_xticks(tick_positions)
plt.tight_layout()
plt.show()
# plt.figure(figsize=(25,10))

##baseline After EMD filtering ori
ax[imfs.shape[0]+1].set_title(f'SDS:{using_SDS}mm ' + f",{start_time}s~{X}s {subject_name} baseline After EMD filtering")
ax[imfs.shape[0]+1].plot(time, plot_data_After_EMD.mean(axis=1)[round(start_time/time_resolution):round(X/time_resolution)], 'b')
ax[imfs.shape[0]+1].set_xticks(tick_positions)
plt.xlabel('time (sec)')
# plt.ylabel('meaured intensity (gray level)')
plt.tight_layout()
plt.show()

##Recovery
plot_data_After_EMD, start_time, X, plot_data_new = data_EMD, 70, 120,data_moving_avg
plt.rcParams.update({'font.size': 30})

fig, ax = plt.subplots(imfs.shape[0]+2, 1, figsize=(30, 60))
ax[0].set_title(f'SDS:{using_SDS}mm ' + f",{start_time}s~{X}s {subject_name} Recovery period After EMD filtering")
ax[0].plot(time, plot_data_After_EMD.mean(axis=1)[round(start_time/time_resolution):round(X/time_resolution)], 'b')
ax[0].set_xticks([])
plt.xlabel('time (sec)')
# tick_positions = np.arange(start_time, X + 1, step=5)
# plt.xticks(tick_positions)
# plt.ylabel('meaured intensity (gray level)')

ax[1].set_title(f'SDS:{using_SDS}mm ' + f",{start_time}s~{X}s {subject_name} Recovery period Before EMD filtering")
ax[1].plot(time, plot_data_Before_EMD.mean(axis=1)[round(start_time/time_resolution):round(X/time_resolution)], 'r')
ax[1].set_ylim(920, 970)
ax[1].set_xticks([])

# Plot and adjust y-axis size for baseline Before EMD filtering
plot_data = plot_data_Before_EMD.mean(axis=1)[round(start_time/time_resolution):round(X/time_resolution)]
ax0.plot(time, plot_data, 'r')
y_axis_size = 5  # Adjust as needed
ax0.set_ylim(min(plot_data) - y_axis_size, max(plot_data) + y_axis_size)

# Hide X-axis ticks for the first subplot
ax0.set_xticks([])

for n, imf in enumerate(imfs):
    # plt.figure(figsize=(25,10))
    #ax[n+1].set_title("imf " + str(n+1))  
    time = np.linspace(start_time,X, imf[round(start_time/time_resolution):round(X/time_resolution)].shape[0])
    ax[n+2].plot(time, imf[round(start_time/time_resolution):round(X/time_resolution)], 'g')
    ax[n+2].set_xticks([])
    #tick_positions = np.arange(start_time, X + 1, step=5)
    #ax[n+1].set_xticks(tick_positions)
    # plt.xlabel('time (sec)')
    # plt.ylabel('meaured intensity (gray level)')
    # plt.show()
tick_positions = np.arange(start_time, X + 1, step=5)
ax[-7].set_xticks(tick_positions)
ax[-1].set_xticks(tick_positions)
plt.tight_layout()
plt.show()


##skip NEW 
plot_data_After_EMD, start_time, find_skip_t1, find_skip_t2,  end_time, plot_data_Before_EMD = data_EMD, 0, 55, 70, data_EMD.shape[0]*time_resolution,data_moving_avg
plt.rcParams.update({'font.size': 30})

time = np.linspace(start_time, find_skip_t1, plot_data_Before_EMD[round(start_time/time_resolution):round(find_skip_t1/time_resolution)].shape[0])
#time = np.linspace(start_time, end_time, plot_data[round(start_time/time_resolution):round(find_skip_t1/time_resolution)].shape[0], plot_data[round(find_skip_t2/time_resolution):round(end_time/time_resolution)].shape[0])
plt.figure(figsize=(35,10))
plt.title(f'SDS:{using_SDS}mm ' + f",{start_time}s~{end_time}s {subject_name} Before EMD filtering")
plt.plot(time, plot_data_Before_EMD.mean(axis=1)[round(start_time/time_resolution):round(find_skip_t1/time_resolution)], 'r')
#plt.plot(time, plot_data_new.mean(axis=1)[round(find_skip_t2/time_resolution):round(end_time/time_resolution)], 'r')
plt.xlabel('time (sec)')
plt.ylabel('meaured intensity (gray level)')
plt.show()


for n, imf in enumerate(imfs):
    plt.figure(figsize=(25,10))
    plt.title(f'SDS:{using_SDS}mm ' + f",{start_time}s~{X}s IMF"+str(n+1)+" signal")
    time = np.linspace(start_time,X, imf[round(start_time/time_resolution):round(X/time_resolution)].shape[0])
    plt.plot(time, imf[round(start_time/time_resolution):round(X/time_resolution)], 'g')
    plt.xlabel('time (sec)')
    plt.ylabel('meaured intensity (gray level)')
    plt.show() 
    
# ##all    
# imfs = EMD().emd(data_moving_avg.mean(axis=1)) 
# imfs[-1] -= imfs[-1].mean()
# artifact = [imfs[0], imfs[1], imfs[-4], imfs[-3], imfs[-2], imfs[-1]]   
# # artifact = imfs[-2:]  
# # remove baseline artifact
# for art in artifact:
#     data_EMD -= art.reshape(-1, 1)

# plot_data, start_time, end_time, plot_data_new = data_EMD, 0, data_EMD.shape[0]*time_resolution,data_no_spike
# plt.rcParams.update({'font.size': 30})
# time = np.linspace(start_time, end_time, plot_data.shape[0])
# fig, ax = plt.subplots(imfs.shape[0]+2, 1, figsize=(60, 30))
# ax[0].set_title(f'SDS:{using_SDS}mm ' + f",{start_time}s~{end_time}s Before EMD")
# ax[0].plot(time, plot_data.mean(axis=1)[round(start_time/time_resolution):round(end_time/time_resolution)], 'r')
# # plt.xlabel('time (sec)')
# # plt.ylabel('meaured intensity (gray level)')
# # plt.show()

# for n, imf in enumerate(imfs):
#     # plt.figure(figsize=(25,10))
#     ax[n+1].set_title("imf " + str(n+1)) 
#     time = np.linspace(start_time,end_time, imf.shape[0])
#     ax[n+1].plot(time, imf[round(start_time/time_resolution):round(end_time/time_resolution)], 'g')
#     # plt.xlabel('time (sec)')
#     # plt.ylabel('meaured intensity (gray level)')
#     # plt.show()

# # plt.figure(figsize=(25,10))
# ax[imfs.shape[0]+1].set_title(f'SDS:{using_SDS}mm ' + f",{start_time}s~{end_time}s After EMD filtering")
# ax[imfs.shape[0]+1].plot(time, plot_data_new.mean(axis=1)[round(start_time/time_resolution):round(end_time/time_resolution)], 'b')
# plt.xlabel('time (sec)')
# # plt.ylabel('meaured intensity (gray level)')
# plt.tight_layout()
# plt.show()

# ## manually remove artifact(?) at index: 565~590 %2024/02/01
# time = np.array([i for i in range(555)])
# f_interpolate = interp1d(time, data_EMD[0:555].transpose(), kind='linear', bounds_error=False, fill_value='extrapolate')
# inter_time = np.array([i for i in range(565,590)])
# data_EMD_int = f_interpolate(inter_time)
# data_EMD[565:590] = data_EMD_int.transpose()



## detect peak 
# get straight signal to find peaks
straight_signal = data_moving_avg.copy()
# remove all-time signal based at first
imfs = EMD().emd(data_moving_avg.mean(axis=1))
imfs[-1] -= imfs[-1].mean()
artifact = imfs[2:] 
# remove artifact
for art in artifact:
    straight_signal -= art.reshape(-1, 1)
is_peak = np.zeros(straight_signal.shape[0])
for ts in range(0, recovery_end, time_interval):
    td = ts+time_interval
    data_signal = straight_signal[round(ts/time_resolution):round(td/time_resolution), :].mean(axis=1)
    max_idx, min_idx = process_raw_data.get_peak_final(data_signal)
    is_peak[min_idx + round(ts/time_resolution)] = -1
    is_peak[max_idx + round(ts/time_resolution)] = 1

# save result 
save_result = {}
time = [i*time_resolution for i in range(data_moving_avg.shape[0])]
save_result['time(s)'] = time
save_result['peak'] = is_peak # max:+1, min:-1
for idx, using_wl in enumerate(moving_avg_wl_data):
    save_result[f'{using_wl}nm'] = data_EMD[:,idx]
save_result = pd.DataFrame(save_result)
save_result.to_csv(os.path.join("dataset", mother_folder_name, f"in_vivo_result_{exp}.csv"), index=False)


# ### Plot Raw Data

# In[31]:


plt.rcParams.update({'font.size': 20})
plt.figure(figsize=(20,8))
time = [i*time_resolution for i in range(raw_data.shape[0])]
plt.plot(time, raw_data.mean(1))
plt.axvline(x=baseline_end, linestyle='--', color='b', label='baseline_end')
plt.axvline(x=exp_end, linestyle='--', color='r', label=f'{exp}_end')
plt.axvline(recovery_end, linestyle='--', color='g', label='recovery_end')
plt.legend(loc='center left', bbox_to_anchor=(1.05, 0.5),
          fancybox=True, shadow=True)
plt.xlabel("time [sec]")
plt.ylabel("meaured intensity (gray level)")
plt.title('Raw Data SDS10mm')
plt.savefig(os.path.join('pic', mother_folder_name, 'time', 'raw_all_time_SDS10.png'), dpi=300, format='png', bbox_inches='tight')
plt.show()


# In[30]:


max_id = np.where(save_result['peak']==1)[0]
min_id = np.where(save_result['peak']==-1)[0]

max_id = max_id[np.where(max_id<round(recovery_end/time_resolution))[0]]
min_id = min_id[np.where(min_id<round(recovery_end/time_resolution))[0]]

plt.figure(figsize=(20,8))
time = save_result['time(s)']
plt.plot(time, data_EMD.mean(1)) 
plt.axvline(x=baseline_end, linestyle='--', color='b', label='baseline_end')
plt.axvline(x=exp_end, linestyle='--', color='r', label=f'{exp}_end')
plt.axvline(recovery_end, linestyle='--', color='g', label='recovery_end')
plt.legend(loc='center left', bbox_to_anchor=(1.05, 0.5),
          fancybox=True, shadow=True)
plt.plot(time[max_id], data_EMD.mean(1)[max_id], 'r.')
plt.plot(time[min_id], data_EMD.mean(1)[min_id], 'b.')
plt.xlabel("time [sec]")
plt.ylabel("meaured intensity (gray level)")
plt.title('After EMD filtering SDS10mm')
plt.savefig(os.path.join('pic', mother_folder_name, 'time', 'processed_all_time_SDS10.png'), dpi=300, format='png', bbox_inches='tight')
plt.show()


# In[32]:


plt.rcParams.update({'font.size': 12})
Processer.long_plot_all_fig(data=raw_data, 
            wavelength=total_wl,
            name='raw')

Processer.long_plot_all_fig(data=data_no_spike, 
            wavelength=total_wl,
            name='remove_spike_and_bg')

Processer.long_plot_all_fig(data=data_moving_avg, 
            wavelength=moving_avg_wl_data,
            name='moving_average')
    
Processer.long_plot_all_fig(data=data_EMD, 
            wavelength=moving_avg_wl_data,
            name='EMD')

Processer.plot_Rmax_Rmin(data=data_EMD,
                         wavelength=moving_avg_wl_data,
                         max_idx_Set=max_idx,
                         min_idx_Set=min_idx,
                         name="get_peak",
                         start_time=0,
                         end_time=recovery_end)

for using_num_IMF in [1,2,3,4,5]:
    Processer.plot_time_EMD(data=data_moving_avg,
                    name='EMD',
                    start_time=0,
                    end_time=recovery_end,
                    using_num_IMF=using_num_IMF)

    Processer.plot_compare_time_EMD(data=data_moving_avg,
                        name='compare',
                        start_time=0,
                        end_time=recovery_end,
                        using_num_IMF=using_num_IMF)

    
    for ts in range(0,recovery_end,time_interval):
        td = ts + time_interval
        Processer.plot_time_EMD(data=data_moving_avg,
                    name='EMD',
                    start_time=ts,
                    end_time=td,
                    using_num_IMF=using_num_IMF)

        Processer.plot_compare_time_EMD(data=data_moving_avg,
                            name='compare',
                            start_time=ts,
                            end_time=td,
                            using_num_IMF=using_num_IMF)

# ## Phantom Calibration

# ### Process spectrapro+EMCCD raw phantom data

# In[36]:


# %% Load target spectrum
folder_list = glob(os.path.join(ifolder, '*'))
stdname = 'standard0.1'
if any(stdname in folder_list[i] for i in range(len(folder_list))):
    path_det_bg = glob(os.path.join(ifolder, stdname, 'background*'))
    path_det_bg = natsorted(path_det_bg)
    
    path_det = glob(os.path.join(ifolder, stdname, 'std*')) 
    path_det = natsorted(path_det)
    
    bg_arr, df_det_mean, df_det_ch1, df_det_ch2, df_det_ch3 = process_raw_data.get_spec(path_det_bg, path_det, row_choose, a, b, img_size=(20, 320))
    df_det_mean.loc[start_wl:stop_wl, :].to_csv(os.path.join(ofolder, stdname+'_det_mean.csv'), index = False)
    df_det_ch1.loc[start_wl:stop_wl, :].to_csv(os.path.join(ofolder, stdname+'_det_ch1.csv'), index = False)
    df_det_ch2.loc[start_wl:stop_wl, :].to_csv(os.path.join(ofolder, stdname+'_det_ch2.csv'), index = False)
    df_det_ch3.loc[start_wl:stop_wl, :].to_csv(os.path.join(ofolder, stdname+'_det_ch3.csv'), index = False)
    
for tar_name in tqdm(phantom_measured_ID):
    path_det_bg = glob(os.path.join(ifolder, tar_name, 'background*'))
    path_det_bg = natsorted(path_det_bg)
    
    path_det = glob(os.path.join(ifolder, tar_name, tar_name + '*')) 
    path_det = natsorted(path_det)
    
    bg_arr, df_det_mean, df_det_ch1, df_det_ch2, df_det_ch3 = process_raw_data.get_spec(path_det_bg, path_det, row_choose, a, b, img_size=(6, 1600))
    df_det_mean.loc[start_wl:stop_wl, :].to_csv(os.path.join(ofolder, tar_name+'_det_mean.csv'), index = False)
    df_det_ch1.loc[start_wl:stop_wl, :].to_csv(os.path.join(ofolder, tar_name+'_det_ch1.csv'), index = False)
    df_det_ch2.loc[start_wl:stop_wl, :].to_csv(os.path.join(ofolder, tar_name+'_det_ch2.csv'), index = False)
    df_det_ch3.loc[start_wl:stop_wl, :].to_csv(os.path.join(ofolder, tar_name+'_det_ch3.csv'), index = False)
    det_list = [df_det_ch1, df_det_ch2, df_det_ch3]
    stat_list = process_raw_data.get_stat(det_list, start_wl, stop_wl)
    fig, ax = plt.subplots(1, 3, figsize=(16, 8), dpi=300)
    for i in range(3):
        for j in range(df_det_ch1.shape[1]-1):
            ax[i].plot(det_list[i].loc[start_wl:stop_wl, 'wl'], det_list[i].loc[start_wl:stop_wl, f'shot_{j}'])
        ax[i].set_title(f'SDS = {SDS_LIST[i]} mm')
        ax[i].set_xticks(np.arange(700, 881, 30))
        ax[i].set_xlabel('Wavelength (nm)')
        ax[i].set_ylabel('Intensity (counts)')
        ax2 = ax[i].twinx()
        ax2.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f%%'))
        ax2.plot(stat_list[i]['wl'], 100 * stat_list[i]['cv'], color='black', linestyle='--')
        ax2.set_ylabel('CV')
        ax2.set_yticks(np.linspace(0, 10, 6))
        ax2.tick_params(axis='y')
        
    fig.suptitle(f'Phantom {tar_name}')
    fig.tight_layout()
    fig.savefig(os.path.join(ofolder, f'{tar_name}.png'))


# ## Sync spectraprofile to QEpro file

# In[37]:


for phantom_ID in phantom_measured_ID:
    data = pd.read_csv(os.path.join(ofolder, f'{phantom_ID}_det_ch3.csv'))
    total_wl =  data['wl'].to_numpy(dtype=np.float64)
    data = data.to_numpy().T
    time = [i*time_resolution for i in range(1,data.shape[0])]
    df = {'Wavelength (nm)': time}
    for idx, wl in enumerate(total_wl):
        df[wl] = data[1:,idx]
    df = pd.DataFrame(df)
    df.to_csv(os.path.join(ofolder, f'{phantom_ID}_SDS1_sync.csv'), index=False)
df


# ### Load measured phantom data

# In[ ]:


# get measured phantom data
phantom_data = {} # CHIK3456
for ID in phantom_measured_ID:
    # define plot savepath
    os.makedirs(os.path.join("pic", mother_folder_name, 'phantom', ID), exist_ok=True)
    
    # import measured data
    data, total_wl = process_raw_data.read_file(os.path.join(ofolder, f'{ID}_SDS1_sync.csv'))
    
    # remove spike
    remove_spike_data = process_phantom.remove_spike(total_wl, data, 
                                                     normalStdTimes=2, 
                                                     savepath=os.path.join("pic", mother_folder_name,'phantom', ID)) # remove spike
    time_mean_data = remove_spike_data.mean(0) # mean of measured signal
    
    # plt.plot(data.mean(1))
    # plt.plot(remove_spike_data.mean(1), 'o--')
    # plt.show()
    phantom_data[ID] = time_mean_data


# ### Load simulated phantom data

# In[39]:


# load used wavelength
with open(os.path.join("OPs_used", "wavelength.json"), "r") as f:
    wavelength = json.load(f)
    wavelength = wavelength['wavelength']


# In[40]:


matplotlib.rcParams.update({'font.size': 18})
## Get the same simulated wavelength point from measured phantom
measured_phantom_data = []
plt.figure(figsize=(12,8))
# Cubic spline interpolation"
for idx, out_k in enumerate(phantom_data.keys()):
    data = phantom_data[out_k]
    f_interpolate = interp1d(total_wl, data, kind='linear', bounds_error=False, fill_value='extrapolate')
    used_wl_data = f_interpolate(wavelength)
    measured_phantom_data.append(used_wl_data)
    plt.plot(wavelength, used_wl_data, 'o--', label=f'phantom_{phantom_measured_ID[idx]}')
plt.title("SDS=10mm, measured phantom spectrum")
plt.xlabel('wavelength (nm)')
plt.ylabel('intensity')
plt.legend()
plt.savefig(os.path.join("pic", mother_folder_name, 'phantom', "measured_phantom_adjust_wl.png"), dpi=300, format='png', bbox_inches='tight')
plt.show()
measured_phantom_data = np.array(measured_phantom_data)

## Get simulated phantom data
sim_phantom_data = []
plt.figure(figsize=(12,8))
for c in phantom_simulated_ID:
    data = np.load(os.path.join("dataset", "phantom_simulated", f'{c}.npy'))
    sim_phantom_data.append(data[:,SDS_idx].tolist())
    plt.plot(wavelength, data[:,SDS_idx], 'o--',label=f'phantom_{c}')
plt.title("SDS=10mm, simulated phantom spectrum")
plt.xlabel('wavelength (nm)')
plt.ylabel('intensity')
plt.legend()
plt.savefig(os.path.join("pic", mother_folder_name, 'phantom', "simulated_phantom_adjust_wl.png"), dpi=300, format='png', bbox_inches='tight')
plt.show()
sim_phantom_data = np.array(sim_phantom_data)


# ### Fit measured phantom and simulated phantom

# In[41]:


fig = plt.figure(figsize=(18,12))
fig.suptitle(f"SDS = {using_SDS} mm", fontsize=16)
count = 1
for idx, used_wl in enumerate(wavelength):
    ## fit measured phantom and simulated phantom
    z = np.polyfit(measured_phantom_data[:, idx], sim_phantom_data[:,idx], 1)
    plotx = np.linspace(measured_phantom_data[-1, idx]*0.8,  measured_phantom_data[0, idx]*1.2,100)
    ploty = plotx*z[0] + z[1]
    calibrate_data = measured_phantom_data[:, idx]*z[0] + z[1]
    R_square = process_phantom.cal_R_square(calibrate_data, sim_phantom_data[:,idx]) # cal R square
    
    ## plot result
    ax = plt.subplot(5,4, count)
    ax.set_title(f"@wavelength={used_wl} nm")
    ax.set_title(f'{used_wl}nm, $R^{2}$={R_square:.2f}')
    for ID_idx, ID in enumerate(phantom_measured_ID):
        ax.plot(measured_phantom_data[ID_idx, idx], sim_phantom_data[ID_idx,idx], 's', label=f'phantom_{ID}')
    ax.plot(plotx, ploty, '--')
    ax.set_xlabel("measure intensity")
    ax.set_ylabel("sim intensity")
    count += 1
plt.legend(loc='center left', bbox_to_anchor=(1.05, 0.5),
                    fancybox=True, shadow=True)
plt.tight_layout()
plt.savefig(os.path.join('pic', mother_folder_name, 'phantom', "all.png"), dpi=300, format='png', bbox_inches='tight')
plt.show()

# ### Save fitting result as csv

# In[42]:


result = {}
for idx, used_wl in enumerate(wavelength):
    z = np.polyfit(measured_phantom_data[:, idx], sim_phantom_data[:,idx], 1)
    result[used_wl] = z
result = pd.DataFrame(result)
os.makedirs(os.path.join("dataset",  subject_name, 'calibration_result', date) , exist_ok=True)
result.to_csv(os.path.join("dataset",  subject_name, 'calibration_result', date, "calibrate_SDS_1.csv"), index=False)


# ### Plot all measured phantom together

# In[43]:


for idx, k in enumerate(phantom_data.keys()):
    data = phantom_data[k]
    plt.plot(total_wl, data, label=f'phantom_{phantom_measured_ID[idx]}')
plt.title('phantom spectrum')
plt.xlabel('wavelength (nm)')
plt.ylabel('intensity')
plt.legend(loc='center left', bbox_to_anchor=(1.05, 0.5),
                fancybox=True, shadow=True)
plt.savefig(os.path.join("pic", mother_folder_name, 'phantom', "measured_2345_phantom_result.png"), dpi=300, format='png', bbox_inches='tight')
plt.show()

