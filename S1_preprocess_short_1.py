#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from PyEMD import EMD 
from scipy.signal import convolve
from scipy.signal import butter, lfilter, freqz
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

# In[54]:


subject_name = 'Julie'
exp = 'VM5s'
date = '20240116'
pname = ['VM5s_Julie']
subject_background_stem_name = "phantom2Background_1_X1"
subject_image_stem_name = 'phantom2Image_1_X1'
phantom_background_stem_name = "PhantomBG"

time_resolution = 0.1 # [sec] # 0.10404, 0.10129
using_SDS = 10 # [mm]
phantom_measured_ID = ['2', '3', '4', '5']
phantom_simulated_ID = ['2', '3', '4', '5']
SDS_idx = 3  # SDS=10 mm, phantom simulated idx

# previous_phantom_measured_ID = ['2', '3', '4', '5']
# previous_date = '20230819'
# previous_ofolder = os.path.join('dataset', subject_name, 'SDS1', f'm_out_{previous_date}')
# previous_mother_folder_name = os.path.join(subject_name, "SDS1", previous_date)

## exp setting
baseline_start = 0 # [sec]
baseline_end = 67 # [sec]
HP_start = 61 # [sec]
HP_end = 67 # [sec]
recovery_start = 67 # [sec]
recovery_end = 840 # [sec] 

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


# Wavelength calibration factor defined : y = ax + b
a = 0.72646
b = 255.82063

# Choose wavelength
start_wl = 115
stop_wl = 190

# Choose fiber output range
row_choose = ((7, 16), (53, 62), (94, 110))


mother_folder_name = os.path.join(subject_name, "SDS1", date, exp)
background_filenpath = os.path.join("dataset", subject_name, "SDS1", date,'background.csv')
data_filepath = os.path.join(ofolder, f'{subject_name}_SDS1_sync.csv')


# ## Process Spectrapro+EMCCD raw data (image to csv file)

# In[44]:


# %% Load target spectrum
folder_list = glob(os.path.join(ifolder, '*'))
stdname = 'standard0.1'
if any(stdname in folder_list[i] for i in range(len(folder_list))):
    path_det_bg = glob(os.path.join(ifolder, stdname, f'{subject_background_stem_name}*'))
    path_det_bg = natsorted(path_det_bg)
    
    path_det = glob(os.path.join(ifolder, stdname, 'std*')) 
    path_det = natsorted(path_det)
    
    bg_arr, df_det_mean, df_det_ch1, df_det_ch2, df_det_ch3 = process_raw_data.get_spec(path_det_bg, path_det, row_choose, a, b, img_size=(18, 1600))
    df_det_mean.loc[start_wl:stop_wl, :].to_csv(os.path.join(ofolder, stdname+'_det_mean.csv'), index = False)
    df_det_ch1.loc[start_wl:stop_wl, :].to_csv(os.path.join(ofolder, stdname+'_det_ch1.csv'), index = False)
    df_det_ch2.loc[start_wl:stop_wl, :].to_csv(os.path.join(ofolder, stdname+'_det_ch2.csv'), index = False)
    df_det_ch3.loc[start_wl:stop_wl, :].to_csv(os.path.join(ofolder, stdname+'_det_ch3.csv'), index = False)
    
for tar_name in tqdm(pname):
    path_det_bg = glob(os.path.join(ifolder, tar_name, f'{subject_background_stem_name}*'))
    path_det_bg = natsorted(path_det_bg)
    
    path_det = glob(os.path.join(ifolder, tar_name, subject_image_stem_name + '*')) 
    path_det = natsorted(path_det)
    
    bg_arr, df_det_mean, df_det_ch1, df_det_ch2, df_det_ch3 = process_raw_data.get_spec(path_det_bg, path_det, row_choose, a, b, img_size=(200, 320))
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
        
    fig.suptitle(f'baseline')
    fig.tight_layout()
    fig.savefig(os.path.join(ofolder, f'{tar_name}.png'))
    plt.show()

# ## Sync spectraprofile to QEpro file

# In[45]:


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


# # ## Initialize Processer Instance

# # In[46]:


# process_raw_data.create_folder(mother_folder_name)

# Processer = process_raw_data(baseline_start=baseline_start,
#                              baseline_end=baseline_end,
#                              HP_start=HP_start,
#                              HP_end=HP_end,
#                              recovery_start=recovery_start,
#                              recovery_end=recovery_end,
#                              time_resolution=time_resolution,
#                              time_interval=time_interval,
#                              mother_folder_name=mother_folder_name,
#                              using_SDS=using_SDS)


# # ## Process Raw in-vivo Data

# # In[47]:


# def remove_spike(wl, data, normalStdTimes, ts):
#     # data : Nx --> time, Ny --> wavelength 
#     data_no_spike = data.copy()
#     mean = data_no_spike.mean(axis=0)
#     std = data_no_spike.std(ddof=1, axis=0)
#     targetSet = []  # save spike idx
#     for idx, s in enumerate(data_no_spike):  # iterate spectrum in every time frame
#         isSpike = np.any(abs(s-mean) > normalStdTimes*std)
#         if isSpike:
#             targetSet.append(idx) 
#     print(f"target = {targetSet}")
#     if len(targetSet) != 0:
#         for target in targetSet:
#             # show target spec and replace that spec by using average of the two adjacents
#             if ((target+1) < data_no_spike.shape[0]) & ((target-1) >= 0 ): 
#                 data_no_spike[target] = (data_no_spike[target-1] + data_no_spike[target+1]) / 2
#             elif (target+1) == data_no_spike.shape[0]:
#                 data_no_spike[target] = data_no_spike[target-1]
#             elif (target-1) <= 0:
#                 data_no_spike[target] = data_no_spike[target+1]
            
#     return data_no_spike


# # In[59]:


# # load raw data
# raw_data, total_wl = Processer.read_file(data_filepath)

# # select range 700nm~850nm
# idx_700nm = np.argmin(np.abs(total_wl-700))
# idx_850nm = np.argmin(np.abs(total_wl-850))
# raw_data, total_wl = raw_data[:, idx_700nm:idx_850nm], total_wl[idx_700nm:idx_850nm]


# def remove_peak(data):
#     mean_data = data.mean()
#     max_data = data.max()
#     min_data = data.min()
#     up_range = abs(max_data - mean_data)
#     down_range = abs(mean_data - min_data)
#     get_peak = up_range if (up_range > down_range) else down_range
#     used_bound = max_data if (up_range > down_range) else min_data
#     remove_idx = []
#     for idx, d in enumerate(data):
#         if abs(d - used_bound) < 0.7*get_peak:
#             # data[idx] = mean_data
#             remove_idx += [idx]
    
#     return remove_idx

# # data_no_peak = raw_data.copy()
# # for ts in range(0, recovery_end, 10):
# #     td = ts + 10
# #     for wl in range(raw_data.shape[1]):
# #         data_no_peak[round(ts/time_resolution):round(td/time_resolution), wl] = remove_peak(data=raw_data[round(ts/time_resolution):round(td/time_resolution), wl])
    

# # # remove spike
# # data_no_spike = data_no_peak.copy()
# # for ts in range(0, recovery_end, 10):
# #     td = ts + 10
# #     data_no_spike[round(ts/time_resolution):round(td/time_resolution)] = remove_spike(total_wl, data_no_peak[round(ts/time_resolution):round(td/time_resolution)], 
# #                                                                                       normalStdTimes=2, 
# #                                                                                       ts = ts)


# ## Low Pass Filter
# # # Filter requirements.
# # order = 6
# # fs = 1/time_resolution # sample rate, Hz
# # cutoff = 3  # desired cutoff frequency of the filter, Hz
# # # Filter the data, and plot both the original and filtered signals.
# # LPF_data = data_moving_avg.copy()
# # for i in range(data_moving_avg.shape[1]):
# #     LPF_data[:,i] = process_raw_data.butter_lowpass_filter(data_moving_avg[:,i], cutoff, fs, order)

# ## EMD
# data_EMD = raw_data.copy()
# # remove all-time signal based at first
# imfs = EMD().emd(raw_data.mean(axis=1))
# imfs[-1] -= imfs[-1].mean()
# artifact = imfs[2:] 
# # remove artifact
# for art in artifact:
#     data_EMD -= art.reshape(-1, 1)


# # remove peaks
# data_no_peak = raw_data.copy()
# for ts in range(0, recovery_end, 10):
#     td = ts + 10
#     for wl in range(raw_data.shape[1]):
#         remove_idx = remove_peak(data=data_EMD[round(ts/time_resolution):round(td/time_resolution), wl])
#         data_no_peak[round(ts/time_resolution)+np.array(remove_idx),wl] = data_no_peak[round(ts/time_resolution):round(td/time_resolution),wl].mean()

# # remove spike
# data_no_spike = data_no_peak.copy()
# for ts in range(0, recovery_end, 10):
#     td = ts + 10
#     data_no_spike[round(ts/time_resolution):round(td/time_resolution)] = remove_spike(total_wl, data_no_peak[round(ts/time_resolution):round(td/time_resolution)], 
#                                                                                       normalStdTimes=3, 
#                                                                                       ts = ts)
# # moving average
# for i in range(data_no_spike.shape[0]):
#     if i == 0:
#         moving_avg_I_data, moving_avg_wl_data = process_raw_data.moving_avg(moving_window, total_wl, data_no_spike[i,:])
#         moving_avg_I_data = moving_avg_I_data.reshape(1,-1)
#         data_moving_avg = moving_avg_I_data
#     else:
#         moving_avg_I_data, moving_avg_wl_data = process_raw_data.moving_avg(moving_window, total_wl, data_no_spike[i,:])
#         moving_avg_I_data = moving_avg_I_data.reshape(1,-1) 
#         data_moving_avg = np.concatenate((data_moving_avg,moving_avg_I_data))

# ## EMD
# data_EMD = data_moving_avg.copy()
# # remove all-time signal based at first
# imfs = EMD().emd(data_moving_avg.mean(axis=1))
# imfs[-1] -= imfs[-1].mean()
# artifact = imfs[2:] 
# # remove artifact
# for art in artifact:
#     data_EMD -= art.reshape(-1, 1)

# # detect peak 
# is_peak = np.zeros(data_EMD.shape[0])
# for ts in range(0, recovery_end, time_interval):
#     td = ts+time_interval
#     data_signal = data_EMD[round(ts/time_resolution):round(td/time_resolution), :].mean(axis=1)
#     max_idx, min_idx = process_raw_data.get_peak_final(data_signal)
#     is_peak[min_idx + round(ts/time_resolution)] = -1
#     is_peak[max_idx + round(ts/time_resolution)] = 1


# # ## EMD used as processed data
# # data_EMD = data_no_spike.copy()
# # # remove all-time signal based at first
# # imfs = EMD().emd(data_no_spike.mean(axis=1))
# # imfs[-1] -= imfs[-1].mean()
# # artifact = imfs[4:] 
# # # remove artifact
# # for art in artifact:
# #     data_EMD -= art.reshape(-1, 1)

# # save result 
# save_result = {}
# time = [i*time_resolution for i in range(data_EMD.shape[0])]
# save_result['time(s)'] = time
# save_result['peak'] = is_peak # max:+1, min:-1
# for idx, using_wl in enumerate(moving_avg_wl_data):
#     save_result[f'{using_wl}nm'] = data_EMD[:,idx]
# save_result = pd.DataFrame(save_result)
# save_result.to_csv(os.path.join("dataset", mother_folder_name, f"in_vivo_result_{exp}.csv"), index=False)


# # In[60]:


# plt.rcParams.update({'font.size': 20})
# plt.figure(figsize=(20,8))
# time = np.linspace(0,recovery_end, raw_data.shape[0])
# plt.plot(time, raw_data.mean(1))
# plt.axvline(x=baseline_end, linestyle='--', color='b', label='baseline_end')
# plt.axvline(x=HP_end, linestyle='--', color='r', label='hyperventilation_end')
# plt.axvline(recovery_end, linestyle='--', color='g', label='recovery_end')
# plt.legend(loc='center left', bbox_to_anchor=(1.05, 0.5),
#           fancybox=True, shadow=True)
# # plt.plot(time[max_id1], used_wl_data.mean(1)[max_id1], 'r.')
# # plt.plot(time[min_id1], used_wl_data.mean(1)[min_id1], 'b.')
# plt.xlabel("time [sec]")
# plt.ylabel("Intensity")
# plt.title('SDS10mm')
# plt.savefig(os.path.join('pic', mother_folder_name, 'time', 'raw', 'raw_all_time_SDS10mm.png'), dpi=300, format='png', bbox_inches='tight')
# plt.show()


# # In[63]:


# plt.rcParams.update({'font.size': 12})
# Processer.long_plot_all_fig(data=raw_data, 
#             wavelength=total_wl,
#             name='raw')


# # for ts in range(0,recovery_end,time_interval):
# #     td = ts + time_interval
# #     plot_LPF_compare(BF_data=data_moving_avg, 
# #                      AF_data=LPF_data,
# #                      start_time=ts,
# #                     end_time=td,
# #                     time_resolution=time_resolution,
# #                     name='compare_LPF')
# # plot_all_fig(data=LPF_data, 
# #         wavelength=moving_avg_wl_data,
# #         name='LPF')  

# Processer.long_plot_all_fig(data=data_no_spike, 
#             wavelength=total_wl,
#             name='remove_spike_and_bg')
# Processer.long_plot_all_fig(data=data_moving_avg, 
#             wavelength=moving_avg_wl_data,
#             name='moving_average')
# Processer.long_plot_all_fig(data=data_EMD, 
#             wavelength=moving_avg_wl_data,
#             name='EMD')

# Processer.plot_Rmax_Rmin(data=data_EMD,
#                          wavelength=moving_avg_wl_data,
#                          max_idx_Set=max_idx,
#                          min_idx_Set=min_idx,
#                          name="get_peak",
#                          start_time=0,
#                          end_time=recovery_end)

# for using_num_IMF in [1,2,3]:
#     Processer.plot_time_EMD(data=data_moving_avg,
#                     name='EMD',
#                     start_time=0,
#                     end_time=recovery_end,
#                     using_num_IMF=using_num_IMF)

#     Processer.plot_compare_time_EMD(data=data_moving_avg,
#                         name='compare',
#                         start_time=0,
#                         end_time=recovery_end,
#                         using_num_IMF=using_num_IMF)

    
#     for ts in range(0,recovery_end,time_interval):
#         td = ts + time_interval
#         Processer.plot_time_EMD(data=data_moving_avg,
#                     name='EMD',
#                     start_time=ts,
#                     end_time=td,
#                     using_num_IMF=using_num_IMF)

#         Processer.plot_compare_time_EMD(data=data_moving_avg,
#                             name='compare',
#                             start_time=ts,
#                             end_time=td,
#                             using_num_IMF=using_num_IMF)


# # ## Phantom Calibration

# # ### Process spectrapro+EMCCD raw phantom data

# # In[56]:


# # %% Load target spectrum
# folder_list = glob(os.path.join(ifolder, '*'))
# stdname = 'standard0.1'
# if any(stdname in folder_list[i] for i in range(len(folder_list))):
#     path_det_bg = glob(os.path.join(ifolder, stdname, f'{phantom_background_stem_name}*'))
#     path_det_bg = natsorted(path_det_bg)
    
#     path_det = glob(os.path.join(ifolder, stdname, 'std*')) 
#     path_det = natsorted(path_det)
    
#     bg_arr, df_det_mean, df_det_ch1, df_det_ch2, df_det_ch3 = process_raw_data.get_spec(path_det_bg, path_det, row_choose, a, b, img_size=(18, 1600))
#     df_det_mean.loc[start_wl:stop_wl, :].to_csv(os.path.join(ofolder, stdname+'_det_mean.csv'), index = False)
#     df_det_ch1.loc[start_wl:stop_wl, :].to_csv(os.path.join(ofolder, stdname+'_det_ch1.csv'), index = False)
#     df_det_ch2.loc[start_wl:stop_wl, :].to_csv(os.path.join(ofolder, stdname+'_det_ch2.csv'), index = False)
#     df_det_ch3.loc[start_wl:stop_wl, :].to_csv(os.path.join(ofolder, stdname+'_det_ch3.csv'), index = False)
    
# for tar_name in tqdm(phantom_measured_ID):
#     path_det_bg = glob(os.path.join(ifolder, tar_name, f'{phantom_background_stem_name}*'))
#     path_det_bg = natsorted(path_det_bg)
    
#     path_det = glob(os.path.join(ifolder, tar_name, tar_name + '*')) 
#     path_det = natsorted(path_det)
    
#     bg_arr, df_det_mean, df_det_ch1, df_det_ch2, df_det_ch3 = process_raw_data.get_spec(path_det_bg, path_det, row_choose, a, b, img_size=(18, 1600))
#     df_det_mean.loc[start_wl:stop_wl, :].to_csv(os.path.join(ofolder, tar_name+'_det_mean.csv'), index = False)
#     df_det_ch1.loc[start_wl:stop_wl, :].to_csv(os.path.join(ofolder, tar_name+'_det_ch1.csv'), index = False)
#     df_det_ch2.loc[start_wl:stop_wl, :].to_csv(os.path.join(ofolder, tar_name+'_det_ch2.csv'), index = False)
#     df_det_ch3.loc[start_wl:stop_wl, :].to_csv(os.path.join(ofolder, tar_name+'_det_ch3.csv'), index = False)
#     det_list = [df_det_ch1, df_det_ch2, df_det_ch3]
#     stat_list = process_raw_data.get_stat(det_list, start_wl, stop_wl)
#     fig, ax = plt.subplots(1, 3, figsize=(16, 8), dpi=300)
#     for i in range(3):
#         for j in range(df_det_ch1.shape[1]-1):
#             ax[i].plot(det_list[i].loc[start_wl:stop_wl, 'wl'], det_list[i].loc[start_wl:stop_wl, f'shot_{j}'])
#         ax[i].set_title(f'SDS = {SDS_LIST[i]} mm')
#         ax[i].set_xticks(np.arange(700, 881, 30))
#         ax[i].set_xlabel('Wavelength (nm)')
#         ax[i].set_ylabel('Intensity (counts)')
#         ax2 = ax[i].twinx()
#         ax2.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f%%'))
#         ax2.plot(stat_list[i]['wl'], 100 * stat_list[i]['cv'], color='black', linestyle='--')
#         ax2.set_ylabel('CV')
#         ax2.set_yticks(np.linspace(0, 10, 6))
#         ax2.tick_params(axis='y')
        
#     fig.suptitle(f'Phantom {tar_name}')
#     fig.tight_layout()
#     fig.savefig(os.path.join(ofolder, f'{tar_name}.png'))


# # ## Sync spectraprofile to QEpro file

# # In[57]:


# for phantom_ID in phantom_measured_ID:
#     data = pd.read_csv(os.path.join(ofolder, f'{phantom_ID}_det_ch3.csv'))
#     total_wl =  data['wl'].to_numpy(dtype=np.float64)
#     data = data.to_numpy().T
#     time = [i*time_resolution for i in range(1,data.shape[0])]
#     df = {'Wavelength (nm)': time}
#     for idx, wl in enumerate(total_wl):
#         df[wl] = data[1:,idx]
#     df = pd.DataFrame(df)
#     df.to_csv(os.path.join(ofolder, f'{phantom_ID}_SDS1_sync.csv'), index=False)
# df


# # In[21]:


# a = df.to_numpy()[:,1:]
# plt.plot(a[:].mean(1))
# plt.show()


# # ### Load measured phantom data

# # In[82]:


# # get measured phantom data
# phantom_data = {} # CHIK3456
# for ID in phantom_measured_ID:
#     # define plot savepath
#     os.makedirs(os.path.join("pic", mother_folder_name, 'phantom', ID), exist_ok=True)
    
#     # import measured data
#     data, total_wl = process_raw_data.read_file(os.path.join(ofolder, f'{ID}_SDS1_sync.csv'))
    
#     # remove spike
#     remove_spike_data = process_phantom.remove_spike(total_wl, data, 
#                                                      normalStdTimes=2, 
#                                                      savepath=os.path.join("pic", mother_folder_name,'phantom', ID)) # remove spike
#     time_mean_data = remove_spike_data.mean(0) # mean of measured signal
    
#     # Do moving avg of spectrum
#     moving_avg_I_data, moving_avg_wl_data = process_phantom.moving_avg(used_wl_bandwidth = 3,
#                                                        used_wl = total_wl,
#                                                        time_mean_arr = time_mean_data)
#     # plt.plot(data.mean(1))
#     # plt.plot(moving_avg_I_data.mean(1), 'o--')
#     # plt.show()
#     phantom_data[ID] = moving_avg_I_data


# # ### Load previous measured phantom data

# # In[7]:


# if previous_date != date:
#     # get measured phantom data
#     previous_phantom_data = {} # CHIK3456
#     for ID in previous_phantom_measured_ID:
#         # define plot savepath
#         os.makedirs(os.path.join("pic", previous_mother_folder_name, 'phantom', ID), exist_ok=True)
        
#         # import measured data
#         data, total_wl = process_raw_data.read_file(os.path.join(previous_ofolder, f'{ID}_SDS1_sync.csv'))
        
#         # remove spike
#         remove_spike_data = process_phantom.remove_spike(total_wl, data, 
#                                                         normalStdTimes=2, 
#                                                         savepath=os.path.join("pic", previous_mother_folder_name,'phantom', ID)) # remove spike
#         time_mean_data = remove_spike_data.mean(0) # mean of measured signal
        
#         plt.plot(data.mean(1))
#         plt.plot(remove_spike_data.mean(1), 'o--')
#         plt.show()
#         previous_phantom_data[ID] = time_mean_data
        
#     for ID in phantom_measured_ID:    
#         corr_raito = phantom_data[ID]/previous_phantom_data[ID]
#         print(f'corr_ratio = {corr_raito}')
#     corr_previous_phantom_data = previous_phantom_data.copy()
#     for ID in previous_phantom_measured_ID:
#         corr_previous_phantom_data[ID] = corr_raito*corr_previous_phantom_data[ID]
        
#     phantom_data = corr_previous_phantom_data


# # ### Load simulated phantom data

# # In[83]:


# # load used wavelength
# with open(os.path.join("OPs_used", "wavelength.json"), "r") as f:
#     wavelength = json.load(f)
#     wavelength = wavelength['wavelength']


# # In[84]:


# matplotlib.rcParams.update({'font.size': 18})
# ## Get the same simulated wavelength point from measured phantom
# measured_phantom_data = []
# plt.figure(figsize=(12,8))
# # Cubic spline interpolation"
# for idx, out_k in enumerate(phantom_data.keys()):
#     data = phantom_data[out_k]
#     f_interpolate = interp1d(moving_avg_wl_data, data, kind='linear', bounds_error=False, fill_value='extrapolate')
#     used_wl_data = f_interpolate(wavelength)
#     measured_phantom_data.append(used_wl_data)
#     plt.plot(wavelength, used_wl_data, 'o--', label=f'phantom_{previous_phantom_measured_ID[idx]}')
# plt.title("SDS=10mm, measured phantom spectrum")
# plt.xlabel('wavelength (nm)')
# plt.ylabel('intensity')
# plt.legend()
# plt.savefig(os.path.join("pic", mother_folder_name, 'phantom', "measured_phantom_adjust_wl.png"), dpi=300, format='png', bbox_inches='tight')
# plt.show()
# measured_phantom_data = np.array(measured_phantom_data)

# ## Get simulated phantom data
# sim_phantom_data = []
# plt.figure(figsize=(12,8))
# for c in phantom_simulated_ID:
#     data = np.load(os.path.join("dataset", "phantom_simulated", f'{c}.npy'))
#     sim_phantom_data.append(data[:,SDS_idx].tolist())
#     plt.plot(wavelength, data[:,SDS_idx], 'o--',label=f'phantom_{c}')
# plt.title("SDS=10mm, simulated phantom spectrum")
# plt.xlabel('wavelength (nm)')
# plt.ylabel('intensity')
# plt.legend()
# plt.savefig(os.path.join("pic", mother_folder_name, 'phantom', "simulated_phantom_adjust_wl.png"), dpi=300, format='png', bbox_inches='tight')
# plt.show()
# sim_phantom_data = np.array(sim_phantom_data)


# # ### Fit measured phantom and simulated phantom

# # In[86]:


# fig = plt.figure(figsize=(18,12))
# fig.suptitle(f"SDS = {using_SDS} mm", fontsize=16)
# count = 1
# for idx, used_wl in enumerate(wavelength):
#     ## fit measured phantom and simulated phantom
#     z = np.polyfit(measured_phantom_data[:, idx], sim_phantom_data[:,idx], 1)
#     plotx = np.linspace(measured_phantom_data[-1, idx]*0.8,  measured_phantom_data[0, idx]*1.2,100)
#     ploty = plotx*z[0] + z[1]
#     calibrate_data = measured_phantom_data[:, idx]*z[0] + z[1]
#     R_square = process_phantom.cal_R_square(calibrate_data, sim_phantom_data[:,idx]) # cal R square
    
#     ## plot result
#     ax = plt.subplot(5,4, count)
#     ax.set_title(f"@wavelength={used_wl} nm")
#     ax.set_title(f'{used_wl}nm, $R^{2}$={R_square:.2f}')
#     for ID_idx, ID in enumerate(previous_phantom_measured_ID):
#         ax.plot(measured_phantom_data[ID_idx, idx], sim_phantom_data[ID_idx,idx], 's', label=f'phantom_{ID}')
#     ax.plot(plotx, ploty, '--')
#     ax.set_xlabel("measure intensity")
#     ax.set_ylabel("sim intensity")
#     count += 1
# plt.legend(loc='center left', bbox_to_anchor=(1.05, 0.5),
#                     fancybox=True, shadow=True)
# plt.tight_layout()
# plt.savefig(os.path.join('pic', mother_folder_name, 'phantom', "all.png"), dpi=300, format='png', bbox_inches='tight')
# plt.show()


# # ### Save fitting result as csv

# # In[87]:


# result = {}
# for idx, used_wl in enumerate(wavelength):
#     z = np.polyfit(measured_phantom_data[:, idx], sim_phantom_data[:,idx], 1)
#     result[used_wl] = z
# result = pd.DataFrame(result)
# os.makedirs(os.path.join("dataset",  subject_name, 'calibration_result', date) , exist_ok=True)
# result.to_csv(os.path.join("dataset",  subject_name, 'calibration_result', date, "calibrate_SDS_1.csv"), index=False)


# # ### Plot all measured phantom together

# # In[89]:


# for idx, k in enumerate(phantom_data.keys()):
#     data = phantom_data[k]
#     plt.plot(moving_avg_wl_data, data, label=f'phantom_{previous_phantom_measured_ID[idx]}')
# plt.title('phantom spectrum')
# plt.xlabel('wavelength (nm)')
# plt.ylabel('intensity')
# plt.legend(loc='center left', bbox_to_anchor=(1.05, 0.5),
#                 fancybox=True, shadow=True)
# plt.savefig(os.path.join("pic", mother_folder_name, 'phantom', "measured_2345_phantom_result.png"), dpi=300, format='png', bbox_inches='tight')
# plt.show()

