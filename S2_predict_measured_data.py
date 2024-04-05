#!/usr/bin/env python
# coding: utf-8

# In[1]:
import numpy as np
import pandas as pd
import os
import matplotlib as mpl
import json
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
# Default settings
mpl.rcParams.update(mpl.rcParamsDefault)
plt.style.use("seaborn-darkgrid")
plt.rcParams.update({'font.size': 20})

 
# ## Format Setting

# In[2]:


date = "20240202"
subject = 'Julie'
exp = 'VM10s'
SDS1_time_resolution = 0.1 # [sec/point]
SDS2_time_resolution = 0.1 # [sec/point]
baseline_end = 61 # [sec]
HP_end = 72 #[sec]
recovery_end = 272 #[sec]


# ## Get processed data

# In[3]:


# get processed long ch. data
data2 = pd.read_csv(os.path.join("dataset", subject, "SDS2", date, exp, f'in_vivo_result_{exp}.csv')) # wl resolution = 0.171 nm, time resolution = 0.1 secs
np_data2 = data2.to_numpy()[:round(recovery_end/SDS2_time_resolution),2:]
used_wl2 = [float(k.split('nm')[0]) for k in data2.keys().to_list()[2:]]
max_id2 = np.where(data2['peak']==1)[0]
min_id2 = np.where(data2['peak']==-1)[0]

max_id2 = max_id2[np.where(max_id2<round(recovery_end/SDS2_time_resolution))[0]]
min_id2 = min_id2[np.where(min_id2<round(recovery_end/SDS2_time_resolution))[0]]

# get processed short ch. data & sync to use same peak.
data = pd.read_csv(os.path.join("dataset", subject, "SDS1", date, exp, f'in_vivo_result_{exp}.csv')) # wl resolution = 0.171 nm, time resolution =  secs
np_data = data.to_numpy()[:round(recovery_end/SDS1_time_resolution),2:]
used_wl = [float(k.split('nm')[0]) for k in data.keys().to_list()[2:]]
max_id1 = np.where(data['peak']==1)[0]
min_id1 = np.where(data['peak']==-1)[0]

max_id1 = max_id1[np.where(max_id1<round(recovery_end/SDS1_time_resolution))[0]]
min_id1 = min_id1[np.where(min_id1<round(recovery_end/SDS1_time_resolution))[0]]

# # # get processed short ch. data & sync to use same peak.
# data = pd.read_csv(os.path.join("dataset", subject, "SDS1_4.5mm", date, 'in_vivo_result.csv')) # wl resolution = 0.171 nm, time resolution =  secs
# np_data_ch1 = data.to_numpy()[:round(recovery_end/SDS1_time_resolution),2:]
# used_wl_ch1 = [float(k.split('nm')[0]) for k in data.keys().to_list()[2:]]
# max_id1_ch1 = np.where(data['peak']==1)[0]
# min_id1_ch1 = np.where(data['peak']==-1)[0]

# max_id1_ch1 = max_id1_ch1[np.where(max_id1_ch1<round(recovery_end/SDS1_time_resolution))[0]]
# min_id1_ch1 = min_id1_ch1[np.where(min_id1_ch1<round(recovery_end/SDS1_time_resolution))[0]]

# # get processed short ch. data & sync to use same peak.
# data = pd.read_csv(os.path.join("dataset", subject, "SDS1_7.5mm", date, 'in_vivo_result.csv')) # wl resolution = 0.171 nm, time resolution =  secs
# np_data_ch2 = data.to_numpy()[:round(recovery_end/SDS1_time_resolution),2:]
# used_wl_ch2 = [float(k.split('nm')[0]) for k in data.keys().to_list()[2:]]
# max_id1_ch2 = np.where(data['peak']==1)[0]
# min_id1_ch2 = np.where(data['peak']==-1)[0]

# max_id1_ch2 = max_id1_ch2[np.where(max_id1_ch2<round(recovery_end/SDS1_time_resolution))[0]]
# min_id1_ch2 = min_id1_ch2[np.where(min_id1_ch2<round(recovery_end/SDS1_time_resolution))[0]]


# ## Adjust Wavelength

# In[4]:


with open(os.path.join("OPs_used", "wavelength.json"), 'r') as f:
    wavelength = json.load(f)
    wavelength = wavelength['wavelength']
wavelength = np.array(wavelength)


# In[5]:


## adjust short ch.
# Cubic spline interpolation
f_interpolate = interp1d(used_wl, np_data, kind='linear', bounds_error=False, fill_value='extrapolate')
used_wl_data = f_interpolate(wavelength)

# f_interpolate_ch1 = interp1d(used_wl_ch1, np_data_ch1, kind='linear', bounds_error=False, fill_value='extrapolate')
# used_wl_data_ch1 = f_interpolate_ch1(wavelength)

# f_interpolate_ch2 = interp1d(used_wl_ch2, np_data_ch2, kind='linear', bounds_error=False, fill_value='extrapolate')
# used_wl_data_ch2 = f_interpolate_ch2(wavelength)

## adjust long ch.
# get cumulate wavelength index
acumulate_table = {}
accmulate_range_of_wl = 2 # [nm]
for comp_wl in wavelength:
    cumulate_index = []
    for idx, each_used_wl in enumerate(used_wl2):
        if abs(float(each_used_wl) - comp_wl) < accmulate_range_of_wl:
            cumulate_index += [idx]
    acumulate_table[comp_wl] = cumulate_index

# used cumulate wavelength index to binning
for idx, wl in enumerate(acumulate_table.keys()):
    accmulate_idx = acumulate_table[wl]
    each_wl_data = np_data2[:, accmulate_idx]
    mean_of_each_wl_data = each_wl_data.mean(1).reshape(-1,1)
    if idx == 0:
        used_wl_data2 = mean_of_each_wl_data
    else:
        used_wl_data2 = np.concatenate((used_wl_data2, mean_of_each_wl_data), axis=1)

#%%
test_1 = np.mean(used_wl_data2[:400,:], axis=0)/np.mean(used_wl_data2[720:750,:], axis=0) #在0-50秒中每個波長強度取平均

combined_matrix = np.vstack([test_1,wavelength])
y_values = combined_matrix[0, :]
x_ticks = combined_matrix[1, :]
plt.plot(x_ticks, y_values)
plt.title(f'{subject} {exp} 20mm delta_OD for each wavelength')
plt.ylabel('$\Delta$ OD')
plt.xlabel('wavelength (nm)')
plt.show()


# In[6]:

os.makedirs(os.path.join('pic', subject, f'{date}_invivo_result', exp), exist_ok=True)
## plot raw data
plt.figure(figsize=(20,8))
time = np.linspace(0,recovery_end, used_wl_data.shape[0])
plt.plot(time, used_wl_data.mean(1))
#plt.axvline(x=420, linestyle='--', color='c')
plt.axvline(x=baseline_end, linestyle='--', color='b', label='baseline_end')
plt.axvline(x=HP_end, linestyle='--', color='r', label="Valsalva's maneuver_end")
#plt.axvline(recovery_end, linestyle='--', color='g', label='recovery_end')
plt.legend(loc='center left', bbox_to_anchor=(1.05, 0.5),
          fancybox=True, shadow=True)
plt.plot(time[max_id1], used_wl_data.mean(1)[max_id1], 'r.')
plt.plot(time[min_id1], used_wl_data.mean(1)[min_id1], 'b.')
plt.xlabel("time [sec]")
plt.ylabel("Intensity")
plt.title('SDS10mm')
plt.savefig(os.path.join('pic', subject, f'{date}_invivo_result', exp, 'raw_SDS1.png'), dpi=300, format='png', bbox_inches='tight')
plt.show()

# plt.figure(figsize=(20,8))
# time = np.linspace(0,recovery_end, used_wl_data_ch1.shape[0])
# plt.plot(time, used_wl_data_ch1.mean(1))
# plt.axvline(x=baseline_end, linestyle='--', color='b', label='baseline_end')
# plt.axvline(x=HP_end, linestyle='--', color='r', label='hyperventilation_end')
# plt.axvline(recovery_end, linestyle='--', color='g', label='recovery_end')
# plt.legend(loc='center left', bbox_to_anchor=(1.05, 0.5),
#           fancybox=True, shadow=True)
# plt.plot(time[max_id1_ch1], used_wl_data_ch1.mean(1)[max_id1_ch1], 'r.')
# plt.plot(time[min_id1_ch1], used_wl_data_ch1.mean(1)[min_id1_ch1], 'b.')
# plt.xlabel("time [sec]")
# plt.ylabel("Intensity")
# plt.title('SDS4.5mm')
# plt.savefig(os.path.join('pic', subject, f'{date}_invivo_result', exp, 'raw_SDS1_4.5mm.png'), dpi=300, format='png', bbox_inches='tight')
# plt.show()

# plt.figure(figsize=(20,8))
# time = np.linspace(0,recovery_end, used_wl_data_ch2.shape[0])
# plt.plot(time, used_wl_data_ch2.mean(1))
# plt.axvline(x=baseline_end, linestyle='--', color='b', label='baseline_end')
# plt.axvline(x=HP_end, linestyle='--', color='r', label='hyperventilation_end')
# plt.axvline(recovery_end, linestyle='--', color='g', label='recovery_end')
# plt.legend(loc='center left', bbox_to_anchor=(1.05, 0.5),
#           fancybox=True, shadow=True)
# plt.plot(time[max_id1_ch2], used_wl_data_ch2.mean(1)[max_id1_ch2], 'r.')
# plt.plot(time[min_id1_ch2], used_wl_data_ch2.mean(1)[min_id1_ch2], 'b.')
# plt.xlabel("time [sec]")
# plt.ylabel("Intensity")
# plt.title('SDS7.5mm')
# plt.savefig(os.path.join('pic', subject, f'{date}_invivo_result', exp, 'raw_SDS1_7.5mm.png'), dpi=300, format='png', bbox_inches='tight')
# plt.show()

plt.figure(figsize=(20,8))
time = np.linspace(0,recovery_end, used_wl_data2.shape[0])
plt.plot(time, used_wl_data2.mean(1))
#plt.axvline(x=430, linestyle='--', color='c')
plt.axvline(x=baseline_end, linestyle='--', color='b', label='baseline_end')
plt.axvline(x=HP_end, linestyle='--', color='r', label="Valsalva's maneuver_end")
#plt.axvline(recovery_end, linestyle='--', color='g', label='recovery_end')
plt.legend(loc='center left', bbox_to_anchor=(1.05, 0.5),
          fancybox=True, shadow=True)
plt.plot(time[max_id2], used_wl_data2.mean(1)[max_id2], 'r.')
plt.plot(time[min_id2], used_wl_data2.mean(1)[min_id2], 'b.')
plt.xlabel("time [sec]")
plt.ylabel("Intensity")
plt.title('SDS20mm')
plt.savefig(os.path.join('pic', subject, f'{date}_invivo_result', exp, 'raw_SDS2.png'), dpi=300, format='png', bbox_inches='tight')
plt.show()


# In[7]:


## plot raw data
ts = 0
td = round(20/SDS1_time_resolution)
plt.figure(figsize=(20,8))
time = np.linspace(0,recovery_end, used_wl_data.shape[0])
plt.plot(time[ts:td], used_wl_data[ts:td].mean(1))
plt.plot(time[max_id1[np.where((max_id1<=td)&(max_id1>=ts))]], used_wl_data.mean(1)[max_id1[np.where((max_id1<=td)&(max_id1>=ts))]], 'r.', ms=10, label='R max')
plt.plot(time[min_id1[np.where((min_id1<=td)&(min_id1>=ts))]], used_wl_data.mean(1)[min_id1[np.where((min_id1<=td)&(min_id1>=ts))]], 'b.', ms=10, label='R min')
plt.xlabel("time [sec]")
plt.ylabel("Intensity")
plt.title('SDS10mm')
plt.legend(loc='upper left', fancybox=True, shadow=True)
plt.savefig(os.path.join('pic', subject, f'{date}_invivo_result', exp, f'raw_SDS1_{ts}_{td}.png'), dpi=300, format='png', bbox_inches='tight')
plt.show()

ts = 0
td = round(20/SDS1_time_resolution)
plt.figure(figsize=(20,8))
time = np.linspace(0,recovery_end, used_wl_data2.shape[0])
plt.plot(time[ts:td], used_wl_data2[ts:td].mean(1))
plt.plot(time[max_id2[np.where((max_id2<=td)&(max_id2>=ts))]], used_wl_data2.mean(1)[max_id2[np.where((max_id2<=td)&(max_id2>=ts))]], 'r.', ms=10, label='R max')
plt.plot(time[min_id2[np.where((min_id2<=td)&(min_id2>=ts))]], used_wl_data2.mean(1)[min_id2[np.where((min_id2<=td)&(min_id2>=ts))]], 'b.', ms=10, label='R min')
plt.xlabel("time [sec]")
plt.ylabel("Intensity")
plt.title('SDS20mm')
plt.legend(loc='upper left', fancybox=True, shadow=True)
plt.savefig(os.path.join('pic', subject, f'{date}_invivo_result', exp, f'raw_SDS2_{ts}_{td}.png'), dpi=300, format='png', bbox_inches='tight')
plt.show()


# In[8]:


plt.rcParams.update({'font.size': 12})
ts = 180
td = round(200/SDS1_time_resolution)
plt.figure(figsize=(8,6))
plt.plot(wavelength, used_wl_data[max_id1[np.where((max_id1<=td)&(max_id1>=ts))]].mean(0), 'o-', label='R max')
plt.plot(wavelength, used_wl_data[min_id1[np.where((min_id1<=td)&(min_id1>=ts))]].mean(0), 'o-', label='R min')
plt.xlabel('wavelength')
plt.ylabel('Intensity')
plt.title('SDS10mm')
plt.legend(loc='upper left', fancybox=True, shadow=True)
plt.savefig(os.path.join('pic', subject, f'{date}_invivo_result', exp, f'raw_SDS1_{ts}_{td}_spec.png'), dpi=300, format='png', bbox_inches='tight')
plt.show()

ts = 180
td = round(200/SDS2_time_resolution)
plt.figure(figsize=(8,6))
plt.plot(wavelength, used_wl_data2[max_id2[np.where((max_id2<=td)&(max_id2>=ts))]].mean(0), 'o-', label='R max')
plt.plot(wavelength, used_wl_data2[min_id2[np.where((min_id2<=td)&(min_id2>=ts))]].mean(0), 'o-', label='R min')
plt.xlabel('wavelength')
plt.ylabel('Intensity')
plt.title('SDS20mm')
plt.legend(loc='upper left', fancybox=True, shadow=True)
plt.savefig(os.path.join('pic', subject, f'{date}_invivo_result', exp, f'raw_SDS2_{ts}_{td}_spec.png'), dpi=300, format='png', bbox_inches='tight')
plt.show()

plt.rcParams.update({'font.size': 20})


# In[9]:

# plot all the wavelength
plt.figure(figsize=(20,8))
time = [i*SDS1_time_resolution for i in range(used_wl_data.shape[0])]
for i in range(20):
    plt.plot(time, used_wl_data[:,i], label=f'{wavelength[i]}nm')

plt.axvline(x=baseline_end, linestyle='--', color='b', label='baseline_end')
plt.axvline(x=HP_end, linestyle='--', color='r', label="Valsalva's maneuver_end")
plt.axvline(x=recovery_end, linestyle='--', color='g', label='recovery_end')
plt.legend(loc='center left', bbox_to_anchor=(1.05, 0.5),
          fancybox=True, shadow=True, ncol=2)
plt.xlabel('time [sec]')
plt.ylabel('Intensity')
plt.title('SDS10mm')
plt.savefig(os.path.join('pic', subject, f'{date}_invivo_result', exp, 'raw_each_wl_SDS1.png'), dpi=300, format='png', bbox_inches='tight')
plt.show()

# plt.figure(figsize=(20,8))
# time = [i*SDS1_time_resolution for i in range(used_wl_data_ch1.shape[0])]
# for i in range(20):
#     plt.plot(time, used_wl_data_ch1[:,i], label=f'{wavelength[i]}nm')

# plt.axvline(x=baseline_end, linestyle='--', color='b', label='baseline_end')
# plt.axvline(x=HP_end, linestyle='--', color='r', label='hyperventilation_end')
# plt.axvline(x=recovery_end, linestyle='--', color='g', label='recovery_end')
# plt.legend(loc='center left', bbox_to_anchor=(1.05, 0.5),
#           fancybox=True, shadow=True, ncol=2)
# plt.xlabel('time [sec]')
# plt.ylabel('Intensity')
# plt.title('SDS4.5mm')
# plt.savefig(os.path.join('pic', subject, f'{date}_invivo_result', exp, 'raw_each_wl_SDS1_4.5mm.png'), dpi=300, format='png', bbox_inches='tight')
# plt.show()

# plt.figure(figsize=(20,8))
# time = [i*SDS1_time_resolution for i in range(used_wl_data_ch2.shape[0])]
# for i in range(20):
#     plt.plot(time, used_wl_data_ch2[:,i], label=f'{wavelength[i]}nm')

# plt.axvline(x=baseline_end, linestyle='--', color='b', label='baseline_end')
# plt.axvline(x=HP_end, linestyle='--', color='r', label='hyperventilation_end')
# plt.axvline(x=recovery_end, linestyle='--', color='g', label='recovery_end')
# plt.legend(loc='center left', bbox_to_anchor=(1.05, 0.5),
#           fancybox=True, shadow=True, ncol=2)
# plt.xlabel('time [sec]')
# plt.ylabel('Intensity')
# plt.title('SDS7.5mm')
# plt.savefig(os.path.join('pic', subject, f'{date}_invivo_result', exp, 'raw_each_wl_SDS1_7.5mm.png'), dpi=300, format='png', bbox_inches='tight')
# plt.show()

plt.figure(figsize=(20,8))
time = [i*SDS2_time_resolution for i in range(used_wl_data2.shape[0])]
for i in range(20):
    plt.plot(time,used_wl_data2[:,i], label=f'{wavelength[i]}nm')

plt.axvline(x=baseline_end, linestyle='--', color='b', label='baseline_end')
plt.axvline(x=HP_end, linestyle='--', color='r', label="Valsalva's maneuver_end")
plt.axvline(x=recovery_end, linestyle='--', color='g', label='recovery_end')
plt.legend(loc='center left', bbox_to_anchor=(1.05, 0.5),
          fancybox=True, shadow=True, ncol=2)
plt.xlabel('time [sec]')
plt.ylabel('Intensity')
plt.title('SDS20mm')
plt.savefig(os.path.join('pic', subject, f'{date}_invivo_result', exp, 'raw_each_wl_SDS2.png'), dpi=300, format='png', bbox_inches='tight')
plt.show()


# In[12]:


# moving avg
def before_after_moving_average(data, avg_points=30):
    '''
    1D array
    '''
    process_data = data.copy()
    original_data = data.copy()
    def moving_average(a, n=3):
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n
    process_data[avg_points - 1:] = moving_average(process_data, n = avg_points)
    process_data[:avg_points - 1] = process_data[avg_points - 1 : avg_points - 2 + avg_points]
    return original_data, process_data


# In[11]:
#####ZOOM IN SDS10 
start_time, end_time,time_resolution = 50, 100, 0.1
plt.figure(figsize=(35,8))
tick_positions = np.arange(start_time, end_time + 1, step=1)
plt.xticks(tick_positions, labels=tick_positions)
time = [i*SDS1_time_resolution for i in range(used_wl_data.shape[0])]
color = ['blue', 'green', 'violet', 'red']
for c_idx, i in enumerate([np.where(wavelength==763)[0][0], np.where(wavelength==805)[0][0], np.where(wavelength==830)[0][0], np.where(wavelength==850)[0][0]]):
    print(i)
    BF_used_wl_data, AF_used_wl_data = before_after_moving_average(data=used_wl_data[:,i])
    plt.plot(time[int(start_time/time_resolution):int(end_time/time_resolution)], BF_used_wl_data[int(start_time/time_resolution):int(end_time/time_resolution)], color=color[c_idx],  alpha=1)
#    plt.plot(time, AF_used_wl_data, color=color[c_idx], label=f'{wavelength[i]}nm')


#plt.axvline(x=baseline_end, linestyle='--', color='b', label='baseline_end')
#plt.axvline(x=HP_end, linestyle='--', color='r', label="Valsalva's maneuver_end")
#plt.axvline(x=recovery_end, linestyle='--', color='g', label='recovery_end')
plt.legend(loc='center left', bbox_to_anchor=(1.05, 0.5),
          fancybox=True, shadow=True)
plt.xlabel('time [sec]')
plt.ylabel('Intensity')
plt.title(f'SDS10mm,{subject} {exp} {start_time}s~{end_time}s')
plt.savefig(os.path.join('pic', subject, f'{date}_invivo_result', exp, 'raw_focus_wl_SDS1.png'), dpi=300, format='png', bbox_inches='tight')
plt.show()

# plot interested wavelength [763nm , 805nm, 830nm, 850nm]
plt.figure(figsize=(20,8))
time = [i*SDS1_time_resolution for i in range(used_wl_data.shape[0])]
color = ['blue', 'green', 'violet', 'red']
for c_idx, i in enumerate([np.where(wavelength==768)[0][0], np.where(wavelength==805)[0][0], np.where(wavelength==830)[0][0], np.where(wavelength==850)[0][0]]):
    BF_used_wl_data, AF_used_wl_data = before_after_moving_average(data=used_wl_data[:,i])
    plt.plot(time, BF_used_wl_data, color=color[c_idx], alpha=1)
#    plt.plot(time, AF_used_wl_data, color=color[c_idx], label=f'{wavelength[i]}nm')

# plt.axvline(x=baseline_end, linestyle='--', color='b', label='baseline_end')
# plt.axvline(x=HP_end, linestyle='--', color='r', label="Valsalva's maneuver_end")
# plt.axvline(x=recovery_end, linestyle='--', color='g', label='recovery_end')
plt.legend(loc='center left', bbox_to_anchor=(1.05, 0.5),
          fancybox=True, shadow=True)
plt.xlabel('time [sec]')
plt.ylabel('Intensity')
plt.title(f'SDS10mm,{subject} {exp} ')
plt.savefig(os.path.join('pic', subject, f'{date}_invivo_result', exp, 'raw_focus_wl_SDS1.png'), dpi=300, format='png', bbox_inches='tight')
plt.show()

#%%
AVG10 = used_wl_data.copy()
columns_to_change = [11, 16, 18, 19]
# 计算指定行和列的均值
new_values = np.mean(AVG10[:620, columns_to_change], axis=0)
# 将均值广播到整个AVG10数据集中指定的位置
AVG10[:620, columns_to_change] = new_values
print(AVG10)

start_time, end_time,time_resolution = 50, 100, 0.1
plt.figure(figsize=(35,8))
tick_positions = np.arange(start_time, end_time + 1, step=1)
plt.xticks(tick_positions, labels=tick_positions)
time = [i*SDS2_time_resolution for i in range(AVG10.shape[0])]
color = ['blue', 'green', 'violet', 'red']
for c_idx, i in enumerate([np.where(wavelength==763)[0][0], np.where(wavelength==805)[0][0], np.where(wavelength==830)[0][0], np.where(wavelength==850)[0][0]]):
    print(i)
    BF_used_wl_data, AF_used_wl_data = before_after_moving_average(data=AVG10[:,i])
    plt.plot(time[int(start_time/time_resolution):int(end_time/time_resolution)], BF_used_wl_data[int(start_time/time_resolution):int(end_time/time_resolution)], color=color[c_idx],  alpha=1) #label=f'{wavelength[i]}nm',
#    plt.plot(time, AF_used_wl_data, color=color[c_idx], label=f'{wavelength[i]}nm')


#plt.axvline(x=baseline_end, linestyle='--', color='b', label='baseline_end')
#plt.axvline(x=HP_end, linestyle='--', color='r', label="Valsalva's maneuver_end")
#plt.axvline(x=recovery_end, linestyle='--', color='g', label='recovery_end')
plt.legend(loc='center left', bbox_to_anchor=(1.05, 0.5),
          fancybox=True, shadow=True)
plt.xlabel('time [sec]')
plt.ylabel('Intensity')
plt.title(f'SDS10mm,{subject} {exp} {start_time}s~{end_time}s')
plt.savefig(os.path.join('pic', subject, f'{date}_invivo_result', exp, 'raw_focus_wl_SDS1.png'), dpi=300, format='png', bbox_inches='tight')
plt.show()

# Reletive_percentage_change = (used_wl_data2 - used_wl_data2[500:620,:].mean(axis=0)) / used_wl_data2[500:620,:].mean(axis=0)* 100
# selected_data = Reletive_percentage_change[rows_to_change, :][:, columns_to_change]
# 计算相对百分比变化时，确保每列的维度为 (620, 1)
relative_percentage_change = np.zeros((380, len(columns_to_change)))

for idx, col in enumerate(columns_to_change):
    relative_percentage_change[:, idx] = ((AVG10[621:1001, col] - AVG10[rows_to_change, col][:, np.newaxis]) / AVG10[rows_to_change, col][:, np.newaxis]) * 100

relative_percentage_change_col11 = ((AVG10[621:1001, 11] - AVG10[rows_to_change, 11][:, np.newaxis]) / AVG10[rows_to_change, 11][:, np.newaxis]) * 100
relative_percentage_change_col16 = ((AVG10[621:1001, 16] - AVG10[rows_to_change, 16][:, np.newaxis]) / AVG10[rows_to_change, 16][:, np.newaxis]) * 100
relative_percentage_change_col18 = ((AVG10[621:1001, 18] - AVG10[rows_to_change, 18][:, np.newaxis]) / AVG10[rows_to_change, 18][:, np.newaxis]) * 100
relative_percentage_change_col19 = ((AVG10[621:1001, 19] - AVG10[rows_to_change, 19][:, np.newaxis]) / AVG10[rows_to_change, 19][:, np.newaxis]) * 100


time_Percentage = np.arange(621, 1001) * time_resolution

columns_to_change = [11, 16, 18, 19]
new_values = np.mean(AVG10[:620, columns_to_change], axis=0)
# 将均值广播到整个AVG10数据集中指定的位置
AVG10[:620, columns_to_change] = new_values
# 绘制图形
start_time, end_time, time_resolution = 50, 100, 0.1
plt.figure(figsize=(35, 8))
tick_positions = np.arange(start_time, end_time + 1, step=1)
plt.xticks(tick_positions, labels=tick_positions)
time = np.arange(AVG10.shape[0]) * time_resolution

# 遍历每个波长
color = ['blue', 'green', 'violet', 'red']
for c_idx, i in enumerate([np.where(wavelength == 763)[0][0], np.where(wavelength == 805)[0][0], np.where(wavelength == 830)[0][0], np.where(wavelength == 850)[0][0]]):
    BF_used_wl_data, AF_used_wl_data = before_after_moving_average(data=AVG10[:, i])
    plt.plot(time[int(start_time / time_resolution):int(end_time / time_resolution)], BF_used_wl_data[int(start_time / time_resolution):int(end_time / time_resolution)], color=color[c_idx], alpha=1)

plt.legend(loc='center left', bbox_to_anchor=(1.05, 0.5),
          fancybox=True, shadow=True)
plt.xlabel('time [sec]')
plt.ylabel('Intensity')
plt.title(f'SDS10mm,{subject} {exp} {start_time}s~{end_time}s')
plt.savefig(os.path.join('pic', subject, f'{date}_invivo_result', exp, 'raw_focus_wl_SDS1.png'), dpi=300, format='png', bbox_inches='tight')
plt.show()

# 计算相对百分比变化
relative_percentage_change = np.zeros((380, len(columns_to_change)))

for idx, col in enumerate(columns_to_change):
    relative_percentage_change[:, idx] = ((AVG10[621:1001, col] - AVG10[:380, col][:, np.newaxis]) / AVG10[:380, col][:, np.newaxis]) * 100

# 绘制图表
plt.figure(figsize=(20,8))
plt.plot(time_Percentage, relative_percentage_change_col11[0, :380], label='763nm')
plt.plot(time_Percentage, relative_percentage_change_col16[0, :380], label='805nm')
plt.plot(time_Percentage, relative_percentage_change_col18[0, :380], label='830nm')
plt.plot(time_Percentage, relative_percentage_change_col19[0, :380], label='850nm')

plt.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), fancybox=True, shadow=True)
plt.xlabel('time [sec]')
plt.ylabel('Relative Percentage Change')
plt.title(f'SDS10mm,{subject} {exp} 63s~100s Relative Percentage Changes for Specific wavelength')
plt.show()
#%%
###SDS20(ZOOM IN SDS20) 
start_time, end_time,time_resolution = 50, 100, 0.1
plt.figure(figsize=(35,8))
tick_positions = np.arange(start_time, end_time + 1, step=1)
plt.xticks(tick_positions, labels=tick_positions)
time = [i*SDS1_time_resolution for i in range(used_wl_data2.shape[0])]
color = ['blue', 'green', 'violet', 'red']
for c_idx, i in enumerate([np.where(wavelength==763)[0][0], np.where(wavelength==805)[0][0], np.where(wavelength==830)[0][0], np.where(wavelength==850)[0][0]]):
    print(i)
    BF_used_wl_data, AF_used_wl_data = before_after_moving_average(data=used_wl_data2[:,i])
    plt.plot(time[int(start_time/time_resolution):int(end_time/time_resolution)], BF_used_wl_data[int(start_time/time_resolution):int(end_time/time_resolution)], color=color[c_idx],  alpha=1)
#    plt.plot(time, AF_used_wl_data, color=color[c_idx], label=f'{wavelength[i]}nm')


#plt.axvline(x=baseline_end, linestyle='--', color='b', label='baseline_end')
#plt.axvline(x=HP_end, linestyle='--', color='r', label="Valsalva's maneuver_end")
#plt.axvline(x=recovery_end, linestyle='--', color='g', label='recovery_end')
plt.legend(loc='center left', bbox_to_anchor=(1.05, 0.5),
          fancybox=True, shadow=True)
plt.xlabel('time [sec]')
plt.ylabel('Intensity')
plt.title(f'SDS20mm,{subject} {exp} {start_time}s~{end_time}s')
plt.savefig(os.path.join('pic', subject, f'{date}_invivo_result', exp, 'raw_focus_wl_SDS2.png'), dpi=300, format='png', bbox_inches='tight')
plt.show()

# SDS20 all plot interested wavelength [763nm , 805nm, 830nm, 850nm] 
plt.figure(figsize=(20,8))
time = [i*SDS1_time_resolution for i in range(used_wl_data2.shape[0])]
color = ['blue', 'green', 'violet', 'red']
for c_idx, i in enumerate([np.where(wavelength==748)[0][0], np.where(wavelength==805)[0][0], np.where(wavelength==830)[0][0], np.where(wavelength==850)[0][0]]):
    BF_used_wl_data, AF_used_wl_data = before_after_moving_average(data=used_wl_data2[:,i])
    plt.plot(time, BF_used_wl_data, color=color[c_idx], alpha=1)
#    plt.plot(time, AF_used_wl_data, color=color[c_idx], label=f'{wavelength[i]}nm')

# plt.axvline(x=baseline_end, linestyle='--', color='b', label='baseline_end')
# plt.axvline(x=HP_end, linestyle='--', color='r', label="Valsalva's maneuver_end")
# plt.axvline(x=recovery_end, linestyle='--', color='g', label='recovery_end')
plt.legend(loc='center left', bbox_to_anchor=(1.05, 0.5),
          fancybox=True, shadow=True)
plt.xlabel('time [sec]')
plt.ylabel('Intensity')
plt.title(f'SDS20mm,{subject} {exp} ')
plt.savefig(os.path.join('pic', subject, f'{date}_invivo_result', exp, 'raw_focus_wl_SDS1.png'), dpi=300, format='png', bbox_inches='tight')
plt.show()

#%%
#####ZOOM IN SDS20 
AVG = used_wl_data2.copy()

columns_to_change = [11, 16, 18, 19]  
rows_to_change = range(0, 620)  
new_values = [922.4047723, 972.0260594, 702.3775743, 533.3225248]

for i, col in enumerate(columns_to_change):
    AVG[rows_to_change, col] = new_values[i]

print(AVG)

start_time, end_time,time_resolution = 50, 100, 0.1
plt.figure(figsize=(35,8))
tick_positions = np.arange(start_time, end_time + 1, step=1)
plt.xticks(tick_positions, labels=tick_positions)
time = [i*SDS2_time_resolution for i in range(AVG.shape[0])]
color = ['blue', 'green', 'violet', 'red']
for c_idx, i in enumerate([np.where(wavelength==763)[0][0], np.where(wavelength==805)[0][0], np.where(wavelength==830)[0][0], np.where(wavelength==850)[0][0]]):
    print(i)
    BF_used_wl_data, AF_used_wl_data = before_after_moving_average(data=AVG[:,i])
    plt.plot(time[int(start_time/time_resolution):int(end_time/time_resolution)], BF_used_wl_data[int(start_time/time_resolution):int(end_time/time_resolution)], color=color[c_idx],  alpha=1) #label=f'{wavelength[i]}nm',
#    plt.plot(time, AF_used_wl_data, color=color[c_idx], label=f'{wavelength[i]}nm')


#plt.axvline(x=baseline_end, linestyle='--', color='b', label='baseline_end')
#plt.axvline(x=HP_end, linestyle='--', color='r', label="Valsalva's maneuver_end")
#plt.axvline(x=recovery_end, linestyle='--', color='g', label='recovery_end')
plt.legend(loc='center left', bbox_to_anchor=(1.05, 0.5),
          fancybox=True, shadow=True)
plt.xlabel('time [sec]')
plt.ylabel('Intensity')
plt.title(f'SDS20mm,{subject} {exp} {start_time}s~{end_time}s')
plt.savefig(os.path.join('pic', subject, f'{date}_invivo_result', exp, 'raw_focus_wl_SDS2.png'), dpi=300, format='png', bbox_inches='tight')
plt.show()

# Reletive_percentage_change = (used_wl_data2 - used_wl_data2[500:620,:].mean(axis=0)) / used_wl_data2[500:620,:].mean(axis=0)* 100

# selected_data = Reletive_percentage_change[rows_to_change, :][:, columns_to_change]

# 计算相对百分比变化时，确保每列的维度为 (620, 1)
relative_percentage_change_col11 = ((AVG[621:1001, 11] - AVG[rows_to_change, 11][:, np.newaxis]) / AVG[rows_to_change, 11][:, np.newaxis]) * 100
relative_percentage_change_col16 = ((AVG[621:1001, 16] - AVG[rows_to_change, 16][:, np.newaxis]) / AVG[rows_to_change, 16][:, np.newaxis]) * 100
relative_percentage_change_col18 = ((AVG[621:1001, 18] - AVG[rows_to_change, 18][:, np.newaxis]) / AVG[rows_to_change, 18][:, np.newaxis]) * 100
relative_percentage_change_col19 = ((AVG[621:1001, 19] - AVG[rows_to_change, 19][:, np.newaxis]) / AVG[rows_to_change, 19][:, np.newaxis]) * 100


time_Percentage = np.arange(621, 1001) * time_resolution

# 绘制图表
plt.figure(figsize=(20,8))
plt.plot(time_Percentage, relative_percentage_change_col11[0, :380], label='763nm')
plt.plot(time_Percentage, relative_percentage_change_col16[0, :380], label='805nm')
plt.plot(time_Percentage, relative_percentage_change_col18[0, :380], label='830nm')
plt.plot(time_Percentage, relative_percentage_change_col19[0, :380], label='850nm')

plt.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), fancybox=True, shadow=True)
plt.xlabel('time [sec]')
plt.ylabel('Relative Percentage Change')
plt.title(f'SDS20mm,{subject} {exp} 63s~100s Relative Percentage Changes for Specific wavelength')
plt.show()

##SDS20 all 
plt.figure(figsize=(20,8))
plt.rcParams.update({'font.size':30})
time = [i*SDS2_time_resolution for i in range(AVG.shape[0])]
color = ['blue', 'green', 'violet', 'red']
for c_idx, i in enumerate([np.where(wavelength==763)[0][0], np.where(wavelength==805)[0][0], np.where(wavelength==830)[0][0], np.where(wavelength==850)[0][0]]):
    BF_used_wl_data, AF_used_wl_data = before_after_moving_average(data=AVG[:,i])
    plt.plot(time, BF_used_wl_data, color=color[c_idx], alpha=1)
#    plt.plot(time, AF_used_wl_data, color=color[c_idx], label=f'{wavelength[i]}nm')

plt.axvline(x=baseline_end, linestyle='--', color='b')#, label='baseline_end')
plt.axvline(x=HP_end, linestyle='--', color='r')#, label="Valsalva's maneuver_end")
#plt.axvline(x=recovery_end, linestyle='--', color='g', label='recovery_end')
plt.legend(loc='center left', bbox_to_anchor=(1.05, 0.5),
          fancybox=True, shadow=True)
plt.xlabel('time [sec]')
plt.ylabel('Intensity')
plt.title(f'SDS20mm,{subject} {exp} ')
plt.savefig(os.path.join('pic', subject, f'{date}_invivo_result', exp, 'raw_focus_wl_SDS2.png'), dpi=300, format='png', bbox_inches='tight')
plt.show()
#%%
##Normalized Intensity SDS10mm
plt.figure(figsize=(20,8))
time = [i*SDS1_time_resolution for i in range(used_wl_data.shape[0])]
color = ['blue', 'green', 'red']
for c_idx, i in enumerate([np.where(wavelength==763)[0][0], np.where(wavelength==805)[0][0], np.where(wavelength==850)[0][0]]):
    BF_used_wl_data, AF_used_wl_data = before_after_moving_average(data=used_wl_data[:,i])
    plt.plot(time, (BF_used_wl_data- BF_used_wl_data.min())/(BF_used_wl_data.max()-BF_used_wl_data.min()), color=color[c_idx], alpha=0.1)
    plt.plot(time, (AF_used_wl_data- BF_used_wl_data.min())/(BF_used_wl_data.max()-BF_used_wl_data.min()), color=color[c_idx], label=f'{wavelength[i]}nm')

plt.axvline(x=baseline_end, linestyle='--', color='b', label='baseline_end')
plt.axvline(x=HP_end, linestyle='--', color='r', label="Valsalva's maneuver_end")
#plt.axvline(x=recovery_end, linestyle='--', color='g', label='recovery_end')
plt.legend(loc='center left', bbox_to_anchor=(1.05, 0.5),
          fancybox=True, shadow=True)
plt.xlabel('time [sec]')
plt.ylabel('Normalized Intensity')
plt.title('SDS10mm')
plt.savefig(os.path.join('pic', subject, f'{date}_invivo_result', exp, 'raw_normalized_focus_wl_SDS1.png'), dpi=300, format='png', bbox_inches='tight')
plt.show()


# plt.figure(figsize=(20,8))
# time = [i*SDS1_time_resolution for i in range(used_wl_data_ch1.shape[0])]
# color = ['blue', 'green', 'red']
# for c_idx, i in enumerate([np.where(wavelength==763)[0][0], np.where(wavelength==805)[0][0], np.where(wavelength==850)[0][0]]):
#     BF_used_wl_data, AF_used_wl_data = before_after_moving_average(data=used_wl_data_ch1[:,i])
#     plt.plot(time, (BF_used_wl_data- BF_used_wl_data.min())/(BF_used_wl_data.max()-BF_used_wl_data.min()), color=color[c_idx], alpha=0.1)
#     plt.plot(time, (AF_used_wl_data- BF_used_wl_data.min())/(BF_used_wl_data.max()-BF_used_wl_data.min()), color=color[c_idx], label=f'{wavelength[i]}nm')

# plt.axvline(x=baseline_end, linestyle='--', color='b', label='baseline_end')
# plt.axvline(x=HP_end, linestyle='--', color='r', label='hyperventilation_end')
# plt.axvline(x=recovery_end, linestyle='--', color='g', label='recovery_end')
# plt.legend(loc='center left', bbox_to_anchor=(1.05, 0.5),
#           fancybox=True, shadow=True)
# plt.xlabel('time [sec]')
# plt.ylabel('Normalized Intensity')
# plt.title('SDS4.5mm')
# plt.savefig(os.path.join('pic', subject, f'{date}_invivo_result', exp, 'raw_normalized_focus_wl_SDS1_4.5mm.png'), dpi=300, format='png', bbox_inches='tight')
# plt.show()

# plt.figure(figsize=(20,8))
# time = [i*SDS1_time_resolution for i in range(used_wl_data_ch2.shape[0])]
# color = ['blue', 'green', 'red']
# for c_idx, i in enumerate([np.where(wavelength==763)[0][0], np.where(wavelength==805)[0][0], np.where(wavelength==850)[0][0]]):
#     BF_used_wl_data, AF_used_wl_data = before_after_moving_average(data=used_wl_data_ch2[:,i])
#     plt.plot(time, (BF_used_wl_data- BF_used_wl_data.min())/(BF_used_wl_data.max()-BF_used_wl_data.min()), color=color[c_idx], alpha=0.1)
#     plt.plot(time, (AF_used_wl_data- BF_used_wl_data.min())/(BF_used_wl_data.max()-BF_used_wl_data.min()), color=color[c_idx], label=f'{wavelength[i]}nm')

# plt.axvline(x=baseline_end, linestyle='--', color='b', label='baseline_end')
# plt.axvline(x=HP_end, linestyle='--', color='r', label='hyperventilation_end')
# plt.axvline(x=recovery_end, linestyle='--', color='g', label='recovery_end')
# plt.legend(loc='center left', bbox_to_anchor=(1.05, 0.5),
#           fancybox=True, shadow=True)
# plt.xlabel('time [sec]')
# plt.ylabel('Normalized Intensity')
# plt.title('SDS7.5mm')
# plt.savefig(os.path.join('pic', subject, f'{date}_invivo_result', exp, 'raw_normalized_focus_wl_SDS1_7.5mm.png'), dpi=300, format='png', bbox_inches='tight')
# plt.show()

##Normalized Intensity SDS20mm
plt.figure(figsize=(20,8))
time = [i*SDS2_time_resolution for i in range(used_wl_data2.shape[0])]
color = ['blue', 'green', 'red']
for c_idx, i in enumerate([np.where(wavelength==763)[0][0], np.where(wavelength==805)[0][0], np.where(wavelength==850)[0][0]]):
    BF_used_wl_data, AF_used_wl_data = before_after_moving_average(data=used_wl_data2[:,i])
    plt.plot(time, (BF_used_wl_data- BF_used_wl_data.min())/(BF_used_wl_data.max()-BF_used_wl_data.min()), color=color[c_idx], alpha=0.1)
    plt.plot(time, (AF_used_wl_data- BF_used_wl_data.min())/(BF_used_wl_data.max()-BF_used_wl_data.min()), color=color[c_idx], label=f'{wavelength[i]}nm')

plt.axvline(x=baseline_end, linestyle='--', color='b', label='baseline_end')
plt.axvline(x=HP_end, linestyle='--', color='r', label="Valsalva's maneuver_end")
#plt.axvline(x=recovery_end, linestyle='--', color='g', label='recovery_end')
plt.legend(loc='center left', bbox_to_anchor=(1.05, 0.5),
          fancybox=True, shadow=True)
plt.xlabel('time [sec]')
plt.ylabel('Normalized Intensity')
plt.title('SDS20mm')
plt.savefig(os.path.join('pic', subject, f'{date}_invivo_result', exp, 'raw_normalized_focus_wl_SDS2.png'), dpi=300, format='png', bbox_inches='tight')
plt.show()

## phantom 
# In[6]:


# cali short ch.
cali = pd.read_csv(os.path.join('dataset', subject, 'calibration_result', date, 'calibrate_SDS_1.csv'))
cali = cali.to_numpy()
cali_used_wl_data = (used_wl_data*cali[0] + cali[1])

# cali long ch.
cali2 = pd.read_csv(os.path.join('dataset', subject, 'calibration_result', date, 'calibrate_SDS_2.csv'))
cali2 = cali2.to_numpy()
cali_used_wl_data2 = (used_wl_data2*cali2[0] + cali2[1])


# In[17]:


# plot calibrated data
plt.figure(figsize=(20,8))
time = np.linspace(0,recovery_end, cali_used_wl_data.shape[0])
plt.axvline(x=baseline_end, linestyle='--', color='b', label='baseline_end')
plt.axvline(x=HP_end, linestyle='--', color='r', label="Valsalva's maneuver_end")
plt.axvline(recovery_end, linestyle='--', color='g', label='recovery_end')
plt.legend(loc='center left', bbox_to_anchor=(1.05, 0.5),
          fancybox=True, shadow=True)
plt.plot(time, cali_used_wl_data.mean(1))
plt.plot(time[max_id1], cali_used_wl_data.mean(1)[max_id1], 'r.')
plt.plot(time[min_id1], cali_used_wl_data.mean(1)[min_id1], 'b.')
plt.xlabel('time [sec]')
plt.ylabel('Intensity')
plt.title('SDS10mm')
plt.savefig(os.path.join('pic', subject, f'{date}_invivo_result', exp, 'cali_SDS1.png'), dpi=300, format='png', bbox_inches='tight')
plt.show()

plt.figure(figsize=(20,8))
time = np.linspace(0,recovery_end, cali_used_wl_data2.shape[0])
plt.axvline(x=baseline_end, linestyle='--', color='b', label='baseline_end')
plt.axvline(x=HP_end, linestyle='--', color='r', label="Valsalva's maneuver_end")
plt.axvline(recovery_end, linestyle='--', color='g', label='recovery_end')
plt.legend(loc='center left', bbox_to_anchor=(1.05, 0.5),
          fancybox=True, shadow=True)
plt.plot(time, cali_used_wl_data2.mean(1))
plt.plot(time[max_id2], cali_used_wl_data2.mean(1)[max_id2], 'r.')
plt.plot(time[min_id2], cali_used_wl_data2.mean(1)[min_id2], 'b.')
plt.xlabel('time [sec]')
plt.ylabel('Intensity')
plt.title('SDS20mm')
plt.savefig(os.path.join('pic', subject, f'{date}_invivo_result', exp, 'cali_SDS2.png'), dpi=300, format='png', bbox_inches='tight')
plt.show()


# In[18]:


# plot all the wavelength
plt.figure(figsize=(20,8))
time = [i*SDS1_time_resolution for i in range(cali_used_wl_data.shape[0])]
for i in range(20):
    plt.plot(time, cali_used_wl_data[:,i], label=f'{wavelength[i]}nm')

plt.axvline(x=baseline_end, linestyle='--', color='b', label='baseline_end')
plt.axvline(x=HP_end, linestyle='--', color='r', label="Valsalva's maneuver_end")
plt.axvline(x=recovery_end, linestyle='--', color='g', label='recovery_end')
plt.legend(loc='center left', bbox_to_anchor=(1.05, 0.5),
          fancybox=True, shadow=True, ncol=2)
plt.xlabel('time [sec]')
plt.ylabel('Intensity')
plt.title('SDS10mm')
plt.savefig(os.path.join('pic', subject, f'{date}_invivo_result', exp, 'cali_each_wl_SDS1.png'), dpi=300, format='png', bbox_inches='tight')
plt.show()

plt.figure(figsize=(20,8))
time = [i*SDS2_time_resolution for i in range(cali_used_wl_data2.shape[0])]
for i in range(20):
    plt.plot(time,cali_used_wl_data2[:,i], label=f'{wavelength[i]}nm')

plt.axvline(x=baseline_end, linestyle='--', color='b', label='baseline_end')
plt.axvline(x=HP_end, linestyle='--', color='r', label="Valsalva's maneuver_end")
plt.axvline(x=recovery_end, linestyle='--', color='g', label='recovery_end')
plt.legend(loc='center left', bbox_to_anchor=(1.05, 0.5),
          fancybox=True, shadow=True, ncol=2)
plt.xlabel('time [sec]')
plt.ylabel('Intensity')
plt.title('SDS20mm')
plt.savefig(os.path.join('pic', subject, f'{date}_invivo_result', exp, 'cali_each_wl_SDS2.png'), dpi=300, format='png', bbox_inches='tight')
plt.show()


# In[19]:


# plot interested wavelength [763nm , 805nm, 850nm]
plt.figure(figsize=(20,8))
time = [i*SDS1_time_resolution for i in range(cali_used_wl_data.shape[0])]
color = ['blue', 'green', 'red']
for c_idx, i in enumerate([np.where(wavelength==763)[0][0], np.where(wavelength==805)[0][0], np.where(wavelength==850)[0][0]]):
    BF_used_wl_data, AF_used_wl_data = before_after_moving_average(data=cali_used_wl_data[:,i])
    plt.plot(time, BF_used_wl_data, color=color[c_idx], alpha=0.3)
    plt.plot(time, AF_used_wl_data, color=color[c_idx], label=f'{wavelength[i]}nm')

plt.axvline(x=baseline_end, linestyle='--', color='b', label='baseline_end')
plt.axvline(x=HP_end, linestyle='--', color='r', label="Valsalva's maneuver_end")
plt.axvline(x=recovery_end, linestyle='--', color='g', label='recovery_end')
plt.legend(loc='center left', bbox_to_anchor=(1.05, 0.5),
          fancybox=True, shadow=True)
plt.xlabel('time [sec]')
plt.ylabel('Intensity')
plt.title('SDS10mm')
plt.savefig(os.path.join('pic', subject, f'{date}_invivo_result', exp, 'cali_focus_wl_SDS1.png'), dpi=300, format='png', bbox_inches='tight')
plt.show()

plt.figure(figsize=(20,8))
time = [i*SDS2_time_resolution for i in range(cali_used_wl_data2.shape[0])]
color = ['blue', 'green', 'red']
for c_idx, i in enumerate([np.where(wavelength==763)[0][0], np.where(wavelength==805)[0][0], np.where(wavelength==850)[0][0]]):
    BF_used_wl_data, AF_used_wl_data = before_after_moving_average(data=cali_used_wl_data2[:,i])
    plt.plot(time, BF_used_wl_data, color=color[c_idx], alpha=0.3)
    plt.plot(time, AF_used_wl_data, color=color[c_idx], label=f'{wavelength[i]}nm')

plt.axvline(x=baseline_end, linestyle='--', color='b', label='baseline_end')
plt.axvline(x=HP_end, linestyle='--', color='r', label="Valsalva's maneuver_end")
plt.axvline(x=recovery_end, linestyle='--', color='g', label='recovery_end')
plt.legend(loc='center left', bbox_to_anchor=(1.05, 0.5),
          fancybox=True, shadow=True)
plt.xlabel('time [sec]')
plt.ylabel('Intensity')
plt.title('SDS20mm')
plt.savefig(os.path.join('pic', subject, f'{date}_invivo_result', exp, 'cali_focus_wl_SDS2.png'), dpi=300, format='png', bbox_inches='tight')
plt.show()



plt.figure(figsize=(20,8))
time = [i*SDS1_time_resolution for i in range(cali_used_wl_data.shape[0])]
color = ['blue', 'green', 'red']
for c_idx, i in enumerate([np.where(wavelength==763)[0][0], np.where(wavelength==805)[0][0], np.where(wavelength==850)[0][0]]):
    BF_used_wl_data, AF_used_wl_data = before_after_moving_average(data=cali_used_wl_data[:,i])
    plt.plot(time, (BF_used_wl_data- BF_used_wl_data.min())/(BF_used_wl_data.max()-BF_used_wl_data.min()), color=color[c_idx], alpha=0.1)
    plt.plot(time, (AF_used_wl_data- BF_used_wl_data.min())/(BF_used_wl_data.max()-BF_used_wl_data.min()), color=color[c_idx], label=f'{wavelength[i]}nm')

plt.axvline(x=baseline_end, linestyle='--', color='b', label='baseline_end')
plt.axvline(x=HP_end, linestyle='--', color='r', label="Valsalva's maneuver_end")
plt.axvline(x=recovery_end, linestyle='--', color='g', label='recovery_end')
plt.legend(loc='center left', bbox_to_anchor=(1.05, 0.5),
          fancybox=True, shadow=True)
plt.xlabel('time [sec]')
plt.ylabel('Normalized Intensity')
plt.title('SDS10mm')
plt.savefig(os.path.join('pic', subject, f'{date}_invivo_result', exp, 'cali_normalized_focus_wl_SDS1.png'), dpi=300, format='png', bbox_inches='tight')
plt.show()

plt.figure(figsize=(20,8))
time = [i*SDS2_time_resolution for i in range(cali_used_wl_data2.shape[0])]
color = ['blue', 'green', 'red']
for c_idx, i in enumerate([np.where(wavelength==763)[0][0], np.where(wavelength==805)[0][0], np.where(wavelength==850)[0][0]]):
    BF_used_wl_data, AF_used_wl_data = before_after_moving_average(data=cali_used_wl_data2[:,i])
    plt.plot(time, (BF_used_wl_data- BF_used_wl_data.min())/(BF_used_wl_data.max()-BF_used_wl_data.min()), color=color[c_idx], alpha=0.1)
    plt.plot(time, (AF_used_wl_data- BF_used_wl_data.min())/(BF_used_wl_data.max()-BF_used_wl_data.min()), color=color[c_idx], label=f'{wavelength[i]}nm')

plt.axvline(x=baseline_end, linestyle='--', color='b', label='baseline_end')
plt.axvline(x=HP_end, linestyle='--', color='r', label="Valsalva's maneuver_end")
plt.axvline(x=recovery_end, linestyle='--', color='g', label='recovery_end')
plt.legend(loc='center left', bbox_to_anchor=(1.05, 0.5),
          fancybox=True, shadow=True)
plt.xlabel('time [sec]')
plt.ylabel('Normalized Intensity')
plt.title('SDS20mm')
plt.savefig(os.path.join('pic', subject, f'{date}_invivo_result', exp, 'cali_normalized_focus_wl_SDS2.png'), dpi=300, format='png', bbox_inches='tight')
plt.show()


# ## Plot Rmax/Rmin

# In[ ]:


# load file Nx : time Ny : Rmax/Rmin
# SDS1_Rmax_Rmin = pd.read_csv(os.path.join('pic', subject, 'SDS1', date, 'spectrum', 'get_peak', 'mean_time_spec.csv')).to_numpy()
# SDS2_Rmax_Rmin = pd.read_csv(os.path.join('pic', subject, 'SDS2', date, 'spectrum', 'get_peak', 'mean_time_spec.csv')).to_numpy()
# SDS1_Rmax_Rmin_data, SDS1_Rmax_Rmin_time = SDS1_Rmax_Rmin[:, 1], SDS1_Rmax_Rmin[:, 0]
# SDS2_Rmax_Rmin_data, SDS2_Rmax_Rmin_time = SDS2_Rmax_Rmin[:, 1], SDS2_Rmax_Rmin[:, 0]


# In[20]:


SDS1_Rmax_Rmin_data = []
SDS2_Rmax_Rmin_data = []
for ts in range(0, recovery_end, 10):
    td = ts + 10
    used_max_idx1 = np.where(((max_id1<round(td/SDS1_time_resolution))&(max_id1>=round(ts/SDS1_time_resolution))))
    used_min_idx1 = np.where(((min_id1<round(td/SDS1_time_resolution))&(min_id1>=round(ts/SDS1_time_resolution))))
    SDS1_Rmax_Rmin_data += [used_wl_data[max_id1[used_max_idx1]].mean() / used_wl_data[min_id1[used_min_idx1]].mean()]
    
    used_max_idx2 = np.where(((max_id2<round(td/SDS1_time_resolution))&(max_id2>=round(ts/SDS1_time_resolution))))
    used_min_idx2 = np.where(((min_id2<round(td/SDS1_time_resolution))&(min_id2>=round(ts/SDS1_time_resolution))))
    SDS2_Rmax_Rmin_data += [used_wl_data2[max_id2[used_max_idx2]].mean() / used_wl_data2[min_id2[used_min_idx2]].mean()]


# In[21]:


plt.rcParams.update({'font.size': 12})
time = np.linspace(0, recovery_end, len(SDS1_Rmax_Rmin_data))
plt.plot(time, SDS1_Rmax_Rmin_data, label='SDS=10mm')
plt.plot(time, SDS2_Rmax_Rmin_data, label='SDS=20mm')
plt.axvline(x=baseline_end, linestyle='--', color='b', label='baseline_end')
plt.axvline(x=HP_end, linestyle='--', color='r', label="Valsalva's maneuver_end")
plt.axvline(x=recovery_end, linestyle='--', color='g', label='recovery_end')
plt.legend(loc='center left', bbox_to_anchor=(1.05, 0.5),
          fancybox=True, shadow=True)
plt.xlabel('time [sec]')
plt.ylabel('Rmax/Rmin')
# plt.savefig(os.path.join('pic', subject, f'{date}_invivo_result', exp,'Rmax_Rmin.png'), dpi=300, format='png', bbox_inches='tight')
plt.show()


# ## Predict in-vivo data

# In[7]:


plt.rcParams.update({'font.size': 12})


# In[8]:


# load model 
import torch
from ANN_models import PredictionModel, PredictionModel2, PredictionModel3, PredictionModel4, PredictionModel5, PredictionModel6
# with open(os.path.join("model_save", "prediction_model_formula24_chromophore_rand_ab", "ctchen", "trlog.json"), 'r') as f:
#     trlog = json.load(f)
#     best_model = trlog['best_model']

# # # model = PredictionModel2().cuda()
# # # model = PredictionModel3().cuda()
# # model = PredictionModel5(neuronsize=5).cuda()
# # # model = PredictionModel6().cuda()
# model.load_state_dict(torch.load(best_model, map_location='cpu'))

model = PredictionModel5(neuronsize=5)
temp = torch.load(os.path.join("model_save", "BU_prediction_model", "70.pth"), map_location='cpu')
model.load_state_dict(temp)


# In[9]:


def get_OD(used_wl_data, used_wl_data2, time, average_time=6):
    time_resolution = SDS1_time_resolution
    time_resolution2 = SDS2_time_resolution
    sds1_ijv_small = used_wl_data[max_id1[np.where(abs(max_id1-round(time/time_resolution))<round(average_time/time_resolution))]].mean(0)
    sds1_ijv_large = used_wl_data[min_id1[np.where(abs(min_id1-round(time/time_resolution))<round(average_time/time_resolution))]].mean(0)
    
    sds2_ijv_small = used_wl_data2[max_id2[np.where(abs(max_id2-round(time/time_resolution2))<round(average_time/time_resolution2))]].mean(0)
    sds2_ijv_large = used_wl_data2[min_id2[np.where(abs(min_id2-round(time/time_resolution2))<round(average_time/time_resolution2))]].mean(0)
    
    # sds1_ijv_small = used_wl_data[max_id[idx:idx+average_point]].mean(0)
    # sds1_ijv_large = used_wl_data[min_id[idx:idx+average_point]].mean(0)

    # sds2_ijv_small = used_wl_data2[max_id2[idx:idx+average_point]].mean(0)
    # sds2_ijv_large = used_wl_data2[min_id2[idx:idx+average_point]].mean(0)
    OD_spec = []

    # for sds2_large in sds2_ijv_large:
    #     for sds1_large in sds1_ijv_large:
    #         OD_spec += [sds1_large/sds2_large]

    # for sds2_small in sds2_ijv_small:
    #     for sds1_small in sds1_ijv_small:
    #         OD_spec += [sds1_small/sds2_small]
    for sds1_large in sds1_ijv_large:
        for sds2_large in sds2_ijv_large:
            OD_spec += [sds2_large/sds1_large]
            
    for sds1_small in sds1_ijv_small:
        for sds2_small in sds2_ijv_small:
            OD_spec += [sds2_small/sds1_small]

    
    return np.array(OD_spec)

# In[10]:
# window size = 10 sec
# e.g. 10 --> 5~15 sec mean of Rmax and mean of Rmin

# predict --> each 10 time point
# e.g. 10sec , 20sec, 30sec


total_predict = []
using_time = []
# baseline_time = [i for i in range(0,50,10)]
# HP_time = [i for i in range(70,230,10)]
# recovery_time = [i for i in range(250,420,10)]
# total_time = baseline_time + HP_time + recovery_time
for idx, time in enumerate(range(10,recovery_end-10,1)):
    using_time += [time]
    OD1_spec = get_OD(used_wl_data=cali_used_wl_data, 
                      used_wl_data2=cali_used_wl_data2, 
                      time=0,
                      average_time=20)

    OD2_spec = get_OD(used_wl_data=cali_used_wl_data, 
                      used_wl_data2=cali_used_wl_data2, 
                      time=time,
                      average_time=5) # window size = average_time*2
    if idx == 0:
        result_OD1_spec = OD1_spec.reshape(1,-1)
        result_OD2_spec = OD2_spec.reshape(1,-1)
    else:
        result_OD1_spec = np.concatenate((result_OD1_spec, OD1_spec.reshape(1,-1)))
        result_OD2_spec = np.concatenate((result_OD2_spec, OD2_spec.reshape(1,-1)))
        
    
    # # # normalize 
    # for i in range(40):
    #     OD1_spec[i*20:i*20+20] = (OD1_spec[i*20:i*20+20] - OD1_spec[i*20:i*20+20].mean()) / (OD1_spec[i*20:i*20+20].max() - OD1_spec[i*20:i*20+20].min())
    
    # for i in range(40):
    #     OD2_spec[i*20:i*20+20] = (OD2_spec[i*20:i*20+20] - OD2_spec[i*20:i*20+20].mean()) / (OD2_spec[i*20:i*20+20].max() - OD2_spec[i*20:i*20+20].min())
        
    # delta_OD = OD2_spec - OD1_spec
    delta_OD = OD1_spec/OD2_spec
    
    # # normalize 
    # for i in range(40):
    #     delta_OD[i*20:i*20+20] = (delta_OD[i*20:i*20+20] - delta_OD[i*20:i*20+20].mean()) / (delta_OD[i*20:i*20+20].max() - delta_OD[i*20:i*20+20].min())
    
    # normalize 
    for i in range(40):
        delta_OD[i*20:i*20+20] = (delta_OD[i*20:i*20+20] - delta_OD[i*20:i*20+20].min() + 1e-9) / (delta_OD[i*20:i*20+20].max() - delta_OD[i*20:i*20+20].min() + 1e-9)
    delta_OD = np.log(delta_OD)
    
    
    
    
    model_input = torch.tensor(delta_OD)
    model_input = model_input.to(torch.float32)
    predict = model(model_input)
    total_predict += [predict.item()]
total_predict = np.array(total_predict)*100

## fix nan value
for nan_idx in np.argwhere(np.isnan(total_predict)):
    
    prev_idx = nan_idx-1
    while np.isnan(total_predict[prev_idx]):
        prev_idx = prev_idx - 1

    next_idx = nan_idx+1
    while np.isnan(total_predict[next_idx]):
        next_idx = next_idx + 1
    
    total_predict[nan_idx] = (total_predict[prev_idx] +  total_predict[next_idx])/2

## save result
save_result = pd.DataFrame({'time [sec]' : using_time, 
              'predict_result' : total_predict.tolist()})
# save_result.to_csv(os.path.join('pic', subject, f'{date}_invivo_result', exp, 'each_predict_result.csv'), index=False)



total_predict


# # In[13]:
X_start = using_time.index(25)    
start_time = using_time.index(25)
find_skip_t1 = using_time.index(59)
find_skip_t2 = using_time.index(70)

# # scaled output
total_predict = total_predict-total_predict[start_time]
total_predict = total_predict/(np.abs(total_predict[find_skip_t2:]).max())

BF_total_predict, AF_total_predict = before_after_moving_average(data=total_predict, avg_points=3)
# plt.plot(using_time[start_time:], AF_total_predict[start_time:], "b.-")
# plt.plot(using_time[start_time:], BF_total_predict[start_time:], "b.-", alpha=0.2)
# cut signal 
#plt.plot(using_time[start_time:find_skip_t1], AF_total_predict[start_time:find_skip_t1], "b.-")
plt.plot(using_time[X_start:find_skip_t1], BF_total_predict[X_start:find_skip_t1], "b.-", alpha=1)
#plt.plot(using_time[find_skip_t2:80], BF_total_predict[find_skip_t2:80], "b.-")
plt.plot(using_time[find_skip_t2:], BF_total_predict[find_skip_t2:], "b.-", alpha=1)
plt.axvline(x=baseline_end, linestyle='--', color='b')#, label='baseline_end'
plt.axvline(x=HP_end, linestyle='--', color='r')#, label="Valsalva's maneuver_end"
#plt.axvline(x=using_time[-1], linestyle='--', color='g', label='recovery_end')
plt.xlabel("time(sec)")
plt.ylabel("predict SO2(%)")
#plt.ylim([-40,25])
plt.ylim([-1.05,1.05])
plt.legend(loc='center left', bbox_to_anchor=(1.05, 0.5),
          fancybox=True, shadow=True)
plt.title(f'$\Delta$SO2 change with time')
# plt.savefig(os.path.join('pic', subject, f'{date}_invivo_result', exp, 'predict_change_with_time.png'), dpi=300, format='png', bbox_inches='tight')
plt.show()



# In[14]:


# shift baseline

AF_total_predict = (AF_total_predict - AF_total_predict[start_time:start_time+5].mean())
# plt.plot(using_time[start_time:], AF_total_predict[start_time:], "b-")
# cut siganl
#plt.plot(using_time[X_start:find_skip_t1], AF_total_predict[X_start:find_skip_t1], "b.-", alpha=1)
#plt.plot(using_time[find_skip_t2:80], AF_total_predict[find_skip_t2:80], "b.-")
plt.plot(using_time[start_time:find_skip_t1], AF_total_predict[start_time:find_skip_t1], "b-")
plt.plot(using_time[find_skip_t2:], AF_total_predict[find_skip_t2:], "b-")
# plt.plot(using_time, BF_total_predict, "b.-", alpha=0.2)
plt.axvline(x=baseline_end, linestyle='--', color='b')#, label='baseline_end')
plt.axvline(x=HP_end, linestyle='--', color='r')#, label="Valsalva's maneuver_end")
#plt.axvline(x=using_time[-1], linestyle='--', color='g', label='recovery_end')
plt.xlabel("time(sec)")
plt.ylabel("predict SO2(%)")
#plt.ylim([-5,30])
plt.ylim([-1.05,1.05])
plt.legend(loc='center left', bbox_to_anchor=(1.05, 0.5),
          fancybox=True, shadow=True)
plt.title(f'shift baseline $\Delta$SO2 change with time')
# plt.savefig(os.path.join('pic', subject, f'{date}_invivo_result', exp, 'predict_change_with_time_shift.png'), dpi=300, format='png', bbox_inches='tight')
plt.show()

# In[ ]:


# BF_total_predict = BF_total_predict/(BF_total_predict[:6].mean()-1)
# accumulate_predict = []
# baseline = AF_total_predict[0]
# accumulate = AF_total_predict[0]
# accumulate_predict += [accumulate]
# for p in AF_total_predict[1:]:
#     accumulate += p 
#     accumulate_predict += [accumulate]


# In[ ]:


# plt.plot(using_time[:6], accumulate_predict[:6], "b.-")
# plt.plot(using_time[6:24], accumulate_predict[6:24], "r.-")
# plt.plot(using_time[24:], accumulate_predict[24:], "g.-")
# plt.axvline(x=60, linestyle='--', color='b', label='baseline_end')
# plt.axvline(x=240, linestyle='--', color='r', label='hyperventilation_end')
# plt.axvline(x=using_time[-1], linestyle='--', color='g', label='recovery_end')
# plt.xlabel("time(sec)")
# plt.ylabel("predict SO2(%)")
# plt.legend(loc='center left', bbox_to_anchor=(1.05, 0.5),
#           fancybox=True, shadow=True)
# plt.title(f'cumulative $\Delta$SO2')
# plt.savefig(os.path.join('pic', subject, f'{date}_invivo_result', 'predict_cumulative.png'), dpi=300, format='png', bbox_inches='tight')
# plt.show()


# In[15]:


total_predict = []
using_time = []
# baseline_time = [i for i in range(0,50,10)]
# HP_time = [i for i in range(70,230,10)]
# recovery_time = [i for i in range(250,420,10)]
# total_time = baseline_time + HP_time + recovery_time
for idx, time in enumerate(range(10,recovery_end-50,10)):
    using_time += [time]
    OD1_spec = get_OD(used_wl_data=cali_used_wl_data, 
                      used_wl_data2=cali_used_wl_data2, 
                      time=0, average_time=30)
    
    OD2_spec = get_OD(used_wl_data=cali_used_wl_data, 
                      used_wl_data2=cali_used_wl_data2, 
                      time=time, average_time=6)
    if idx == 0:
        result_OD1_spec = OD1_spec.reshape(1,-1)
        result_OD2_spec = OD2_spec.reshape(1,-1)
    else:
        result_OD1_spec = np.concatenate((result_OD1_spec, OD1_spec.reshape(1,-1)))
        result_OD2_spec = np.concatenate((result_OD2_spec, OD2_spec.reshape(1,-1)))
        
    
    # # # normalize 
    # for i in range(40):
    #     OD1_spec[i*20:i*20+20] = (OD1_spec[i*20:i*20+20] - OD1_spec[i*20:i*20+20].mean()) / (OD1_spec[i*20:i*20+20].max() - OD1_spec[i*20:i*20+20].min())
    
    # for i in range(40):
    #     OD2_spec[i*20:i*20+20] = (OD2_spec[i*20:i*20+20] - OD2_spec[i*20:i*20+20].mean()) / (OD2_spec[i*20:i*20+20].max() - OD2_spec[i*20:i*20+20].min())
    
    # delta_OD = OD2_spec - OD1_spec
 
    delta_OD = OD1_spec/OD2_spec
    
    # # normalize 
    # for i in range(40):
    #     delta_OD[i*20:i*20+20] = (delta_OD[i*20:i*20+20] - delta_OD[i*20:i*20+20].mean()) / (delta_OD[i*20:i*20+20].max() - delta_OD[i*20:i*20+20].min())
    
    # normalize 
    for i in range(40):
        delta_OD[i*20:i*20+20] = (delta_OD[i*20:i*20+20] - delta_OD[i*20:i*20+20].min() + 1e-9) / (delta_OD[i*20:i*20+20].max() - delta_OD[i*20:i*20+20].min() + 1e-9)
    delta_OD = np.log(delta_OD)
    
    
    model_input = torch.tensor(delta_OD)
    model_input = model_input.to(torch.float32)
    predict = model(model_input)
    total_predict += [predict.item()]
total_predict = np.array(total_predict)*100

## fix nan value
for nan_idx in np.argwhere(np.isnan(total_predict)):
    
    prev_idx = nan_idx-1
    while np.isnan(total_predict[prev_idx]):
        prev_idx = prev_idx - 1

    next_idx = nan_idx+1
    while np.isnan(total_predict[next_idx]):
        next_idx = next_idx + 1
    
    total_predict[nan_idx] = (total_predict[prev_idx] +  total_predict[next_idx])/2


save_result = pd.DataFrame({'time [sec]' : using_time, 
              'predict_result' : total_predict.tolist()})
# save_result.to_csv(os.path.join('pic', subject, f'{date}_invivo_result', exp, 'cum_predict_result.csv'), index=False)
total_predict


# In[16]:


fitting_SO2 = np.load(os.path.join('pic', subject, f'{date}_invivo_result', exp, "fitting", "SDS2", "fit_result_OPs", "chromophore", f"fitting_SO2.npy"))
fitting_SO2 = fitting_SO2[:62]*100
BF_fitting_SO2, AF_fitting_SO2 = before_after_moving_average(data=fitting_SO2, avg_points=3)

plt.plot(using_time, AF_fitting_SO2, "b.-")
plt.plot(using_time, BF_fitting_SO2, "b.-", alpha=0.2)
plt.axvline(x=baseline_end, linestyle='--', color='b', label='baseline_end')
plt.axvline(x=HP_end, linestyle='--', color='r', label='valsalva_end')
plt.axvline(x=using_time[-1], linestyle='--', color='g', label='recovery_end')
plt.xlabel("time(sec)")
plt.ylabel("SO2(%)")
plt.legend(loc='center left', bbox_to_anchor=(1.05, 0.5),
          fancybox=True, shadow=True)
plt.title(f'Iterative curve fitting SO2(%)')
# plt.savefig(os.path.join('pic', subject, f'{date}_invivo_result', exp, 'predict_cumulative.png'), dpi=300, format='png', bbox_inches='tight')
plt.show()


# In[17]:


change_fitting_SO2 = fitting_SO2 - fitting_SO2[:4].mean()
BF_change_fitting_SO2, AF_change_fitting_SO2 = before_after_moving_average(data=change_fitting_SO2, avg_points=3)

plt.plot(using_time, AF_change_fitting_SO2, "b.-")
plt.plot(using_time, BF_change_fitting_SO2, "b.-", alpha=0.2)
plt.axvline(x=baseline_end, linestyle='--', color='b', label='baseline_end')
plt.axvline(x=HP_end, linestyle='--', color='r', label='valsalva_end')
plt.axvline(x=using_time[-1], linestyle='--', color='g', label='recovery_end')
plt.xlabel("time(sec)")
plt.ylabel("SO2(%)")
plt.legend(loc='center left', bbox_to_anchor=(1.05, 0.5),
          fancybox=True, shadow=True)
plt.title(f'Iterative curve fitting $\Delta$SO2(%)')
# plt.savefig(os.path.join('pic', subject, f'{date}_invivo_result', exp, 'predict_cumulative.png'), dpi=300, format='png', bbox_inches='tight')
plt.show()


# In[18]:


BF_total_predict, AF_total_predict = before_after_moving_average(data=total_predict, avg_points=3)

plt.plot(using_time, AF_total_predict, "b.-")
plt.plot(using_time, BF_total_predict, "b.-", alpha=0.2)
plt.axvline(x=baseline_end, linestyle='--', color='b', label='baseline_end')
plt.axvline(x=HP_end, linestyle='--', color='r', label='valsalva_end')
plt.axvline(x=using_time[-1], linestyle='--', color='g', label='recovery_end')
plt.xlabel("time(sec)")
plt.ylabel("predict \u0394SO2(%)")
plt.legend(loc='center left', bbox_to_anchor=(1.05, 0.5),
          fancybox=True, shadow=True)
plt.title(f'prediction model $\Delta$SO2(%)')
# plt.savefig(os.path.join('pic', subject, f'{date}_invivo_result', exp, 'predict_cumulative.png'), dpi=300, format='png', bbox_inches='tight')
plt.show()


# In[19]:


# shift baseline
# AF_total_predict = (AF_total_predict - AF_total_predict[:4].mean())
# AF_change_fitting_SO2 = (AF_change_fitting_SO2 - AF_change_fitting_SO2[:4].mean())

# plt.plot(using_time, AF_change_fitting_SO2, "r.-", label='curve fitting')
plt.plot(using_time, AF_total_predict*0.1, "b.-", label='prediction model')
# plt.plot(using_time, BF_total_predict, "b.-", alpha=0.2)
plt.axvline(x=baseline_end, linestyle='--', color='b', label='baseline_end')
plt.axvline(x=HP_end, linestyle='--', color='r', label='valsalva_end')
plt.axvline(x=using_time[-1], linestyle='--', color='g', label='recovery_end')
plt.xlabel("time(sec)")
plt.ylabel("predict \u0394SO2(%)")
plt.legend(loc='center left', bbox_to_anchor=(1.05, 0.5),
          fancybox=True, shadow=True)
# plt.title(f'$\Delta$SO2')
plt.ylim([-15,15])
# plt.savefig(os.path.join('pic', subject, f'{date}_invivo_result', exp, 'predict_cumulative_shift.png'), dpi=300, format='png', bbox_inches='tight')
plt.show()


# In[21]:


AF_total_predict = AF_total_predict*0.1
np.save("S2_AF_total_predict.npy", AF_total_predict)


# In[31]:


plt.plot(MBLL_SO2['time'], MBLL_SO2['delta_SO2_20_10'], 'r.--', alpha=0.9, label='MBLL 20mm-10mm')
# plt.plot(MBLL_SO2['time'], MBLL_SO2['delta_SO2_20_7'], 'r.--', alpha=0.6, label='MBLL 20mm-7.5mm')
# plt.plot(MBLL_SO2['time'], MBLL_SO2['delta_SO2_20_4'], 'r.--', alpha=0.3, label='MBLL 20mm-4.5mm')
plt.plot(using_time, AF_total_predict*0.1, "b.-", label='ANN prediction model')
# plt.plot(using_time, BF_total_predict, "b.-", alpha=0.2)
plt.axvline(x=baseline_end, linestyle='--', color='b', label='baseline_end')
plt.axvline(x=HP_end, linestyle='--', color='r', label='hyperventilation_end')
plt.axvline(x=using_time[-1], linestyle='--', color='g', label='recovery_end')
plt.xlabel("time(sec)")
plt.ylabel("predict $\Delta$SO2(%)")
plt.legend(loc='center left', bbox_to_anchor=(1.05, 0.5),
          fancybox=True, shadow=True)
plt.title(f'$\Delta$SO2 (t1=0s~30s)')
# plt.savefig(os.path.join('pic', subject, f'{date}_invivo_result', 'MBLL', 'predict_cumulative_shift_with_MBLL.png'), dpi=300, format='png', bbox_inches='tight')
plt.show()


# In[ ]:




