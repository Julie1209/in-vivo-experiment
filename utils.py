import numpy as np
import pandas as pd
from PyEMD import EMD 
from scipy.signal import convolve
from scipy.signal import butter, lfilter, freqz
import skimage.io 
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
# Default settings
mpl.rcParams.update(mpl.rcParamsDefault)
plt.style.use("seaborn-darkgrid")

class process_raw_data():
    def __init__(self, baseline_start, baseline_end, exp_start, exp_end, recovery_start, recovery_end,
                 time_resolution, time_interval,mother_folder_name, using_SDS) -> None:
        self.baseline_start = baseline_start
        self.baseline_end = baseline_end
        self.exp_start = exp_start
        self.exp_end = exp_end
        self.recovery_start = recovery_start
        self.recovery_end = recovery_end
        self.time_resolution = time_resolution
        self.time_interval = time_interval
        self.mother_folder_name = mother_folder_name
        self.using_SDS = using_SDS
        
    @staticmethod
    def create_folder(mother_folder_name):
        os.makedirs(os.path.join("pic", mother_folder_name, "spectrum", 'background'), exist_ok=True)
        os.makedirs(os.path.join("pic", mother_folder_name, "spectrum", 'raw'), exist_ok=True)
        os.makedirs(os.path.join("pic", mother_folder_name, "spectrum", 'remove_spike_and_bg'), exist_ok=True)
        os.makedirs(os.path.join("pic", mother_folder_name, "spectrum", 'moving_average'), exist_ok=True)
        os.makedirs(os.path.join("pic", mother_folder_name, "spectrum", 'LPF'), exist_ok=True)
        os.makedirs(os.path.join("pic", mother_folder_name, "spectrum", 'EMD'), exist_ok=True)
        os.makedirs(os.path.join("pic", mother_folder_name, "spectrum", 'compare'), exist_ok=True)
        os.makedirs(os.path.join("pic", mother_folder_name, "spectrum", 'get_peak'), exist_ok=True)
        os.makedirs(os.path.join("pic", mother_folder_name, "time", 'background'), exist_ok=True)
        os.makedirs(os.path.join("pic", mother_folder_name, "time", 'raw'), exist_ok=True)
        os.makedirs(os.path.join("pic", mother_folder_name, "time", 'remove_spike_and_bg'), exist_ok=True)
        os.makedirs(os.path.join("pic", mother_folder_name, "time", 'moving_average'), exist_ok=True)
        os.makedirs(os.path.join("pic", mother_folder_name, "time", 'LPF'), exist_ok=True)
        os.makedirs(os.path.join("pic", mother_folder_name, "time", 'compare_LPF'), exist_ok=True)
        os.makedirs(os.path.join("pic", mother_folder_name, "time", 'EMD'), exist_ok=True)
        os.makedirs(os.path.join("pic", mother_folder_name, "time", 'compare'), exist_ok=True)
        os.makedirs(os.path.join("pic", mother_folder_name, "time", 'get_peak'), exist_ok=True)
        os.makedirs(os.path.join("pic", mother_folder_name, "cvp", 'raw'), exist_ok=True)
        os.makedirs(os.path.join("pic", mother_folder_name, "cvp", 'remove_spike_and_bg'), exist_ok=True)
        os.makedirs(os.path.join("pic", mother_folder_name, "cvp", 'moving_average'), exist_ok=True)
        os.makedirs(os.path.join("pic", mother_folder_name, "cvp", 'LPF'), exist_ok=True)
        os.makedirs(os.path.join("pic", mother_folder_name, "cvp", 'EMD'), exist_ok=True)
        os.makedirs(os.path.join("pic", mother_folder_name, "cvp", 'compare'), exist_ok=True)
        os.makedirs(os.path.join("pic", mother_folder_name, "cvp", 'get_peak'), exist_ok=True)
        os.makedirs(os.path.join("pic", mother_folder_name, "EMD", 'raw'), exist_ok=True)
        os.makedirs(os.path.join("pic", mother_folder_name, "EMD", 'remove_spike_and_bg'), exist_ok=True)
        os.makedirs(os.path.join("pic", mother_folder_name, "EMD", 'moving_average'), exist_ok=True)
        os.makedirs(os.path.join("pic", mother_folder_name, "EMD", 'LPF'), exist_ok=True)
        os.makedirs(os.path.join("pic", mother_folder_name, "EMD", 'EMD'), exist_ok=True)
        os.makedirs(os.path.join("dataset", mother_folder_name), exist_ok=True)
    
    def remove_spike(self, wl, data, normalStdTimes, ts):
        # data : Nx --> time, Ny --> wavelength 
        data_no_spike = data.copy()
        mean = data_no_spike.mean(axis=0)
        std = data_no_spike.std(ddof=1, axis=0)
        targetSet = []  # save spike idx
        for idx, s in enumerate(data_no_spike):  # iterate spectrum in every time frame
            isSpike = np.any(abs(s-mean) > normalStdTimes*std)
            if isSpike:
                targetSet.append(idx) 
        print(f"target = {targetSet}")
        if len(targetSet) != 0:
            for target in targetSet:
                # show target spec and replace that spec by using average of the two adjacents
                plt.plot(wl, data_no_spike[target])
                plt.xlabel("Wavelength [nm]")
                plt.ylabel("Intensity [counts]")    
                plt.title(f"spike idx: {target}")
                plt.savefig(os.path.join("pic", self.mother_folder_name, "spectrum", 'remove_spike_and_bg', f'{ts+target*self.time_resolution:.2f}s_spike.png'), dpi=300, format='png', bbox_inches='tight')
                plt.close()
                if ((target+1) < data_no_spike.shape[0]) & ((target-1) >= 0 ): 
                    data_no_spike[target] = (data_no_spike[target-1] + data_no_spike[target+1]) / 2
                elif (target+1) == data_no_spike.shape[0]:
                    data_no_spike[target] = data_no_spike[target-1]
                elif (target-1) <= 0:
                    data_no_spike[target] = data_no_spike[target+1]
                
        return data_no_spike
    
    @staticmethod
    def get_peak_final(data):
        '''
        Detect peak (rmax and rmin) position in invivo-signal

        '''
        max_idx_set = []
        min_idx_set = []
        data_mean = data.mean()
        
        idx = 0
        break_out_flag = False
        state = data[idx] < data_mean  
        while (data[idx] < data_mean) == state:
            idx += 1
        idx_s = idx
        while True:
            
            state = data[idx] < data_mean
            # while ((data[idx-1] < data_mean) != state) | ((data[idx] < data_mean) == state) | (idx-idx_s <= 3):
            while ((data[idx] < data_mean) == state) | (idx-idx_s <= 3):
            # while (data[idx] < data_mean) == state:
                idx += 1
                if idx >= len(data):
                    break_out_flag = True
                    break
            if break_out_flag:
                break
            idx_e = idx
            
            minimum_interval = 6
            data_local = data[idx_s:idx_e]
            if data_local.mean() > data_mean:
                max_idx = idx_s + np.argmax(data_local)
                if len(max_idx_set) == 0:
                    max_idx_set.append(max_idx)
                elif max_idx - max_idx_set[-1] >= minimum_interval:
                    max_idx_set.append(max_idx)
            if data_local.mean() <= data_mean:
                min_idx = idx_s + np.argmin(data_local)
                if len(min_idx_set) == 0:
                    min_idx_set.append(min_idx)
                elif min_idx - min_idx_set[-1] >= minimum_interval:
                    min_idx_set.append(min_idx)
            
            idx_s = idx_e
        
        max_idx_set, min_idx_set = np.array(max_idx_set), np.array(min_idx_set)    
        
        # # remove improper peak idx
        # data_std = data.std(ddof=1)    
        # max_idx_set = max_idx_set[abs(data[max_idx_set] - data_mean) > (0.5*data_std)]
        # min_idx_set = min_idx_set[abs(data[min_idx_set] - data_mean) > (0.5*data_std)]
        
        # fine-tuning
        for _ in range(5):
            for idx, max_idx in enumerate(max_idx_set):
                start = 0 if max_idx-(minimum_interval-1) < 0 else max_idx-(minimum_interval-1)
                end = max_idx+minimum_interval
                # print((start, end))
                tmp_idx = np.argmax(data[start:end])
                max_idx = start + tmp_idx
                max_idx_set[idx] = max_idx
            for idx, min_idx in enumerate(min_idx_set):
                start = 0 if min_idx-(minimum_interval-1) < 0 else min_idx-(minimum_interval-1)
                end = min_idx+minimum_interval
                # print((start, end))
                tmp_idx = np.argmin(data[start:end])
                min_idx = start + tmp_idx
                min_idx_set[idx] = min_idx
        
        # remove improper peak idx
        # cutbackforthpercent = 0.1
        # max_cutnum = round(len(max_idx_set)*cutbackforthpercent)
        # min_cutnum = round(len(min_idx_set)*cutbackforthpercent)
        # data_max, data_min = data[max_idx_set], data[min_idx_set]
        # sort_max, sort_min = np.argsort(data_max), np.argsort(data_min)
        # max_idx_set = max_idx_set[sort_max][max_cutnum:-max_cutnum]
        # min_idx_set = min_idx_set[sort_min][min_cutnum:-min_cutnum]
        
        max_idx_set.sort()
        min_idx_set.sort()
        
        return np.unique(max_idx_set), np.unique(min_idx_set)
    
    
    @staticmethod
    def moving_avg(used_wl_bandwidth, used_wl, time_mean_arr):
        # time_mean_arr : Nx --> wavelength
        # used_wl_bandwidth --> nm the range want to average
        resolution = used_wl[1] - used_wl[0]
        half_average_points = int(0.5*used_wl_bandwidth/resolution)
        moving_avg_I = []
        moving_avg_wl = []
        for i in range(half_average_points, time_mean_arr.shape[0]-half_average_points):
            moving_avg_I += [time_mean_arr[i-half_average_points:i+half_average_points].mean()]
            if (2*half_average_points) % 2 == 1: # odd 
                moving_avg_wl += [used_wl[i+(2*half_average_points)//2]]
            else: # even
                even = (used_wl[i+(2*half_average_points)//2] + used_wl[i+(2*half_average_points)//2-1]) * 0.5
                moving_avg_wl += [even]
        
        return np.array(moving_avg_I), np.array(moving_avg_wl)
    
    @staticmethod
    def read_file(filename):
        data = pd.read_csv(filename)
        wl = np.array(data.columns[1:].values, dtype=float) # zero-idx is wavelength(nm), so we skip it.
        data = data.iloc[:, 1:]
        data = np.array(data)
        
        return data, wl
    
    @staticmethod
    def butter_lowpass(cutoff, fs, order=5):
        return butter(order, cutoff, fs=fs, btype='low', analog=False)
    
    @classmethod
    def butter_lowpass_filter(cls, data, cutoff, fs, order=5):
        b, a = cls.butter_lowpass(cutoff, fs, order=order)
        y = lfilter(b, a, data)
        return y
    
    def plot_mean_time_spectrum(self, data, wavelength, name, start_time, end_time, is_show=False):
        spectrum = data[round(start_time/self.time_resolution):round(end_time/self.time_resolution)].mean(0)
        plt.figure(figsize=(8,6))
        plt.plot(wavelength, spectrum)
        plt.xlabel("wavelength (nm)")
        plt.ylabel("intensity")
        plt.title(f"SDS={self.using_SDS}mm {start_time}s~{end_time}s {name} spectrum")
        plt.tight_layout()
        plt.savefig(os.path.join("pic", self.mother_folder_name, "spectrum", name, f"mean_time_spec_{start_time}_{end_time}s.png"), dpi=300, format='png', bbox_inches='tight')
        if is_show:
            plt.show()
        else:
            plt.close()
    
    def plot_all_time_spectrum(self, data, wavelength, name, start_time, end_time, is_show=False):
        spectrum = data[round(start_time/self.time_resolution):round(end_time/self.time_resolution)]
        plt.figure(figsize=(8,6))
        for using_spectrum in spectrum:
            plt.plot(wavelength, using_spectrum)
            plt.xlabel("wavelength (nm)")
            plt.ylabel("intensity")
            plt.title(f"SDS={self.using_SDS}mm {start_time}s~{end_time}s {name} spectrum")
        plt.tight_layout()
        plt.savefig(os.path.join("pic", self.mother_folder_name, "spectrum", name, f"all_time_spec_{start_time}_{end_time}s.png"), dpi=300, format='png', bbox_inches='tight')
        if is_show:
            plt.show()
        else:
            plt.close()
    
    def plot_signal(self, data, name, start_time, end_time, is_show=False):
        # ts [sec]
        signal = data.mean(axis=1)
        time = np.linspace(start_time, end_time, signal[round(start_time/self.time_resolution):round(end_time/self.time_resolution)].shape[0])
        plt.figure(figsize=(10,6))    
        plt.plot(time, signal[round(start_time/self.time_resolution):round(end_time/self.time_resolution)])
        plt.xlabel(f"Time [s], integration time = {self.time_resolution} s")
        plt.ylabel("intensity [-]")
        plt.title(f"SDS={self.using_SDS}mm {name} {start_time}s~{end_time}s signal waveform")
        plt.savefig(os.path.join("pic", self.mother_folder_name, "time", name, f"signal_{start_time}s_{end_time}s.png"), dpi=300, format='png', bbox_inches='tight')
        if is_show:
            plt.show()
        else:
            plt.close()
    
    def plot_cvp(self, data, name, start_time, end_time, is_show=False):
        cvp = data.mean(axis=1)
        cvp = 1 / cvp
        time = np.linspace(start_time, end_time, cvp[round(start_time/self.time_resolution):round(end_time/self.time_resolution)].shape[0])
        plt.figure(figsize=(10,6))    
        plt.plot(time, cvp[round(start_time/self.time_resolution):round(end_time/self.time_resolution)])
        plt.xlabel(f"Time [s], integration time = {self.time_resolution} s")
        plt.ylabel("1 / Sum(intensity)  [-]")
        plt.title(f"SDS={self.using_SDS}mm {start_time}s~{end_time}s CVP waveform of {name} signal")
        plt.savefig(os.path.join("pic", self.mother_folder_name, "cvp", name, f"cvp_{start_time}s_{end_time}s.png"), dpi=300, format='png', bbox_inches='tight')
        if is_show:
            plt.show()
        else:
            plt.close()
    
    def plot_LPF_compare(self, BF_data, AF_data, name, start_time, end_time, is_show=False):
        BF_signal = BF_data[round(start_time/self.time_resolution):round(end_time/self.time_resolution)].mean(axis=1)
        AF_signal = AF_data[round(start_time/self.time_resolution):round(end_time/self.time_resolution)].mean(axis=1)
        time = np.linspace(start_time, end_time, BF_signal.shape[0])
        plt.figure()
        plt.plot(time, BF_signal, 'b-', label='data')
        plt.plot(time, AF_signal, 'g-', linewidth=2, label='filtered data')
        plt.title(f"SDS={self.using_SDS}mm {name} {start_time}s~{end_time}s LPF signal waveform")
        plt.ylabel("intensity [-]")
        plt.xlabel('Time [sec]')
        plt.grid()
        plt.legend()
        plt.savefig(os.path.join("pic", self.mother_folder_name, "time", name, f"LPF_{start_time}s_{end_time}s.png"), dpi=300, format='png', bbox_inches='tight')
        if is_show:
            plt.show()
        else:
            plt.show()
    
    def plot_EMD(self, data, name, start_time, end_time, is_show=False):
        imfs = EMD().emd(data.mean(axis=1)[round(start_time/self.time_resolution):round(end_time/self.time_resolution)])
        imfs[-1] -= imfs[-1].mean()
        fig, ax = plt.subplots(imfs.shape[0]+1, 1, figsize=(16, 12))
        time = np.linspace(start_time,end_time, data.mean(axis=1)[round(start_time/self.time_resolution):round(end_time/self.time_resolution)].shape[0])
        ax[0].plot(time, data.mean(axis=1)[round(start_time/self.time_resolution):round(end_time/self.time_resolution)], 'r')
        ax[0].set_title(f'SDS:{self.using_SDS}mm ' + f",{start_time}s~{end_time}s {name} signal")
        for n, imf in enumerate(imfs):
            time = np.linspace(start_time,end_time, imf.shape[0])
            ax[n+1].plot(time, imf, 'g')
            ax[n+1].set_title("imf " + str(n+1))        
        plt.xlabel("time [sec]")
        plt.tight_layout()
        plt.savefig(os.path.join("pic", self.mother_folder_name, "EMD", name, f"{name}_EMD_{start_time}s_{end_time}s.png"), dpi=300, format='png', bbox_inches='tight')
        if is_show:
            plt.show()
        else:
            plt.close()
    
    def plot_time_EMD(self, data, name, start_time, end_time, using_num_IMF, is_show=False):
        # get artifact
        imfs = EMD().emd(data[round(start_time/self.time_resolution):round(end_time/self.time_resolution)].mean(axis=1))
        imfs[-1] -= imfs[-1].mean()
        artifact = imfs[using_num_IMF+1:] 

        # remove artifact
        data_EMD = data[round(start_time/self.time_resolution):round(end_time/self.time_resolution)].copy()
        for art in artifact:
            data_EMD -= art.reshape(-1, 1)

        # plot after EMD
        time = np.linspace(start_time,end_time, data_EMD.mean(1).shape[0])
        plt.figure(figsize=(9,5))
        plt.plot(time, data_EMD.mean(1))
        plt.title(f'SDS:{self.using_SDS}mm ' + f",{start_time}s~{end_time}s after EMD 0~{using_num_IMF}")
        plt.xlabel("time [sec]")
        plt.tight_layout()
        plt.savefig(os.path.join("pic", self.mother_folder_name, "time", name, f"EMD_0_{using_num_IMF}_IMF_{start_time}s_{end_time}s.png"), dpi=300, format='png', bbox_inches='tight')
        if is_show:
            plt.show()
        else:
            plt.close()
    
    
    def plot_compare_time_EMD(self, data, name, start_time, end_time, using_num_IMF, is_show=False):
        # get artifact
        imfs = EMD().emd(data[round(start_time/self.time_resolution):round(end_time/self.time_resolution)].mean(axis=1))
        imfs[-1] -= imfs[-1].mean()
        artifact = imfs[using_num_IMF+1:] 

        # remove artifact
        data_EMD = data[round(start_time/self.time_resolution):round(end_time/self.time_resolution)].copy()
        for art in artifact:
            data_EMD -= art.reshape(-1, 1)
            
        # detect peak 
        data_signal = data_EMD.mean(axis=1) # do average w.r.t wavelength
        max_idx, min_idx = self.get_peak_final(data_signal)
        
        fig, ax = plt.subplots(2,1, figsize=(12,8))
        # plot before EMD
        time = np.linspace(start_time,end_time, data[round(start_time/self.time_resolution):round(end_time/self.time_resolution)].mean(1).shape[0])
        ax[0].plot(time, data[round(start_time/self.time_resolution):round(end_time/self.time_resolution)].mean(1))
        ax[0].set_title(f'SDS:{self.using_SDS}mm ' + f",{start_time}s~{end_time}s before EMD 0~{using_num_IMF}")
        # plot after EMD
        time = np.linspace(start_time,end_time, data_EMD.mean(1).shape[0])
        ax[1].plot(time, data_EMD.mean(1))
        ax[1].set_title(f'SDS:{self.using_SDS}mm ' + f",{start_time}s~{end_time}s after EMD 0~{using_num_IMF}")
        if max_idx.size > 0:
            ax[1].scatter(time[max_idx], 
                        data_signal[max_idx], s=11, 
                        color="red", label="Max")
        if min_idx.size > 0:
            ax[1].scatter(time[min_idx], 
                        data_signal[min_idx], s=11, 
                        color="tab:orange", label="Min")
        ax[1].legend()
        
        plt.xlabel("time [sec]")
        plt.tight_layout()
        plt.savefig(os.path.join("pic", self.mother_folder_name, "time", name, f"compare_0_{using_num_IMF}_IMF_{start_time}s_{end_time}s.png"), dpi=300, format='png', bbox_inches='tight')
        if is_show:
            plt.show()
        else:
            plt.close()
        
    # plot all script
    def long_plot_all_fig(self, data, wavelength, name, is_show=False):
        self.plot_all_time_spectrum(data=data,
                            wavelength=wavelength,
                            name=name,
                            start_time=self.baseline_start,
                            end_time=self.recovery_end,
                            is_show=is_show)

        self.plot_mean_time_spectrum(data=data,
                            wavelength=wavelength,
                            name=name,
                            start_time=self.baseline_start,
                            end_time=self.recovery_end,
                            is_show=is_show)

        self.plot_mean_time_spectrum(data=data,
                            wavelength=wavelength,
                            name=name,
                            start_time=self.baseline_start,
                            end_time=self.baseline_end,
                            is_show=is_show)

        self.plot_mean_time_spectrum(data=data,
                            wavelength=wavelength,
                            name=name,
                            start_time=self.exp_start,
                            end_time=self.exp_end,
                            is_show=is_show)

        self.plot_mean_time_spectrum(data=data,
                            wavelength=wavelength,
                            name=name,
                            start_time=self.recovery_start,
                            end_time=self.recovery_end,
                            is_show=is_show)
        
        self.plot_signal(data=data,
                    name=name,
                    start_time=0,
                    end_time=self.recovery_end,
                    is_show=is_show)
        
        self.plot_cvp(data=data,
                name=name,
                start_time=0,
                end_time=self.recovery_end,
                is_show=is_show)
        
        self.plot_EMD(data=data,
                name=name,
                start_time=0,
                end_time=self.recovery_end,
                is_show=is_show)

        for ts in range(0,self.recovery_end,self.time_interval):
            td = ts + self.time_interval
            # if ts == 0:
            #         is_show = True
            # else:
            #         is_show = False
            self.plot_signal(data=data,
                    name=name,
                    start_time=ts,
                    end_time=td,
                    is_show = is_show)
            self.plot_cvp(data=data,
                    name=name,
                    start_time=ts,
                    end_time=td,
                    is_show = is_show)
            self.plot_EMD(data=data,
                    name=name,
                    start_time=ts,
                    end_time=td,
                    is_show = is_show)
            
    # def plot_Rmax_Rmin(self, data, wavelength, max_idx_Set, min_idx_Set, name, start_time, end_time, is_show=False):
    #   window_size = 10  
    #   R_max_Rmin = {'time': [], 'values': []}
  
    #   data_window = []  
  
    #   for ts in range(start_time, end_time):
    #       if len(data_window) == window_size:
    #           data_window.pop(0)
  
    #       data_window.append(data[:, ts])  
  
    #       max_idx_subset = max_idx_Set[
    #           np.where(
    #               (max_idx_Set > round(ts / self.time_resolution))
    #               & (max_idx_Set < round((ts + 1) / self.time_resolution))
    #           )[0]
    #       ]
    #       min_idx_subset = min_idx_Set[
    #           np.where(
    #               (min_idx_Set > round(ts / self.time_resolution))
    #               & (min_idx_Set < round((ts + 1) / self.time_resolution))
    #           )[0]
    #       ]
  
    #       R_max_spec = data[max_idx_subset, :].mean(0)
    #       R_min_spec = data[min_idx_subset, :].mean(0)
  
    #       plt.figure()
    #       plt.plot(wavelength, R_max_spec, label="IJV small")
    #       plt.plot(wavelength, R_min_spec, label="IJV large")
    #       plt.xlabel("wavelength (nm)")
    #       plt.ylabel("intensity")
    #       plt.title(
    #           f"SDS={self.using_SDS}mm {name} spectrum, {ts}~{ts + 1}s, R_max/R_min:{(R_max_spec / R_min_spec).mean():.4f}"
    #       )
    #       plt.tight_layout()
    #       plt.legend()
    #       plt.savefig(
    #           os.path.join(
    #               "pic",
    #               self.mother_folder_name,
    #               "spectrum",
    #               name,
    #               f"mean_time_spec_{ts}_{ts + 1}s.png",
    #           ),
    #           dpi=300,
    #           format="png",
    #           bbox_inches="tight",
    #       )
    #       if is_show:
    #           plt.show()
    #       else:
    #           plt.close()
  
    #       R_max_Rmin["time"] += [ts]
    #       R_max_Rmin["values"] += [(R_max_spec / R_min_spec).mean()]
  
    #   R_max_Rmin = pd.DataFrame(R_max_Rmin)
    #   R_max_Rmin.to_csv(
    #       os.path.join(
    #           "pic",
    #           self.mother_folder_name,
    #           "spectrum",
    #           name,
    #           f"mean_time_spec.csv",
    #       ),
    #       index=False,
    #   )          
                 
    
    def plot_Rmax_Rmin(self, data, wavelength, max_idx_Set, min_idx_Set, name, start_time, end_time, is_show=False):
        R_max_Rmin = {'time' : [],
                      'values' : []}
        for ts in range(start_time, end_time, 10):
            td = ts + 10
            max_idx_subset = max_idx_Set[np.where((max_idx_Set>round(ts/self.time_resolution))&(max_idx_Set<round(td/self.time_resolution)))[0]]
            min_idx_subset = min_idx_Set[np.where((min_idx_Set>round(ts/self.time_resolution))&(min_idx_Set<round(td/self.time_resolution)))[0]]
            R_max_spec = data[max_idx_subset, :].mean(0)
            R_min_spec = data[min_idx_subset, :].mean(0)
            plt.figure()
            plt.plot(wavelength, R_max_spec, label="IJV small")
            plt.plot(wavelength, R_min_spec, label="IJV large")
            plt.xlabel("wavelength (nm)")
            plt.ylabel("intensity")
            plt.title(f"SDS={self.using_SDS}mm {name} spectrum, {ts}~{td}s, R_max/R_min:{(R_max_spec/R_min_spec).mean():.4f}")
            plt.tight_layout()
            plt.legend()
            plt.savefig(os.path.join("pic", self.mother_folder_name, "spectrum", name, f"mean_time_spec_{ts}_{td}s.png"), dpi=300, format='png', bbox_inches='tight')
            if is_show:
                plt.show()
            else:
                plt.close()
            
            R_max_Rmin['time'] += [ts]
            R_max_Rmin['values'] += [(R_max_spec/R_min_spec).mean()]
        R_max_Rmin = pd.DataFrame(R_max_Rmin)
        R_max_Rmin.to_csv(os.path.join("pic", self.mother_folder_name, "spectrum", name, f"mean_time_spec.csv"), index=False)

    @staticmethod
    def get_spec(path_background, path_tar, row_choose, a, b, *, img_size=(200, 1600)): 
        """_summary_

        Args:
            path_background (_type_): _description_
            path_tar (_type_): _description_
            row_choose (_type_): _description_
            a (_type_): _description_
            b (_type_): _description_
            img_size (tuple, optional): _description_. Defaults to (200, 1600).

        Returns:
            list: [img_background, dfmean, df_chN...]
        """
        img_background = np.zeros(img_size, dtype = 'float64')
        tar_list = len(path_tar) * [np.zeros(img_size, dtype = 'float64')]
        
        # Calculate BG means
        for bgid in range(len(path_background)):
            img_background += skimage.io.imread(path_background[bgid], plugin = 'tifffile').astype(np.float64) 
        img_background /= len(path_background)
        
        # If forget BG
        # row1 = list(range(57, 71))
        # row2 = list(range(100, 120))
        # row3 = list(range(142, 175))
        # img_background[[row1+row2+row3], :] = img_background[83, :]
        
        # Each shot substract mean BG
        for tarid in range(len(path_tar)):
            tar_list[tarid] = skimage.io.imread(path_tar[tarid], plugin = 'tifffile').astype(np.float64) 
            tar_list[tarid] -= img_background
            tar_list[tarid][tar_list[tarid] < 0] = 0
            
        # Split in 3 channel           
        tar_ch_list = []        # ch1, 2, 3
        for chid in range(len(row_choose)):
            # axis 0: each shot; axis 1: wl
            temp_arr = np.zeros((len(path_tar), img_size[1]))
            for tarid in range(len(path_tar)):
                row_tar = np.sum(tar_list[tarid][row_choose[chid][0] : row_choose[chid][1], :], 0)
                temp_arr[tarid, :] = row_tar
            tar_ch_list.append(temp_arr)
        
        # Process output
        df_temp = pd.DataFrame(np.array([np.arange(img_size[1])]).T, columns=['wl'])
        # df_temp = a * (2*df_temp) + b
        df_temp = a * df_temp + b
        
        # each channel mean
        dmean = {f'ch{i+1}_mean': np.mean(tar_ch_list[i], 0) for i in range(len(row_choose))}
        dfmean = pd.DataFrame(data=dmean, dtype='float64')
        dfmean = pd.concat([df_temp, dfmean], axis = 1)
        out = [img_background, dfmean]
        
        # each channel
        for chid in range(len(row_choose)):
            d = {f'shot_{i}': tar_ch_list[chid][i, :] for i in range(len(tar_ch_list[chid]))}
            df = pd.DataFrame(data=d, dtype='float64')
            df = pd.concat([df_temp, df], axis = 1)
            out.append(df)
        return out

    @staticmethod
    def get_stat(ref_list, start_wl, stop_wl):
        stat_list = []
        for i in range(len(ref_list)):
            # get average, standard deviation and CV
            avg = ref_list[i].loc[start_wl:stop_wl, 'shot_0':].mean(axis=1)
            std = ref_list[i].loc[start_wl:stop_wl, 'shot_0':].std(axis=1)
            cv = std / avg    
            d = {'wl': ref_list[i].loc[start_wl:stop_wl, 'wl'], 'avg': avg, 'std': std, 'cv': cv}
            df_stat = pd.DataFrame(d)
            stat_list.append(df_stat)
            
        return stat_list

class process_phantom():
    def __init__(self) -> None:
        pass
    
    @staticmethod
    def moving_avg(used_wl_bandwidth, used_wl, time_mean_arr):
        ''' 
        used_wl_bandwidth --> nm the range want to average
        used_wl --> 1D array
        time_mean_arr --> 1D array (spectrum) correspond with used_wl
        '''
        resolution = used_wl[1] - used_wl[0]
        average_points = int(used_wl_bandwidth/resolution)
        moving_avg_I = []
        moving_avg_wl = []
        for i in range(time_mean_arr.shape[0]-average_points):
            moving_avg_I += [time_mean_arr[i:i+average_points].mean()]
            if average_points % 2 == 1: # odd 
                moving_avg_wl += [used_wl[i+average_points//2]]
            else: # even
                even = (used_wl[i+average_points//2] + used_wl[i+average_points//2-1]) * 0.5
                moving_avg_wl += [even]
        
        return np.array(moving_avg_I), np.array(moving_avg_wl)
    
    @staticmethod
    def remove_spike(used_wl, data, normalStdTimes, savepath):
        data_no_spikes = data.copy()
        mean = data_no_spikes.mean(axis=0)
        std = data_no_spikes.std(ddof=1, axis=0)
        targetSet = []  # save spike idx
        for idx, s in enumerate(data_no_spikes):  # iterate spectrum in every time frame
            isSpike = np.any(abs(s-mean) > normalStdTimes*std)
            if isSpike:
                targetSet.append(idx) 
        print(f"target = {targetSet}")
        if len(targetSet) != 0:
            for target in targetSet:
                # show target spec and replace that spec by using average of the two adjacents
                plt.plot(used_wl, data_no_spikes[target])
                plt.xlabel("Wavelength [nm]")
                plt.ylabel("Intensity [counts]")    
                plt.title(f"spike idx: {target}")
                plt.savefig(os.path.join(savepath, f"spike_at_time_stamp_{target}.png"), dpi=300, format='png', bbox_inches='tight')
                plt.close()
                
                if ((target+1) < data_no_spikes.shape[0]) & ((target-1) >= 0 ): 
                    data_no_spikes[target] = (data_no_spikes[target-1] + data_no_spikes[target+1]) / 2
                elif (target+1) == data_no_spikes.shape[0]:
                    data_no_spikes[target] = data_no_spikes[target-1]
                elif (target-1) <= 0:
                    data_no_spikes[target] = data_no_spikes[target+1]
                
        return data_no_spikes

    @staticmethod
    def plot_individual_phantom(used_wl, time_mean_np_arr, time_mean_np_arr_remove_bg, moving_avg_I, moving_avg_wl, name, savepath):
        plt.plot(used_wl, time_mean_np_arr, label='raw data')
        plt.title(f'phantom_{name} spectrum')
        plt.xlabel('wavelength (nm)')
        plt.ylabel('intensity')
        plt.legend(loc='center left', bbox_to_anchor=(1.05, 0.5),
                        fancybox=True, shadow=True)
        plt.savefig(os.path.join("pic", savepath, "raw_data.png"), dpi=300, format='png', bbox_inches='tight')
        plt.show()
        
        plt.plot(used_wl, time_mean_np_arr_remove_bg, label='raw data - bg')
        plt.title(f'phantom_{name} spectrum')
        plt.xlabel('wavelength (nm)')
        plt.ylabel('intensity')
        plt.legend(loc='center left', bbox_to_anchor=(1.05, 0.5),
                        fancybox=True, shadow=True)
        plt.savefig(os.path.join("pic", savepath, "raw_data_remove_bg.png"), dpi=300, format='png', bbox_inches='tight')
        plt.show()
        
        plt.plot(moving_avg_wl, moving_avg_I, label='moving average')
        plt.title(f'phantom_{name} spectrum')
        plt.xlabel('wavelength (nm)')
        plt.ylabel('intensity')
        plt.legend(loc='center left', bbox_to_anchor=(1.05, 0.5),
                        fancybox=True, shadow=True)
        plt.savefig(os.path.join("pic", savepath, "moving_avg.png"), dpi=300, format='png', bbox_inches='tight')
        plt.show()
    
    @classmethod
    def plot_each_time_and_remove_spike(cls, used_wl, np_arr, name, savepath):
        for data in np_arr:
            plt.plot(used_wl, data)
        plt.title(f'phantom_{name} spectrum')
        plt.xlabel('wavelength (nm)')
        plt.ylabel('intensity')
        plt.legend(loc='center left', bbox_to_anchor=(1.05, 0.5),
                        fancybox=True, shadow=True)
        plt.savefig(os.path.join("pic", savepath, "each_time_raw_data.png"), dpi=300, format='png', bbox_inches='tight')
        plt.show()
        
        remove_spike_data = cls.remove_spike(used_wl, np_arr, name, normalStdTimes=10, showTargetSpec=True, savepath=savepath)
        
        for data in remove_spike_data:
            plt.plot(used_wl, data)
        plt.title(f'phantom_{name} spectrum')
        plt.xlabel('wavelength (nm)')
        plt.ylabel('intensity')
        plt.legend(loc='center left', bbox_to_anchor=(1.05, 0.5),
                        fancybox=True, shadow=True)
        plt.savefig(os.path.join("pic", savepath, "each_time_raw_data_remove_spike.png"), dpi=300, format='png', bbox_inches='tight')
        plt.show()
        
        return remove_spike_data

    @staticmethod
    def cal_R_square(y_true, y_pred):
        y_bar = np.mean(y_true)
        numerator = np.sum(np.square(y_true-y_pred))
        denominator = np.sum(np.square(y_true-y_bar))
        R_square = 1 - numerator/denominator
        
        return R_square
