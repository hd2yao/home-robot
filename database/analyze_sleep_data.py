#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MIT-BIH 多导睡眠数据分析脚本
使用 wfdb 库读取和分析 MIT-BIH Polysomnographic Database 数据
"""

import os
import numpy as np
import wfdb
import matplotlib.pyplot as plt
from scipy import signal
from matplotlib.gridspec import GridSpec

class SleepDataAnalyzer:
    """MIT-BIH 睡眠数据分析类"""
    
    def __init__(self, data_dir, record_name):
        """
        初始化分析器
        
        参数:
            data_dir: 数据目录
            record_name: 记录名称
        """
        self.data_dir = data_dir
        self.record_name = record_name
        self.full_path = os.path.join(data_dir, record_name)
        self.record = None
        self.load_record()
    
    def load_record(self):
        """加载记录数据"""
        try:
            self.record = wfdb.rdrecord(self.full_path)
            print(f"成功加载记录: {self.full_path}")
            self.print_record_info()
        except Exception as e:
            print(f"加载记录失败: {e}")
    
    def print_record_info(self):
        """打印记录信息"""
        if self.record is None:
            print("记录未加载")
            return
        
        print("\n记录信息:")
        print(f"采样频率: {self.record.fs} Hz")
        print(f"信号数量: {self.record.n_sig}")
        print(f"信号名称: {self.record.sig_name}")
        print(f"信号单位: {self.record.units}")
        print(f"信号长度: {self.record.sig_len} 样本")
        print(f"记录时长: {self.record.sig_len / self.record.fs:.2f} 秒 "
              f"({self.record.sig_len / self.record.fs / 60:.2f} 分钟, "
              f"{self.record.sig_len / self.record.fs / 3600:.2f} 小时)")
    
    def plot_signals(self, time_range=(0, 30), save_fig=True):
        """
        绘制信号数据
        
        参数:
            time_range: 时间范围（秒），默认为前30秒
            save_fig: 是否保存图像
        """
        if self.record is None:
            print("记录未加载")
            return
        
        # 计算样本范围
        start_sample = int(time_range[0] * self.record.fs)
        end_sample = int(time_range[1] * self.record.fs)
        
        # 确保范围有效
        if end_sample > self.record.sig_len:
            end_sample = self.record.sig_len
            print(f"警告: 请求的结束时间超出记录范围，已调整为记录结束时间 "
                  f"({end_sample / self.record.fs:.2f} 秒)")
        
        # 时间向量
        time = np.arange(start_sample, end_sample) / self.record.fs
        
        # 创建图形
        plt.figure(figsize=(16, 12))
        gs = GridSpec(self.record.n_sig, 1)
        
        for i in range(self.record.n_sig):
            ax = plt.subplot(gs[i, 0])
            ax.plot(time, self.record.p_signal[start_sample:end_sample, i])
            ax.set_ylabel(f"{self.record.sig_name[i]}\n({self.record.units[i]})")
            ax.grid(True)
            
            # 只在最后一个子图上显示x轴标签
            if i == self.record.n_sig - 1:
                ax.set_xlabel("时间 (秒)")
        
        plt.suptitle(f"{self.record_name} 信号波形 ({time_range[0]}-{time_range[1]}秒)")
        plt.tight_layout()
        
        if save_fig:
            fig_path = os.path.join(self.data_dir, f"{self.record_name}_signals_{time_range[0]}-{time_range[1]}s.png")
            plt.savefig(fig_path)
            print(f"图像已保存至: {fig_path}")
        
        plt.show()
    
    def analyze_ecg(self, time_range=(0, 60), save_fig=True):
        """
        分析ECG信号
        
        参数:
            time_range: 时间范围（秒）
            save_fig: 是否保存图像
        """
        if self.record is None:
            print("记录未加载")
            return
        
        # 寻找ECG信号索引
        ecg_idx = None
        for i, name in enumerate(self.record.sig_name):
            if "ECG" in name:
                ecg_idx = i
                break
        
        if ecg_idx is None:
            print("未找到ECG信号")
            return
        
        # 计算样本范围
        start_sample = int(time_range[0] * self.record.fs)
        end_sample = int(time_range[1] * self.record.fs)
        
        # 确保范围有效
        if end_sample > self.record.sig_len:
            end_sample = self.record.sig_len
        
        # 提取ECG信号
        ecg = self.record.p_signal[start_sample:end_sample, ecg_idx]
        time = np.arange(start_sample, end_sample) / self.record.fs
        
        # 使用Pan-Tompkins算法检测R波
        # 带通滤波
        lowcut = 5.0  # Hz
        highcut = 15.0  # Hz
        nyquist = 0.5 * self.record.fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = signal.butter(2, [low, high], btype='band')
        filtered_ecg = signal.filtfilt(b, a, ecg)
        
        # 导数
        diff_ecg = np.diff(filtered_ecg)
        
        # 平方
        squared_ecg = diff_ecg ** 2
        
        # 移动平均
        window_size = int(0.15 * self.record.fs)  # 150ms窗口
        if window_size % 2 == 0:
            window_size += 1  # 确保窗口大小为奇数
        moving_avg = signal.convolve(squared_ecg, np.ones(window_size)/window_size, mode='same')
        
        # 寻找峰值
        peaks, _ = signal.find_peaks(moving_avg, height=0.35*np.max(moving_avg), distance=int(0.2*self.record.fs))
        
        # 计算心率
        if len(peaks) > 1:
            # 计算RR间隔（秒）
            rr_intervals = np.diff(peaks) / self.record.fs
            # 计算每分钟心跳次数
            heart_rates = 60 / rr_intervals
            avg_hr = np.mean(heart_rates)
            std_hr = np.std(heart_rates)
            print(f"\nECG分析结果:")
            print(f"检测到的R波数量: {len(peaks)}")
            print(f"平均心率: {avg_hr:.2f} ± {std_hr:.2f} BPM")
            print(f"最小心率: {np.min(heart_rates):.2f} BPM")
            print(f"最大心率: {np.max(heart_rates):.2f} BPM")
        else:
            print("检测到的R波数量不足，无法计算心率")
        
        # 绘图
        plt.figure(figsize=(16, 12))
        
        # 原始ECG
        plt.subplot(4, 1, 1)
        plt.plot(time, ecg)
        plt.title("原始ECG信号")
        plt.ylabel(f"振幅 ({self.record.units[ecg_idx]})")
        plt.grid(True)
        
        # 滤波后的ECG
        plt.subplot(4, 1, 2)
        plt.plot(time[:-1], filtered_ecg[:-1])
        plt.title("带通滤波后的ECG信号 (5-15 Hz)")
        plt.ylabel("振幅")
        plt.grid(True)
        
        # 处理后的信号
        plt.subplot(4, 1, 3)
        plt.plot(time[1:], moving_avg)
        plt.title("处理后的信号")
        plt.ylabel("振幅")
        plt.grid(True)
        
        # 带有R波标记的原始信号
        plt.subplot(4, 1, 4)
        plt.plot(time, ecg)
        plt.plot(time[peaks], ecg[peaks], "ro")
        if len(peaks) > 1:
            for i, hr in enumerate(heart_rates):
                plt.text(time[peaks[i+1]], ecg[peaks[i+1]], f"{hr:.1f}", fontsize=8)
        plt.title(f"R波检测结果 (平均心率: {avg_hr:.2f} BPM)" if len(peaks) > 1 else "R波检测结果")
        plt.xlabel("时间 (秒)")
        plt.ylabel(f"振幅 ({self.record.units[ecg_idx]})")
        plt.grid(True)
        
        plt.tight_layout()
        
        if save_fig:
            fig_path = os.path.join(self.data_dir, f"{self.record_name}_ecg_analysis_{time_range[0]}-{time_range[1]}s.png")
            plt.savefig(fig_path)
            print(f"ECG分析图像已保存至: {fig_path}")
        
        plt.show()
        
        return {
            "peaks": peaks,
            "heart_rates": heart_rates if len(peaks) > 1 else None,
            "avg_hr": avg_hr if len(peaks) > 1 else None,
            "std_hr": std_hr if len(peaks) > 1 else None
        }
    
    def analyze_eeg(self, time_range=(0, 30), save_fig=True):
        """
        分析EEG信号频谱
        
        参数:
            time_range: 时间范围（秒）
            save_fig: 是否保存图像
        """
        if self.record is None:
            print("记录未加载")
            return
        
        # 寻找EEG信号索引
        eeg_idx = None
        for i, name in enumerate(self.record.sig_name):
            if "EEG" in name:
                eeg_idx = i
                break
        
        if eeg_idx is None:
            print("未找到EEG信号")
            return
        
        # 计算样本范围
        start_sample = int(time_range[0] * self.record.fs)
        end_sample = int(time_range[1] * self.record.fs)
        
        # 确保范围有效
        if end_sample > self.record.sig_len:
            end_sample = self.record.sig_len
        
        # 提取EEG信号
        eeg = self.record.p_signal[start_sample:end_sample, eeg_idx]
        time = np.arange(start_sample, end_sample) / self.record.fs
        
        # 频谱分析
        n = len(eeg)
        fft_result = np.fft.rfft(eeg)
        magnitude = np.abs(fft_result)
        freq = np.fft.rfftfreq(n, 1/self.record.fs)
        
        # 计算各频段能量
        bands = {
            'Delta': (0.5, 4),    # 0.5-4 Hz
            'Theta': (4, 8),      # 4-8 Hz
            'Alpha': (8, 13),     # 8-13 Hz
            'Beta': (13, 30),     # 13-30 Hz
            'Gamma': (30, 100)    # 30-100 Hz
        }
        
        band_powers = {}
        total_power = np.sum(magnitude**2)
        
        for band, (low, high) in bands.items():
            # 找到该频段范围内的频率索引
            idx = np.where((freq >= low) & (freq < high))[0]
            # 计算该频段能量
            if len(idx) > 0:
                band_power = np.sum(magnitude[idx]**2)
                band_powers[band] = band_power / total_power * 100  # 百分比
            else:
                band_powers[band] = 0
        
        # 打印频段能量分布
        print("\nEEG频段能量分布:")
        for band, power in band_powers.items():
            print(f"{band}: {power:.2f}%")
        
        # 绘图
        plt.figure(figsize=(16, 12))
        
        # 原始EEG信号
        plt.subplot(3, 1, 1)
        plt.plot(time, eeg)
        plt.title(f"原始EEG信号 ({self.record.sig_name[eeg_idx]})")
        plt.ylabel(f"振幅 ({self.record.units[eeg_idx]})")
        plt.grid(True)
        
        # 频谱
        plt.subplot(3, 1, 2)
        plt.plot(freq, magnitude)
        plt.title("EEG频谱")
        plt.xlabel("频率 (Hz)")
        plt.ylabel("振幅")
        plt.xlim(0, 50)  # 只显示0-50Hz的频率
        plt.grid(True)
        
        # 频段能量分布
        plt.subplot(3, 1, 3)
        bands_list = list(bands.keys())
        powers_list = [band_powers[band] for band in bands_list]
        plt.bar(bands_list, powers_list)
        plt.title("EEG频段能量分布")
        plt.ylabel("能量百分比 (%)")
        plt.ylim(0, max(powers_list) * 1.2)
        
        for i, v in enumerate(powers_list):
            plt.text(i, v + 1, f"{v:.1f}%", ha='center')
        
        plt.tight_layout()
        
        if save_fig:
            fig_path = os.path.join(self.data_dir, f"{self.record_name}_eeg_analysis_{time_range[0]}-{time_range[1]}s.png")
            plt.savefig(fig_path)
            print(f"EEG分析图像已保存至: {fig_path}")
        
        plt.show()
        
        return band_powers

if __name__ == "__main__":
    # 数据目录和记录名称
    data_dir = "MIT-BIH Polysomnographic Database"
    record_name = "slp01a"
    
    # 创建分析器
    analyzer = SleepDataAnalyzer(data_dir, record_name)
    
    # 显示一段信号数据
    analyzer.plot_signals(time_range=(0, 30))
    
    # 分析ECG数据
    analyzer.analyze_ecg(time_range=(0, 60))
    
    # 分析EEG数据
    analyzer.analyze_eeg(time_range=(0, 30)) 