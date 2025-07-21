#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MIT-BIH 多导睡眠数据导出脚本
将 MIT-BIH Polysomnographic Database 的生理信号数据导出为CSV格式
"""

import os
import numpy as np
import pandas as pd
import wfdb
import argparse
from tqdm import tqdm

def export_record_to_csv(record_path, record_name, output_dir=None, time_range=None, sample_rate=None):
    """
    将WFDB记录导出为CSV文件
    
    参数:
        record_path: 记录文件所在目录
        record_name: 记录名称（不包含扩展名）
        output_dir: 输出目录，默认为record_path
        time_range: 时间范围（秒），格式为(start_time, end_time)，默认导出全部数据
        sample_rate: 输出的采样率，如果提供，将对数据进行重采样，默认不重采样
    """
    try:
        # 构建完整路径
        full_path = os.path.join(record_path, record_name)
        
        # 读取记录数据
        print(f"读取记录: {full_path}")
        record = wfdb.rdrecord(full_path)
        
        # 输出目录设置
        if output_dir is None:
            output_dir = record_path
        os.makedirs(output_dir, exist_ok=True)
        
        # 打印记录信息
        print("记录信息:")
        print(f"采样频率: {record.fs} Hz")
        print(f"信号数量: {record.n_sig}")
        print(f"信号名称: {record.sig_name}")
        print(f"信号单位: {record.units}")
        print(f"信号长度: {record.sig_len} 样本")
        print(f"记录时长: {record.sig_len / record.fs:.2f} 秒")
        
        # 处理时间范围
        if time_range is not None:
            start_time, end_time = time_range
            start_sample = int(start_time * record.fs)
            end_sample = int(end_time * record.fs)
            
            # 确保范围有效
            if end_sample > record.sig_len:
                end_sample = record.sig_len
                print(f"警告: 请求的结束时间超出记录范围，已调整为记录结束时间 ({end_sample / record.fs:.2f} 秒)")
            
            # 提取指定范围的数据
            data = record.p_signal[start_sample:end_sample, :]
            # 创建时间列
            time = np.arange(start_sample, end_sample) / record.fs
        else:
            # 使用全部数据
            data = record.p_signal
            time = np.arange(record.sig_len) / record.fs
        
        # 如果需要重采样
        if sample_rate is not None and sample_rate != record.fs:
            print(f"重采样数据从 {record.fs} Hz 到 {sample_rate} Hz")
            # 计算新的时间点
            new_time = np.arange(0, time[-1], 1/sample_rate)
            # 对每个信号进行插值重采样
            new_data = np.zeros((len(new_time), record.n_sig))
            for i in range(record.n_sig):
                # 使用线性插值进行重采样
                new_data[:, i] = np.interp(new_time, time, data[:, i])
            
            # 更新数据和时间
            data = new_data
            time = new_time
        
        # 创建DataFrame
        df = pd.DataFrame()
        df['Time(s)'] = time
        
        # 添加每个信号的数据
        for i, name in enumerate(record.sig_name):
            column_name = f"{name}({record.units[i]})"
            df[column_name] = data[:, i]
        
        # 导出为CSV
        if time_range is not None:
            csv_filename = f"{record_name}_{time_range[0]}-{time_range[1]}s.csv"
        else:
            csv_filename = f"{record_name}_full.csv"
        
        csv_path = os.path.join(output_dir, csv_filename)
        df.to_csv(csv_path, index=False)
        print(f"数据已导出至: {csv_path}")
        print(f"导出 {len(df)} 行 × {len(df.columns)} 列")
        
        return csv_path
        
    except Exception as e:
        print(f"导出数据时出错: {e}")
        return None

def export_individual_signals(record_path, record_name, output_dir=None, time_range=None):
    """
    将WFDB记录中的每个信号分别导出为单独的CSV文件
    
    参数:
        record_path: 记录文件所在目录
        record_name: 记录名称（不包含扩展名）
        output_dir: 输出目录，默认为record_path
        time_range: 时间范围（秒），格式为(start_time, end_time)，默认导出全部数据
    """
    try:
        # 构建完整路径
        full_path = os.path.join(record_path, record_name)
        
        # 读取记录数据
        print(f"读取记录: {full_path}")
        record = wfdb.rdrecord(full_path)
        
        # 输出目录设置
        if output_dir is None:
            output_dir = os.path.join(record_path, f"{record_name}_csv")
        os.makedirs(output_dir, exist_ok=True)
        
        # 处理时间范围
        if time_range is not None:
            start_time, end_time = time_range
            start_sample = int(start_time * record.fs)
            end_sample = int(end_time * record.fs)
            
            # 确保范围有效
            if end_sample > record.sig_len:
                end_sample = record.sig_len
                print(f"警告: 请求的结束时间超出记录范围，已调整为记录结束时间 ({end_sample / record.fs:.2f} 秒)")
            
            # 提取指定范围的数据
            data = record.p_signal[start_sample:end_sample, :]
            # 创建时间列
            time = np.arange(start_sample, end_sample) / record.fs
            time_suffix = f"_{time_range[0]}-{time_range[1]}s"
        else:
            # 使用全部数据
            data = record.p_signal
            time = np.arange(record.sig_len) / record.fs
            time_suffix = "_full"
        
        # 为每个信号创建单独的CSV
        csv_files = []
        for i, name in enumerate(record.sig_name):
            df = pd.DataFrame()
            df['Time(s)'] = time
            column_name = f"{name}({record.units[i]})"
            df[column_name] = data[:, i]
            
            # 导出为CSV
            csv_filename = f"{record_name}_{name}{time_suffix}.csv"
            csv_path = os.path.join(output_dir, csv_filename)
            df.to_csv(csv_path, index=False)
            print(f"{name} 数据已导出至: {csv_path}")
            csv_files.append(csv_path)
        
        return csv_files
        
    except Exception as e:
        print(f"导出数据时出错: {e}")
        return None

def export_sections(record_path, record_name, output_dir=None, section_length=30, overlap=0):
    """
    将WFDB记录按固定时间段分段导出为多个CSV文件
    
    参数:
        record_path: 记录文件所在目录
        record_name: 记录名称（不包含扩展名）
        output_dir: 输出目录，默认为record_path/record_name_sections
        section_length: 每段长度（秒），默认为30秒
        overlap: 段之间的重叠时间（秒），默认为0秒
    """
    try:
        # 构建完整路径
        full_path = os.path.join(record_path, record_name)
        
        # 读取记录数据
        print(f"读取记录: {full_path}")
        record = wfdb.rdrecord(full_path)
        
        # 输出目录设置
        if output_dir is None:
            output_dir = os.path.join(record_path, f"{record_name}_sections")
        os.makedirs(output_dir, exist_ok=True)
        
        # 计算总段数
        total_duration = record.sig_len / record.fs
        step_size = section_length - overlap
        num_sections = int((total_duration - overlap) / step_size)
        
        print(f"总记录时长: {total_duration:.2f} 秒")
        print(f"分段长度: {section_length} 秒，重叠: {overlap} 秒")
        print(f"将分割为 {num_sections} 个段落")
        
        # 分段导出
        csv_files = []
        for i in tqdm(range(num_sections)):
            start_time = i * step_size
            end_time = start_time + section_length
            
            # 确保不超出记录末尾
            if end_time > total_duration:
                end_time = total_duration
            
            # 提取当前段的样本索引
            start_sample = int(start_time * record.fs)
            end_sample = int(end_time * record.fs)
            
            # 创建DataFrame
            df = pd.DataFrame()
            # 时间从0开始，相对于段落开始
            df['Time(s)'] = np.arange(end_sample - start_sample) / record.fs
            
            # 添加每个信号的数据
            for j, name in enumerate(record.sig_name):
                column_name = f"{name}({record.units[j]})"
                df[column_name] = record.p_signal[start_sample:end_sample, j]
            
            # 导出为CSV
            csv_filename = f"{record_name}_section_{i+1}_{start_time:.1f}-{end_time:.1f}s.csv"
            csv_path = os.path.join(output_dir, csv_filename)
            df.to_csv(csv_path, index=False)
            csv_files.append(csv_path)
        
        print(f"已将数据分段导出至: {output_dir}")
        return csv_files
        
    except Exception as e:
        print(f"导出数据时出错: {e}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将MIT-BIH生理信号数据导出为CSV格式")
    parser.add_argument("--mode", choices=["all", "individual", "sections"], default="all",
                        help="导出模式: all=所有信号一个文件, individual=每个信号单独文件, sections=按时间段分割")
    parser.add_argument("--record_dir", default="MIT-BIH Polysomnographic Database",
                        help="记录文件所在目录")
    parser.add_argument("--record_name", default="slp01a",
                        help="记录名称（不包含扩展名）")
    parser.add_argument("--output_dir", default=None,
                        help="输出目录，默认为记录文件所在目录")
    parser.add_argument("--start_time", type=float, default=None,
                        help="开始时间（秒）")
    parser.add_argument("--end_time", type=float, default=None,
                        help="结束时间（秒）")
    parser.add_argument("--section_length", type=float, default=30.0,
                        help="分段长度（秒），仅在sections模式下有效")
    parser.add_argument("--overlap", type=float, default=0.0,
                        help="段落重叠（秒），仅在sections模式下有效")
    parser.add_argument("--sample_rate", type=float, default=None,
                        help="输出采样率（Hz），不指定则保持原始采样率")
    
    args = parser.parse_args()
    
    # 处理时间范围参数
    time_range = None
    if args.start_time is not None or args.end_time is not None:
        start_time = args.start_time if args.start_time is not None else 0
        # 如果没有指定end_time，则读取整个文件获取记录长度
        if args.end_time is None:
            record = wfdb.rdrecord(os.path.join(args.record_dir, args.record_name))
            end_time = record.sig_len / record.fs
        else:
            end_time = args.end_time
        time_range = (start_time, end_time)
    
    # 根据模式执行相应的导出操作
    if args.mode == "all":
        export_record_to_csv(args.record_dir, args.record_name, args.output_dir, time_range, args.sample_rate)
    elif args.mode == "individual":
        export_individual_signals(args.record_dir, args.record_name, args.output_dir, time_range)
    elif args.mode == "sections":
        export_sections(args.record_dir, args.record_name, args.output_dir, args.section_length, args.overlap) 