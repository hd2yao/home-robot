#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MIT-BIH 多导睡眠数据读取与可视化脚本
使用 wfdb 库读取和处理 MIT-BIH Polysomnographic Database 数据
"""

import os
import wfdb
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def read_and_visualize_record(record_path, record_name):
    """
    读取并可视化 WFDB 记录
    
    参数:
        record_path: 记录文件所在目录
        record_name: 记录名称（不包含扩展名）
    """
    # 构建完整路径
    full_path = os.path.join(record_path, record_name)
    
    # 读取记录数据
    print(f"读取记录: {full_path}")
    record = wfdb.rdrecord(full_path)
    
    # 打印记录信息
    print("记录信息:")
    print(f"采样频率: {record.fs} Hz")
    print(f"信号数量: {record.n_sig}")
    print(f"信号名称: {record.sig_name}")
    print(f"信号单位: {record.units}")
    print(f"信号长度: {record.sig_len} 样本")
    print(f"记录时长: {record.sig_len / record.fs:.2f} 秒")
    
    # 绘制各个信号
    plt.figure(figsize=(16, 10))
    gs = GridSpec(record.n_sig, 1)
    
    # 计算显示的数据点数量（显示前30秒的数据）
    display_samples = min(30 * record.fs, record.sig_len)
    
    for i in range(record.n_sig):
        ax = plt.subplot(gs[i, 0])
        ax.plot(np.arange(display_samples) / record.fs, record.p_signal[:display_samples, i])
        ax.set_title(f"{record.sig_name[i]} ({record.units[i]})")
        ax.set_ylabel(record.units[i])
        
        # 只在最后一个子图上显示 x 轴标签
        if i == record.n_sig - 1:
            ax.set_xlabel("时间 (秒)")
    
    plt.tight_layout()
    plt.savefig(os.path.join(record_path, f"{record_name}_visualization.png"))
    plt.show()
    
    # 打印记录的注释信息（如果有）
    try:
        ann = wfdb.rdann(full_path, 'atr')
        print("\n注释信息:")
        print(f"注释数量: {len(ann.sample)}")
        print(f"注释类型: {set(ann.symbol)}")
    except Exception as e:
        print(f"\n没有找到注释文件或读取注释时出错: {e}")

if __name__ == "__main__":
    # 数据库目录
    database_dir = "MIT-BIH Polysomnographic Database"
    
    # 读取并可视化 slp01a 记录
    read_and_visualize_record(database_dir, "slp01a") 