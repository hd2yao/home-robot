import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def load_data(file_path):
    """加载原始数据"""
    return pd.read_csv(file_path)

def add_timestamps(df):
    """
    添加时间戳到数据中
    - 睡眠相关数据(heart_rate,resp_rate,spo2,body_movement,breath_wave,sleep_stage)：每天晚上9:30到次日5:30
    - 环境数据(temperature,humidity)：全天
    """
    # 设置基准日期
    base_date = datetime(2023, 1, 1)
    
    # 创建结果DataFrame
    result_df = pd.DataFrame()
    
    # 获取唯一的健康风险水平记录数
    unique_health_records = df['health_risk_level'].value_counts().to_dict()
    
    # 计算每个健康风险水平每天的记录数
    days_needed = 30  # 生成30天的数据
    daily_records = {level: count // days_needed for level, count in unique_health_records.items()}
    
    # 当前处理的行索引
    current_idx = 0
    
    # 为每一天生成数据
    for day in range(days_needed):
        current_date = base_date + timedelta(days=day)
        
        # 1. 生成睡眠时间段的数据（晚上9:30到次日5:30）
        sleep_start = current_date.replace(hour=21, minute=30) + timedelta(minutes=random.randint(-30, 30))
        sleep_end = (current_date + timedelta(days=1)).replace(hour=5, minute=30) + timedelta(minutes=random.randint(-30, 30))
        
        # 计算睡眠时间内的采样点数（假设每15分钟采样一次）
        sleep_duration = sleep_end - sleep_start
        sleep_samples = int(sleep_duration.total_seconds() / (15 * 60))  # 每15分钟一个样本
        
        # 为每个健康风险水平分配睡眠数据
        for health_level, count in daily_records.items():
            # 确保我们有足够的数据
            if current_idx + count > len(df):
                break
                
            # 获取当前健康级别的数据
            level_data = df[df['health_risk_level'] == health_level].iloc[current_idx:current_idx+count].copy()
            current_idx += count
            
            # 只保留睡眠相关的数据
            sleep_data = level_data.copy()
            
            # 为睡眠数据分配时间戳
            sleep_timestamps = [sleep_start + timedelta(minutes=15*i) for i in range(sleep_samples)]
            
            # 如果睡眠样本数量不足，则随机选择时间点
            if len(sleep_timestamps) > len(sleep_data):
                sleep_timestamps = random.sample(sleep_timestamps, len(sleep_data))
            else:
                # 如果数据多于时间点，随机重复一些时间点
                while len(sleep_timestamps) < len(sleep_data):
                    sleep_timestamps.append(random.choice(sleep_timestamps))
            
            # 随机打乱时间戳以模拟真实情况
            random.shuffle(sleep_timestamps)
            sleep_data['timestamp'] = sleep_timestamps[:len(sleep_data)]
            
            # 添加到结果DataFrame
            result_df = pd.concat([result_df, sleep_data])
        
        # 2. 生成全天环境数据（每小时采样）
        for hour in range(24):
            # 每小时随机选择1-3个时间点
            for _ in range(random.randint(1, 3)):
                env_time = current_date.replace(hour=hour, minute=random.randint(0, 59))
                
                # 从原始数据中随机选择一行作为环境数据基础
                env_row = df.iloc[random.randint(0, len(df)-1)].copy()
                
                # 只保留环境相关的列和健康风险水平
                env_data = pd.DataFrame({
                    'heart_rate': [np.nan],
                    'resp_rate': [np.nan],
                    'spo2': [np.nan],
                    'body_movement': [np.nan],
                    'breath_wave': [np.nan],
                    'sleep_stage': [np.nan],
                    'temperature': [env_row['temperature']],
                    'humidity': [env_row['humidity']],
                    'health_risk_level': [env_row['health_risk_level']],
                    'timestamp': [env_time]
                })
                
                # 添加到结果DataFrame
                result_df = pd.concat([result_df, env_data])
    
    # 按时间戳排序
    result_df = result_df.sort_values('timestamp').reset_index(drop=True)
    
    return result_df

def process_data(input_file, output_file):
    """处理数据并保存到新文件"""
    df = load_data(input_file)
    df_with_timestamps = add_timestamps(df)
    df_with_timestamps.to_csv(output_file, index=False)
    print(f"数据处理完成，已保存到 {output_file}")
    return df_with_timestamps

if __name__ == "__main__":
    input_file = "/Users/dysania/program/AI/home-robot/v2/docs/simulated_health_data_high_quality.csv"
    output_file = "/Users/dysania/program/AI/home-robot/v2/processed_health_data.csv"
    process_data(input_file, output_file) 