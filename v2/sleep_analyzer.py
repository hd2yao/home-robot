import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta, date
import json
from matplotlib_config import configure_chinese_font

class SleepAnalyzer:
    def __init__(self):
        self.sleep_data = None
        self.user_profile = {}
        
    def load_data(self, file_path):
        """加载带有时间戳的健康数据"""
        df = pd.read_csv(file_path)
        
        # 确保timestamp列是datetime类型
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        else:
            raise ValueError("数据中缺少timestamp列")
            
        self.sleep_data = df
        return df
    
    def identify_sleep_sessions(self):
        """识别睡眠会话"""
        # 筛选出睡眠相关数据（非NaN的心率数据通常表示睡眠监测）
        sleep_df = self.sleep_data.dropna(subset=['heart_rate']).copy()
        
        # 按时间排序
        sleep_df = sleep_df.sort_values('timestamp')
        
        # 添加日期列，用于按天分组
        sleep_df['date'] = sleep_df['timestamp'].dt.date
        
        # 计算相邻记录的时间差
        sleep_df['time_diff'] = sleep_df['timestamp'].diff().dt.total_seconds() / 60  # 转换为分钟
        
        # 如果时间差大于30分钟，认为是新的睡眠会话
        sleep_df['new_session'] = (sleep_df['time_diff'] > 30) | (sleep_df['time_diff'].isna())
        sleep_df['session_id'] = sleep_df['new_session'].cumsum()
        
        return sleep_df
    
    def analyze_sleep_patterns(self):
        """分析睡眠模式"""
        sleep_sessions = self.identify_sleep_sessions()
        
        # 按会话分组，计算每个睡眠会话的统计信息
        session_stats = sleep_sessions.groupby('session_id').agg({
            'timestamp': ['min', 'max', 'count'],
            'heart_rate': ['mean', 'std', 'min', 'max'],
            'resp_rate': ['mean', 'std'],
            'spo2': ['mean', 'min'],
            'body_movement': ['mean', 'sum'],
            'breath_wave': ['mean'],
            'sleep_stage': ['mean']
        })
        
        # 重命名列
        session_stats.columns = ['_'.join(col).strip() for col in session_stats.columns.values]
        
        # 计算每个会话的持续时间（分钟）
        session_stats['duration_minutes'] = (
            (session_stats['timestamp_max'] - session_stats['timestamp_min']).dt.total_seconds() / 60
        )
        
        # 识别睡眠会话的开始和结束时间
        session_stats['start_time'] = session_stats['timestamp_min'].dt.time
        session_stats['end_time'] = session_stats['timestamp_max'].dt.time
        session_stats['date'] = session_stats['timestamp_min'].dt.date
        
        # 计算睡眠质量得分（简单示例）
        # 基于心率稳定性、呼吸率、血氧水平和身体运动
        session_stats['sleep_quality_score'] = (
            (100 - session_stats['heart_rate_std']) +  # 心率越稳定越好
            (session_stats['spo2_mean'] - 90) * 5 +    # 血氧水平高于90分
            (20 - session_stats['resp_rate_std']) * 2 - # 呼吸率稳定性
            (session_stats['body_movement_mean'] / 10)  # 身体运动越少越好
        ).clip(0, 100)  # 限制在0-100范围内
        
        return session_stats
    
    def detect_sleep_anomalies(self, session_stats):
        """检测睡眠异常"""
        anomalies = []
        
        # 1. 检测睡眠时间异常（过长或过短）
        long_sleep = session_stats[session_stats['duration_minutes'] > 600]  # 超过10小时
        if not long_sleep.empty:
            for _, row in long_sleep.iterrows():
                anomalies.append({
                    'type': 'long_sleep',
                    'date': row['date'],
                    'duration_minutes': row['duration_minutes'],
                    'details': f"睡眠时间过长: {row['duration_minutes']:.1f}分钟 ({row['start_time']} - {row['end_time']})"
                })
        
        short_sleep = session_stats[session_stats['duration_minutes'] < 300]  # 少于5小时
        if not short_sleep.empty:
            for _, row in short_sleep.iterrows():
                anomalies.append({
                    'type': 'short_sleep',
                    'date': row['date'],
                    'duration_minutes': row['duration_minutes'],
                    'details': f"睡眠时间过短: {row['duration_minutes']:.1f}分钟 ({row['start_time']} - {row['end_time']})"
                })
        
        # 2. 检测睡眠中断
        sleep_sessions = self.identify_sleep_sessions()
        
        # 按日期分组
        for date, day_data in sleep_sessions.groupby(sleep_sessions['timestamp'].dt.date):
            # 检查同一天内是否有多个睡眠会话
            sessions = day_data['session_id'].unique()
            if len(sessions) > 1:
                # 计算会话之间的间隔
                for i in range(len(sessions) - 1):
                    session1 = day_data[day_data['session_id'] == sessions[i]]
                    session2 = day_data[day_data['session_id'] == sessions[i + 1]]
                    
                    # 计算会话间隔（分钟）
                    gap_minutes = (session2['timestamp'].min() - session1['timestamp'].max()).total_seconds() / 60
                    
                    # 如果间隔超过30分钟但小于3小时，视为睡眠中断
                    if 30 < gap_minutes < 180:
                        anomalies.append({
                            'type': 'sleep_interruption',
                            'date': date,
                            'gap_minutes': gap_minutes,
                            'details': f"睡眠中断: {gap_minutes:.1f}分钟 ({session1['timestamp'].max().time()} - {session2['timestamp'].min().time()})"
                        })
        
        # 3. 检测心率异常
        high_hr = session_stats[session_stats['heart_rate_mean'] > 90]
        if not high_hr.empty:
            for _, row in high_hr.iterrows():
                anomalies.append({
                    'type': 'high_heart_rate',
                    'date': row['date'],
                    'heart_rate': row['heart_rate_mean'],
                    'details': f"睡眠期间心率偏高: {row['heart_rate_mean']:.1f} bpm"
                })
        
        # 4. 检测血氧异常
        low_spo2 = session_stats[session_stats['spo2_min'] < 90]
        if not low_spo2.empty:
            for _, row in low_spo2.iterrows():
                anomalies.append({
                    'type': 'low_oxygen',
                    'date': row['date'],
                    'spo2': row['spo2_min'],
                    'details': f"睡眠期间血氧偏低: 最低 {row['spo2_min']:.1f}%"
                })
                
        return anomalies
    
    def build_user_profile(self):
        """构建用户睡眠行为画像"""
        session_stats = self.analyze_sleep_patterns()
        anomalies = self.detect_sleep_anomalies(session_stats)
        
        # 计算平均睡眠时长
        avg_sleep_duration = session_stats['duration_minutes'].mean() / 60  # 转换为小时
        
        # 计算平均入睡和起床时间
        avg_sleep_time = self._average_time([t.hour * 60 + t.minute for t in session_stats['start_time']])
        avg_wake_time = self._average_time([t.hour * 60 + t.minute for t in session_stats['end_time']])
        
        # 计算平均睡眠质量
        avg_sleep_quality = session_stats['sleep_quality_score'].mean()
        
        # 计算异常睡眠的比例
        anomaly_dates = set([a['date'] for a in anomalies])
        anomaly_ratio = len(anomaly_dates) / len(session_stats['date'].unique())
        
        # 构建用户画像
        self.user_profile = {
            "average_sleep_duration_hours": round(avg_sleep_duration, 2),
            "average_sleep_time": f"{avg_sleep_time[0]:02d}:{avg_sleep_time[1]:02d}",
            "average_wake_time": f"{avg_wake_time[0]:02d}:{avg_wake_time[1]:02d}",
            "average_sleep_quality": round(avg_sleep_quality, 2),
            "sleep_anomaly_ratio": round(anomaly_ratio, 2),
            "common_anomalies": self._summarize_anomalies(anomalies),
            "sleep_pattern": self._determine_sleep_pattern(session_stats),
            "sleep_consistency": self._calculate_sleep_consistency(session_stats),
            "detailed_anomalies": anomalies
        }
        
        return self.user_profile
    
    def _average_time(self, minutes_list):
        """计算平均时间（处理跨日的情况）"""
        # 处理跨日的情况（如有些入睡时间是前一天的23点，有些是当天的0点）
        for i in range(len(minutes_list)):
            if minutes_list[i] < 6 * 60:  # 如果时间是凌晨（小于6点）
                minutes_list[i] += 24 * 60  # 加上24小时以便正确计算平均值
        
        avg_minutes = sum(minutes_list) / len(minutes_list)
        
        # 转换回24小时制
        avg_minutes = avg_minutes % (24 * 60)
        
        # 返回小时和分钟
        return int(avg_minutes // 60), int(avg_minutes % 60)
    
    def _summarize_anomalies(self, anomalies):
        """汇总常见的异常类型"""
        if not anomalies:
            return []
            
        anomaly_types = {}
        for anomaly in anomalies:
            anomaly_type = anomaly['type']
            if anomaly_type in anomaly_types:
                anomaly_types[anomaly_type] += 1
            else:
                anomaly_types[anomaly_type] = 1
        
        # 按频率排序
        sorted_anomalies = sorted(anomaly_types.items(), key=lambda x: x[1], reverse=True)
        
        return [{"type": t, "count": c} for t, c in sorted_anomalies]
    
    def _determine_sleep_pattern(self, session_stats):
        """确定睡眠模式（规律/不规律）"""
        # 计算入睡时间和起床时间的标准差
        sleep_time_minutes = [t.hour * 60 + t.minute for t in session_stats['start_time']]
        wake_time_minutes = [t.hour * 60 + t.minute for t in session_stats['end_time']]
        
        # 处理跨日的情况
        for i in range(len(sleep_time_minutes)):
            if sleep_time_minutes[i] < 6 * 60:  # 凌晨
                sleep_time_minutes[i] += 24 * 60
                
        for i in range(len(wake_time_minutes)):
            if wake_time_minutes[i] < 6 * 60:  # 凌晨
                wake_time_minutes[i] += 24 * 60
        
        sleep_time_std = np.std(sleep_time_minutes) / 60  # 转换为小时
        wake_time_std = np.std(wake_time_minutes) / 60
        
        # 如果标准差小于1小时，认为是规律的
        if sleep_time_std < 1 and wake_time_std < 1:
            return "规律"
        elif sleep_time_std < 1.5 and wake_time_std < 1.5:
            return "较规律"
        else:
            return "不规律"
    
    def _calculate_sleep_consistency(self, session_stats):
        """计算睡眠一致性得分（0-100）"""
        # 计算入睡时间和起床时间的标准差
        sleep_time_std = np.std([t.hour * 60 + t.minute for t in session_stats['start_time']]) / 60
        wake_time_std = np.std([t.hour * 60 + t.minute for t in session_stats['end_time']]) / 60
        duration_std = np.std(session_stats['duration_minutes']) / 60
        
        # 计算一致性得分
        # 入睡时间和起床时间越稳定，得分越高
        time_consistency = 100 - (sleep_time_std + wake_time_std) * 20
        duration_consistency = 100 - duration_std * 15
        
        # 综合得分
        consistency_score = (time_consistency + duration_consistency) / 2
        
        # 限制在0-100范围内
        return max(0, min(100, consistency_score))
    
    def visualize_sleep_patterns(self, output_path=None):
        """可视化睡眠模式"""
        session_stats = self.analyze_sleep_patterns()
        
        # 配置中文字体
        configure_chinese_font()
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 睡眠时长趋势
        axes[0, 0].plot(session_stats['date'], session_stats['duration_minutes'] / 60, marker='o')
        axes[0, 0].set_title('睡眠时长趋势')
        axes[0, 0].set_xlabel('日期')
        axes[0, 0].set_ylabel('睡眠时长（小时）')
        axes[0, 0].grid(True)
        
        # 2. 睡眠质量趋势
        axes[0, 1].plot(session_stats['date'], session_stats['sleep_quality_score'], marker='o', color='green')
        axes[0, 1].set_title('睡眠质量趋势')
        axes[0, 1].set_xlabel('日期')
        axes[0, 1].set_ylabel('睡眠质量得分（0-100）')
        axes[0, 1].grid(True)
        
        # 3. 睡眠时间分布
        sleep_minutes = [(t.hour * 60 + t.minute) % (24 * 60) for t in session_stats['start_time']]
        wake_minutes = [(t.hour * 60 + t.minute) % (24 * 60) for t in session_stats['end_time']]
        
        # 转换为小时表示
        sleep_hours = [m / 60 for m in sleep_minutes]
        wake_hours = [m / 60 for m in wake_minutes]
        
        axes[1, 0].hist(sleep_hours, bins=24, alpha=0.7, label='入睡时间')
        axes[1, 0].hist(wake_hours, bins=24, alpha=0.7, label='起床时间')
        axes[1, 0].set_title('入睡和起床时间分布')
        axes[1, 0].set_xlabel('时间（小时）')
        axes[1, 0].set_ylabel('频率')
        axes[1, 0].set_xticks(range(0, 25, 2))
        axes[1, 0].set_xticklabels([f'{h:02d}:00' for h in range(0, 25, 2)])
        axes[1, 0].legend()
        
        # 4. 睡眠阶段分布
        sleep_sessions = self.identify_sleep_sessions()
        axes[1, 1].hist(sleep_sessions['sleep_stage'].dropna(), bins=3, rwidth=0.8)
        axes[1, 1].set_title('睡眠阶段分布')
        axes[1, 1].set_xlabel('睡眠阶段')
        axes[1, 1].set_ylabel('频率')
        axes[1, 1].set_xticks([0, 1, 2])
        axes[1, 1].set_xticklabels(['浅睡眠', '中度睡眠', '深睡眠'])
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path)
            print(f"睡眠模式可视化已保存到 {output_path}")
        else:
            plt.show()
    
    def save_user_profile(self, output_path):
        """保存用户画像到JSON文件"""
        if not self.user_profile:
            self.build_user_profile()
            
        # 转换日期对象为字符串
        profile_copy = self.user_profile.copy()
        
        if 'detailed_anomalies' in profile_copy:
            for anomaly in profile_copy['detailed_anomalies']:
                if isinstance(anomaly.get('date'), date):
                    anomaly['date'] = anomaly['date'].isoformat()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(profile_copy, f, ensure_ascii=False, indent=2)
            
        print(f"用户画像已保存到 {output_path}")

if __name__ == "__main__":
    # 加载处理后的数据
    data_path = "/Users/dysania/program/AI/home-robot/v2/processed_health_data.csv"
    
    # 创建睡眠分析器
    analyzer = SleepAnalyzer()
    analyzer.load_data(data_path)
    
    # 构建用户画像
    user_profile = analyzer.build_user_profile()
    
    # 可视化睡眠模式
    analyzer.visualize_sleep_patterns("/Users/dysania/program/AI/home-robot/v2/sleep_patterns.png")
    
    # 保存用户画像
    analyzer.save_user_profile("/Users/dysania/program/AI/home-robot/v2/user_sleep_profile.json")
    
    # 打印用户画像摘要
    print("\n用户睡眠画像摘要:")
    print(f"平均睡眠时长: {user_profile['average_sleep_duration_hours']}小时")
    print(f"平均入睡时间: {user_profile['average_sleep_time']}")
    print(f"平均起床时间: {user_profile['average_wake_time']}")
    print(f"睡眠质量: {user_profile['average_sleep_quality']}/100")
    print(f"睡眠模式: {user_profile['sleep_pattern']}")
    print(f"睡眠一致性: {user_profile['sleep_consistency']:.1f}/100")
    
    if user_profile['common_anomalies']:
        print("\n常见睡眠异常:")
        for anomaly in user_profile['common_anomalies']:
            print(f"- {anomaly['type']}: {anomaly['count']}次") 