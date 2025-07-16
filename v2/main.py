import os
import pandas as pd
import matplotlib.pyplot as plt
from data_processor import process_data
from health_predictor import HealthPredictor, prepare_data_for_prediction
from sleep_analyzer import SleepAnalyzer
from matplotlib_config import configure_chinese_font

def create_directory_if_not_exists(path):
    """创建目录（如果不存在）"""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"创建目录: {path}")

def main():
    """主函数，运行整个分析流程"""
    # 设置路径
    base_dir = "/Users/dysania/program/AI/home-robot/v2"
    input_file = os.path.join(base_dir, "docs/simulated_health_data_high_quality.csv")
    output_dir = os.path.join(base_dir, "results")
    
    # 创建输出目录
    create_directory_if_not_exists(output_dir)
    
    # 1. 数据处理：添加时间戳
    print("\n=== 第一步：数据处理 ===")
    processed_data_file = os.path.join(output_dir, "processed_health_data.csv")
    df = process_data(input_file, processed_data_file)
    
    # 2. 健康状态预测
    print("\n=== 第二步：健康状态预测 ===")
    # 准备数据
    df = prepare_data_for_prediction(df)
    
    # 训练模型
    predictor = HealthPredictor()
    predictor.train(df)
    
    # 保存模型
    model_path = os.path.join(output_dir, "health_predictor_model.joblib")
    predictor.save_model(model_path)
    
    # 预测健康状态
    predictions = predictor.predict(df)
    predictions_file = os.path.join(output_dir, "health_predictions.csv")
    predictions.to_csv(predictions_file, index=False)
    print(f"健康状态预测完成，结果已保存到 {predictions_file}")
    
    # 3. 睡眠行为分析
    print("\n=== 第三步：睡眠行为分析 ===")
    # 创建睡眠分析器
    analyzer = SleepAnalyzer()
    analyzer.load_data(processed_data_file)
    
    # 构建用户画像
    user_profile = analyzer.build_user_profile()
    
    # 可视化睡眠模式
    sleep_patterns_file = os.path.join(output_dir, "sleep_patterns.png")
    analyzer.visualize_sleep_patterns(sleep_patterns_file)
    
    # 保存用户画像
    user_profile_file = os.path.join(output_dir, "user_sleep_profile.json")
    analyzer.save_user_profile(user_profile_file)
    
    # 4. 结果汇总
    print("\n=== 分析结果汇总 ===")
    print(f"1. 处理后的数据已保存到: {processed_data_file}")
    print(f"2. 健康预测模型已保存到: {model_path}")
    print(f"3. 健康状态预测结果已保存到: {predictions_file}")
    print(f"4. 睡眠模式可视化已保存到: {sleep_patterns_file}")
    print(f"5. 用户睡眠画像已保存到: {user_profile_file}")
    
    # 打印用户画像摘要
    print("\n用户睡眠画像摘要:")
    print(f"平均睡眠时长: {user_profile['average_sleep_duration_hours']}小时")
    print(f"平均入睡时间: {user_profile['average_sleep_time']}")
    print(f"平均起床时间: {user_profile['average_wake_time']}")
    print(f"睡眠质量: {user_profile['average_sleep_quality']}/100")
    print(f"睡眠模式: {user_profile['sleep_pattern']}")
    print(f"睡眠一致性: {user_profile['sleep_consistency']:.1f}/100")
    
    # 打印健康状态分布
    health_distribution = predictions['predicted_health_level'].value_counts().to_dict()
    print("\n预测健康状态分布:")
    for level, count in health_distribution.items():
        print(f"{level}: {count}次 ({count/len(predictions)*100:.1f}%)")
    
    # 绘制健康状态分布饼图
    plt.figure(figsize=(10, 6))
    # 配置中文字体
    configure_chinese_font()
    # 将字典转换为列表，以解决类型错误
    values = list(health_distribution.values())
    labels = list(health_distribution.keys())
    plt.pie(values, labels=labels, autopct='%1.1f%%', 
            startangle=90, colors=['lightgreen', 'yellow', 'salmon'])
    plt.title('预测健康状态分布')
    plt.axis('equal')
    health_pie_file = os.path.join(output_dir, "health_distribution.png")
    plt.savefig(health_pie_file)
    print(f"健康状态分布图已保存到: {health_pie_file}")

if __name__ == "__main__":
    main()
    print("\n分析流程执行完毕!") 