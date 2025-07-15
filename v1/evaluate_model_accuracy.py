import pandas as pd
import numpy as np
from enhanced_health_user_model import EnhancedHealthUserProfileSystem
from sklearn.metrics import accuracy_score, classification_report

def main():
    """
    评估增强版健康用户画像模型在新数据上的准确率
    """
    print("=== 增强版健康用户画像模型评估 ===")
    
    # 创建系统实例
    system = EnhancedHealthUserProfileSystem()
    
    # 训练模型
    print("\n[1] 使用原始数据训练模型...")
    train_data_path = "docs/simulated_health_data_high_quality.csv"
    system.train_health_risk_model(train_data_path)
    
    # 训练行为异常检测模型
    print("\n[2] 训练行为异常检测模型...")
    np.random.seed(42)
    n_samples = 200
    behavior_train_data = np.random.normal(size=(n_samples, 5))
    behavior_train_data[-20:] = np.random.normal(loc=3, scale=2, size=(20, 5))
    system.behavioral_anomaly_model.train(behavior_train_data)
    
    # 加载新数据
    print("\n[3] 加载新数据样本...")
    new_data_path = "docs/new_health_data_samples.csv"
    new_data = pd.read_csv(new_data_path)
    print(f"加载了 {len(new_data)} 条新数据样本")
    
    # 检查数据
    print("\n数据类别分布:")
    print(new_data['health_risk_level'].value_counts())
    
    # 数据清理 - 去除每个值末尾可能的空格
    new_data['health_risk_level'] = new_data['health_risk_level'].str.strip()
    print("\n清理后的类别分布:")
    print(new_data['health_risk_level'].value_counts())
    
    # 提取特征和标签
    X = new_data[system.health_risk_model.base_features]
    y_true = new_data['health_risk_level']
    
    # 预测
    print("\n[4] 使用模型进行预测...")
    y_pred = system.health_risk_model.predict(X)
    
    # 查看唯一的预测值
    print("\n预测的唯一值:")
    print(np.unique(y_pred))
    
    # 计算准确率
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\n在新数据上的准确率: {accuracy:.4f}")
    
    # 打印分类报告
    print("\n分类报告:")
    # 获取所有唯一的标签
    unique_labels = sorted(set(list(y_true) + list(y_pred)))
    print(classification_report(y_true, y_pred, target_names=unique_labels))
    
    # 打印预测结果对比
    print("\n预测结果对比:")
    results = pd.DataFrame({
        '实际风险等级': y_true,
        '预测风险等级': y_pred
    })
    print(results)
    
    # 统计每个类别的准确率
    print("\n各风险等级准确率:")
    for level in unique_labels:
        level_mask = (y_true == level)
        if sum(level_mask) > 0:  # 避免除零错误
            level_accuracy = accuracy_score(y_true[level_mask], y_pred[level_mask])
            print(f"{level} 风险等级准确率: {level_accuracy:.4f} ({sum(level_mask)} 个样本)")

if __name__ == "__main__":
    main() 