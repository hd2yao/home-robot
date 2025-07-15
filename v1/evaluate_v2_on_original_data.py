import pandas as pd
import numpy as np
from enhanced_health_user_model_v2 import ImprovedHealthRiskModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

def main():
    """
    评估增强版健康用户画像系统V2在原始数据集上的准确率
    """
    print("=== 增强版健康用户画像系统V2在原始数据集上的评估 ===")
    
    # 加载原始数据
    print("\n[1] 加载原始数据...")
    data_path = "docs/simulated_health_data_high_quality.csv"
    data = pd.read_csv(data_path)
    print(f"加载了 {len(data)} 条数据")
    
    # 数据分析
    print("\n[2] 数据分析...")
    print("健康风险等级分布:")
    risk_distribution = data['health_risk_level'].value_counts()
    print(risk_distribution)
    
    # 划分训练集和测试集
    print("\n[3] 划分训练集和测试集...")
    X = data[['heart_rate', 'resp_rate', 'spo2', 'body_movement',
              'breath_wave', 'sleep_stage', 'temperature', 'humidity']]
    y = data['health_risk_level']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"训练集大小: {len(X_train)}")
    print(f"测试集大小: {len(X_test)}")
    
    # 创建并训练模型
    print("\n[4] 创建并训练模型...")
    model = ImprovedHealthRiskModel()
    model.train(X_train, y_train)
    
    # 评估模型
    print("\n[5] 评估模型...")
    y_pred = model.predict(X_test)
    
    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    print(f"模型在测试集上的准确率: {accuracy:.4f}")
    
    # 打印分类报告
    print("\n分类报告:")
    print(classification_report(y_test, y_pred))
    
    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('健康风险模型混淆矩阵')
    plt.colorbar()
    classes = sorted(y.unique())
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.tight_layout()
    
    # 各类别准确率
    print("\n各风险等级准确率:")
    for level in sorted(y.unique()):
        level_mask = (y_test == level)
        if sum(level_mask) > 0:  # 避免除零错误
            level_accuracy = accuracy_score(y_test[level_mask], y_pred[level_mask])
            print(f"{level} 风险等级准确率: {level_accuracy:.4f} ({sum(level_mask)} 个样本)")
    
    # 全量数据评估
    print("\n[6] 全量数据评估...")
    full_pred = model.predict(X)
    full_accuracy = accuracy_score(y, full_pred)
    print(f"模型在全量数据上的准确率: {full_accuracy:.4f}")
    
    # 打印分类报告
    print("\n全量数据分类报告:")
    print(classification_report(y, full_pred))
    
    print("\n=== 评估完成 ===")

if __name__ == "__main__":
    main() 