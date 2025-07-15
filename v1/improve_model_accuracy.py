import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import joblib
import os

def main():
    """
    改进健康风险预测模型，提高在新数据上的准确率
    """
    print("=== 健康风险预测模型改进 ===")
    
    # 加载原始训练数据
    print("\n[1] 加载原始训练数据...")
    train_data_path = "docs/simulated_health_data_high_quality.csv"
    train_data = pd.read_csv(train_data_path)
    print(f"加载了 {len(train_data)} 条原始训练数据")
    
    # 加载新数据
    print("\n[2] 加载新数据...")
    new_data_path = "docs/new_health_data_samples.csv"
    new_data = pd.read_csv(new_data_path)
    new_data['health_risk_level'] = new_data['health_risk_level'].str.strip()
    print(f"加载了 {len(new_data)} 条新数据")
    
    # 合并数据集
    print("\n[3] 合并数据集...")
    combined_data = pd.concat([train_data, new_data], ignore_index=True)
    print(f"合并后共有 {len(combined_data)} 条数据")
    
    # 特征和标签
    features = [
        'heart_rate', 'resp_rate', 'spo2', 'body_movement',
        'breath_wave', 'sleep_stage', 'temperature', 'humidity'
    ]
    
    # 添加工程特征
    print("\n[4] 添加工程特征...")
    combined_data['heart_resp_ratio'] = combined_data['heart_rate'] / combined_data['resp_rate']
    combined_data['spo2_heart_product'] = combined_data['spo2'] * combined_data['heart_rate']
    combined_data['temp_humidity_ratio'] = combined_data['temperature'] / combined_data['humidity']
    combined_data['movement_sleep_product'] = combined_data['body_movement'] * combined_data['sleep_stage']
    
    # 为新数据单独添加工程特征
    new_data['heart_resp_ratio'] = new_data['heart_rate'] / new_data['resp_rate']
    new_data['spo2_heart_product'] = new_data['spo2'] * new_data['heart_rate']
    new_data['temp_humidity_ratio'] = new_data['temperature'] / new_data['humidity']
    new_data['movement_sleep_product'] = new_data['body_movement'] * new_data['sleep_stage']
    
    # 扩展特征列表
    extended_features = features + [
        'heart_resp_ratio', 'spo2_heart_product', 
        'temp_humidity_ratio', 'movement_sleep_product'
    ]
    
    # 划分数据集
    print("\n[5] 划分训练集和测试集...")
    X = combined_data[extended_features]
    y = combined_data['health_risk_level']
    
    # 使用分层抽样，保持类别比例
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 创建改进的模型
    print("\n[6] 创建和训练改进的模型...")
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(
            n_estimators=500,
            max_depth=15,
            min_samples_split=4,
            min_samples_leaf=2,
            class_weight='balanced',  # 处理类别不平衡
            random_state=42
        ))
    ])
    
    # 训练模型
    model.fit(X_train, y_train)
    
    # 评估模型
    print("\n[7] 评估模型...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"改进后模型在测试集上的准确率: {accuracy:.4f}")
    
    print("\n分类报告:")
    print(classification_report(y_test, y_pred))
    
    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('改进模型混淆矩阵')
    plt.colorbar()
    classes = sorted(y.unique())
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.tight_layout()
    
    # 在新数据上评估
    print("\n[8] 在新数据上评估...")
    X_new = new_data[extended_features]  # 使用相同的扩展特征列表
    y_new = new_data['health_risk_level']
    y_new_pred = model.predict(X_new)
    
    new_accuracy = accuracy_score(y_new, y_new_pred)
    print(f"改进后模型在新数据上的准确率: {new_accuracy:.4f}")
    
    print("\n新数据分类报告:")
    print(classification_report(y_new, y_new_pred))
    
    # 保存模型
    print("\n[9] 保存改进的模型...")
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/improved_health_risk_model.pkl')
    
    # 预测结果对比
    print("\n新数据预测结果对比:")
    results = pd.DataFrame({
        '实际风险等级': y_new,
        '预测风险等级': y_new_pred
    })
    print(results)
    
    print("\n=== 模型改进完成 ===")

if __name__ == "__main__":
    main() 