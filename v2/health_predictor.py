import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
import xgboost as xgb
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from tqdm import tqdm
from matplotlib_config import configure_chinese_font

class HealthPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.imputer = SimpleImputer(strategy='mean')
        
    def preprocess_data(self, df):
        """预处理数据，提取时间特征并处理缺失值"""
        # 确保timestamp列是datetime类型
        if isinstance(df['timestamp'].iloc[0], str):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
        # 提取时间特征
        df['hour'] = df['timestamp'].dt.hour
        df['minute'] = df['timestamp'].dt.minute
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
        df['is_night'] = ((df['hour'] >= 21) | (df['hour'] <= 5)).astype(int)
        
        # 计算滚动窗口特征（过去24小时的平均值和标准差）
        # 按照时间排序
        df = df.sort_values('timestamp')
        
        # 创建滚动窗口特征
        window_features = []
        for subject in df['subject_id'].unique() if 'subject_id' in df.columns else [1]:
            subject_df = df if 'subject_id' not in df.columns else df[df['subject_id'] == subject].copy()
            
            # 对于每个数值型特征计算滚动窗口统计量
            for col in ['heart_rate', 'resp_rate', 'spo2', 'body_movement', 'breath_wave', 
                       'sleep_stage', 'temperature', 'humidity']:
                if col in subject_df.columns:
                    # 24小时窗口（假设数据是按小时记录的）
                    subject_df[f'{col}_24h_mean'] = subject_df[col].rolling(window=24, min_periods=1).mean()
                    subject_df[f'{col}_24h_std'] = subject_df[col].rolling(window=24, min_periods=1).std().fillna(0)
            
            window_features.append(subject_df)
        
        df = pd.concat(window_features) if len(window_features) > 1 else window_features[0]
        
        # 处理缺失值
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        df[numerical_cols] = self.imputer.fit_transform(df[numerical_cols])
        
        return df
    
    def extract_features(self, df):
        """提取用于预测的特征"""
        # 基本特征
        feature_cols = [
            'heart_rate', 'resp_rate', 'spo2', 'body_movement', 'breath_wave', 
            'sleep_stage', 'temperature', 'humidity',
            'hour', 'minute', 'day_of_week', 'is_weekend', 'is_night'
        ]
        
        # 添加滚动窗口特征
        rolling_cols = [col for col in df.columns if ('_24h_mean' in col or '_24h_std' in col)]
        feature_cols.extend(rolling_cols)
        
        # 保留存在的列
        feature_cols = [col for col in feature_cols if col in df.columns]
        
        X = df[feature_cols]
        
        # 编码目标变量
        if 'health_risk_level' in df.columns:
            y = self.label_encoder.fit_transform(df['health_risk_level'])
            return X, y
        else:
            return X, None
    
    def train(self, df):
        """训练健康状态预测模型"""
        # 预处理数据
        processed_df = self.preprocess_data(df)
        
        # 提取特征和目标变量
        X, y = self.extract_features(processed_df)
        
        # 标准化特征
        X = self.scaler.fit_transform(X)
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 训练XGBoost模型
        self.model = xgb.XGBClassifier(
            objective='multi:softprob',
            num_class=len(np.unique(y)),
            learning_rate=0.1,
            max_depth=6,
            n_estimators=100,
            random_state=42
        )
        
        self.model.fit(X_train, y_train)
        
        # 评估模型
        y_pred = self.model.predict(X_test)
        print("模型评估报告:")
        print(classification_report(y_test, y_pred, 
                                   target_names=self.label_encoder.classes_))
        
        # 配置中文字体
        configure_chinese_font()
        
        # 绘制混淆矩阵
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_)
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.title('混淆矩阵')
        plt.savefig('/Users/dysania/program/AI/home-robot/v2/confusion_matrix.png')
        
        # 特征重要性
        plt.figure(figsize=(12, 8))
        feature_cols = [col for col in processed_df.columns if col in X]
        xgb.plot_importance(self.model, max_num_features=15)
        plt.title('特征重要性')
        plt.savefig('/Users/dysania/program/AI/home-robot/v2/feature_importance.png')
        
        return self.model
    
    def predict(self, df):
        """预测健康状态"""
        # 预处理数据
        processed_df = self.preprocess_data(df)
        
        # 提取特征
        X, _ = self.extract_features(processed_df)
        
        # 标准化特征
        X = self.scaler.transform(X)
        
        # 预测
        y_pred = self.model.predict(X)
        y_prob = self.model.predict_proba(X)
        
        # 转换回原始标签
        predictions = self.label_encoder.inverse_transform(y_pred)
        
        # 添加预测结果到原始数据
        result_df = df.copy()
        result_df['predicted_health_level'] = predictions
        
        # 添加预测概率
        for i, class_name in enumerate(self.label_encoder.classes_):
            result_df[f'prob_{class_name}'] = y_prob[:, i]
        
        return result_df
    
    def save_model(self, path):
        """保存模型"""
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'imputer': self.imputer
        }, path)
        print(f"模型已保存到 {path}")
    
    def load_model(self, path):
        """加载模型"""
        model_data = joblib.load(path)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoder = model_data['label_encoder']
        self.imputer = model_data['imputer']
        print(f"模型已从 {path} 加载")

def prepare_data_for_prediction(df):
    """准备用于预测的数据，添加subject_id"""
    if 'subject_id' not in df.columns:
        df['subject_id'] = 1
    return df

if __name__ == "__main__":
    # 加载处理后的数据
    data_path = "/Users/dysania/program/AI/home-robot/v2/processed_health_data.csv"
    df = pd.read_csv(data_path)
    
    # 准备数据
    df = prepare_data_for_prediction(df)
    
    # 训练模型
    predictor = HealthPredictor()
    predictor.train(df)
    
    # 保存模型
    predictor.save_model("/Users/dysania/program/AI/home-robot/v2/health_predictor_model.joblib")
    
    # 预测健康状态
    predictions = predictor.predict(df)
    
    # 保存预测结果
    predictions.to_csv("/Users/dysania/program/AI/home-robot/v2/health_predictions.csv", index=False)
    print("健康状态预测完成，结果已保存") 