import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
import joblib
import os

class HealthRiskModel:
    """
    健康风险预测模型（Health Risk Classification）
    基于结构化生理指标与环境数据，实时评估健康风险等级（low / medium / high）
    """
    def __init__(self):
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('mlp', MLPClassifier(
                hidden_layer_sizes=(16, 8),
                activation='relu',
                solver='adam',
                alpha=0.0001,
                batch_size=32,
                learning_rate='adaptive',
                max_iter=500,
                random_state=42
            ))
        ])
        self.features = [
            'heart_rate', 'resp_rate', 'spo2', 'body_movement',
            'breath_wave', 'sleep_stage', 'temperature', 'humidity'
        ]
        self.label_mapping = {0: 'low', 1: 'medium', 2: 'high'}
        self.reverse_mapping = {'low': 0, 'medium': 1, 'high': 2}
        
    def train(self, X, y):
        """训练健康风险预测模型"""
        # 将字符串标签转换为数字
        y_encoded = np.array([self.reverse_mapping[label] for label in y])
        self.model.fit(X, y_encoded)
        
    def predict(self, X):
        """预测健康风险等级"""
        y_pred_encoded = self.model.predict(X)
        # 将数字标签转换回字符串
        return np.array([self.label_mapping[label] for label in y_pred_encoded])
    
    def predict_proba(self, X):
        """预测健康风险概率分布"""
        return self.model.predict_proba(X)
    
    def evaluate(self, X_test, y_test):
        """评估模型性能"""
        y_test_encoded = np.array([self.reverse_mapping[label] for label in y_test])
        y_pred = self.model.predict(X_test)
        
        accuracy = accuracy_score(y_test_encoded, y_pred)
        print(f"模型准确率: {accuracy:.4f}")
        print("\n分类报告:")
        print(classification_report(y_test_encoded, y_pred, 
                                   target_names=['low', 'medium', 'high']))
        
        # 混淆矩阵
        cm = confusion_matrix(y_test_encoded, y_pred)
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('健康风险模型混淆矩阵')
        plt.colorbar()
        tick_marks = np.arange(3)
        plt.xticks(tick_marks, ['low', 'medium', 'high'])
        plt.yticks(tick_marks, ['low', 'medium', 'high'])
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.tight_layout()
        
        return accuracy
    
    def save(self, filepath='models/health_risk_model.pkl'):
        """保存模型"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.model, filepath)
        
    def load(self, filepath='models/health_risk_model.pkl'):
        """加载模型"""
        self.model = joblib.load(filepath)


class BehavioralAnomalyModel:
    """
    行为异常识别模型（Behavioral Anomaly Detection）
    识别用户行为偏离其画像特征的异常模式
    """
    def __init__(self, contamination=0.1):
        # 使用IsolationForest进行异常检测
        self.model = IsolationForest(
            n_estimators=100,
            contamination=contamination,  # 异常样本的预期比例
            random_state=42
        )
        
        # 示例特征，实际应用中需要根据真实数据调整
        self.features = [
            'remote_usage_freq', 'voice_wakeup_count', 
            'routine_deviation', 'content_interaction_time',
            'interaction_switch_freq'
        ]
    
    def train(self, X):
        """训练行为异常检测模型"""
        self.model.fit(X)
    
    def predict(self, X):
        """
        预测行为是否异常
        返回: 1表示正常, -1表示异常
        """
        return self.model.predict(X)
    
    def predict_score(self, X):
        """
        获取异常分数
        分数越低，越可能是异常
        """
        return self.model.decision_function(X)
    
    def save(self, filepath='models/behavioral_anomaly_model.pkl'):
        """保存模型"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.model, filepath)
        
    def load(self, filepath='models/behavioral_anomaly_model.pkl'):
        """加载模型"""
        self.model = joblib.load(filepath)


class InterventionSuggestionModel:
    """
    干预建议生成模型（Intervention Suggestion Model）
    根据用户当前状态、健康风险等级和行为偏差，生成个性化干预建议
    """
    def __init__(self):
        # 使用随机森林分类器作为干预建议模型
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        # 干预建议类型
        self.suggestion_types = [
            'health_reminder',  # 健康提醒
            'service_recommendation',  # 服务推荐
            'content_recommendation',  # 内容推荐
            'device_action'  # 设备联动
        ]
        
    def train(self, X, y):
        """
        训练干预建议模型
        X: 特征矩阵，包含健康风险、行为异常和用户画像特征
        y: 标签，表示干预建议类型的索引
        """
        self.model.fit(X, y)
    
    def predict(self, X):
        """预测最合适的干预建议类型"""
        suggestion_idx = self.model.predict(X)
        return [self.suggestion_types[idx] for idx in suggestion_idx]
    
    def predict_proba(self, X):
        """预测各干预建议类型的概率分布"""
        return self.model.predict_proba(X)
    
    def generate_suggestion(self, health_risk, behavior_anomaly, user_profile, current_context):
        """
        生成具体的干预建议
        
        参数:
        - health_risk: 健康风险等级 ('low', 'medium', 'high')
        - behavior_anomaly: 行为是否异常 (1: 正常, -1: 异常)
        - user_profile: 用户画像信息 (字典)
        - current_context: 当前情境信息 (字典)
        
        返回:
        - suggestion: 干预建议内容
        """
        # 示例规则逻辑
        if health_risk == 'high':
            if 'age' in user_profile and user_profile['age'] > 70:
                return "检测到健康指标异常，建议立即联系家庭医生。"
            else:
                return "您的健康指标出现波动，建议休息并多喝水。"
        
        elif health_risk == 'medium':
            if behavior_anomaly == -1:
                return "您今天活动较少，建议适当进行轻度运动。"
            else:
                return "建议您查看今日健康报告，了解身体状况。"
        
        else:  # low risk
            if 'preference' in user_profile and 'music' in user_profile['preference']:
                return "播放您喜爱的轻音乐，帮助放松心情。"
            else:
                return "天气不错，建议适当户外活动增强身体免疫力。"
    
    def save(self, filepath='models/intervention_model.pkl'):
        """保存模型"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.model, filepath)
        
    def load(self, filepath='models/intervention_model.pkl'):
        """加载模型"""
        self.model = joblib.load(filepath)


class HealthUserProfileSystem:
    """
    健康用户画像系统
    整合三个核心模型，提供完整的用户健康状态分析和干预建议
    """
    def __init__(self):
        self.health_risk_model = HealthRiskModel()
        self.behavioral_anomaly_model = BehavioralAnomalyModel()
        self.intervention_model = InterventionSuggestionModel()
        
    def train_health_risk_model(self, data_path):
        """训练健康风险预测模型"""
        # 加载数据
        df = pd.read_csv(data_path)
        
        # 提取特征和标签
        X = df[self.health_risk_model.features]
        y = df['health_risk_level']
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # 训练模型
        self.health_risk_model.train(X_train, y_train)
        
        # 评估模型
        accuracy = self.health_risk_model.evaluate(X_test, y_test)
        
        return accuracy
    
    def predict_health_risk(self, user_data):
        """预测用户健康风险等级"""
        features = np.array([user_data[f] for f in self.health_risk_model.features]).reshape(1, -1)
        return self.health_risk_model.predict(features)[0]
    
    def detect_behavioral_anomaly(self, behavior_data):
        """检测用户行为是否异常"""
        features = np.array([behavior_data[f] for f in self.behavioral_anomaly_model.features]).reshape(1, -1)
        return self.behavioral_anomaly_model.predict(features)[0]
    
    def generate_intervention(self, health_risk, behavior_anomaly, user_profile, current_context):
        """生成干预建议"""
        return self.intervention_model.generate_suggestion(
            health_risk, behavior_anomaly, user_profile, current_context
        )
    
    def analyze_user_health(self, physiological_data, behavior_data, user_profile, current_context):
        """
        分析用户健康状态并生成干预建议
        
        参数:
        - physiological_data: 生理数据 (字典)
        - behavior_data: 行为数据 (字典)
        - user_profile: 用户画像 (字典)
        - current_context: 当前情境 (字典)
        
        返回:
        - analysis_result: 分析结果 (字典)
        """
        # 预测健康风险
        health_risk = self.predict_health_risk(physiological_data)
        
        # 检测行为异常
        behavior_anomaly = self.detect_behavioral_anomaly(behavior_data)
        
        # 生成干预建议
        intervention = self.generate_intervention(
            health_risk, behavior_anomaly, user_profile, current_context
        )
        
        # 检查是否需要直接触发报警
        alarm_triggered = False
        alarm_type = None
        
        if 'is_fall_detected' in current_context and current_context['is_fall_detected']:
            alarm_triggered = True
            alarm_type = 'fall_detected'
        elif 'is_sos_detected' in current_context and current_context['is_sos_detected']:
            alarm_triggered = True
            alarm_type = 'sos_detected'
        elif 'is_fire_detected' in current_context and current_context['is_fire_detected']:
            alarm_triggered = True
            alarm_type = 'fire_detected'
        
        # 整合分析结果
        analysis_result = {
            'health_risk_level': health_risk,
            'behavior_status': 'normal' if behavior_anomaly == 1 else 'anomaly',
            'intervention_suggestion': intervention,
            'alarm_triggered': alarm_triggered,
            'alarm_type': alarm_type,
            'timestamp': pd.Timestamp.now()
        }
        
        return analysis_result
    
    def save_models(self, directory='models'):
        """保存所有模型"""
        os.makedirs(directory, exist_ok=True)
        self.health_risk_model.save(f"{directory}/health_risk_model.pkl")
        self.behavioral_anomaly_model.save(f"{directory}/behavioral_anomaly_model.pkl")
        self.intervention_model.save(f"{directory}/intervention_model.pkl")
        
    def load_models(self, directory='models'):
        """加载所有模型"""
        self.health_risk_model.load(f"{directory}/health_risk_model.pkl")
        self.behavioral_anomaly_model.load(f"{directory}/behavioral_anomaly_model.pkl")
        self.intervention_model.load(f"{directory}/intervention_model.pkl") 