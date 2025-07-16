# 老人健康状态预测与睡眠行为分析系统

该系统基于老人的体征信息、房间环境信息以及行为信息，预测老人的健康状态并分析睡眠行为，构建用户画像。

## 功能特点

1. **数据处理**：为原始数据添加时间戳信息
   - 睡眠相关数据（心率、呼吸率、血氧等）：每天晚上9:30到次日5:30
   - 环境数据（温度、湿度）：全天

2. **健康状态预测**：基于时间序列特征预测老人健康水平
   - 使用XGBoost模型进行分类（低风险、中风险、高风险）
   - 融合多模态数据（体征、环境、行为）
   - 提取时间序列特征（滚动窗口统计量）

3. **睡眠行为分析**：构建用户睡眠行为画像
   - 识别睡眠会话和模式
   - 检测睡眠异常（过长、过短、中断等）
   - 计算睡眠质量和一致性
   - 可视化睡眠模式

## 项目结构

```
v2/
├── docs/                         # 原始数据
│   └── simulated_health_data_high_quality.csv
├── results/                      # 分析结果输出目录
├── data_processor.py             # 数据处理模块
├── health_predictor.py           # 健康状态预测模块
├── sleep_analyzer.py             # 睡眠行为分析模块
├── main.py                       # 主程序
├── requirements.txt              # 项目依赖
└── README.md                     # 项目说明
```

## 安装与运行

1. 安装依赖：

```bash
pip install -r requirements.txt
```

2. 运行分析流程：

```bash
python main.py
```

## 输出结果

1. **处理后的数据**：`results/processed_health_data.csv`
2. **健康预测模型**：`results/health_predictor_model.joblib`
3. **健康状态预测结果**：`results/health_predictions.csv`
4. **睡眠模式可视化**：`results/sleep_patterns.png`
5. **用户睡眠画像**：`results/user_sleep_profile.json`
6. **健康状态分布图**：`results/health_distribution.png`

## 模型说明

本系统采用XGBoost模型进行健康状态预测，该模型具有以下优势：

1. 能够处理多种类型的特征（数值、类别）
2. 对缺失值有较好的容忍度
3. 能够捕捉特征间的复杂关系
4. 可解释性较好，能够输出特征重要性

睡眠分析部分使用基于规则和统计的方法，识别睡眠模式和异常，构建用户画像。

## 未来改进方向

1. 引入更多传感器数据（如活动量、声音、光线等）
2. 采用更复杂的时序模型（如LSTM、Transformer）
3. 增加实时预警功能
4. 开发移动应用界面，方便家属和医护人员查看 