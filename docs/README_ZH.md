# 数据中心预测性冷却优化：基于温度感知的冷水机组调度与能耗降低
[English](README.md) | [தமிழ்](README_TA.md) | 中文 | [हिन्दी](README_HI.md) | [Bahasa Indonesia](README_ID.md)

![GitHub top language](https://img.shields.io/github/languages/top/vk22006/predictive-cooling-optimizer-for-data-centers)
![GitHub language count](https://img.shields.io/github/languages/count/vk22006/predictive-cooling-optimizer-for-data-centers)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![GitHub forks](https://img.shields.io/github/forks/vk22006/predictive-cooling-optimizer-for-data-centers)

本项目旨在解决数据中心冷却系统中的能源效率问题。项目通过开发一种基于温度感知的预测模型，对冷水机组的运行调度进行优化，从而在确保热安全的同时降低能源消耗。传统的响应式冷却系统通常会在温度变化发生后才进行响应，这容易造成能源浪费以及冷水机组运行效率低下。

![主页](img/home_page.PNG "主页")

## 项目方法

本项目首先对 13,615 条 HVAC 样本进行了全面的数据预处理，包括使用 IQR 方法进行异常值检测、通过 MinMaxScaler 进行归一化处理，以及按照时间顺序进行 80-20 的训练集与测试集划分，以保持数据的时间完整性。

通过特征工程共构建了 46 个增强特征，包括 16 个滞后特征（Lag Features）、12 个滚动平均特征（Rolling Averages）、6 个周期性时间编码特征（Cyclical Temporal Encodings）以及 4 个交互特征（Interaction Features），从而更好地捕捉系统复杂的动态变化。

两个 XGBoost 回归模型构成了核心预测引擎：

* **能源预测模型**：R² = 0.9891，MAE = 1.222 kWh
* **温度预测模型**：R² = 0.6853，其中 89.24% 的预测结果位于 ±1°C 的误差范围内

两个模型都表现出了较高的训练效率。能源预测模型的训练时间为 2.12 秒，温度预测模型的训练时间为 1.87 秒，因此具备实时部署的可行性。

`PredictiveCoolingOptimizer` 类将两个模型进行集成，并通过基于约束的温度管理与能源最小化策略，实现整个冷却系统的优化。

## 测试

项目共进行了 11 项测试，分为五个类别，具体如下：

|            测试类型           |        测试目标        |   状态    |
| :--------------------------: | :--------------------: | :------: |
| 单元测试（Unit Tests）        | 能源与温度模型、优化引擎 | ✅ 通过  |
| 集成测试（Integration Tests） | 端到端流水线、系统集成   | ✅ 通过  |
| 功能测试（Functional Tests）  | 准确率、响应时间与逻辑   | ✅ 通过  |
| 白盒测试（White Box Test）    | 超参数、特征工程         | ✅ 通过  |
| 黑盒测试（Black Box Test）    | 边界值、输出一致性       | ✅ 通过  |
|                              | 通过的测试              | 11/11    |
|                              | 失败的测试              | 0/11     |
|                              | 成功率                  | 100.0% |

## 执行步骤

程序的运行过程非常简单，只需按照以下步骤操作即可。

1. 安装所需的库：

```bash
pip install xgboost streamlit
```

2. 在命令提示符（Command Prompt）或 PowerShell 中进入项目目录：

```bash
cd <your-file-path>
```

3. 使用以下命令启动应用程序：

```bash
streamlit run 1_Home.py
```

## 使用的工具

1. Anaconda Jupyter：用于模型训练与测试
2. Streamlit：用于前端实现
3. Joblib：用于处理 `.pkl` 模型文件
4. NumPy
5. Pandas
6. Scikit-Learn
7. XGBoost

## 使用的算法

### 1. 预测算法

* XGBoost（Extreme Gradient Boosting，极端梯度提升）
* Random Forest Regressor（随机森林回归）

### 2. 支持算法

* Min-Max Normalization（最小-最大归一化）
* Rolling Average（滚动平均，用于特征工程）

该工具成功验证了基于软件的数据中心预测性冷却优化方案的可行性。训练完成的模型可进一步部署于基于 Streamlit 的交互式 Web 应用中，从而便于实际操作、系统展示以及向相关利益相关者进行演示。

## 许可证

本项目采用 MIT License 许可证。详情请参阅 [LICENSE](LICENSE) 文件。
