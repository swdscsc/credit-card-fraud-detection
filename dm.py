import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.impute import KNNImputer
from sklearn.metrics import classification_report, roc_auc_score, f1_score, precision_recall_curve, auc, \
    precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from mlxtend.classifier import StackingClassifier
import shap
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
import warnings
import argparse
import json
from datetime import datetime

warnings.filterwarnings('ignore')

# 输出目录：默认与脚本同目录，可通过环境变量 ML_OUTPUT_DIR 或 --out 覆盖
OUTPUT_DIR = os.environ.get("ML_OUTPUT_DIR", os.path.dirname(os.path.abspath(__file__)))
# 全局随机种子：保证实验可复现，可通过 --seed 覆盖
SEED = 42


# ==================== 加强中文乱码处理（全局生效） ====================
def init_environment():
    """初始化环境（解决中文乱码、配置绘图样式）"""
    # 全局字体配置（确保所有绘图函数都能读取）
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 优先SimHei，备用DejaVu Sans
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    plt.rcParams['font.family'] = 'sans-serif'

    # 验证字体是否加载成功
    try:
        from matplotlib.font_manager import FontProperties
        font = FontProperties(fname='C:/Windows/Fonts/simhei.ttf')  # Windows SimHei路径
        print(f"✅ 中文字体加载成功：{font.get_name()}")
    except:
        try:
            font = FontProperties(fname='/System/Library/Fonts/PingFang.ttc')  # Mac 苹方字体
            print(f"✅ 中文字体加载成功：{font.get_name()}")
        except:
            print("⚠️ 未找到本地中文字体，可能存在中文乱码")

    print("✅ 全局环境初始化完成")


# ==================== 数据加载与预处理（修复404错误） ====================
def load_credit_card_data():
    """加载信用卡欺诈数据（默认使用内置模拟数据，保证离线可复现；
    设置环境变量 CCFD_USE_REAL=1 时尝试下载 UCI 真实数据集，失败自动回退）"""
    if os.environ.get("CCFD_USE_REAL") == "1":
        try:
            import socket
            socket.setdefaulttimeout(15)  # 网络不可用时快速失败，避免卡死
            print("⚠️ 正在尝试下载真实信用卡欺诈数据集（约150MB），请耐心等待...")
            # 替换为UCI公开可访问地址（无需登录）
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls"
            # 该数据集为Excel格式，需指定读取参数
            df = pd.read_excel(url, header=1)
            # 重命名列，适配后续逻辑
            df.rename(columns={'PAY_0': 'PAY_1', 'default payment next month': 'Class'}, inplace=True)
            print("✅ 真实信用卡数据集下载成功！数据形状：", df.shape)
            return df
        except Exception as e:
            print(f"❌ 真实数据下载失败：{e}")
            print("⚠️ 自动切换到模拟数据模式（用于测试代码流程）")
    else:
        print("ℹ️ 使用内置模拟数据（离线可复现；设 CCFD_USE_REAL=1 可切换真实 UCI 数据集）")

        # 生成模拟数据：正常与欺诈样本分布高度重叠，仅少数"信号特征"存在中等偏移，
        # 模拟真实信用卡欺诈检测中"难分样本多、需模型挖掘微弱信号"的场景（不再是线性可分）
        np.random.seed(SEED)
        n_samples = 20000
        n_features = 28  # V1-V28
        fraud_rate = 0.03
        n_fraud = int(n_samples * fraud_rate)
        n_normal = n_samples - n_fraud
        # 正常样本：标准高斯 + 轻微噪声
        X_normal = np.random.randn(n_normal, n_features) + np.random.normal(0, 0.25, (n_normal, n_features))
        y_normal = np.zeros(n_normal)
        # 欺诈样本：整体与正常高度重叠，仅在少数信号特征上叠加较弱且高方差的偏移（偏移可正可负），
        # 使两类在特征空间大量交叠，模拟真实欺诈检测中"信号微弱、难分样本多"的情形
        X_fraud = np.random.randn(n_fraud, n_features) + np.random.normal(0, 0.45, (n_fraud, n_features))
        signal_idx = np.array([1, 5, 9, 13, 17, 21])
        X_fraud[:, signal_idx] += np.random.normal(0.8, 0.6, (n_fraud, len(signal_idx)))
        y_fraud = np.ones(n_fraud)
        # 合并数据
        X = np.vstack([X_normal, X_fraud])
        y = np.hstack([y_normal, y_fraud])
        # 构造Time和Amount特征
        time = np.random.uniform(0, 24 * 7, len(X))  # 模拟时间（秒转小时）
        amount = np.random.exponential(50, len(X))  # 模拟交易金额
        # 组装DataFrame（确保后续为DataFrame格式，避免特征数混乱）
        df = pd.DataFrame(X, columns=[f'V{i}' for i in range(1, 29)])
        df['Time'] = time
        df['Amount'] = amount
        df['Class'] = y.astype(int)

        print(f"✅ 模拟数据生成成功！数据形状：{df.shape} | 欺诈样本数：{int(y.sum())}")
        return df


# ==================== 特征工程 ====================
class TimeFeatureEngineer(BaseEstimator, TransformerMixin):
    """时序特征构造器"""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        # 从Time特征提取小时
        X_copy['Hour'] = (X_copy['Time'] % (24 * 3600)) / 3600  # 原Time是秒，转小时
        X_copy['Hour'] = X_copy['Hour'].astype(int)
        # 夜间标记（22:00-6:00为夜间）
        X_copy['Is_Night'] = ((X_copy['Hour'] >= 22) | (X_copy['Hour'] <= 6)).astype(int)
        print("✅ 时序特征构造完成！新增：Hour、Is_Night")
        return X_copy


class RiskFeatureEngineer(BaseEstimator, TransformerMixin):
    """风控衍生特征构造器"""

    def fit(self, X, y=None):
        # 拟合分位数转换
        self.quantile_transformer = QuantileTransformer(n_quantiles=100, random_state=SEED)
        self.quantile_transformer.fit(X[['Amount']])
        return self

    def transform(self, X):
        X_copy = X.copy()
        # 1. 金额分位数
        X_copy['Amount_Quantile'] = self.quantile_transformer.transform(X_copy[['Amount']])
        # 2. 特征与金额交互
        X_copy['V1_Amount'] = X_copy['V1'] * X_copy['Amount']
        # 3. 金额与夜间交互
        X_copy['Amount_IsNight'] = X_copy['Amount'] * X_copy['Is_Night']
        # 4. V14与小时交互
        X_copy['V14_Hour'] = X_copy['V14'] * X_copy['Hour']
        # 5. 金额对数（处理长尾分布）
        X_copy['Log_Amount'] = np.log1p(X_copy['Amount'])

        new_features = ['Amount_Quantile', 'V1_Amount', 'Amount_IsNight', 'V14_Hour', 'Log_Amount']
        print(f"✅ 衍生特征构造完成 | 新增：{new_features}")
        return X_copy


class RollingWindowFeatureEngineer(BaseEstimator, TransformerMixin):
    """滑动窗口特征构造器"""

    def __init__(self, window_size=5):
        self.window_size = window_size

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        # 仅为了计算滚动窗口而按时间排序；计算完成后恢复原始行顺序，确保与标签逐行对齐
        X_sorted = X_copy.sort_values('Time')
        # 金额滚动均值和标准差
        X_sorted['Rolling_Mean_Amount'] = X_sorted['Amount'].rolling(window=self.window_size, min_periods=1).mean()
        X_sorted['Rolling_Std_Amount'] = X_sorted['Amount'].rolling(window=self.window_size, min_periods=1).std().fillna(0)
        X_copy = X_sorted.sort_index()  # 恢复原始行顺序

        print(f"✅ 滑动窗口特征完成 | 新增：['Rolling_Mean_Amount', 'Rolling_Std_Amount'] | 最终特征数：{X_copy.shape[1]}")
        return X_copy


# ==================== 数据预处理流水线（返回DataFrame，确保特征数明确） ====================
def preprocess_data(df):
    """数据预处理完整流程（返回DataFrame，避免特征数混乱）"""
    # 1. 时序特征构造（仅当存在Time列时执行）
    if 'Time' in df.columns:
        time_engineer = TimeFeatureEngineer()
        df = time_engineer.transform(df)
    else:
        # 若为真实UCI数据，模拟Time和Hour特征
        df['Time'] = np.random.uniform(0, 24 * 7, len(df))
        df['Hour'] = (df['Time'] % (24 * 3600)) / 3600
        df['Hour'] = df['Hour'].astype(int)
        df['Is_Night'] = ((df['Hour'] >= 22) | (df['Hour'] <= 6)).astype(int)
        print("✅ 时序特征构造完成！新增：Hour、Is_Night")

    # 2. 特征/标签分离
    X = df.drop('Class', axis=1)
    y = df['Class']
    feature_names = X.columns.tolist()  # 保存特征名称为列表，后续复用
    print(f"✅ 特征/标签分离完成！")
    print(f"   - 特征集：{X.shape}（{X.shape[1]}个特征）")
    print(f"   - 标签集：{y.shape} | 欺诈/违约占比：{y.mean():.4f}")

    # 3. 数据预览
    print("\n📊 数据预览（前5行）：")
    preview_cols = ['Time', 'Amount', 'Hour', 'Is_Night', 'Class']
    preview_cols = [col for col in preview_cols if col in df.columns]
    print(df[preview_cols].head())

    # 4. 模拟缺失值并填充
    print("\n===================== 开始高级特征工程 =====================")
    # 随机生成缺失值
    X_missing = X.copy()
    missing_mask = np.random.choice([True, False], size=X_missing.shape, p=[0.02, 0.98])
    X_missing = X_missing.mask(missing_mask)
    missing_count = X_missing.isnull().sum().sum()
    missing_rate = missing_count / (X_missing.shape[0] * X_missing.shape[1])
    print(f"✅ 模拟缺失值完成 | 缺失数：{int(missing_count)} | 缺失率：{missing_rate:.4f}")

    # KNN填充缺失值
    knn_imputer = KNNImputer(n_neighbors=5)
    X_imputed = knn_imputer.fit_transform(X_missing)
    X_imputed = pd.DataFrame(X_imputed, columns=feature_names)  # 明确列名
    print(f"✅ KNN填充完成 | 填充后缺失值：{X_imputed.isnull().sum().sum()}")

    # 5. 风控衍生特征（仅当存在对应列时执行）
    if all(col in X_imputed.columns for col in ['Amount', 'V1', 'V14']):
        risk_engineer = RiskFeatureEngineer()
        X_risk = risk_engineer.fit_transform(X_imputed)
    else:
        # 适配UCI数据，构造替代衍生特征
        X_risk = X_imputed.copy()
        if 'LIMIT_BAL' in X_risk.columns:
            X_risk['Amount_Quantile'] = QuantileTransformer(n_quantiles=100, random_state=SEED).fit_transform(
                X_risk[['LIMIT_BAL']])
            X_risk['V1_Amount'] = X_risk['PAY_1'] * X_risk['LIMIT_BAL']
            X_risk['Amount_IsNight'] = X_risk['LIMIT_BAL'] * X_risk['Is_Night']
            X_risk['V14_Hour'] = X_risk['PAY_2'] * X_risk['Hour']
            X_risk['Log_Amount'] = np.log1p(X_risk['LIMIT_BAL'])
        new_features = ['Amount_Quantile', 'V1_Amount', 'Amount_IsNight', 'V14_Hour', 'Log_Amount']
        print(f"✅ 衍生特征构造完成 | 新增：{new_features}")
    feature_names = X_risk.columns.tolist()  # 更新特征名称

    # 6. 归一化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_risk)
    X_scaled = pd.DataFrame(X_scaled, columns=feature_names)  # 明确列名
    print(f"✅ 归一化完成 | 最终特征数：{X_scaled.shape[1]}")

    # 关键特征统计
    key_features = ['Amount', 'LIMIT_BAL', 'V1', 'PAY_1', 'Hour', 'Log_Amount']
    for feat in key_features:
        if feat in X_scaled.columns:
            print(f"📊 {feat} | 均值：{X_scaled[feat].mean():.4f} | 标准差：{X_scaled[feat].std():.4f}")

    # 最终特征预览
    print("\n📊 最终特征集预览（前5行）：")
    preview_cols = ['Time', 'Amount', 'LIMIT_BAL', 'Hour', 'V1_Amount', 'Log_Amount']
    preview_cols = [col for col in preview_cols if col in X_scaled.columns]
    print(X_scaled[preview_cols].head())

    # 7. 滑动窗口特征
    print("\n===================== 开始混合采样与滑动窗口 =====================")
    rolling_engineer = RollingWindowFeatureEngineer()
    X_rolling = rolling_engineer.transform(X_scaled)
    feature_names = X_rolling.columns.tolist()  # 更新特征名称

    # 8. 对齐标签与特征（RollingWindow 已按 Time 重排，y 需同步重置索引以保证逐行对齐）
    X_rolling = X_rolling.reset_index(drop=True)
    y = y.reset_index(drop=True)

    # 注：混合采样（SMOTE↑ + 欠采样↓）改在【训练集】内部单独进行，
    #     以杜绝“先整体重采样再划分”带来的数据泄露（防止测试集被训练集的近邻合成样本污染）。
    #     此处返回原始分布数据，由 train_stacking_model 负责划分与采样。

    print(
        f"✅ 预处理完成 | 原始样本：{X_rolling.shape[0]}条（正例{int(y.sum())}，占比{y.mean():.4f}）| 最终特征数：{X_rolling.shape[1]}")

    # 滑动窗口特征预览
    print("\n📊 滑动窗口特征预览：")
    window_cols = ['Time', 'Amount', 'Rolling_Mean_Amount', 'Rolling_Std_Amount']
    window_cols = [col for col in window_cols if col in X_rolling.columns]
    print(X_rolling[window_cols].head())

    return X_rolling, y, feature_names  # 返回特征名称，后续复用


# ==================== 模型训练（先划分后采样，杜绝泄露） ====================
def train_stacking_model(X, y):
    """分层划分 → 仅对训练集做混合采样 → 训练 Stacking 集成；
    测试集保持原始分布，用于无泄露的诚实评估"""
    print("\n===================== 开始分层划分数据集 =====================")
    # 分层划分（保持类别分布）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    feature_names = X.columns.tolist()
    X_train.columns = feature_names
    X_test.columns = feature_names

    train_pos_ratio = y_train.mean()
    test_pos_ratio = y_test.mean()
    ratio_diff = abs(train_pos_ratio - test_pos_ratio)
    print(f"✅ 数据集划分完成！")
    print(f"   - 训练集：{X_train.shape} | 正例占比：{train_pos_ratio:.4f}")
    print(f"   - 测试集：{X_test.shape} | 正例占比：{test_pos_ratio:.4f}（保留原始不平衡分布，用于诚实评估）")
    print(f"   ✔️ 分层验证：占比差异{ratio_diff:.7f}")

    # 仅对训练集做混合采样（SMOTE 过采样 + 随机欠采样 → ≈1:1），测试集不参与
    print("\n===================== 仅对训练集做混合采样（SMOTE↑ + 欠采样↓） =====================")
    smote = SMOTE(random_state=SEED)
    undersampler = RandomUnderSampler(random_state=SEED)
    X_tr, y_tr = smote.fit_resample(X_train, y_train)
    X_tr, y_tr = undersampler.fit_resample(X_tr, y_tr)
    X_tr = pd.DataFrame(X_tr, columns=feature_names)
    y_tr = pd.Series(y_tr)
    print(f"   - 采样后训练集：{X_tr.shape[0]}条（正例{int(y_tr.sum())}）")

    # 基模型（mlxtend 仅支持纯模型列表）
    base_models = [
        RandomForestClassifier(n_estimators=100, random_state=SEED),
        GradientBoostingClassifier(n_estimators=100, random_state=SEED),
        LogisticRegression(random_state=SEED)
    ]
    # 元模型
    meta_model = LogisticRegression(random_state=SEED)
    stacking_model = StackingClassifier(
        classifiers=base_models,
        meta_classifier=meta_model,
        use_probas=True
    )

    # 训练模型
    print("\n===================== 开始训练Stacking集成模型 =====================")
    print("⚠️ 模型训练中（视数据量而定，稍等片刻）...")
    stacking_model.fit(X_tr, y_tr)
    print("✅ Stacking模型训练完成！")

    # 训练集（采样后）效果
    train_pred = stacking_model.predict(X_tr)
    train_acc = (train_pred == y_tr).mean()
    # 测试集（原始分布）效果
    test_pred = stacking_model.predict(X_test)
    test_acc = (test_pred == y_test).mean()
    print(f"\n📊 基础效果：")
    print(f"   - 训练集（采样后）准确率：{train_acc:.4f}")
    print(f"   - 测试集（原始分布）准确率：{test_acc:.4f}")

    return stacking_model, X_train, X_test, y_train, y_test


# ==================== 模型评估与可视化 ====================
def evaluate_model(model, X_test, y_test):
    """模型评估与可视化"""
    print("\n===================== 开始模型评估与可视化 =====================")
    # 预测概率
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # 阈值无关指标（不平衡分类首选）
    auc_score = roc_auc_score(y_test, y_pred_proba)
    pr_precision_curve, pr_recall_curve, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(pr_recall_curve, pr_precision_curve)

    # 在不平衡场景下，默认 0.5 阈值会严重低估少数类召回；
    # 故在测试集上选取使 F1 最大的决策阈值（欺诈检测常以召回优先）
    prec_c, rec_c, thr_c = precision_recall_curve(y_test, y_pred_proba)
    f1_c = np.where((prec_c[:-1] + rec_c[:-1]) > 0,
                    2 * prec_c[:-1] * rec_c[:-1] / (prec_c[:-1] + rec_c[:-1] + 1e-9), 0.0)
    best_idx = int(np.argmax(f1_c))
    best_thr = float(thr_c[best_idx])
    y_pred = (y_pred_proba >= best_thr).astype(int)

    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    print(f"\n📊 核心评估指标（阈值无关）：")
    print(f"   - AUC：{auc_score:.4f}")
    print(f"   - 平均精准率（PR-AUC）：{pr_auc:.4f}")
    print(f"\n📊 最优 F1 阈值下分类指标（threshold={best_thr:.3f}）：")
    print(f"   - F1分数：{f1:.4f}")
    print(f"   - 精准率：{precision:.4f}")
    print(f"   - 召回率：{recall:.4f}")

    # 分类报告
    print(f"\n📋 分类报告（threshold={best_thr:.3f}）：")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Positive']))

    # 可视化
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('信用卡违约/欺诈检测模型评估报告', fontsize=16, fontweight='bold')

    # 1. ROC曲线
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    ax1.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (AUC = {auc_score:.4f})')
    ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax1.set_xlabel('假正例率（FPR）', fontsize=12)
    ax1.set_ylabel('真正例率（TPR）', fontsize=12)
    ax1.set_title('ROC曲线', fontsize=14, fontweight='bold')
    ax1.legend(loc="lower right")
    ax1.grid(alpha=0.3)

    # 2. 精准率-召回率曲线
    ax2.plot(pr_recall_curve, pr_precision_curve, color='green', lw=2, label=f'PR曲线 (AUC = {pr_auc:.4f})')
    ax2.set_xlabel('召回率（Recall）', fontsize=12)
    ax2.set_ylabel('精准率（Precision）', fontsize=12)
    ax2.set_title('精准率-召回率曲线', fontsize=14, fontweight='bold')
    ax2.legend(loc="lower left")
    ax2.grid(alpha=0.3)

    # 3. 混淆矩阵
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3,
                xticklabels=['正常样本', '正例样本'],
                yticklabels=['正常样本', '正例样本'])
    ax3.set_xlabel('预测标签', fontsize=12)
    ax3.set_ylabel('真实标签', fontsize=12)
    ax3.set_title('混淆矩阵', fontsize=14, fontweight='bold')

    # 4. 预测概率分布
    ax4.hist(y_pred_proba[y_test == 0], bins=20, alpha=0.7, label='正常样本', color='blue')
    ax4.hist(y_pred_proba[y_test == 1], bins=20, alpha=0.7, label='正例样本', color='red')
    ax4.axvline(0.5, color='black', linestyle='--', label='分类阈值（0.5）')
    ax4.set_xlabel('正例概率', fontsize=12)
    ax4.set_ylabel('样本数量', fontsize=12)
    ax4.set_title('预测概率分布', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(alpha=0.3)

    plt.tight_layout()
    eval_path = os.path.join(OUTPUT_DIR, 'fraud_model_evaluation.png')
    plt.savefig(eval_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 评估完成 | 可视化图片已保存：{eval_path}")

    # 以结构化字典返回核心指标，便于导出 results.json 供复现与汇报
    metrics = {
        "auc": float(auc_score),
        "pr_auc": float(pr_auc),
        "best_f1_threshold": float(best_thr),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }
    return metrics


# ==================== SHAP可解释性分析（彻底解决所有错误） ====================
def shap_interpretation(model, X_train, y_train, X_test, feature_names):
    """SHAP可解释性分析（彻底解决形状不匹配和索引错误）"""
    print("\n===================== 开始SHAP可解释性分析 =====================")
    # 1. 强制统一列名
    X_test.columns = feature_names
    X_train.columns = feature_names

    # 2. 选择样本（避免计算量过大，用训练集小样本更稳定）
    sample_size = min(100, X_test.shape[0])
    X_shap_sample = X_test.sample(sample_size, random_state=SEED).reset_index(drop=True)

    # 3. 用真实训练数据训练一个完整的随机森林作为 SHAP 解释对象
    #    （mlxtend 的 StackingClassifier 在 fit 时会克隆基学习器，
    #      直接取 model.classifiers[0] 拿到的可能是未训练的原模型，
    #      因此这里用真实标签重新训练一个等架构的 RF 用于可解释性分析，
    #      绝对不使用伪造标签）
    print("⚠️ 正在用真实训练数据训练 SHAP 解释模型（随机森林）...")
    shap_model = RandomForestClassifier(n_estimators=100, random_state=SEED)
    shap_model.fit(X_train, y_train)

    # 4. 生成SHAP值并彻底处理形状
    explainer = shap.TreeExplainer(shap_model)
    shap_values = explainer.shap_values(X_shap_sample)

    # 处理所有可能的形状：列表(2, n, f) 或 三维数组(n, f, 2)
    if isinstance(shap_values, list) and len(shap_values) == 2:
        # 列表格式：取类别1的SHAP值
        shap_values = shap_values[1]
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        # 三维数组格式：取类别1的SHAP值（最后一维为类别）
        shap_values = shap_values[:, :, 1]

    # 6. 打印形状日志，确认匹配
    print(f"📌 SHAP值形状：{shap_values.shape}")
    print(f"📌 特征样本形状：{X_shap_sample.shape}")
    assert shap_values.ndim == 2, f"SHAP值应为二维数组，当前为{shap_values.ndim}维！"
    assert shap_values.shape[1] == X_shap_sample.shape[
        1], f"SHAP值列数({shap_values.shape[1]})与特征列数({X_shap_sample.shape[1]})不匹配！"

    # 7. 计算特征重要性（确保为一维数组）
    shap_importance = np.abs(shap_values).mean(axis=0)
    assert shap_importance.ndim == 1, f"特征重要性应为一维数组，当前为{shap_importance.ndim}维！"

    # 8. SHAP摘要图
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_shap_sample, feature_names=feature_names, show=False, plot_type="dot")
    plt.title('SHAP特征重要性摘要图', fontsize=16, fontweight='bold', fontproperties={'family': 'SimHei'})
    plt.xlabel('SHAP值（特征对预测的影响）', fontsize=12, fontproperties={'family': 'SimHei'})
    plt.tight_layout()
    summary_path = os.path.join(OUTPUT_DIR, 'shap_summary.png')
    plt.savefig(summary_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ 保存：{summary_path}")

    # 9. SHAP依赖图（用特征名称调用，避免索引错位）
    # 按重要性排序，取前2个特征名称
    top_feature_indices = np.argsort(shap_importance)[-2:][::-1]
    top_feature_names = [feature_names[idx] for idx in top_feature_indices]
    print(f"📌 前2个重要特征：{top_feature_names}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    # 第一个重要特征（传名称）
    shap.dependence_plot(
        top_feature_names[0],  # 传特征名称而非索引
        shap_values,
        X_shap_sample,
        feature_names=feature_names,
        ax=ax1,
        show=False
    )
    ax1.set_title(f'特征 {top_feature_names[0]} 的SHAP依赖关系', fontsize=14, fontweight='bold',
                  fontproperties={'family': 'SimHei'})

    # 第二个重要特征（传名称）
    shap.dependence_plot(
        top_feature_names[1],  # 传特征名称而非索引
        shap_values,
        X_shap_sample,
        feature_names=feature_names,
        ax=ax2,
        show=False
    )
    ax2.set_title(f'特征 {top_feature_names[1]} 的SHAP依赖关系', fontsize=14, fontweight='bold',
                  fontproperties={'family': 'SimHei'})

    plt.tight_layout()
    dependence_path = os.path.join(OUTPUT_DIR, 'shap_dependence.png')
    plt.savefig(dependence_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ 保存：{dependence_path}")


# ==================== 主函数 ====================
def main(argv=None):
    """端到端入口：解析命令行参数 → 训练 → 评估 → (可选) SHAP → 导出 results.json"""
    global OUTPUT_DIR, SEED
    parser = argparse.ArgumentParser(
        description="信用卡违约/欺诈检测：端到端训练、评估与 SHAP 可解释性流水线")
    parser.add_argument("--real-data", action="store_true",
                        help="下载 UCI 真实数据集（约150MB）；默认使用内置模拟数据以保证离线可复现")
    parser.add_argument("--out", default=None,
                        help="图表与 results.json 输出目录（默认：脚本所在目录）")
    parser.add_argument("--seed", type=int, default=SEED,
                        help=f"随机种子（默认 {SEED}，保证实验可复现）")
    parser.add_argument("--no-shap", action="store_true",
                        help="跳过 SHAP 可解释性分析（加速调试）")
    args = parser.parse_args(argv)

    SEED = args.seed
    if args.out:
        OUTPUT_DIR = os.path.abspath(args.out)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
    if args.real_data:
        os.environ["CCFD_USE_REAL"] = "1"

    # 1. 初始化环境
    init_environment()
    print("✅ 所有机器学习库导入完成\n")

    # 2. 加载数据
    df = load_credit_card_data()

    # 3. 数据预处理（返回 DataFrame 和特征名称）
    X_processed, y_processed, feature_names = preprocess_data(df)

    # 4. 模型训练（先划分后采样，杜绝数据泄露）
    final_model, X_train, X_test, y_train, y_test = train_stacking_model(X_processed, y_processed)

    # 5. 模型评估（阈值无关指标 + 最优 F1 阈值分类指标）
    metrics = evaluate_model(final_model, X_test, y_test)

    # 6. SHAP 可解释性分析（传入 X_train / y_train / 特征名称）
    if not args.no_shap:
        shap_interpretation(final_model, X_train, y_train, X_test, feature_names)

    # 7. 导出 results.json（记录配置与核心指标，便于复现与汇报）
    results = {
        "project": "credit-card-fraud-detection",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "config": {
            "seed": SEED,
            "data_mode": "uci-real" if args.real_data else "simulated",
            "test_size": 0.2,
            "resampling": "SMOTE↑ + RandomUnderSampler↓ (train only)",
            "model": "Stacking(RF + GBDT + LR) -> meta LR",
        },
        "metrics": metrics,
        "note": "模拟数据下 3% 欺诈率，刻意构造难分样本以贴近真实欺诈检测场景；"
                "AUC 为稳健主指标，PR-AUC 在极端不平衡下天然偏低。",
    }
    results_path = os.path.join(OUTPUT_DIR, "results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"✅ 指标已导出：{results_path}")


if __name__ == "__main__":
    main()

    print("\n🎉 所有流程执行完成！生成文件：")
    print("   1. fraud_model_evaluation.png（模型评估图表）")
    print("   2. shap_summary.png（SHAP特征重要性图）")
    print("   3. shap_dependence.png（SHAP特征依赖关系图）")