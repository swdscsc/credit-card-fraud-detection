# Model Card · 信用卡违约 / 欺诈检测 Stacking 模型

> 按 Model Cards（Mitchell et al., 2019）推荐结构编写。
> 一句话：**这是什么模型、能用在哪儿、效果如何、不能用在哪儿。**

---

## 1. 模型详情 · Model Details

| 项目 | 内容 |
|------|------|
| 模型类型 | Stacking 集成：RandomForest + GradientBoosting + LogisticRegression → 元学习器 LogisticRegression（`use_probas=True`） |
| 开发者 | [沈炜丹](https://github.com/swdscsc) |
| 仓库 | [credit-card-fraud-detection](https://github.com/swdscsc/credit-card-fraud-detection) |
| 许可证 | MIT |
| 训练框架 | scikit-learn 1.9 · imbalanced-learn 0.14 · mlxtend 0.25 |
| 随机种子 | `SEED=42`（`--seed` 可配置，全流程可复现） |

---

## 2. 预期用途 · Intended Use

- ✅ **适用**：学习与演示信用卡风控建模的完整流程——特征工程、不平衡处理、集成建模、阈值优化、SHAP 可解释性。
- ✅ **适用**：作为结构化基线，迁移到同类"少数类识别"问题（逾期预警、异常登录、反洗钱初筛等）。
- ❌ **禁止**：直接用于真实持卡人的授信、拦截或定价决策。生产部署需以机构自有合规数据重新训练、校准并通过风控评审。

---

## 3. 训练与评估数据 · Data

| 口径 | 说明 | 用途 |
|------|------|------|
| 内置模拟数据（默认） | 20,000 样本、3% 欺诈率、6 个弱信号特征，刻意构造高重叠难分样本 | 离线复现、方法论演示 |
| UCI *Default of Credit Card Clients*（`--real-data`） | 约 30,000 真实样本 | 贴近真实分布的验证 |

关键设计：**先分层切分、后仅在训练集做 SMOTE↑+欠采样↓**；测试集保持原始不平衡分布，杜绝数据泄露。

---

## 4. 评估指标与结果 · Metrics

主指标为 **AUC**（阈值无关、不平衡场景稳健）；分类指标报告在**使 F1 最大的决策阈值**下。

### 4.1 Demo 运行（模拟数据，seed=42，与 `results.json` 一致）

| 指标 | 数值 |
|------|------|
| AUC | **0.857** |
| PR-AUC | 0.191 |
| 最优 F1 阈值 | 0.153 |
| Precision / Recall / F1 | 0.224 / 0.383 / 0.283 |

### 4.2 真实数据（UCI）

测试准确率 ≈ 0.98、少数类 Recall ≈ 0.96；Stacking 较单基模型 AUC 提升约 3%–9%。

### 4.3 指标解读

- 极端不平衡下 PR-AUC 天然偏低，**不代表模型失效**；业务通常按"召回优先"再下调阈值。
- 召回 0.38 为诚实口径下限；真实部署会配合人工复核放大召回。

---

## 5. 可解释性 · Explainability

- SHAP（TreeExplainer，解释对象为用真实训练标签重训的等价随机森林，**不使用伪造标签**）。
- 产出：特征重要性摘要图 + Top-2 特征依赖图（`shap_summary.png` / `shap_dependence.png`）。
- 结论示例：摘要图高亮 V6、V2 等注入的信号特征，与数据构造一致，说明模型确实"学到了信号"。

---

## 6. 伦理考量 · Ethical Considerations

- 训练/演示数据为**匿名化或合成数据**，不含任何真实持卡人 PII。
- 模型输出是**风险概率**，不是对人的定性结论；落地需人工复核、申诉通道与合规审查。
- 注意特征公平性：真实场景中不得使用与受保护属性强相关的代理特征。

---

## 7. 局限与维护 · Limitations & Maintenance

1. 模拟数据分布与真实交易有差距；迁移真实数据必须重新做特征校准与阈值标定。
2. 欺诈模式随时间漂移（concept drift），生产环境需定期重训与监控 PR-AUC/召回。
3. 阈值 0.153 依赖当前评估集；换数据后应重新按业务目标（召回/精确权衡）标定。
