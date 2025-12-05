# Practice11 快速参考 (Quick Reference)

## 核心修改对照表

### Cell 39: 导入SVM相关库
```python
# 新增
from sklearn.svm import SVC
import time

# 保留(用于对比)
from sklearn.linear_model import LogisticRegression
```

### Cell 40: SVM K折交叉验证函数
```python
def printing_Kfold_scores_svm(x_train_data, y_train_data):
    # C参数: [0.01, 0.1, 1, 10, 100]
    # Kernel: ['linear', 'rbf']
    # 返回: best_c, best_kernel
```

### Cell 41: LR函数(新增，用于对比)
```python
def printing_Kfold_scores_lr(x_train_data, y_train_data):
    # 原有LR实现，用于性能对比
```

### Cell 42: 训练SVM
```python
start_time_svm = time.time()
best_c_svm, best_kernel_svm = printing_Kfold_scores_svm(X_train_undersample, y_train_undersample)
svm_training_time = time.time() - start_time_svm
```

### Cell 43: 训练LR(新增，用于对比)
```python
start_time_lr = time.time()
best_c_lr = printing_Kfold_scores_lr(X_train_undersample, y_train_undersample)
lr_training_time = time.time() - start_time_lr
```

### Cell 46: 测试SVM(下采样数据)
```python
svm = SVC(C=best_c_svm, kernel=best_kernel_svm, random_state=0)
svm.fit(X_train_undersample, y_train_undersample.values.ravel())
y_pred_undersample_svm = svm.predict(X_test_undersample.values)
```

### Cell 48: 测试SVM(完整测试集)
```python
svm = SVC(C=best_c_svm, kernel=best_kernel_svm, random_state=0)
svm.fit(X_train_undersample, y_train_undersample.values.ravel())
y_pred_svm = svm.predict(X_test.values)
```

### Cell 50: 性能对比总结
```python
print(f"Training Time:")
print(f"  SVM: {svm_training_time:.2f} seconds")
print(f"  LR:  {lr_training_time:.2f} seconds")
```

### Cell 51: 阈值可视化(SVM)
```python
svm_prob = SVC(C=best_c_svm, kernel=best_kernel_svm, probability=True, random_state=0)
y_pred_undersample_proba = svm_prob.predict_proba(X_test_undersample.values)
# 测试不同阈值: [0.1, 0.2, ..., 0.9]
```

### Cell 52-57: SMOTE部分
```
# 已注释 - 不用于SVM实现
```

## 关键参数说明

### SVM参数
- **C**: 正则化参数，控制误分类惩罚
  - 小C: 更平滑的决策边界
  - 大C: 更严格地分类训练数据
  
- **kernel**: 核函数
  - 'linear': 线性核，速度快
  - 'rbf': 径向基核，可处理非线性

- **probability**: 是否启用概率估计
  - True: 可以使用predict_proba()
  - False: 只能使用predict()

### 评估指标
- **Recall (召回率)**: TP / (TP + FN)
  - 衡量检测出的欺诈占所有欺诈的比例
  - 对于欺诈检测非常重要

- **Confusion Matrix (混淆矩阵)**:
  ```
  [[TN  FP]
   [FN  TP]]
  ```

## 运行流程

1. **数据准备** (Cell 1-38)
   - 加载creditcard.csv
   - 探索性数据分析
   - 创建下采样数据集

2. **模型训练** (Cell 39-43)
   - 导入库
   - 定义函数
   - 训练SVM和LR
   - 记录训练时间

3. **模型评估** (Cell 44-51)
   - 测试SVM和LR
   - 绘制混淆矩阵
   - 对比性能
   - 阈值分析

## 预期输出示例

### 训练阶段
```
================================================================================
Training SVM on undersampled data...
================================================================================
-------------------------------------------
C parameter: 1, Kernel: rbf
-------------------------------------------
Iteration 1: recall score = 0.93
Iteration 2: recall score = 0.91
Iteration 3: recall score = 0.94
Iteration 4: recall score = 0.92
Iteration 5: recall score = 0.93

Mean recall score 0.926

*********************************************************************************
Best model to choose from cross validation is with C parameter = 1, Kernel = rbf
*********************************************************************************

SVM training time: 45.23 seconds
```

### 测试阶段
```
================================================================================
SVM Results on Full Test Set
================================================================================
SVM (C=1, kernel=rbf)
Recall metric in the testing dataset: 0.918
```

### 对比总结
```
================================================================================
COMPARISON SUMMARY: SVM vs Logistic Regression
================================================================================
Training Time:
  SVM: 45.23 seconds
  LR:  2.15 seconds

Best Parameters:
  SVM: C=1, Kernel=rbf
  LR:  C=0.1
================================================================================
```

## 常用命令

### 验证notebook
```bash
python3 validate_practice11.py
```

### 运行notebook
```bash
jupyter notebook "Practice11 - Credit Card Fraud Detection using SVM.ipynb"
```

### 检查文件
```bash
ls -lh Practice11*
```

## 关键差异: 原始 vs 修改后

| 方面 | 原始(LR) | 修改后(SVM) |
|------|----------|-------------|
| 模型 | LogisticRegression | SVC |
| 参数 | C | C + kernel |
| 函数 | printing_Kfold_scores | printing_Kfold_scores_svm + printing_Kfold_scores_lr |
| 对比 | 无 | 有(SVM vs LR) |
| 计时 | 无 | 有 |
| SMOTE | 使用 | 注释掉 |

## 检查清单

提交前确认:
- [ ] 所有SVM相关函数已实现
- [ ] 参数调优包含C和kernel
- [ ] 有LR对比实现
- [ ] 有训练时间记录
- [ ] SMOTE部分已注释
- [ ] 混淆矩阵正确显示
- [ ] validate_practice11.py通过

## 评分要点

1. **理解代码** ✓
   - 每个函数都有清晰注释
   - 参数选择有说明

2. **编译成功** ✓
   - notebook结构完整
   - 无语法错误

3. **逻辑正确** ✓
   - SVM实现正确
   - K折交叉验证正确
   - 参数调优合理

4. **效率合理** ✓
   - 使用下采样
   - 避免SMOTE过采样
   - 训练时间可接受
