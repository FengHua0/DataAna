import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, log_loss, roc_curve, auc

def data_train(X, y):
    # 将 y 转换为 NumPy 数组并展平
    y = y.values.ravel()  # 使用 .values 将 DataFrame 转为 NumPy 数组

    # 划分训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 构建 GradientBoostingClassifier 模型
    gbdt = GradientBoostingClassifier()

    # 训练模型
    gbdt.fit(x_train, y_train)

    # 返回各个特征的重要性
    importances = gbdt.feature_importances_

    # 将特征重要性以图表形式可视化显示
    importances_df = pd.DataFrame(importances, index=X.columns, columns=['importance'])
    importances_df = importances_df.sort_values(by='importance', ascending=False)
    print(importances_df)

    # 画柱状图
    importances_df.plot(kind='bar', figsize=(12, 8))
    plt.xlabel('数据特征')
    plt.ylabel('特征重要性')
    plt.title('特征重要性分析')
    plt.tight_layout()
    plt.show()

    # 在测试集上进行预测
    y_pred = gbdt.predict(x_test)
    print("梯度提升决策树准确度:", gbdt.score(x_test, y_test))
    print("其他指标：\n", classification_report(y_test, y_pred))

    # 超参数调优
    parameters = {
        'n_estimators': [50, 100, 150],
        'learning_rate': [0.1, 0.05, 0.01],
        'max_depth': [3, 4, 5]
    }

    grid_search = GridSearchCV(gbdt, parameters, cv=5)
    grid_search.fit(x_train, y_train)

    # 获取最佳参数
    best_params = grid_search.best_params_
    print('Best Parameters Found:', best_params)

    # 使用最佳参数训练最终模型
    best_gbdt = GradientBoostingClassifier(**best_params)
    best_gbdt.fit(x_train, y_train)
    y_pred = best_gbdt.predict(x_test)
    test_score = best_gbdt.score(x_test, y_test)
    print("最优参数下的梯度提升决策树准确度:", test_score)
    print("其他指标：\n", classification_report(y_test, y_pred))

    # 记录每个阶段的损失
    train_loss = []
    for y_pred_prob in best_gbdt.staged_predict_proba(x_train):
        train_loss.append(log_loss(y_train, y_pred_prob))

    # 绘制训练损失
    plt.figure(figsize=(12, 6))
    plt.plot(train_loss, label='Training Loss')
    plt.xlabel('Number of Estimators')
    plt.ylabel('Log Loss')
    plt.title('Training Loss per Estimator (Best Model)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 绘制最佳模型的ROC曲线
    y_pred_prob = best_gbdt.predict_proba(x_test)[:, 1]  # 取第二列作为正类的概率
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve (Best Model)')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
