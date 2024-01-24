import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_curve, roc_auc_score, auc
import xgboost as xgb

# 1. Logistic Regression
data_lr = pd.read_csv("C:/Users/92579/Desktop/MATH/2024-MCM/Problem/Problem/output.csv")  # 替换为实际文件路径

X_lr = data_lr.drop('cardio', axis=1)
y_lr = data_lr['cardio']

X_train_lr, X_test_lr, y_train_lr, y_test_lr = train_test_split(X_lr, y_lr, test_size=0.7, random_state=42)

scaler_lr = StandardScaler()
X_train_scaled_lr = scaler_lr.fit_transform(X_train_lr)
X_test_scaled_lr = scaler_lr.transform(X_test_lr)

model_lr = LogisticRegression(random_state=42)
model_lr.fit(X_train_scaled_lr, y_train_lr)

y_probs_lr = model_lr.predict_proba(X_test_scaled_lr)[:, 1]

# 2. Naive Bayes
data_nb = pd.read_csv("C:/Users/92579/Desktop/MATH/2024-MCM/Problem/Problem/cardio_train.csv")

features_nb = ['age', 'height', 'weight', 'ap_hi', 'ap_lo', 'gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']

data_encoded_nb = pd.get_dummies(data_nb[features_nb])
X_nb = pd.concat([data_encoded_nb, data_nb.drop(features_nb, axis=1)], axis=1)
y_nb = data_nb['cardio']

X_train_nb, X_test_nb, y_train_nb, y_test_nb = train_test_split(X_nb, y_nb, test_size=0.2, random_state=42)

nb_model = GaussianNB()
nb_model.fit(X_train_nb, y_train_nb)

y_probs_nb = nb_model.predict_proba(X_test_nb)[:, 1]

# 3. XGBoost
data_xgb = pd.read_csv("C:/Users/92579/Desktop/MATH/2024-MCM/Problem/Problem/output.csv")  # 替换为实际文件路径

X_xgb = data_xgb.drop('cardio', axis=1)
y_xgb = data_xgb['cardio']

X_train_xgb, X_test_xgb, y_train_xgb, y_test_xgb = train_test_split(X_xgb, y_xgb, test_size=0.2, random_state=42)

model_xgb = xgb.XGBClassifier(objective="binary:logistic", random_state=42)
model_xgb.fit(X_train_xgb, y_train_xgb)

y_probs_xgb = model_xgb.predict_proba(X_test_xgb)[:, 1]

# 绘制 ROC 曲线
fpr_lr, tpr_lr, _ = roc_curve(y_test_lr, y_probs_lr)
fpr_nb, tpr_nb, _ = roc_curve(y_test_nb, y_probs_nb)
fpr_xgb, tpr_xgb, _ = roc_curve(y_test_xgb, y_probs_xgb)

roc_auc_lr = auc(fpr_lr, tpr_lr)
roc_auc_nb = auc(fpr_nb, tpr_nb)
roc_auc_xgb = auc(fpr_xgb, tpr_xgb)

plt.figure(figsize=(8, 6))
plt.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC = {roc_auc_lr:.2f})')
plt.plot(fpr_nb, tpr_nb, label=f'Naive Bayes (AUC = {roc_auc_nb:.2f})')
plt.plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC = {roc_auc_xgb:.2f})')

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()
