import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 加载数据集
data = pd.read_csv("C:/Users/92579/Desktop/MATH/2024-MCM/Problem/Problem/cardio_train.csv")

# 选择用于预测的特征
features = ['age', 'height', 'weight', 'ap_hi', 'ap_lo', 'gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']

# 对类别特征进行独热编码
data_encoded = pd.get_dummies(data[features])

# 将编码后的数据和目标变量整合
X = pd.concat([data_encoded, data.drop(features, axis=1)], axis=1)
y = data['cardio']

# 数据预处理
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 使用支持向量机进行训练
svm_model = SVC(verbose=1)
svm_model.fit(X_train, y_train)

# 进行预测
predictions = svm_model.predict(X_test)

# 计算准确度
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy}")

# 计算混淆矩阵
conf_matrix = confusion_matrix(y_test, predictions)
print("Confusion Matrix:")
print(conf_matrix)

# 计算精确度、召回率和 F1 分数
print("Classification Report:")
print(classification_report(y_test, predictions))

# 使用交叉验证计算准确度
cv_accuracy = cross_val_score(svm_model, X_scaled, y, cv=5, scoring='accuracy')
print("Cross-validated Accuracy:", cv_accuracy.mean())

