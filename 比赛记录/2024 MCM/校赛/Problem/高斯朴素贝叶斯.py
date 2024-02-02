import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
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

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 使用高斯朴素贝叶斯进行训练
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

# 进行预测
predictions = nb_model.predict(X_test)

# 计算准确度
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy}")

# 输出混淆矩阵、分类报告等
conf_matrix = confusion_matrix(y_test, predictions)
class_report = classification_report(y_test, predictions)

print("Confusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)
