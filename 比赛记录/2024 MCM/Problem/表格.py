from prettytable import PrettyTable

# 创建 PrettyTable 对象
table = PrettyTable()

# 添加列名
table.field_names = ["Variable", "Definition", "Levels"]

# 添加数据行
data = [
    ["Gender", "Sex", "{female, male}"],
    ["Age Level", "Age groups", "min-35, 36-40, 41-45, 46-50, 51-55, 56-60, 60-max"],
    ["BMI Level", "Body Mass Index (BMI)", "{[18.5,23.9], (23.9,27.9], (27.9,32.0], (32.0,max]}"],
    ["AP Level", "Blood Pressure", "{Yes, No} (High pressure >= 140 and low pressure >= 90 is Yes, otherwise No)"],
    ["Cholesterol", "Cholesterol levels", "{1 (Normal), 2 (Above normal), 3 (Significantly above normal)}"],
    ["Glucose", "Glucose levels", "{1 (Normal), 2 (Above normal), 3 (Significantly above normal)}"],
    ["Smoke", "Smoking habit", "{0 (Not often), 1 (Often)}"],
    ["Alco", "Alcohol consumption", "{0 (Not often), 1 (Often)}"],
    ["Active", "Physical activity", "{0 (Not often), 1 (Often)}"],
    ["Cardio", "Cardiovascular disease", "{0 (No), 1 (Yes)}"],
    # ["Physiological", "Physiological indicators", "-"],
    # ["Medical", "Medical indicators", "-"],
    # ["Subjective", "Subjective information", "-"]
]

for row in data:
    table.add_row(row)

# 打印表格
print(table)
