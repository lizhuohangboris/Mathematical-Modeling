from prettytable import PrettyTable

# Create PrettyTable object
table = PrettyTable()

# Add column names
table.field_names = ["Variable", "States", "Marginal"]

# Add data rows
data = [
    ["Gender", "female", 0.649396],
    ["", "male", 0.350604],
    ["Age Level", "1", 0.0281861],
    ["", "2", 0.119536],
    ["", "3", 0.163149],
    ["", "4", 0.249454],
    ["", "5", 0.256052],
    ["", "6", 0.183622],
    ["BMI Level", "normal", 0.269733],
    ["", "overweight", 0.357396],
    ["", "obese", 0.199922],
    ["", "severely obese", 0.172949],
    ["AP Level", "Yes", 0.794402],
     ["", "No", 0.205598],
    # ["Cholesterol", "1", 0.744433],
    # ["", "2", 0.138165],
    # ["", "3", 0.117402],
    # ["Gluc", "1", 0.849852],
    # ["", "2", 0.072527],
    # ["", "3", 0.0776209],
    # ["Smoke", "0", 0.911949],
    # ["", "1", 0.0880512],
    # ["Alco", "0", 0.94683],
    # ["", "1", 0.0531703],
    # ["Active", "0", 0.195653],
    # ["", "1", 0.804347],
    # ["Cardio", "0", 0.9524],
    # ["", "1", 0.0476],
]

# Add rows to the table
for row in data:
    table.add_row(row)

# Print the table
print(table)
