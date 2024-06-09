import pandas as pd

# Load prediction results
prediction_results_path = 'D://Normal_tools/Github_desktop/Clone_shop/Mathematical-Modeling/比赛记录/2024 国赛校赛/第二问预测结果_VAR.xlsx'
prediction_data = pd.read_excel(prediction_results_path)

# Calculate the difference to find the reduction amount
prediction_data['二氧化碳排放量减少量（百万吨）'] = prediction_data['二氧化碳排放量（百万吨）'].diff()

# Save the results to a new Excel file
output_diff_path = 'D://Normal_tools/Github_desktop/Clone_shop/Mathematical-Modeling/比赛记录/2024 国赛校赛/第二问预测结果_VAR_差分.xlsx'
prediction_data.to_excel(output_diff_path, index=False)

output_diff_path
