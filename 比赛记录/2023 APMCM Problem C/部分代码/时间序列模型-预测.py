import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore", message="kurtosistest only valid for n>=20", category=UserWarning)

data = pd.read_excel(r"C:\Users\92579\Desktop\MATH\第二题.xlsx")
df = pd.DataFrame(data)

from statsmodels.tsa.ar_model import AutoReg

# Fitting an autoregressive model for X1
X1_model = AutoReg(df['X1'], lags=1).fit()

# Fitting an autoregressive model with X2
X2_model = AutoReg(df['X2'], lags=1).fit()

# Construction of the TSLM model
X = pd.DataFrame({
    'const': 1,
    'X1': df['X1'],
    'X2': df['X2']
})
model = sm.OLS(df['Y'], X).fit()

# Construct the independent variable matrix X
X = pd.DataFrame({
    'const': 1,
    'X1': df['X1'],
    'X2': df['X2']
})

# Construct the vector of dependent variables Y
Y = df['Y']

# Construction of the TSLM model
model = sm.OLS(Y, X).fit()

# Constructing regression formulas
formula = "Y = {:.3f} + {:.3f} * X1 + {:.3f} * X2".format(model.params[0], model.params[1], model.params[2])
print("The regression equation for the TSLM model is:")
print(formula)

# Data projected for the next decade
future_years = range(df['year'].max() + 1, df['year'].max() + 11)
pred_X1 = X1_model.predict(start=len(df), end=len(df) + 9)
pred_X2 = X2_model.predict(start=len(df), end=len(df) + 9)

future_X = pd.DataFrame({
    'const': 1,
    'X1': pred_X1,
    'X2': pred_X2
})
pred_values = model.predict(future_X)

# Report Display
fig, ax = plt.subplots(figsize=(10, 5))

# Plot a line graph of the TSLM model with the actual data together.
ax.plot(df['year'], df['Y'], marker='o', linestyle='-', label='raw data')
ax.plot(df['year'], model.fittedvalues, marker='o', linestyle='--', label='Fitting data')
ax.plot(list(future_years), list(pred_values), marker='o', linestyle='--', label='Predictive data')
ax.set_xlabel('Year')
ax.set_ylabel('Sales')
ax.set_title('TSLM model: comparison of actual, fitted and predicted data')
ax.legend()

plt.show()

# Show forecast results
future_df = pd.DataFrame({
    'year': np.append(df['year'], future_years),
    'predicted_Y': np.append(df['Y'], pred_values)
})
print("Projected data for the next decade:")
print(future_df)