import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore", message="kurtosistest only valid for n>=20", category=UserWarning)

data = pd.read_excel(r"C:/Users/....xlsx")
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
print("The regression equation for the TSLM model is：")
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
fig, ax = plt.subplots(2, 1, figsize=(10, 8))

# Plot a line graph of the TSLM model with the actual data together.
ax[0].plot(df['year'], df['Y'], marker='o', linestyle='-', label='raw data')
ax[0].plot(df['year'], model.fittedvalues, marker='o', linestyle='--', label='TSLM Model')
ax[0].set_xlabel('Year')
ax[0].set_ylabel('Sales')
ax[0].set_title('TSLM model and actual data')
ax[0].legend()

# Mapping of data 10 years after projections
ax[1].plot(df['year'], df['Y'], marker='o', linestyle='-', label='raw data')
ax[1].plot(future_years, pred_values, marker='o', linestyle='--', label='Predicted results')
ax[1].set_xlabel('Year')
ax[1].set_ylabel('Sales')
ax[1].set_title('Forecasting sales figures ten years from now')
ax[1].legend()

plt.tight_layout()
plt.show()