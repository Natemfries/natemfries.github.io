import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

#Load Dataset 
ds = xr.open_dataset("archive/full_combined_processed.nc") 

df = pd.DataFrame({
    "time": ds.time.values,
    "temp": ds.EXFatemp.mean(dim=("latitude","longitude")).values,
    "qh": ds.EXFaqh.mean(dim=("latitude","longitude")).values,
    "uwind": ds.EXFewind.mean(dim=("latitude","longitude")).values,
    "vwind": ds.EXFnwind.mean(dim=("latitude","longitude")).values,
    "wspeed": ds.EXFwspee.mean(dim=("latitude","longitude")).values,
    "pressure": ds.EXFpress.mean(dim=("latitude","longitude")).values,
})

df = df.set_index("time")
df.head()
#print(df.head()) 

######### 1st Round: Using Specific Heat to Predict Temperature #########

df_reduced = df[["temp","qh"]] 
train = df_reduced.iloc[:250] #Split into training and test data
test = df_reduced.iloc[250:]

X_train = train[["qh"]]
Y_train = train[["temp"]]

X_test = test[["qh"]]
Y_test = test[["temp"]]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

#Linear Regression
lin = LinearRegression()
lin.fit(X_train_scaled, Y_train)
y_pred_lin = lin.predict(X_test_scaled)

#Ridge Regression
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, Y_train)
y_pred_ridge = ridge.predict(X_test_scaled)

#SVR
svr = SVR(kernel='rbf', C=10, epsilon=0.1)
svr.fit(X_train_scaled, Y_train)
y_pred_svr = svr.predict(X_test_scaled)

#Evaluate Models Using RMSE and R^2
def evaluate(name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    print(f"{name:12s}  RMSE = {rmse:.3f}   R² = {r2:.3f}")

evaluate("Linear", Y_test, y_pred_lin)
evaluate("Ridge", Y_test, y_pred_ridge)
evaluate("SVR",   Y_test, y_pred_svr)

#Plots
#Scatter Plot of temp vs time
plt.figure(figsize=(6,4))
plt.plot(df.index, df["temp"])
plt.xlabel("Year")
plt.ylabel("Temperature (K)")
plt.title("Temperature vs Time")
#plt.show()
plt.savefig("time_vs_temp.png", dpi=300)

#Scatter Plot of qh vs temp
plt.figure(figsize=(6,4))
plt.scatter(df_reduced["qh"], df_reduced["temp"], s=12)
plt.xlabel("Specific Humidity")
plt.ylabel("Temperature (K)")
plt.title("Temperature vs Specific Humidity")
#plt.show()
plt.savefig("qh_vs_temp.png", dpi=300)

#Predictions vs Actual
plt.figure(figsize=(12,5))
plt.plot(Y_test.index, Y_test, label="Actual", linewidth=3)
plt.plot(Y_test.index, y_pred_lin, label="Linear")
plt.plot(Y_test.index, y_pred_svr, label="SVR")
plt.legend()
plt.title("Temperature Prediction from Humidity")
plt.ylabel("Temperature (K)")
plt.xticks(rotation=45)
plt.tight_layout()
#plt.show()
plt.savefig("qh_models.png", dpi=300)

######### Second Round: Pressure and QH to Predict Temperature #########
df_two = df[["temp", "qh", "pressure"]]

train_two = df_two.iloc[:250] #split into training and test
test_two  = df_two.iloc[250:]

X_train = train_two[["qh", "pressure"]]
Y_train = train_two["temp"]

X_test  = test_two[["qh", "pressure"]]
Y_test  = test_two["temp"]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

#Linear Regression
lin2 = LinearRegression()
lin2.fit(X_train_scaled, Y_train)
y_pred_lin2 = lin2.predict(X_test_scaled)

#Ridge Regression
ridge2 = Ridge(alpha=1.0)
ridge2.fit(X_train_scaled, Y_train)
y_pred_ridge2 = ridge2.predict(X_test_scaled)

#SVR
svr2 = SVR(kernel='rbf', C=10, epsilon=0.1)
svr2.fit(X_train_scaled, Y_train)
y_pred_svr2 = svr2.predict(X_test_scaled)

#Evaluate Performance Using RMSE and R^2
evaluate("Linear-2feat", Y_test, y_pred_lin2)
evaluate("Ridge-2feat", Y_test, y_pred_ridge2)
evaluate("SVR-2feat",   Y_test, y_pred_svr2)

#Scatter Plot of qh vs temp
plt.figure(figsize=(6,4))
plt.scatter(df_two["pressure"] / 10000, df_two["temp"], s=12)
plt.xlabel(r"Pressure (x $10^4$ Pa)")
plt.ylabel("Temperature (K)")
plt.title("Temperature vs Pressure")
#plt.show()
plt.savefig("p_vs_temp.png", dpi=300)

#Plot Predicted Temperature
plt.figure(figsize=(14,5))
plt.plot(Y_test.index, Y_test, label="Actual", linewidth=3)
plt.plot(Y_test.index, y_pred_lin2, label="Linear")
plt.plot(Y_test.index, y_pred_svr2, label="SVR")
plt.title("Temperature Prediction Using Humidity + Pressure")
plt.ylabel("Temperature (K)")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
#plt.show()
plt.savefig("2feat_model.png", dpi=300)

######### Phase Folding to Look at Cycle Throughout the Year #########
df["month"] = df.index.month #Organize by month
df["phase"] = (df["month"] - 1) / 12 #Use a period of 12 months

#Plots
plt.figure(figsize=(8,5))
plt.scatter(df["month"], df["temp"], s=15, alpha=0.7)
plt.xlabel("Month")
plt.ylabel("Temperature (K)")
plt.title("Phase-Folded Annual Cycle of Global Mean Temperature")
plt.xticks(range(1,13))
#plt.show()
plt.savefig("phase_folded_temp.png", dpi=300)

plt.figure(figsize=(8,5))
plt.scatter(df["month"], df["qh"], s=15, alpha=0.7)
plt.xlabel("Month")
plt.ylabel("Specific Humidity (kg/kg)")
plt.title("Phase-Folded Annual Cycle of Global Mean Humidity")
plt.xticks(range(1,13))
#plt.show()
plt.savefig("phase_folded_qh.png", dpi=300)


df["phase"] = df.index.month
X_phase = df[["phase", "qh"]]
y_full  = df["temp"]
# 80% train, 20% test (chronological)
split = int(0.8 * len(df))

X_train_phase = X_phase.iloc[:split]
X_test_phase  = X_phase.iloc[split:]

y_train_full  = y_full.iloc[:split]
y_test_full   = y_full.iloc[split:]

X_train_scaled = scaler.fit_transform(X_train_phase)
X_test_scaled  = scaler.transform(X_test_phase)

model_phase = LinearRegression()
model_phase.fit(X_train_scaled, y_train_full)

y_pred_phase = model_phase.predict(X_test_scaled)

evaluate("Phase-fold model", y_test_full, y_pred_phase)

plt.figure(figsize=(14,5))
plt.plot(y_test_full.index, y_test_full, label="Actual", linewidth=3)
plt.plot(y_test_full.index, y_pred_phase, label="Phase-Fold Model")
plt.legend()
plt.title("Phase-Folded Model")
plt.ylabel("Temperature (K)")
plt.xticks(rotation=45)
#plt.show()
plt.savefig("phase_folded_model.png", dpi=300)

######### Model Performance Table #########
models = ["Linear(qh)", "Ridge(qh)", "SVR(qh)",
          "Linear(qh+P)", "Ridge(qh+P)", "SVR(qh+P)",
          "Phase-fold"]

rmse = [0.561, 0.559, 0.661,
        0.611, 0.607, 0.602,
        0.509]

r2 = [0.638, 0.641, 0.498,
      0.572, 0.577, 0.584,
      0.698]

fig, ax = plt.subplots(figsize=(10, 3))
ax.axis('off')

table_data = [["Model", "RMSE", "R²"]]
for m, r, s in zip(models, rmse, r2):
    table_data.append([m, f"{r:.3f}", f"{s:.3f}"])

table = ax.table(cellText=table_data, loc='center', cellLoc='center')

table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, 2)

plt.title("Model Performance", pad=20)
plt.tight_layout()
#plt.show()
plt.savefig("model_evals.png", dpi=300)