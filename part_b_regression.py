#part_b_regression.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold,cross_val_score
from sklearn.preprocessing import StandardScaler
from src.shipment_prep import FEATURES
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

model_table = pd.read_csv("/content/data/modelling_table.csv")
model_table.head()
model_table.columns.to_list()
x= model_table[FEATURES]
y= model_table["delivery_hours"]
#Split into training, validation and test
x_train,x_temp,y_train,y_temp=train_test_split(x,y,test_size = 0.3,random_state=42)
x_val,x_test,y_val,y_test=train_test_split(x_temp,y_temp,test_size=0.5,random_state=42)
#Scaling without leakage
#Fit Scaler on training data only
scaler_obj = StandardScaler().fit(x_train)
#Transform all three splits using the same scaler
x_train_scaled = scaler_obj.transform(x_train)
x_val_scaled = scaler_obj.transform(x_val)
x_test_scaled = scaler_obj.transform(x_test)

print("Scaling complete!")
print("x_train:",len(x_train))
print("x_val:",len(x_val))
print("x_test:",len(x_test))
split_check = pd.DataFrame({"metric":["train_rows", "val_rows", "test_rows", "train_scaled_max_abs_mean", "test_scaled_max_abs_mean"],
                            "value":[len(x_train),len(x_val),len(x_test),np.abs(x_train_scaled.mean(axis=0)).max(),np.abs(x_test_scaled.mean(axis=0)).max()]}).round(4).to_csv("/content/outputs/part_b_checks.csv")

def mean_scores(y_true, pred):
    mse = mean_squared_error(y_true, pred)
    return {"MAE": mean_absolute_error(y_true, pred), "MSE": mse, "RMSE": float(np.sqrt(mse)), "R2": r2_score(y_true, pred)}

#Baseline
scores = mean_scores(y_val, np.full(len(y_val), y_train.mean()))
rows = [{"model": "mean_baseline", "split": "val", **{"scores":scores}}]

depth_rows,best_depth,best_val = [],None,np.inf #positive infinity
for d in [1, 2, 3, 4, 6, 8, 12,None]:
  model = DecisionTreeRegressor(max_depth=d, random_state=42)
  model.fit(x_train,y_train)
  y_train_rmse = np.sqrt(mean_squared_error(y_train,model.predict(x_train)))
  y_val_rmse = np.sqrt(mean_squared_error(y_val,model.predict(x_val)))
  depth_rows.append({"max_depth":"unlimited" if d is None else d,"train_rmse":y_train_rmse,"val_rmse":y_val_rmse})
  if y_val_rmse < best_val - 1e-12:           # strict: ties keep the smaller depth
   best_val, best_depth = y_val_rmse, d
pd.DataFrame(depth_rows).round(4).to_csv("/content/outputs/part_b_tree_depth.csv", index=False)
#print("best_val:",best_val)
#print("best_depth:",best_depth)
four_candidates = {
    "linear_regression": Pipeline([("scaler",StandardScaler()),("model",LinearRegression())]),
    "decision_tree":DecisionTreeRegressor(max_depth=best_depth,random_state=42),
    "decision_tree": DecisionTreeRegressor(max_depth=best_depth, random_state=42),
    "random_forest":RandomForestRegressor(n_estimators=100, random_state=42),
    "gradient_boosting":GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
}

val_rmse ={}
for name, m in four_candidates.items():
  m.fit(x_train,y_train)
  s = mean_scores(y_val,m.predict(x_val))
  val_rmse[name] = s["RMSE"]
  rows.append({"model":name,"split":"val",**s})
# 6. learning-rate experiment
lr_rows = []
for lr in [0.01, 0.1, 1.0]:
    g = GradientBoostingRegressor(n_estimators=100, learning_rate=lr, random_state=42)
    g.fit(x_train, y_train)
    p = g.predict(x_val)
    lr_rows.append({"learning_rate": lr, "val_rmse": np.sqrt(mean_squared_error(y_val, p)), "val_r2": r2_score(y_val, p)})
pd.DataFrame(lr_rows).round(4).to_csv("/content/outputs/part_b_gb_learning_rate.csv", index=False)
print("val_rmse::",val_rmse)
#cross validation
rank = sorted(val_rmse,key=val_rmse.get)
print("rank::",rank)
top2 = rank[:2]
print("top2:",top2)
x_train_val,y_train_val = pd.concat([x_train,x_val]), pd.concat([y_train,y_val])
kf= KFold(n_splits=5, shuffle=True, random_state=42)
cv_rows=[]
for model_name in top2:
  r2score=cross_val_score(four_candidates[model_name],x_train_val,y_train_val,cv=kf)
  cv_rows.append({"model":model_name,"meanr2":r2score.mean(),"std_r2":r2score.std()})
pd.DataFrame(cv_rows).round(4).to_csv("/content/outputs/part_b_cv.csv", index=False)

final_model = rank[0]
print("final_model::",final_model)
rows.append({"model": final_model, "split": "test", **mean_scores(y_test, four_candidates[final_model].predict(x_test))})
pd.DataFrame(rows)[["model", "split", "MAE", "MSE", "RMSE", "R2"]].round(4).to_csv("/content/outputs/part_b_metrics.csv", index=False)
print("best depth:", best_depth, "| ranking:", rank, "| final:", final_model)
