#part_c_classification.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split,RandomizedSearchCV, GridSearchCV 
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest,f_classif
import json
from src.shipment_prep import FEATURES, NOISE_COLUMNS

model_table = pd.read_csv("/content/data/modelling_table.csv")
model_table.head()
model_table.columns.to_list()
x= model_table[FEATURES]
y= model_table["is_delayed"]

#Split into training, validation and test
x_train,x_temp,y_train,y_temp=train_test_split(x,y,test_size = 0.3,random_state=42,stratify=y)
x_val,x_test,y_val,y_test=train_test_split(x_temp,y_temp,test_size=0.5,random_state=42,stratify=y_temp)

print("sizes", len(x_train), len(x_val), len(x_test), "| delayed", y_train.sum(), y_val.sum(), y_test.sum())

def confusion_matrix_rows(y_true,pred):
  tn, fp, fn, tp = confusion_matrix(y_true,pred,labels=[0,1]).ravel()
  return {"TN":int(tn),"FP":int(fp),"FN":int(fn),"TP":int(tp),
          "accuracy":accuracy_score(y_true,pred),
          "precision":precision_score(y_true,pred,zero_division=0),
          "recall":recall_score(y_true,pred,zero_division=0),
          "f1_score":f1_score(y_true,pred,zero_division=0)}
#baseline with three classifier models on validation data

model_rows = [{"model":"blind_all_on_time",**confusion_matrix_rows(y_val,np.zeros(len(y_val),dtype=int))}]

logr = Pipeline([("scaler",StandardScaler()),  ("model", LogisticRegression())]).fit(x_train,y_train)
model_rows.append({"model":"logistic_regression",**confusion_matrix_rows(y_val,logr.predict(x_val))})
#display(pd.DataFrame(model_rows))

best_d,best_f1 = None,-1
for d in [2, 3, 4, 5, 6, 8, 10]:
  dtree = DecisionTreeClassifier(criterion="gini", max_depth=d, random_state=42).fit(x_train,y_train)
  current_f1 = f1_score(y_val,dtree.predict(x_val))

  if current_f1 > best_f1 + 1e-12:
    best_d,best_f1 = d,current_f1

tree = DecisionTreeClassifier(criterion="gini", max_depth=best_d, random_state=42).fit(x_train,y_train)
model_rows.append({"model":"DecisionTreeClassifier",**confusion_matrix_rows(y_val,tree.predict(x_val))})

rf_default = RandomForestClassifier(n_estimators=100, random_state=42).fit(x_train,y_train)
model_rows.append({"model":"random_forest_default",**confusion_matrix_rows(y_val,rf_default.predict(x_val))})
pd.DataFrame(model_rows).round(4).to_csv("/content/outputs/part_c_models.csv", index=False)

#Feature Ranking
sk_all = SelectKBest(f_classif, k="all").fit(x_train,y_train)
feature_scale = pd.DataFrame({"feature":FEATURES,"f_score":sk_all.scores_,
                              "random_forest_importance":rf_default.feature_importances_})
feature_scale["f_rank"] = feature_scale["f_score"].rank(ascending=False,method="first").astype(int)
feature_scale["is_noise"] = feature_scale["feature"].isin(NOISE_COLUMNS)
feature_scale = feature_scale[["feature", "f_score", "f_rank", "random_forest_importance", "is_noise"]]
feature_scale.round(4).to_csv("/content/outputs/part_c_feature_scores.csv", index=False)
select_rows = []
for k in [5, 7, 10, 12, 17]:
    p = Pipeline([("sel", SelectKBest(f_classif, k=k)), ("rf", RandomForestClassifier(n_estimators=100, random_state=42))])
    p.fit(x_train, y_train)
    select_rows.append({"k": k, "val_f1": f1_score(y_val, p.predict(x_val))})
pd.DataFrame(select_rows).round(4).to_csv("/content/outputs/part_c_selectk.csv", index=False)


# 4. tuning (selector fitted on training rows only; test rows untouched)
K = 7
selector = SelectKBest(f_classif, k=K).fit(x_train, y_train)
xt_train, xt_val = selector.transform(x_train), selector.transform(x_val)
grid = {"n_estimators": [50, 100, 150], "max_depth": [4, 6, 8], "min_samples_leaf": [1, 5, 10]}
gs = GridSearchCV(RandomForestClassifier(random_state=42), grid, cv=5).fit(xt_train, y_train)
rs = RandomizedSearchCV(RandomForestClassifier(random_state=42), grid, n_iter=9, cv=5, random_state=42).fit(xt_train, y_train)
default7 = RandomForestClassifier(random_state=42).fit(xt_train, y_train)

DEFAULT_PARAMS = {"n_estimators": 100, "max_depth": None, "min_samples_leaf": 1}
cands = {
    "grid_search": (gs.best_params_, gs.best_score_, len(gs.cv_results_["params"]), 5, f1_score(y_val, gs.best_estimator_.predict(xt_val))),
    "randomized_search": (rs.best_params_, rs.best_score_, len(rs.cv_results_["params"]), 5, f1_score(y_val, rs.best_estimator_.predict(xt_val))),
    "default_forest_same_7_features": (DEFAULT_PARAMS, np.nan, 1, 0, f1_score(y_val, default7.predict(xt_val))),
}
tuned_rows = []
for name, (params, cv_acc, n_cand, folds, vf1) in cands.items():
    tuned_rows.append({"search": name, "candidates": n_cand, "cv_folds": folds, "total_fits": (n_cand * folds) if folds else 1,
                      "best_params": json.dumps({k: (None if params[k] is None else int(params[k])) for k in sorted(params)}),
                      "best_cv_accuracy": cv_acc, "val_f1": vf1})
tuning = pd.DataFrame(tuned_rows).round(4)
tuning.to_csv("/content/outputs/part_c_tuning.csv", index=False)

# 5. final configuration: highest val_f1 (as written in the CSV); ties -> default, grid, randomized
priority = ["default_forest_same_7_features", "grid_search", "randomized_search"]
best_val = tuning["val_f1"].max()
#print("best_val:",best_val)
chosen = next(n for n in priority if tuning.loc[tuning["search"] == n, "val_f1"].iloc[0] == best_val)
#print("chosen:",cands[chosen][0])
params = cands[chosen][0]
final_model = Pipeline([("selector", SelectKBest(f_classif, k=K)),
                        ("model", RandomForestClassifier(random_state=42, **params))]).fit(x_train, y_train)

 # alert threshold on validation probabilities
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
p_val = final_model.predict_proba(x_val)[:, 1]
th_rows = [{"threshold": t, **confusion_matrix_rows(y_val, (p_val >= t).astype(int))} for t in thresholds]
th = pd.DataFrame(th_rows).round(4)
th.to_csv("/content/outputs/part_c_thresholds.csv", index=False)
display(th)
ok = th.loc[th["recall"] >= 0.80, "threshold"]
threshold = float(ok.max()) if len(ok) else 0.1

#One test evaluation
p_test = final_model.predict_proba(x_test)[:, 1]
pd.DataFrame([{"threshold": threshold, **confusion_matrix_rows(y_test, (p_test >= threshold).astype(int))}]).round(4) \
    .to_csv("outputs/part_c_final_test.csv", index=False)
pd.DataFrame([{"selector_k": K, "n_estimators": params["n_estimators"],
               "max_depth": "unlimited" if params["max_depth"] is None else params["max_depth"],
               "min_samples_leaf": params["min_samples_leaf"], "threshold": threshold,
               "chosen_from": chosen}]).to_csv("outputs/part_c_final_config.csv", index=False)

print("chosen:", chosen, params, "threshold:", threshold)