import sys
import importlib
import numpy as np
import pandas as pd
import os

# Ensure the latest changes to the module are loaded
import src.shipment_prep
importlib.reload(src.shipment_prep)

from src.shipment_prep import clean_shipments, build_features, FEATURES, PROMISED_HOURS

os.makedirs("outputs", exist_ok=True)
raw = pd.read_csv("/content/data/raw_shipments.csv")
hubs = pd.read_csv("/content/data/hubs.csv")
dup_raw = raw.drop_duplicates()

clean, medians = clean_shipments(raw)
table = build_features(clean, hubs)
print("medians::",medians)

# Verify if 'promised_hours' exists in the columns
#print("Is 'promised_hours' in the table?", 'promised_hours' in table.columns)

#display(table.head())
table["is_delayed"] = (table["delivery_hours"] > table["promised_hours"]).astype(int)
assert len(table)==len(clean), "hub merge changed row count"
table.to_csv("data/modelling_table.csv",index=False)

q1,q3 = table["delivery_hours"].quantile(0.25),table["delivery_hours"].quantile(0.75)
iqr = q3-q1
low,high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
metrics = [("raw_rows", len(raw)),
           ("duplicate_rows_removed", len(raw)-len(dup_raw)),
           ("rows_after_dedup",len(dup_raw)),
           ("missing_distance_km_before",int(dup_raw["distance_km"].isnull().sum())),
           ("missing_parcel_weight_before",int(dup_raw["parcel_weight_kg"].isnull().sum())),
           ("sentinel_weight_rows",int((dup_raw["parcel_weight_kg"]>100).sum())),
           ("missing_declared_value_before",int(dup_raw["declared_value"].isnull().sum())),
           ("missing_payment_mode_before",int(dup_raw["payment_mode"].isnull().sum())),
           ("service_type_spellings_before",int(dup_raw["service_type"].nunique())),
           ("service_type_spellings_after",int(table["service_type"].nunique())),
           ("declared_value_mean_inr",round(table["declared_value_inr"].mean(),2)),
           ("log_declared_value_mean",round(table["log_declared_value"].mean(),3)),
           ("median_distance_km",medians["distance_km"]),
           ("median_parcel_weight_kg",medians["parcel_weight_kg"]),
           ("median_declared_value_inr",medians["declared_value_inr"]),
           ("rows_after_hub_merge",len(table)),
           ("nulls_in_features_and_target",int(table[FEATURES +["delivery_hours","is_delayed"]].isna().sum().sum())),
           ("delayed_count",int(table["is_delayed"].sum())),
           ("delay_rate",float(table["is_delayed"].mean())),
           ("hours_q1", round(q1,2)),
           ("hours_q3",round(q3,2)),
           ("hours_lower_bound", round(low,2)),
           ("hours_upper_bound",round(high,2)),
           ("hours_outlier_count",int(((table["delivery_hours"]<low)|(table["delivery_hours"]>high)).sum())),
           ("declared_value_median_inr",table["declared_value_inr"].median()),
           ("log_declared_value_median",round(table["log_declared_value"].median(),3))]


checks_file = pd.DataFrame(metrics,columns=["metric","value"])
checks_file["value"]=checks_file["value"].astype(float).round(4)
checks_file.to_csv("outputs/part_a_checks.csv",index=False)
table.columns.tolist()
table["distance_band"] = pd.cut(table["distance_km"],bins=[0, 200, 400, 600, 2000],labels=["0-200","201-400","401-600","601+"],right=True)
results = []
for dim in ["service_type", "hub_tier", "distance_band"]:
    # Step 1: Aggregate is_delayed column
    g1 = table.groupby(dim, observed=True)["is_delayed"].agg(
        shipments="size", 
        delayed="sum", 
        delay_rate="mean"
    )
    
    # Step 2: If you need mean_hours later, you can calculate it (but we'll skip for this output)
    # g2 = table.groupby(dim, observed=True)["delivery_hours"].mean()
    # g = g1.join(g2.rename("mean_hours"))
    
    # Step 3: Reset index to convert group column back to regular column
    g = g1.reset_index()
    g = g.rename(columns={dim: "group"})
    
    # Step 4: Add dimension column
    g["dimension"] = dim
    results.append(g)

by_group = pd.concat(results, ignore_index=True)

# Reorder to match desired output
by_group = by_group[["dimension", "group", "shipments", "delayed", "delay_rate"]]

# Round delay_rate
by_group["delay_rate"] = by_group["delay_rate"].round(4)

# Save to CSV
by_group.to_csv("outputs/part_a_delay_by_group.csv", index=False)
numeric_cols = table.select_dtypes(include=['number'])
corr_matrix = numeric_cols.corr().reset_index().round(4)
corr_matrix.to_csv("outputs/part_a_correlations.csv",index=False)
