import os
import pandas as pd

base = "Metric_Results"
out_dir = os.path.join(base, "Combined")
os.makedirs(out_dir, exist_ok=True)

models = [m for m in os.listdir(base) if os.path.isdir(os.path.join(base, m)) and m != "Combined"]
rows = []
for m in models:
    for h in [24, 48, 96, 168]:
        fp = os.path.join(base, m, f"{m}_{h}")
        if os.path.isfile(fp):
            df = pd.read_csv(fp)
            for _, r in df.iterrows():
                rows.append({"Horizon": h, "Model": m, "Metric": r["Metric"], "Value": r["Average"]})

for h in sorted(set(r["Horizon"] for r in rows)):
    tmp = [r for r in rows if r["Horizon"] == h]
    df = pd.DataFrame(tmp)
    smapes = {}
    for mdl in df["Model"].unique():
        s = df[(df["Model"] == mdl) & (df["Metric"] == "smape")]["Value"].values
        smapes[mdl] = s[0] if len(s) > 0 else float("inf")
    sorted_models = sorted(smapes, key=smapes.get)
    data = []
    for mdl in sorted_models:
        sub = df[df["Model"] == mdl]
        for m in ["smape", "mape", "mase", "mse", "mae"]:
            r = sub[sub["Metric"] == m]
            if not r.empty:
                data.append(r)
    out = pd.concat(data)[["Model", "Metric", "Value"]]
    out.to_csv(os.path.join(out_dir, f"Combined_{h}.csv"), index=False)

all_df = pd.DataFrame(rows)
avg_df = all_df.groupby(["Model", "Metric"], as_index=False)["Value"].mean()
smapes = {}
for mdl in avg_df["Model"].unique():
    s = avg_df[(avg_df["Model"] == mdl) & (avg_df["Metric"] == "smape")]["Value"].values
    smapes[mdl] = s[0] if len(s) > 0 else float("inf")
sorted_models = sorted(smapes, key=smapes.get)

data = []
for mdl in sorted_models:
    sub = avg_df[avg_df["Model"] == mdl]
    for m in ["smape", "mape", "mase", "mse", "mae"]:
        r = sub[sub["Metric"] == m]
        if not r.empty:
            data.append(r)
out = pd.concat(data)[["Model", "Metric", "Value"]]
out.to_csv(os.path.join(out_dir, "Combined_all.csv"), index=False, float_format="%.3f")
