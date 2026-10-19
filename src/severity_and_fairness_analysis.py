"""
Severity-score comparison and fairness analysis for the locked Z2_tuned model.
Run ON THE PROJECT SERVER from ~/fyp_project after the DuckDB build and the two exports:
    python severity_and_fairness_analysis.py
Writes AGGREGATE CSVs only (no stay-level rows) to outputs/severity_fairness/.
"""
import os, numpy as np, pandas as pd
from scipy import stats
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = os.path.expanduser("~/fyp_project")
EICU = os.path.join(ROOT, "external_validation_data/eicu-collaborative-research-database-2.0/eicu-collaborative-research-database-2.0")
P = dict(
    z2_test=os.path.join(ROOT, "outputs/model_z_family/Z2_tuned_test_predictions.csv"),
    scores=os.path.join(ROOT, "first_day_scores.csv"),
    demo=os.path.join(ROOT, "stay_demographics.csv"),
    eicu_z2=os.path.join(ROOT, "outputs/external_eicu_z2/z2_tuned_eicu_predictions_locked_recipe.csv"),
    eicu_apache=os.path.join(EICU, "apachePatientResult.csv.gz"),
    eicu_patient=os.path.join(EICU, "patient.csv.gz"),
    out=os.path.join(ROOT, "outputs/severity_fairness"),
)
B, SEED, MIN_N, MIN_EVENTS = 1000, 20261003, 50, 10
os.makedirs(P["out"], exist_ok=True)
rng = np.random.default_rng(SEED)
TASKS = ["mortality_24h", "ventilation_onset_24h", "vasopressor_onset_24h", "readmission_30d_24h", "deterioration_24h"]

# ---------------- metrics
def ece(y, p, bins=10):
    idx = np.minimum((p * bins).astype(int), bins - 1)
    return sum((idx == b).mean() * abs(y[idx == b].mean() - p[idx == b].mean()) for b in range(bins) if (idx == b).any())

def calib(y, p):
    import statsmodels.api as sm
    p = np.clip(p, 1e-6, 1 - 1e-6); lp = np.log(p / (1 - p))
    slope = sm.GLM(y, sm.add_constant(lp), family=sm.families.Binomial()).fit().params[1]
    inter = sm.GLM(y, np.ones((len(y), 1)), family=sm.families.Binomial(), offset=lp).fit().params[0]
    return slope, inter

def boot(y, s, fn):
    v = []
    for _ in range(B):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i): v.append(fn(y[i], s[i]))
    return np.percentile(v, [2.5, 97.5])

def fast_delong(y, s1, s2):
    """Paired DeLong (Sun & Xu 2014). Returns (auc1-auc2, p)."""
    y = np.asarray(y).astype(int); m, n = y.sum(), (1 - y).sum()
    order = np.argsort(-y, kind="stable"); X = np.vstack([s1, s2])[:, order]
    pos, neg = X[:, :m], X[:, m:]
    tx = np.array([rankdata(r) for r in pos]); ty = np.array([rankdata(r) for r in neg])
    tz = np.array([rankdata(r) for r in X])
    aucs = tz[:, :m].sum(1) / m / n - (m + 1) / (2 * n)
    v01 = (tz[:, :m] - tx) / n; v10 = 1 - (tz[:, m:] - ty) / m
    S = np.cov(v01) / m + np.cov(v10) / n
    d = aucs[0] - aucs[1]; var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    return d, 2 * stats.norm.sf(abs(d) / np.sqrt(var))

def summary(y, score, prob=None):
    r = dict(n=len(y), events=int(y.sum()), prevalence=round(y.mean(), 4),
             auroc=roc_auc_score(y, score), auprc=average_precision_score(y, score))
    r["auroc_lo"], r["auroc_hi"] = boot(y, score, roc_auc_score)
    r["auprc_lo"], r["auprc_hi"] = boot(y, score, average_precision_score)
    if prob is not None:
        r["brier"] = np.mean((prob - y) ** 2); r["ece"] = ece(y, prob)
        r["cal_slope"], r["cal_intercept"] = calib(y, prob)
    return r

# ---------------- load MIMIC-IV
z2 = pd.read_csv(P["z2_test"])
z2 = z2[(z2.validity.astype(str).str.lower() == "true") & (z2.target >= 0)].copy()
z2["y"] = z2.target.astype(int)
# derived deterioration (noisy-OR of the three acute components; valid only where all three are valid)
_w = z2[z2.task.isin(["mortality_24h", "ventilation_onset_24h", "vasopressor_onset_24h"])].pivot_table(
    index="stay_id", columns="task", values=["y", "probability_mean"], aggfunc="first").dropna()
if len(_w) and "deterioration_24h" not in set(z2.task):
    _p = 1 - (1 - _w["probability_mean"]).prod(axis=1)
    _det = pd.DataFrame({"stay_id": _w.index, "task": "deterioration_24h",
                         "y": _w["y"].max(axis=1).astype(int).values,
                         "prediction": _p.values, "probability_mean": _p.values})
    z2 = pd.concat([z2, _det], ignore_index=True)
sc = pd.read_csv(P["scores"]); demo = pd.read_csv(P["demo"]).drop_duplicates("stay_id")
print("test rows per task:", z2.task.value_counts().to_dict())

# ---------------- 1. Z2 vs SOFA / OASIS / APS-III / SAPS-II (MIMIC-IV test, same stays)
rows = []
SC = [("SOFA", "sofa", None), ("OASIS", "oasis", "oasis_prob"), ("APS-III", "apsiii", "apsiii_prob"), ("SAPS-II", "sapsii", "sapsii_prob")]
for task in ["mortality_24h", "deterioration_24h", "ventilation_onset_24h", "vasopressor_onset_24h"]:
    d = z2[z2.task == task].merge(sc, on="stay_id", how="inner").dropna(subset=[c for _, c, _ in SC])
    y = d.y.values
    r = summary(y, d.prediction.values, d.probability_mean.values); r.update(task=task, model="Z2_tuned"); rows.append(r)
    for name, col, pcol in SC:
        r = summary(y, d[col].values.astype(float), d[pcol].values if (pcol and task == "mortality_24h") else None)
        r["delta_auroc_z2_minus_score"], r["delong_p"] = fast_delong(y, d.prediction.values, d[col].values.astype(float))
        r.update(task=task, model=name); rows.append(r)
    print(task, "common stays:", len(d))
pd.DataFrame(rows).to_csv(os.path.join(P["out"], "1_severity_scores_mimiciv.csv"), index=False)

# ---------------- 2. eICU: Z2 vs APACHE IVa (hospital mortality)
e = pd.read_csv(P["eicu_z2"])
pt = pd.read_csv(P["eicu_patient"], usecols=["patientunitstayid", "hospitaldischargestatus"])
pt = pt[pt.hospitaldischargestatus.isin(["Alive", "Expired"])]
pt["y"] = (pt.hospitaldischargestatus == "Expired").astype(int)
ap = pd.read_csv(P["eicu_apache"], usecols=["patientunitstayid", "apacheversion", "predictedhospitalmortality"])
ap = ap[(ap.apacheversion == "IVa") & (ap.predictedhospitalmortality >= 0)].drop_duplicates("patientunitstayid")
d = e.merge(pt, on="patientunitstayid").merge(ap, on="patientunitstayid")
y = d.y.values; er = []
r = summary(y, d.mortality_24h_prediction.values, d.mortality_24h_probability_mean.values); r.update(model="Z2_tuned (frozen)"); er.append(r)
r = summary(y, d.predictedhospitalmortality.values, d.predictedhospitalmortality.values.astype(float)); r.update(model="APACHE IVa")
r["delta_auroc_z2_minus_score"], r["delong_p"] = fast_delong(y, d.mortality_24h_prediction.values, d.predictedhospitalmortality.values)
er.append(r)
pd.DataFrame(er).to_csv(os.path.join(P["out"], "2_eicu_z2_vs_apache_iva.csv"), index=False)
print("eICU stays with Z2, label and APACHE IVa:", len(d), "of", len(e))

# ---------------- 3. fairness: Z2 by sex, age, race, insurance (MIMIC-IV test)
dm = demo.copy()
dm["Sex"] = dm.gender.map({"F": "Female", "M": "Male"})
dm["Age"] = pd.cut(dm.anchor_age, [17, 44, 64, 79, 200], labels=["18-44", "45-64", "65-79", "80+"]).astype(str)
rc = dm.race.fillna("").str.upper()
dm["Race"] = np.select([rc.str.contains("WHITE"), rc.str.contains("BLACK"), rc.str.contains("HISPANIC|LATINO"),
                        rc.str.contains("ASIAN")], ["White", "Black", "Hispanic/Latino", "Asian"], "Other/unknown")
dm["Insurance"] = dm.insurance.fillna("Other").replace({"No charge": "Other"})
fr = []
for task in TASKS:
    d = z2[z2.task == task].merge(dm[["stay_id", "Sex", "Age", "Race", "Insurance"]], on="stay_id", how="left")
    r = summary(d.y.values, d.prediction.values, d.probability_mean.values); r.update(task=task, attribute="All", group="All"); fr.append(r)
    for attr in ["Sex", "Age", "Race", "Insurance"]:
        for g, dd in d.groupby(attr):
            y = dd.y.values
            if len(dd) < MIN_N or y.sum() < MIN_EVENTS or y.sum() == len(y): continue
            r = summary(y, dd.prediction.values, dd.probability_mean.values); r.update(task=task, attribute=attr, group=g); fr.append(r)
f = pd.DataFrame(fr); f.to_csv(os.path.join(P["out"], "3_fairness_by_group.csv"), index=False)
g = f[f.attribute != "All"].groupby(["task", "attribute"]).agg(
    auroc_min=("auroc", "min"), auroc_max=("auroc", "max"), ece_max=("ece", "max"),
    slope_min=("cal_slope", "min"), slope_max=("cal_slope", "max"),
    int_min=("cal_intercept", "min"), int_max=("cal_intercept", "max")).reset_index()
g["auroc_gap"] = g.auroc_max - g.auroc_min
g.to_csv(os.path.join(P["out"], "4_fairness_gaps.csv"), index=False)
print("Done ->", P["out"])
