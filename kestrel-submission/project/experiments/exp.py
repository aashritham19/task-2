import pandas as pd, numpy as np, warnings; warnings.filterwarnings("ignore")
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from returns_risk.features import *
cu=pd.read_csv("data/customers.csv"); pr=pd.read_csv("data/products.csv")
tr=pd.read_csv("data/train.csv").drop_duplicates("order_id")
F=build_features(tr,cu,pr); y=F.returned.values; t=F.order_placed_at
def hgb():
    ct=ColumnTransformer([("c",OrdinalEncoder(handle_unknown="use_encoded_value",unknown_value=-1),CAT_COLS)],remainder="passthrough")
    return make_pipeline(ct,HistGradientBoostingClassifier(learning_rate=0.05,max_iter=250,max_leaf_nodes=8,min_samples_leaf=40,l2_regularization=1.0,categorical_features=list(range(len(CAT_COLS))),random_state=0))
def lr():
    ct=ColumnTransformer([("c",OneHotEncoder(handle_unknown="ignore",min_frequency=30),CAT_COLS),("n",StandardScaler(),NUM_COLS)])
    return make_pipeline(ct,LogisticRegression(C=0.3,max_iter=2000))
def report(name,p,yy):
    best=max(((np.mean((p>=th)==yy),th) for th in np.linspace(0.05,0.9,86)))
    print(f"{name:10s} AUC {roc_auc_score(yy,p):.3f} PR-AUC {average_precision_score(yy,p):.3f} brier {brier_score_loss(yy,p):.4f} bestAcc {best[0]:.4f}@{best[1]:.2f} base {1-yy.mean():.4f}")
# rolling-origin: train up to cutoff, test following 3 months
for cut in ["2025-12-31","2026-03-31"]:
    a=t<=cut; b=(t>cut)&(t<=pd.Timestamp(cut)+pd.Timedelta(days=92))
    print("cutoff",cut,a.sum(),b.sum())
    for n,m in [("lr",lr()),("hgb",hgb())]:
        m.fit(F[a][FEATURES],y[a]); report(n,m.predict_proba(F[b][FEATURES])[:,1],y[b])
# last 3 months of train held out as mimic of test
a=t<"2026-04-01"; b=~a
for n,m in [("lr",lr()),("hgb",hgb())]:
    m.fit(F[a][FEATURES],y[a]); p=m.predict_proba(F[b][FEATURES])[:,1]; report(n,p,y[b])
# with leaky features, for the memo
L=F.copy(); 
