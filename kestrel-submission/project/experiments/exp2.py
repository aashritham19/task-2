import pandas as pd, numpy as np, warnings; warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, average_precision_score
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from returns_risk.features import *
cu=pd.read_csv("data/customers.csv"); pr=pd.read_csv("data/products.csv")
tr=pd.read_csv("data/train.csv").drop_duplicates("order_id")
F=build_features(tr,cu,pr); y=F.returned.values; t=F.order_placed_at
a=(t<"2026-04-01").values; b=~a
# leaky version
L=F.copy(); L["ev"]=tr.set_index("order_id").loc[F.order_id,"last_service_event_type"].values; L["pk"]=tr.set_index("order_id").loc[F.order_id,"pickup_scheduled_at"].notna().astype(int).values
ct=ColumnTransformer([("c",OneHotEncoder(handle_unknown="ignore",min_frequency=30),CAT_COLS+["ev"]),("n",StandardScaler(),NUM_COLS+["pk"])])
m=make_pipeline(ct,LogisticRegression(C=0.3,max_iter=3000)).fit(L[a],y[a]); p=m.predict_proba(L[b])[:,1]
print("LEAKY acc",np.mean((p>=.5)==y[b]),"AUC",roc_auc_score(y[b],p))
# C sweep, feature subsets
for C in [0.03,0.1,0.3,1]:
    ct=ColumnTransformer([("c",OneHotEncoder(handle_unknown="ignore",min_frequency=30),CAT_COLS),("n",StandardScaler(),NUM_COLS)])
    m=make_pipeline(ct,LogisticRegression(C=C,max_iter=3000)).fit(F[a][FEATURES],y[a]); p=m.predict_proba(F[b][FEATURES])[:,1]
    print("LR C",C,round(roc_auc_score(y[b],p),4),round(average_precision_score(y[b],p),4))
# drop noisy features
small=["family","sku","sales_channel","payment_mode","pin2","discount_pct","promised_delivery_days","is_gift_flag","customer_prior_orders","customer_prior_returns","prior_return_rate","shield_flag","calc_value_inr","qty"]
cs=[c for c in small if c in CAT_COLS]; ns=[c for c in small if c in NUM_COLS]
ct=ColumnTransformer([("c",OneHotEncoder(handle_unknown="ignore",min_frequency=30),cs),("n",StandardScaler(),ns)])
m=make_pipeline(ct,LogisticRegression(C=0.3,max_iter=3000)).fit(F[a][small],y[a]); p=m.predict_proba(F[b][small])[:,1]
print("LR small",round(roc_auc_score(y[b],p),4),round(average_precision_score(y[b],p),4))
# coefficients on full LR
ct=ColumnTransformer([("c",OneHotEncoder(handle_unknown="ignore",min_frequency=30),CAT_COLS),("n",StandardScaler(),NUM_COLS)])
m=make_pipeline(ct,LogisticRegression(C=0.3,max_iter=3000)).fit(F[FEATURES],y)
names=m[0].get_feature_names_out(); co=m[1].coef_[0]
for i in np.argsort(-abs(co))[:25]: print(names[i],round(co[i],3))
# hgb shallow
ct=ColumnTransformer([("c",OrdinalEncoder(handle_unknown="use_encoded_value",unknown_value=-1),CAT_COLS)],remainder="passthrough")
for lr_,it,leaf in [(0.03,200,4),(0.03,300,6),(0.05,100,4)]:
    h=make_pipeline(ct,HistGradientBoostingClassifier(learning_rate=lr_,max_iter=it,max_leaf_nodes=leaf,min_samples_leaf=60,l2_regularization=3,categorical_features=list(range(len(CAT_COLS))),random_state=0)).fit(F[a][FEATURES],y[a]); p=h.predict_proba(F[b][FEATURES])[:,1]
    print("hgb",lr_,it,leaf,round(roc_auc_score(y[b],p),4),round(average_precision_score(y[b],p),4))
