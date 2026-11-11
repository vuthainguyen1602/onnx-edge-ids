"""Hunt for a tree-level export/runtime divergence the parity gate catches.

For each model family and converter, train on CICIDS2017, export to ONNX, and
measure three disagreements with ONNX Runtime on the 447,275-row test split,
all with float32 inputs:

  faithful_f32_vs_ort : our float32 traversal of the exported graph vs ORT.
                        Nonzero means ORT and a faithful reading of the same
                        graph disagree -- an ORT or graph-encoding bug, the most
                        interesting case.
  framework_f32scaled_vs_ort : the training framework, fed the graph's own
                        float32 Scaler output, vs ORT. Nonzero means the
                        exported trees route differently from the framework
                        (threshold rounding or similar), independent of the
                        front-end scaler.
  framework_raw_vs_ort : the framework fed the float64 scikit-learn scaler vs
                        ORT (includes the front end); context only.

Any nonzero faithful_f32_vs_ort, or a nonzero framework_f32scaled_vs_ort under
the latest converter, is a candidate worth filing.
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, 'lncs/code'); sys.path.insert(0, 'lncs/code/experiments')
from edgeids.data import load_test_split
from edgeids.onnx_forest import load_onnx_forest, predict as onnx_predict
from exp5_generalization import export_pipeline, ort_outputs

def flips(a,b): return int((np.asarray(a).astype(np.int64)!=np.asarray(b).astype(np.int64)).sum())

def build(seed):
    from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
        HistGradientBoostingClassifier, GradientBoostingClassifier)
    from sklearn.tree import DecisionTreeClassifier
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier
    return {
      "sklearn_rf":   RandomForestClassifier(n_estimators=200, max_depth=15, n_jobs=-1, random_state=seed),
      "sklearn_et":   ExtraTreesClassifier(n_estimators=200, max_depth=15, n_jobs=-1, random_state=seed),
      "sklearn_dt":   DecisionTreeClassifier(max_depth=20, random_state=seed),
      "sklearn_gb":   GradientBoostingClassifier(n_estimators=100, random_state=seed),
      "sklearn_hgb":  HistGradientBoostingClassifier(max_iter=200, max_bins=255, random_state=seed),
      "lightgbm":     LGBMClassifier(n_estimators=200, max_bin=255, random_state=seed, n_jobs=8, verbose=-1),
      "xgboost_hist": XGBClassifier(n_estimators=200, max_depth=8, tree_method="hist", max_bin=256, random_state=seed, n_jobs=8),
    }

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--train",required=True); ap.add_argument("--test",required=True)
    ap.add_argument("--models",default="sklearn_rf,sklearn_et,sklearn_dt,sklearn_gb,sklearn_hgb,lightgbm,xgboost_hist")
    ap.add_argument("--train-rows",type=int,default=400000); ap.add_argument("--out",required=True)
    a=ap.parse_args()
    import skl2onnx, onnxmltools, sklearn
    from sklearn.pipeline import Pipeline; from sklearn.preprocessing import StandardScaler
    x,_,_=load_test_split(a.test); xtr,ytr,_=load_test_split(a.train)
    rng=np.random.default_rng(0)
    if len(xtr)>a.train_rows:
        idx=np.sort(rng.choice(len(xtr),size=a.train_rows,replace=False)); xtr,ytr=xtr[idx],ytr[idx]
    md=os.path.join(os.path.dirname(a.out),"exp15_models"); os.makedirs(md,exist_ok=True)
    rep={"versions":{"skl2onnx":skl2onnx.__version__,"onnxmltools":onnxmltools.__version__,"sklearn":sklearn.__version__},
         "test_rows":int(len(x)),"models":{}}
    for key in a.models.split(","):
        try:
            pipe=Pipeline([("scaler",StandardScaler()),("model",build(0)[key])])
            pipe.fit(xtr.astype(np.float64),ytr)
            onnx_path=os.path.join(md,f"{key}.onnx")
            ops=export_pipeline(pipe,x.shape[1],onnx_path)
            f=load_onnx_forest(onnx_path)
            lab_out,lo=ort_outputs(onnx_path,x)   # lo = label from P>0.5
            l32=onnx_predict(f,x,np.float32)
            z=(x-f.offset.astype(np.float32))*f.scale.astype(np.float32)
            nat_raw=pipe.predict(x.astype(np.float64))
            nat_f32=pipe.named_steps["model"].predict(z.astype(np.float64))
            rep["models"][key]={"ops":ops,"split_modes":sorted(set(int(m) for m in f.mode[~f.is_leaf])),
                "post_transform":f.post_transform,"n_targets":int(f.n_targets),
                "faithful_f32_vs_ort":flips(l32,lo),
                "ort_labelout_vs_probthreshold":flips(lab_out,lo),
                "framework_f32scaled_vs_ort":flips(nat_f32,lo),
                "framework_raw_vs_ort":flips(nat_raw,lo)}
            print(key,rep["models"][key],flush=True)
        except Exception as e:
            rep["models"][key]={"error":f"{type(e).__name__}: {str(e)[:200]}"}; print(key,"ERR",str(e)[:160],flush=True)
    json.dump(rep,open(a.out,"w"),indent=2); print("wrote",a.out)

if __name__=="__main__": main()
