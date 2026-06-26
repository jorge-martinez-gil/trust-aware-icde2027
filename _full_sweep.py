import json, os
from trust_aware.federated.testbed import load_testbed
from trust_aware.federated.experiment import run_one, aggregate
tb=load_testbed()
seeds=[7,11,13]; counts=[0,2,4,6,8]
out={"seeds":seeds,"counts":counts,"by_seed":{}}
for s in seeds:
    out["by_seed"][s]={}
    for n in counts:
        res=run_one(tb,s,n,1500)
        out["by_seed"][s][n]=aggregate(res["records"])
        print(f"seed {s} n {n} done",flush=True)
d="results/federated"; os.makedirs(d,exist_ok=True)
json.dump(out,open(os.path.join(d,"sweep_multiseed.json"),"w"))
print("SAVED")
