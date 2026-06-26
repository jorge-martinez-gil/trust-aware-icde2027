import json, os, sys
from trust_aware.federated.testbed import load_testbed
from trust_aware.federated.experiment import run_one, aggregate
seed=int(sys.argv[1]); tb=load_testbed()
res={n: aggregate(run_one(tb,seed,n,1500)["records"]) for n in [0,2,4,6,8]}
os.makedirs("results/federated",exist_ok=True)
json.dump(res,open(f"results/federated/sweep_agg_{seed}.json","w"))
print("done",seed)
