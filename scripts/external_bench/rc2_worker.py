"""MAP via MaxSAT: RC2Stratified on hard CNF + soft units with integer-scaled -log weights."""
import json, math, sys, time
from pysat.formula import WCNF
from pysat.examples.rc2 import RC2Stratified

spec = json.load(open(sys.argv[1]))
t = time.perf_counter()
w = WCNF()
for c in spec["clauses"]:
    w.append(c)
for i, k in enumerate(spec["k"]):
    p = k / 1000
    w.append([i + 1], weight=round(-math.log(1 - p) * 1e6))
    w.append([-(i + 1)], weight=round(-math.log(p) * 1e6))
with RC2Stratified(w, solver="g3", adapt=True, exhaust=True, minz=True) as s:
    model = s.compute()
print(json.dumps({"model": model, "seconds": time.perf_counter() - t}))
