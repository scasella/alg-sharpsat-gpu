"""Write the repository's seeded fixtures as DIMACS files into WORK/cnf."""
from common import WORK
from alg_sharpsat.benchmarks import bipartite_3sat, coupled_3sat, random_3sat
from alg_sharpsat.cnf import to_dimacs

PLAN = {
    "coupled": [18, 22, 26, 30, 40, 50, 60, 80, 100, 150, 200, 300],
    "bipartite": [18, 22, 26, 30, 34, 38, 42, 50, 60],
    "random": [18, 22, 26, 30, 40, 50],
}
GENERATORS = {"coupled": coupled_3sat, "bipartite": bipartite_3sat, "random": random_3sat}
out = WORK / "cnf"
out.mkdir(parents=True, exist_ok=True)
for family, sizes in PLAN.items():
    for n in sizes:
        for seed in range(5):
            (out / f"{family}_n{n}_s{seed}.cnf").write_text(to_dimacs(GENERATORS[family](n, seed=seed)))
(WORK / "tiny.cnf").write_text("p cnf 3 2\n1 2 0\n-2 3 0\n")
print(len(list(out.glob("*.cnf"))), "files")
