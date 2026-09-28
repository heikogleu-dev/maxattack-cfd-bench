#!/usr/bin/env python3
# Heiko 28.09.2026: "wirklich nur die 1-Layer-Zellen vom Solid aus". baue_facetten nimmt JEDE Fluidzelle mit
# einem Wandnachbarn unter den 18 D3Q19-Richtungen (setup.cpp:4163) -- also auch Zellen, die das Solid nur
# ueber eine KANTE beruehren. Hier je Facette aus dem Flag-Feld des Nahfeld-VTK: Zahl der Wandnachbarn ueber
# FLAECHEN (6 Achsrichtungen) und ueber Kanten (12). Lage 1 := mindestens ein Flaechennachbar ist Wand.
# Wand = flags == TYPE_S|TYPE_X (0x41), exakt wie ist_wand im Zensus des Nahfelds (setup.cpp:8662).
# Ausgabe: <export>/facetten_lage.csv (n, achs_wand, kanten_wand, lage1).
import sys, numpy as np, csv
vtk, pers, out = sys.argv[1], sys.argv[2], sys.argv[3]
with open(vtk, 'rb') as f:
    kopf = f.read(4096).decode('latin1')
    dims = [int(v) for v in kopf.split('DIMENSIONS')[1].split('\n')[0].split()]
Nx, Ny, Nz = dims; N = Nx*Ny*Nz
with open(vtk, 'rb') as f:
    d = f.read()
i = d.find(b'SCALARS flags unsigned_char 1'); i = d.find(b'LOOKUP_TABLE default\n', i) + len(b'LOOKUP_TABLE default\n')
fl = np.frombuffer(d, dtype=np.uint8, count=N, offset=i)
del d
wert, anz = np.unique(fl, return_counts=True)
print("Flagwerte im Nahfeld:", dict(zip([hex(int(w)) for w in wert], [int(a) for a in anz])))
wand = (fl == 0x41)
n = np.array([int(r['n']) for r in csv.DictReader(l for l in open(pers) if not l.startswith('#'))], dtype=np.int64)
x = n % Nx; y = (n//Nx) % Ny; z = n//(Nx*Ny)
def w(dx, dy, dz):
    xx, yy, zz = x+dx, y+dy, z+dz
    ok = (xx >= 0)&(xx < Nx)&(yy >= 0)&(yy < Ny)&(zz >= 0)&(zz < Nz)
    r = np.zeros(len(n), dtype=bool); r[ok] = wand[(xx+Nx*(yy+Ny*zz))[ok]]; return r
achs = sum(w(*v).astype(int) for v in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)])
kant = sum(w(*v).astype(int) for v in [(a,b,0) for a in (1,-1) for b in (1,-1)] + [(a,0,c) for a in (1,-1) for c in (1,-1)] + [(0,b,c) for b in (1,-1) for c in (1,-1)])
print(f"Facetten {len(n)}: Lage 1 (>=1 Flaechennachbar Wand) {int((achs>0).sum())}, nur Kante {int(((achs==0)&(kant>0)).sum())}, weder noch {int(((achs==0)&(kant==0)).sum())}")
with open(out, 'w') as f:
    f.write("n,achs_wand,kanten_wand,lage1\n")
    for a, b, c in zip(n, achs, kant): f.write(f"{a},{b},{c},{1 if b>0 else 0}\n")
