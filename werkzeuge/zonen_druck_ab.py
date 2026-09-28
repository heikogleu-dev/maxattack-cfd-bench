#!/usr/bin/env python3
"""zonen_druck_ab.py <bezug> <arm> [<arm2> ...] -- WO aendert sich der Abtrieb? (Heiko 28.09.2026, Option 2)

Je Lauf aus den Feldschnappschuessen feld_nah_*ms.vtk (Momentanfelder, hier 451/602/740 ms -> Mittel ueber 3):
  (1) Druckbeitrag zu Fz je Zone:  Fz_zone = - Sum (rho-1)/3 * n_z * A  ueber die LAGE-1-Wandzellen
      (facetten_lage.csv des Bezugs; Normale aus facetten_persistenz.csv; A = 1/max(|n|_max, 1/sqrt3) Voxelflaechen
      je Zelle wie d3q27_zensus.faca). Radband wie cz_rest ausgeschlossen: Zellen mit z*dx < 20 mm.
      Auf die cz_rest-Skala KALIBRIERT ueber den Bezug: k = cz_rest(Bezug, cd_bericht.csv) / Sum_zonen Fz(Bezug).
      Das ist ein PROXY (Zellmittenpruck statt Wanddruck, Momentanfelder); belastbar ist die VERTEILUNG der Differenz,
      nicht die dritte Stelle.
  (2) Rueckstroemung: Anteil der Lage-1-Zellen einer Zone mit u_x < 0 (Anstroemung +x) -- direktes Abloesemass.
Zonen: oberste 0,45 m mit n_z > 0,3 -> Scheibe (n_x < -0,25) / Dach / Heck (n_x > 0,25); sonst oben (n_z > 0,3),
unten (n_z < -0,3), Seite (|n_y| dominant), Front (n_x < 0), Heckflaeche (n_x > 0).
"""
import sys, os, csv, glob
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from zonen_kraft import kopf

def pfad(l): return l if os.path.isdir(l) else f"export/{l}"
bez = pfad(sys.argv[1]); arme = [pfad(a) for a in sys.argv[2:]]
vtks0 = sorted(glob.glob(os.path.join(bez, "feld_nah_*ms.vtk")))
d = kopf(vtks0[0]); Nx, Ny, Nz = d["dims"]; dx = d["spac"][0]
lage = {int(r['n']): int(r['lage1']) for r in csv.DictReader(open(os.path.join("export/s0_f8_nbexp", "facetten_lage.csv")))}
P = [r for r in csv.DictReader(l for l in open(os.path.join(bez, "facetten_persistenz.csv")) if not l.startswith('#'))]
n = np.array([int(r['n']) for r in P], dtype=np.int64)
if not all(int(x) in lage for x in n[:1000]): raise SystemExit("facetten_lage.csv passt nicht zum Gitter des Bezugs")
l1 = np.array([lage[int(x)] for x in n]) == 1
nv = np.array([[float(r['nx']), float(r['ny']), float(r['nz'])] for r in P])
z = n//(Nx*Ny); zg = z.max() - 0.45/dx
A = 1.0/np.maximum(np.abs(nv).max(1), 1.0/np.sqrt(3.0))
rad = (z*dx < 0.020)
nx_, ny_, nz_ = nv[:, 0], nv[:, 1], nv[:, 2]
zone = np.full(n.size, 'Heckflaeche', dtype=object)
zone[nx_ < 0] = 'Front'
zone[(np.abs(ny_) >= np.abs(nx_)) & (np.abs(ny_) >= np.abs(nz_))] = 'Seite'
zone[nz_ < -0.3] = 'unten'
zone[nz_ > 0.3] = 'oben'
top = (z >= zg) & (nz_ > 0.3)
zone[top & (nx_ < -0.25)] = 'Scheibe'
zone[top & (nx_ > 0.25)] = 'Heck'
zone[top & (nx_ >= -0.25) & (nx_ <= 0.25)] = 'Dach'
sel = l1 & ~rad
ZON = ['Scheibe', 'Dach', 'Heck', 'oben', 'Front', 'Heckflaeche', 'Seite', 'unten']

def lauf(p):
    vs = sorted(glob.glob(os.path.join(p, "feld_nah_*ms.vtk")))
    fz = {k: [] for k in ZON}; rueck = {k: [] for k in ZON}
    for v in vs:
        dv = kopf(v)
        if dv["dims"] != (Nx, Ny, Nz): raise SystemExit(f"{v}: anderes Gitter")
        rho = np.memmap(v, dtype=">f4", mode="r", offset=dv["off_rho"], shape=(Nx*Ny*Nz,))
        U = np.memmap(v, dtype=">f4", mode="r", offset=dv["off_u"], shape=(Nx*Ny*Nz, 3))
        r = np.asarray(rho[n]).astype(np.float64); ux = np.asarray(U[n, 0]).astype(np.float64)
        beitrag = -(r-1.0)/3.0*nz_*A
        for k in ZON:
            m = sel & (zone == k)
            fz[k].append(beitrag[m].sum()); rueck[k].append(np.mean(ux[m] < 0.0) if m.any() else np.nan)
    return {k: np.mean(fz[k]) for k in ZON}, {k: np.mean(rueck[k]) for k in ZON}, [os.path.basename(v)[9:15] for v in vs]

def cz_rest(p):
    r = [x for x in csv.DictReader(z_ for z_ in open(os.path.join(p, "cd_bericht.csv")) if not z_.startswith('#')) if x['warmup'] == '0']
    return float(np.mean([float(x['cz_rest_mittel']) for x in r]))

fb, rb, tb = lauf(bez); czb = cz_rest(bez); k = czb/sum(fb.values())
print(f"Bezug {os.path.basename(bez)}: cz_rest {czb:+.4f}, Proxy-Summe kalibriert (k = {k:.4g}), Schnappschuesse {tb}")
print(f"Lage-1-Zellen ohne Radband je Zone: " + ", ".join(f"{z_} {int((sel & (zone == z_)).sum())}" for z_ in ZON))
for a in arme:
    fa, ra, ta = lauf(a); cza = cz_rest(a)
    dges = sum(k*(fa[z_]-fb[z_]) for z_ in ZON)
    print(f"\n{os.path.basename(a)}: cz_rest {cza:+.4f} (d {cza-czb:+.4f}); Proxy-Summe d {dges:+.4f}  [Schnappschuesse {ta}]")
    print(f"  {'Zone':12s} {'cz Bezug':>9s} {'cz Arm':>9s} {'d cz':>8s} {'Anteil':>7s} | Rueckstroemung u_x<0: Bezug -> Arm")
    for z_ in ZON:
        dz = k*(fa[z_]-fb[z_])
        print(f"  {z_:12s} {k*fb[z_]:+9.4f} {k*fa[z_]:+9.4f} {dz:+8.4f} {100*dz/dges if dges else 0:6.0f}% | {100*rb[z_]:5.1f} % -> {100*ra[z_]:5.1f} %")
