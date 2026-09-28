#!/usr/bin/env python3
"""lage1_rang_vtk.py <lauf> [ausgabe.vtk] -- Punkt-VTK der LAGE-1-Wandzellen (mindestens ein FLAECHEN-Nachbar ist Solid)
in Weltkoordinaten, zum Projizieren auf die Facettenflaeche in ParaView (Heiko 28.09.2026).

Eingaben aus export/<lauf>/: facetten_persistenz.csv (ab Binary d4bba99: Spalte rang), facetten_lage.csv
(lage1_aus_vtk.py), lage1_rang1_richtung.npz (lage1_rang1_richtung.py), Gitter aus feld_nah_*.vtk.
SKALARE je Punkt:
  rang               statischer Tangentialrang nach ALPHA2 (0/1/2) -- derselbe, den der Zensus des Laufs zaehlt
  wandmodell_anteil  Anteil des Stroemungs-Wandschubs, den das Modell aufpraegen KANN: Rang 2 -> 1, Rang 0 -> 0,
                     Rang 1 -> cos^2(Wirkrichtung, u_t) aus dem Momentanfeld (Indikator, kein Zeitmittel)
  cos_wirkrichtung   |cos| fuer Rang 1, sonst -1
  zerschnitten       1 = die Ausgleichsebene schneidet den Zellwuerfel (y_w < 0,5*sum|n_i|)
  rueckfallrate      Anteil der Besuche mit Rueckfall auf Bounce-Back (1 - wandmodell_aktiv)
  eigene_links       Zahl der Solid-Links der Zelle
ParaView: diese Datei + remesh_flaeche.vtk laden, Filter "Point Dataset Interpolator" (Source = diese Punkte,
Input = Flaeche, Kernel Voronoi/Nearest) -- dann traegt die Flaeche den Rang der naechsten Lage-1-Zelle.
"""
import sys, os, csv, glob
import numpy as np

lauf = sys.argv[1]
d = lauf if os.path.isdir(lauf) else f"export/{lauf}"
aus = sys.argv[2] if len(sys.argv) > 2 else os.path.join(d, "lage1_rang.vtk")
kand = sorted(glob.glob(os.path.join(d, "feld_nah_*.vtk")))
if not kand: raise SystemExit("kein feld_nah_*.vtk -- Gitter unbekannt")
org = spc = dim = None
with open(kand[0], "rb") as fh:
    for _ in range(12):
        z = fh.readline().decode("ascii", "replace").strip()
        if z.startswith("ORIGIN"): org = [float(v) for v in z.split()[1:4]]
        elif z.startswith("SPACING"): spc = [float(v) for v in z.split()[1:4]]
        elif z.startswith("DIMENSIONS"): dim = [int(v) for v in z.split()[1:4]]
NX, NY = dim[0], dim[1]
lage = {int(r['n']): int(r['lage1']) for r in csv.DictReader(open(os.path.join(d, "facetten_lage.csv")))}
R = np.load(os.path.join(d, "lage1_rang1_richtung.npz"))
cosd = dict(zip(R["n"].tolist(), R["cos"].tolist())); rangd = dict(zip(R["n"].tolist(), R["rang"].tolist()))
P = []; n_wid = 0
for r in csv.DictReader(l for l in open(os.path.join(d, "facetten_persistenz.csv")) if not l.startswith('#')):
    n = int(r['n'])
    if not lage[n]: continue
    rg = int(r['rang'])
    if rangd.get(n, rg) != rg: n_wid += 1
    nx, ny, nz, yw = float(r['nx']), float(r['ny']), float(r['nz']), float(r['yw'])
    c = cosd.get(n, -1.0)
    anteil = 1.0 if rg == 2 else (0.0 if rg == 0 else c*c)
    x = n % NX; y = (n//NX) % NY; z = n//(NX*NY)
    P.append((org[0]+x*spc[0], org[1]+y*spc[1], org[2]+z*spc[2], rg, anteil, c,
              1 if yw < 0.5*(abs(nx)+abs(ny)+abs(nz)) else 0, float(r['rate']), int(r['eigene_links'])))
if n_wid: raise SystemExit(f"Rang aus CSV und aus npz weichen an {n_wid} Zellen ab")
with open(aus, "w") as f:
    f.write("# vtk DataFile Version 3.0\n")
    f.write(f"Lage-1-Wandzellen ({os.path.basename(os.path.normpath(d))}): Tangentialrang 0-2 und Wandmodell-Anteil\n")
    f.write("ASCII\nDATASET POLYDATA\n")
    f.write(f"POINTS {len(P)} float\n")
    for p in P: f.write(f"{p[0]:.5f} {p[1]:.5f} {p[2]:.5f}\n")
    f.write(f"VERTICES {len(P)} {2*len(P)}\n")
    for i in range(len(P)): f.write(f"1 {i}\n")
    f.write(f"POINT_DATA {len(P)}\n")
    for name, i, typ in (("rang", 3, "int"), ("wandmodell_anteil", 4, "float"), ("cos_wirkrichtung", 5, "float"),
                         ("zerschnitten", 6, "int"), ("rueckfallrate", 7, "float"), ("eigene_links", 8, "int")):
        f.write(f"SCALARS {name} {typ} 1\nLOOKUP_TABLE default\n")
        for p in P: f.write(f"{p[i]:.4f}\n" if typ == "float" else f"{p[i]}\n")
print(f"{len(P):,} Lage-1-Zellen -> {aus}  (Rang 0/1/2: {sum(p[3]==0 for p in P)}/{sum(p[3]==1 for p in P)}/{sum(p[3]==2 for p in P)})")
