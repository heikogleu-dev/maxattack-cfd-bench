#!/usr/bin/env python3
# Heiko 28.09.2026: (1) Winkel zwischen jeder Rang-0-Zelle im Dachbereich und ihren Wand-Nachbarn (D3Q27-Stencil)
# messen, um eine Flachheitsschwelle ueberhaupt erst zu SEHEN; (2) exemplarisch je Rang-0-Zelle rechnen, was sie
# bekaeme, wenn sie den Mittelwert ihrer (flachen Rang-2-)Nachbarn uebernimmt. Nur Daten, kein Modell.
# Eingabe: facetten_persistenz.csv ab Binary d4bba99 (Spalten ut, ut_ab, yw_ab, rang). Aufruf: <csv> <Nx> <Ny> <dx_m>
import csv, math, random, statistics as st, sys

csvp, Nx, Ny, dx = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])
F = {}
for r in csv.DictReader(l for l in open(csvp) if not l.startswith('#')):
    n = int(r['n'])
    F[n] = dict(x=n % Nx, y=(n//Nx) % Ny, z=n//(Nx*Ny), nx=float(r['nx']), ny=float(r['ny']), nz=float(r['nz']),
                L=int(r['eigene_links']), yw=float(r['yw']), ut=float(r['ut']), rate=float(r['rate']), rang=int(r['rang']))
zg = max(f['z'] for f in F.values()) - 0.45/dx
def zone(f):
    if f['z'] < zg or f['nz'] <= 0.3: return 'rest'
    return 'Scheibe' if f['nx'] < -0.25 else ('Heck' if f['nx'] > 0.25 else 'Dach')
for f in F.values(): f['zone'] = zone(f)
def winkel(a, b):
    c = max(-1.0, min(1.0, a['nx']*b['nx'] + a['ny']*b['ny'] + a['nz']*b['nz']))
    return math.degrees(math.acos(c))
def nachbarn(n):
    for dz in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dxx in (-1, 0, 1):
                if dxx == dy == dz == 0: continue
                g = F.get(n + dxx + Nx*(dy + Ny*dz))
                if g: yield g

GR = [1, 2, 5, 10, 15, 20, 30, 45, 90, 180]
print("(1) WINKEL Rang-0-Zelle -> Wand-Nachbar (Grad), Anteil der Nachbarpaare je Band")
for zn in ('Scheibe', 'Dach', 'Heck'):
    for rg in (2, 1, 0):
        w = [winkel(f, g) for n, f in F.items() if f['rang'] == 0 and f['zone'] == zn for g in nachbarn(n) if g['rang'] == rg]
        if not w: continue
        h = [0]*len(GR)
        for v in w:
            for i, b in enumerate(GR):
                if v < b: h[i] += 1; break
        print(f"  {zn:8s} Nachbar Rang {rg}: n={len(w):6d}  Median {st.median(w):5.2f}  q95 {sorted(w)[int(.95*(len(w)-1))]:5.2f}  " +
              "  ".join(f"<{b}:{100*c/len(w):4.1f}%" for b, c in zip(GR, h)))

print("\n(2) UEBERNAHME DES MITTELWERTS der flachen Rang-2-Nachbarn (Winkel < SCHWELLE), je Zone:")
for schwelle in (5.0, 15.0):
    print(f"  Schwelle {schwelle:.0f} Grad:")
    for zn in ('Scheibe', 'Dach', 'Heck'):
        d, ohne = [], 0
        for n, f in F.items():
            if f['rang'] != 0 or f['zone'] != zn: continue
            nb = [g for g in nachbarn(n) if g['rang'] == 2 and winkel(f, g) < schwelle]
            if not nb: ohne += 1; continue
            d.append((st.mean(g['ut'] for g in nb), f['ut'], st.mean(g['yw'] for g in nb), f['yw']))
        print(f"    {zn:8s} Rang-0 mit Nachbarn {len(d):6d}, ohne {ohne:5d}:  eigenes ut {st.median(x[1] for x in d):.4f} @ y_w {st.median(x[3] for x in d):.2f}"
              f"   Nachbarmittel ut {st.median(x[0] for x in d):.4f} @ y_w {st.median(x[2] for x in d):.2f}   -> neu/alt {st.median(x[0]/x[1] for x in d if x[1] > 1e-6):.2f}")

print("\n(3) BEISPIELE: 6 zufaellige Rang-0-Zellen am Dach (Schwelle 5 Grad), jeder Wand-Nachbar einzeln")
random.seed(28)
kand = [n for n, f in F.items() if f['rang'] == 0 and f['zone'] == 'Dach']
for n in random.sample(kand, 6):
    f = F[n]
    print(f"  Zelle ({f['x']},{f['y']},{f['z']})  Links {f['L']}  y_w {f['yw']:.2f}  ut {f['ut']:.4f}  Rueckfall {100*f['rate']:.0f} %")
    nb2 = []
    for g in nachbarn(n):
        w = winkel(f, g)
        tag = ""
        if g['rang'] == 2 and w < 5.0: nb2.append(g); tag = "  <- flach, Rang 2"
        print(f"      Nachbar ({g['x']-f['x']:+d},{g['y']-f['y']:+d},{g['z']-f['z']:+d}) Rang {g['rang']}  Winkel {w:5.2f}  y_w {g['yw']:.2f}  ut {g['ut']:.4f}{tag}")
    if nb2:
        m = st.mean(g['ut'] for g in nb2)
        print(f"      => Mittel der {len(nb2)} flachen Rang-2-Nachbarn: ut {m:.4f} (y_w {st.mean(g['yw'] for g in nb2):.2f})  statt {f['ut']:.4f}  -> du {m-f['ut']:+.4f}")
    else:
        print("      => keine flachen Rang-2-Nachbarn")
