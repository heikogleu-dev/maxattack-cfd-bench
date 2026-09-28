#!/usr/bin/env python3
"""dach_statistik.py <bezug> <arm> [t_ab_s=0.24] -- Dachfeld je SCHNAPPSCHUSS, Mittel +- Standardfehler, Arm gegen Bezug.

Heiko 28.09.2026: mit 3 Momentanfeldern lag die R1Q=4-Verbesserung am Rauschniveau (V_Z = Bezug streute >0,28 m im
Abloeseort). Hier je Lauf JEDER feld_nah_*ms.vtk ab t_ab einzeln ausgewertet (fx_band.py -> fx_profil2.py mit genau
einem Band -> prof2_<lauf>_t<ms>.npz), dann je Groesse Mittel, Standardfehler und die Differenz Arm - Bezug in
Einheiten ihres kombinierten Standardfehlers.
Groessen (Definitionen WORTGLEICH aus bericht3.py / zonen_h.py, Band +-40 mm um die Fahrzeug-Mittelebene):
  x_s      Abloeseort: Beginn der durchgehenden Rueckstroemung (u_x < 0, erste Fluidzelle, +-24 mm geglaettet) bis x = 3,62 m;
           ohne Rueckstroemung bis dort -> 3,62 (Zensur, gezaehlt)
  H_pl     H = delta*/theta, Median ueber das Dachplateau x 2,30-2,90   (OF13 1,174)
  H_sp     dito Saugspitze x 2,00-2,30
  d99_pl   delta99 Median Dachplateau [mm]                             (OF13 29-39 mm)
  ut25/29  utg (geglaettet +-24 mm) auf dem Wandstrahl bei s = 4 mm (wie bericht3 Tabelle D), x = 2,50 / 2,90 m [m/s]     (OF13 24-32 / 20-28 m/s bei 2-8 mm)
  neg      Anteil u_x < 0 in der ersten Fluidzelle ueber x 2,9-3,6 m   (Rueckstromflaeche am hinteren Dach)
"""
import sys, os, glob, subprocess
import numpy as np
D = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(D, "..", ".."))
XE = 3.62
def glatt(a, w):
    k = np.ones(2*w+1)/(2*w+1); return np.convolve(np.pad(a, w, mode="edge"), k, mode="valid")
def durch(x, v, xend=XE):
    m = x <= xend; xx, vv = x[m], v[m]; ok = vv < 0
    if not ok.any() or not ok[-1]: return np.nan
    i = len(ok)-1
    while i > 0 and ok[i-1]: i -= 1
    return xx[i]

def groessen(npz):
    F = np.load(npz); x = F["x"]; w = int(round(0.024/float(F["dx"])))
    xs = durch(x, glatt(np.nan_to_num(F["ux1"]), w))
    zens = np.isnan(xs)
    if zens: xs = XE
    def med(key, a, b):
        m = (x >= a) & (x < b) & np.isfinite(F[key]); return float(np.nanmedian(F[key][m])) if m.any() else np.nan
    k4 = int(np.argmin(np.abs(F["s"]-0.004)))
    def at(key, xq): return float(F[key][int(np.argmin(np.abs(x-xq))), k4])   # wie bericht3.py Tabelle D: utg (+-24 mm in x geglaettet) auf dem Wandstrahl bei s = 4 mm -- das rohe ut trifft lokal Stufenkanten
    ux = np.nan_to_num(F["ux1"]); m = (x >= 2.9) & (x <= 3.6)
    return dict(x_s=xs, H_pl=med("H", 2.30, 2.90), H_sp=med("H", 2.00, 2.30), d99_pl=1e3*med("d99", 2.30, 2.90),
                ut25=at("utg", 2.50), ut29=at("utg", 2.90), neg=float(np.mean(ux[m] < 0))), zens

def lauf(name, t_ab):
    ex = os.path.join(ROOT, "export", name)
    vs = sorted(glob.glob(os.path.join(ex, "feld_nah_*ms.vtk")))
    R = []; nz = 0; ts = []
    for v in vs:
        tms = os.path.basename(v)[9:15]
        if int(tms)/1000.0 < t_ab: continue
        band = os.path.join(ex, f"dach_band_{tms}ms.npz")
        if not os.path.exists(band): subprocess.run([sys.executable, os.path.join(D, "fx_band.py"), v, band], check=True, capture_output=True)
        pn = f"{name}_t{tms}"
        pf = os.path.join(D, f"prof2_{pn}.npz")
        if not os.path.exists(pf): subprocess.run([sys.executable, os.path.join(D, "fx_profil2.py"), pn, band], check=True, capture_output=True)
        g, z = groessen(pf); R.append(g); nz += int(z); ts.append(int(tms))
    return R, nz, ts

bez, arm = sys.argv[1], sys.argv[2]; t_ab = float(sys.argv[3]) if len(sys.argv) > 3 else 0.24
RB, zb, tb = lauf(bez, t_ab); RA, za, ta = lauf(arm, t_ab)
print(f"Bezug {bez}: {len(RB)} Schnappschuesse {tb} (x_s zensiert bei {XE}: {zb})")
print(f"Arm   {arm}: {len(RA)} Schnappschuesse {ta} (x_s zensiert: {za})")
print(f"{'Groesse':8s} | {'Bezug Mittel +- SE':>20s} | {'Arm Mittel +- SE':>20s} | {'Arm - Bezug':>11s} | {'in SE':>6s} | OF13")
OF = dict(x_s="3,63", H_pl="1,17", H_sp="--", d99_pl="29-39", ut25="24-32", ut29="20-28", neg="--")
for k in ("x_s", "H_pl", "H_sp", "d99_pl", "ut25", "ut29", "neg"):
    b = np.array([r[k] for r in RB], float); a = np.array([r[k] for r in RA], float)
    b = b[np.isfinite(b)]; a = a[np.isfinite(a)]
    mb, ma = b.mean(), a.mean(); sb = b.std(ddof=1)/np.sqrt(len(b)); sa = a.std(ddof=1)/np.sqrt(len(a))
    se = np.hypot(sb, sa); d = ma-mb
    print(f"{k:8s} | {mb:10.3f} +- {sb:7.3f} | {ma:10.3f} +- {sa:7.3f} | {d:+11.3f} | {d/se if se > 0 else np.nan:+6.1f} | {OF[k]}")
print("Hinweis: aufeinanderfolgende Schnappschuesse (60 ms) sind nicht voll unabhaengig -- SE ist eine UNTERgrenze der Unsicherheit.")
