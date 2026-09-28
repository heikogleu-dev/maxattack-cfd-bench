#!/usr/bin/env python3
"""export_zwischendumps.py [--los] -- Heiko 28.09.2026: Exportordner von unnoetigen Feld-VTKs befreien.
REGEL: je Lauf bleibt der ENDDUMP (groesster Zeitpunkt, nah UND fern) IMMER (Lehre 22.09.: Endzeit-Dump bleibt).
Zwischendumps feld_nah_<t>ms.vtk / feld_fern_<t>ms.vtk werden geloescht -- aber erst, nachdem aus dem Nahfeld-Dump das
Dachband herausgeschnitten ist (werkzeuge/abl_dach/fx_band.py -> dach_band_<t>ms.npz). Schlaegt der Bandschnitt fehl,
bleibt der Dump stehen. Ohne --los nur Probedurchlauf. Jede Loeschung steht in export/loeschliste_zwischendumps_<datum>.txt."""
import glob, os, re, subprocess, sys, datetime
LOS = "--los" in sys.argv
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EX = os.path.join(ROOT, "export"); FXB = os.path.join(ROOT, "werkzeuge", "abl_dach", "fx_band.py")
liste = os.path.join(EX, f"loeschliste_zwischendumps_{datetime.date.today().isoformat()}.txt")
zeit = lambda n: int(re.search(r"_(\d+)ms\.vtk$", n).group(1))
frei = 0; n_del = 0; n_band = 0; n_fehl = 0; log = []
for d in sorted(glob.glob(os.path.join(EX, "*/"))):
    fs = glob.glob(os.path.join(d, "feld_*_*ms.vtk"))
    if not fs: continue
    tend = max(zeit(f) for f in fs)
    for t in sorted({zeit(f) for f in fs if zeit(f) != tend}):
        nah = os.path.join(d, f"feld_nah_{t:06d}ms.vtk"); fern = os.path.join(d, f"feld_fern_{t:06d}ms.vtk")
        band = os.path.join(d, f"dach_band_{t:06d}ms.npz")
        ok = True
        # ★ 28.09. Pruefbefund D-N2: dieselbe Veraltet-Regel wie dach_statistik.py -- ein wiederverwendeter Laufname hinterliesse sonst
        # ein ALTES Band neben einem geloeschten NEUEN Dump, und nach dem Loeschen kann niemand mehr erkennen, dass das Band veraltet ist.
        if os.path.exists(nah) and (not os.path.exists(band) or os.path.getmtime(band) < os.path.getmtime(nah)):
            if LOS:
                r = subprocess.run([sys.executable, FXB, nah, band], capture_output=True, text=True)
                ok = (r.returncode == 0 and os.path.exists(band))
                if ok: n_band += 1
                else: n_fehl += 1; log.append(f"BAND FEHLGESCHLAGEN, Dump bleibt: {nah}: {r.stderr.strip()[-200:]}")
            else: n_band += 1
        if not ok: continue
        for f in (nah, fern):
            if os.path.exists(f):
                s = os.path.getsize(f); frei += s; n_del += 1
                log.append(f"{'GELOESCHT' if LOS else 'wuerde loeschen'} {s/1e9:6.2f} GB {os.path.relpath(f, ROOT)}")
                if LOS: os.remove(f)
print(f"{'LOESCHUNG' if LOS else 'PROBEDURCHLAUF'}: {n_del} Zwischendumps, {frei/1e9:.1f} GB; Dachbaender neu geschnitten: {n_band}; fehlgeschlagen (Dump bleibt): {n_fehl}")
if LOS:
    with open(liste, "a") as f: f.write("\n".join(log)+"\n")
    print(f"Liste: {os.path.relpath(liste, ROOT)}")
