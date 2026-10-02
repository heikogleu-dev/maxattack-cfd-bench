#!/usr/bin/env python3
"""fx_band.py -- schneidet aus einem feld_nah_*.vtk (STRUCTURED_POINTS, BINARY, big-endian)
ein y-BAND um die Mittelebene heraus und legt es als .npz ab.

Datei-Layout laut src/setup.cpp schreibe_vtk_feld (Zeilen 356-395):
  <ASCII-Kopf>            endet nach "VECTORS u float\n"
  u      : np*3 float32 big-endian   (SI, m/s)
  "\nSCALARS rho float 1\nLOOKUP_TABLE default\n"
  rho    : np   float32 big-endian   (Gittereinheiten)
  "\nSCALARS flags unsigned_char 1\nLOOKUP_TABLE default\n"
  flags  : np   uint8   (TYPE_S=0x01, TYPE_E=0x02, TYPE_F=0x08, ...)

Index n = x + Nx*(y + Ny*z)  -> eine (y,z)-Zeile ist in x zusammenhaengend,
und aufeinanderfolgende y bei festem z liegen ebenfalls zusammenhaengend.

Aufruf: fx_band.py <vtk> <out.npz> [halb_mm]   (Default 40 mm zu jeder Seite)

★ 17.09.2026 (SKALIERUNG-BEFUNDE Nebenbefund 10): das Band lag um WELT-y = 0. Mit CFD_Y_VERSATZ=1 steht der Koerper eine halbe
feine Zelle weiter +y -- das Band war dann um eine halbe Zelle asymmetrisch zur Fahrzeug-Mittelebene. Jetzt koerperbezogen:
Mittelebene m = (y_versatz - y0)/dx aus dem Laufprotokoll (werkzeuge/lauf_meta.py; unbekannt -> 0 mit WARNUNG). Liegt m auf einer
Zellmitte, bleibt alles wie bisher (jy0 +- hb, 2 hb + 1 Ebenen); liegt m auf einer Zellflaeche, werden die 2 hb Ebenen
symmetrisch darum genommen. Die npz traegt zusaetzlich y_versatz_m, y_mitte_index und u_lat (NaN, wenn nicht bestimmbar) samt
Quellen -- fx_dach.py/fx_profil.py lesen u_lat von dort.
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import lauf_meta

def kopf(pfad):
    with open(pfad, "rb") as f:
        roh = f.read(1024)
    txt = roh.decode("ascii", "replace")
    i = txt.index("VECTORS u float\n") + len("VECTORS u float\n")
    d = {}
    for zeile in txt[:i].split("\n"):
        t = zeile.split()
        if not t: continue
        if t[0] == "DIMENSIONS": d["dims"] = tuple(int(v) for v in t[1:4])
        elif t[0] == "ORIGIN":   d["orig"] = tuple(float(v) for v in t[1:4])
        elif t[0] == "SPACING":  d["spac"] = tuple(float(v) for v in t[1:4])
    d["off_u"] = i
    Nx, Ny, Nz = d["dims"]
    np_ = Nx*Ny*Nz
    s1 = b"\nSCALARS rho float 1\nLOOKUP_TABLE default\n"
    s2 = b"\nSCALARS flags unsigned_char 1\nLOOKUP_TABLE default\n"
    d["off_rho"]   = i + np_*12 + len(s1)
    d["off_flags"] = d["off_rho"] + np_*4 + len(s2)
    soll = d["off_flags"] + np_
    ist  = os.path.getsize(pfad)
    if soll != ist:
        raise SystemExit(f"Layout passt nicht: erwartet {soll} Bytes, Datei hat {ist}. "
                         f"Marker pruefen (SCALARS-Zeilen).")
    # Marker gegenlesen (Beweis statt Annahme)
    with open(pfad, "rb") as f:
        f.seek(i + np_*12); assert f.read(len(s1)) == s1, "rho-Marker fehlt"
        f.seek(d["off_rho"] + np_*4); assert f.read(len(s2)) == s2, "flags-Marker fehlt"
    return d

def main():
    vtk, out = sys.argv[1], sys.argv[2]
    halb_mm = float(sys.argv[3]) if len(sys.argv) > 3 else 40.0
    d = kopf(vtk)
    Nx, Ny, Nz = d["dims"]; x0, y0, z0 = d["orig"]; dx = d["spac"][0]
    lauf = lauf_meta.lauf_dir_aus(vtk)
    yv, q_yv = lauf_meta.y_versatz_m(lauf, vtk)
    if yv is None:
        print(f"WARNUNG: Y-Versatz unbekannt ({q_yv}) -- Band um Welt-y = 0; bei CFD_Y_VERSATZ=1 eine halbe Zelle asymmetrisch.", file=sys.stderr)
        yv = 0.0
    try: ul, q_ul = lauf_meta.u_lat(lauf)
    except SystemExit as e: ul, q_ul = float("nan"), f"UNBEKANNT ({e})"
    m   = (yv - y0)/dx                                  # Fahrzeug-Mittelebene als (gebrochener) y-Index
    hb  = int(round(0.001*halb_mm/dx))
    if abs(m - round(m)) < 0.25:                        # Zellmitte: wie bisher
        jy0 = int(round(m)); ja, jb = jy0-hb, jy0+hb
    else:                                               # Zellflaeche (Versatz): 2 hb Ebenen symmetrisch
        jl = int(np.floor(m)); jy0 = jl; ja, jb = jl+1-hb, jl+hb
    ja, jb = max(0, ja), min(Ny-1, jb)
    ny = jb - ja + 1
    print(f"{os.path.basename(vtk)}: {Nx}x{Ny}x{Nz} dx={dx*1000:.4g} mm  "
          f"y-Band j={ja}..{jb} (y={y0+ja*dx:+.4f}..{y0+jb*dx:+.4f} m, koerperbezogen {y0+ja*dx-yv:+.4f}..{y0+jb*dx-yv:+.4f} m), {ny} Ebenen; "
          f"Mittelebene Index {m:.3f} (Y-Versatz {yv*1e3:.4g} mm aus {q_yv}); u_lat {ul} aus {q_ul}")
    U   = np.empty((Nz, ny, Nx, 3), dtype=np.float32)
    RHO = np.empty((Nz, ny, Nx),    dtype=np.float32)
    FL  = np.empty((Nz, ny, Nx),    dtype=np.uint8)
    with open(vtk, "rb") as f:
        for z in range(Nz):
            basis = (z*Ny + ja)*Nx
            f.seek(d["off_u"] + basis*12)
            U[z] = np.frombuffer(f.read(ny*Nx*12), dtype=">f4").reshape(ny, Nx, 3)
            f.seek(d["off_rho"] + basis*4)
            RHO[z] = np.frombuffer(f.read(ny*Nx*4), dtype=">f4").reshape(ny, Nx)
            f.seek(d["off_flags"] + basis)
            FL[z] = np.frombuffer(f.read(ny*Nx), dtype=np.uint8).reshape(ny, Nx)
    # ★ 02.10. Pruefbefund E-N1: atomar schreiben -- ein abgebrochener Schreibvorgang hinterliess sonst ein abgeschnittenes Band, das
    # NEUER als der Dump ist; export_zwischendumps.py haette den Dump danach geloescht.
    tmp = out + ".tmp.npz"
    np.savez_compressed(tmp, u=U, rho=RHO, flags=FL,
                        dims=np.array([Nx, Ny, Nz]), orig=np.array([x0, y0, z0]),
                        dx=dx, ja=ja, jb=jb, jy0=jy0, y_versatz_m=yv, y_mitte_index=m, y_versatz_quelle=q_yv,
                        u_lat=ul, u_lat_quelle=q_ul)
    os.replace(tmp, out)
    print("geschrieben:", out, os.path.getsize(out)//1048576, "MB")

main()
