#!/usr/bin/env python3
# Nachbar-Amplitude Stufe 0 (28.09.2026): offline je Facette, welche Geschwindigkeitskorrektur du eine
# Rang-0-Zelle bekaeme -- OHNE Kerneleingriff. Eingabe: facetten_persistenz.csv mit den Spalten
# ut, ut_ab, yw_ab, rang (ab Binary nach 96ac9b9) und das Nahfeldgitter Nx, Ny (aus dem Log).
#
# Variante A (Plan 28.09.): Spalding an der Abtaststelle (ut_ab, yw_ab) -> u_tau, am eigenen y_w
#   ausgewertet: du_A = u_tau*u+(y_w u_tau/nu) - ut.
# Variante B (Heikos Formulierung "die Geschwindigkeit einer ordentlichen Rang-2-Zelle dort"):
#   Ziel aus den Rang-2-Facetten unter den 26 Gitternachbarn: je Nachbar j u_tau,j aus Spalding an
#   (ut_j, yw_j), am eigenen y_w ausgewertet, gemittelt. du_B = Mittel - ut. Nachbarschaft = D3Q27-
#   Stencil, keine frei gewaehlte Reichweite. Ohne Rang-2-Nachbarn: kein du_B (gezaehlt).
#   28.09. Heiko: nur Nachbarn, die FLACH zur Zelle liegen (n_i.n_j >= cosmin, 6. Argument); die Schwelle
#   wird als Leiter gefahren, damit sie kein Handwert ist -- das Ergebnis darf von ihr nicht abhaengen.
# Konstanten: kappa 0,41, B 5,5 (Spalding 1961, wie kernel.cpp wf_spalding_uplus); nu aus dem Log.
#
# ENTSCHEIDREGEL (VOR den Daten festgelegt, Tagesprotokoll 28.09.):
#   Kernel-Arm wird nur gebaut, wenn fuer eine Variante an den Rang-0-Facetten ALLER drei Zonen
#   (Scheibe, Dach, Heck) gilt:
#   (1) Vorzeichen: Median(du) > 0 (beschleunigend, wie +eps, das Abtrieb bringt);
#   (2) Selektivitaet: Median(du) Rang 0 in der Zone > Median(du) Rang 0 ausserhalb der Dachzonen;
#   (3) Rang-Trennung: Median(du) Rang 0 > Median(du) Rang 2 derselben Zone (sonst misst die Formel
#       den Loeser, nicht den Rueckfall).
#   Faellt (3) fuer A und B durch, ist die Praemisse "Rang-0-Zellen sind zu langsam" am Dach widerlegt.
import csv, math, sys, statistics as st

K, BB = 0.41, 5.5
def yplus(u):
    k = K*u
    return u + math.exp(-K*BB)*(math.exp(k) - 1 - k - k*k/2 - k*k*k/6)
def bis(f, a, b):
    fa = f(a)
    for _ in range(80):
        m = 0.5*(a+b); fm = f(m)
        if fa*fm <= 0: b = m
        else: a, fa = m, fm
    return 0.5*(a+b)
def utau_aus(u, y, nu):  # Spalding nach u+ bei gegebenem Y = u*y/nu
    Y = u*y/nu
    up = bis(lambda p: p*yplus(p) - Y, 1e-9, 300.0)
    return u/up
def u_bei(utau, y, nu):  # u an y fuer gegebenes u_tau
    yp = y*utau/nu
    return utau*bis(lambda p: yplus(p) - yp, 0.0, 300.0)

def main(csvp, Nx, Ny, nu, dx, cosmin=-2.0):
    F = {}
    for r in csv.DictReader(l for l in open(csvp) if not l.startswith('#')):
        n = int(r['n']); x = n % Nx; y = (n//Nx) % Ny; z = n//(Nx*Ny)
        F[n] = dict(x=x, y=y, z=z, L=int(r['eigene_links']), yw=float(r['yw']), nx=float(r['nx']), nz=float(r['nz']),
                    rate=float(r['rate']), ny=float(r['ny']), ut=float(r['ut']), ub=float(r['ut_ab']), yb=float(r['yw_ab']), rang=int(r['rang']))
    zmax = max(f['z'] for f in F.values()); zgr = zmax - 0.45/dx
    def zone(f):
        if f['z'] < zgr or f['nz'] <= 0.3: return 'rest'
        return 'Scheibe' if f['nx'] < -0.25 else ('Heck' if f['nx'] > 0.25 else 'Dach')
    for f in F.values():
        f['zone'] = zone(f)
        f['A'] = None
        if f['ub'] > 1e-6 and f['yb'] > f['yw']:
            f['A'] = u_bei(utau_aus(f['ub'], f['yb'], nu), f['yw'], nu) - f['ut']
        f['utau_eig'] = utau_aus(f['ut'], f['yw'], nu) if f['ut'] > 1e-6 else None
    n_ohne = 0
    for n, f in F.items():
        f['B'] = None
        if f['rang'] not in (0, 2): continue  # Rang 2 ebenso, als Vergleich fuer Regel (3)
        z_ = []
        for dz in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dxx in (-1, 0, 1):
                    if dxx == dy == dz == 0: continue
                    g = F.get(n + dxx + Nx*(dy + Ny*dz))
                    if g and g['rang'] == 2 and g['utau_eig'] and f['nx']*g['nx']+f['ny']*g['ny']+f['nz']*g['nz'] >= cosmin: z_.append(u_bei(g['utau_eig'], f['yw'], nu))
        if z_: f['B'] = st.mean(z_) - f['ut']
        else: n_ohne += 1
    print(f"Flachheit: nur Rang-2-Nachbarn mit n_i.n_j >= {cosmin}")
    print(f"Facetten {len(F)}, Rang 0 {sum(f['rang']==0 for f in F.values())}, ohne Rang-2-Nachbarn (Rang 0+2) {n_ohne}, 255 {sum(f['rang']==255 for f in F.values())}")
    def med(v): return st.median(v) if v else float('nan')
    ergebnis = {}
    for var in ('A', 'B'):
        print(f"\nVariante {var}:  Zone  | Rang0 n  Median  q25  q75 | Rang2 Median | Rang0 Rueckfall")
        aussen = [f[var] for f in F.values() if f['rang'] == 0 and f['zone'] == 'rest' and f[var] is not None]
        m_aus = med(aussen)
        for zn in ('Scheibe', 'Dach', 'Heck', 'rest'):
            v0 = sorted(f[var] for f in F.values() if f['rang'] == 0 and f['zone'] == zn and f[var] is not None)
            v2 = [f[var] for f in F.values() if f['rang'] == 2 and f['zone'] == zn and f[var] is not None]
            rr = [f['rate'] for f in F.values() if f['rang'] == 0 and f['zone'] == zn]
            q = (lambda p: v0[int(p*(len(v0)-1))] if v0 else float('nan'))
            print(f"  {zn:8s} {len(v0):7d} {med(v0):+.3e} {q(.25):+.3e} {q(.75):+.3e} | {med(v2):+.3e} | {med(rr):.3f}")
            if zn != 'rest':
                ergebnis[(var, zn)] = (med(v0) > 0, med(v0) > m_aus, med(v0) > med(v2))
        ok = all(all(ergebnis[(var, zn)]) for zn in ('Scheibe', 'Dach', 'Heck'))
        print(f"  Regel (1)Vorzeichen (2)Selektiv (3)Rangtrennung je Zone: " +
              ", ".join(f"{zn} {''.join('J' if b else 'N' for b in ergebnis[(var, zn)])}" for zn in ('Scheibe', 'Dach', 'Heck')) +
              f"  -> {'BAUEN' if ok else 'NICHT bauen'}")

if __name__ == '__main__':
    # Aufruf: nb_amplitude_s0.py <facetten_persistenz.csv> <Nx> <Ny> <nu_lat> <dx_m>
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]), float(sys.argv[6]) if len(sys.argv) > 6 else -2.0)
