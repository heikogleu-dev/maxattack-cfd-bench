#!/usr/bin/env python3
# Heiko 28.09.2026: "die First-Layer-Zellen, die vielleicht auch durch die Facetten zerschnitten sind, auf Rang 2 heben".
# Fragen je Lage-1-Zelle in Scheibe/Dach/Heck mit D3Q19-Rang 1:
#   (a) In welche tangentiale Richtung d kann das Wandmodell dort ueberhaupt wirken (Eigenvektor zu lmax des 2x2 nach
#       ALPHA2-Downdate und Schur-Elimination -- exakt der Klassifikator, der den Laufzensus trifft), und wie gut liegt d
#       in der Stroemung? |cos(d, u_t)| aus dem MOMENTANFELD (feld_nah_<t>ms.vtk) -- ein Indikator, kein Zeitmittel.
#   (b) Hebt der D3Q27-Satz (8 Eckrichtungen) die Zelle auf Rang 2?
# Rang/Entkopplung aus d3q27_zensus.klassifiziere (dort gegen den Log abgenommen, B3: Rang, Wanderung, Entkopplung exakt).
# Aufruf: lage1_rang1_richtung.py <export/lauf> <t_ms>   (braucht facetten_normalen.npz, facetten_lage.csv, facetten_persistenz.csv)
import sys, os, csv
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import d3q27_zensus as Z
from zonen_kraft import kopf

lauf, tms = sys.argv[1], sys.argv[2]
vtk = os.path.join(lauf, f"feld_nah_{tms}ms.vtk")
d = kopf(vtk); Nx, Ny, Nz = d["dims"]; dx = d["spac"][0]
C, _ = Z.lies_csv(os.path.join(lauf, "facetten_histogramme.csv"))
n_alle = C["n"].astype(np.int64)
NR = np.load(os.path.join(lauf, "facetten_normalen.npz"))
assert np.array_equal(NR["n"], n_alle), "Normalen passen nicht zur CSV"
aktiv = C["klasse"].astype(np.int64) == 0
n = n_alle[aktiv]
nf = NR["nf"][aktiv].astype(np.float64); nv = nf/np.linalg.norm(nf, axis=1)[:, None]
FL = np.fromfile(vtk, dtype=np.uint8, count=Nx*Ny*Nz, offset=d["off_flags"])
LK = np.zeros((n.size, 26), bool)
for b, (cx, cy, cz) in enumerate(Z.ALLE): LK[:, b] = FL[n + cx + Nx*(cy + Ny*cz)] == 0x41
del FL
m19 = LK.copy(); m19[:, 18:] = False
rg19, _ = Z.klassifiziere(m19, Z.W19, nv, True)
rg27, _ = Z.klassifiziere(LK, Z.W27, nv, True)
soll = [336847, 243867, 137159]
ist = [int((rg19 == r).sum()) for r in (2, 1, 0)]
if ist != soll: raise SystemExit(f"Rang weicht vom Laufzensus ab: {ist} statt {soll}")
print(f"Abnahme: D3Q19-Rang 2/1/0 = {ist} == Laufzensus (logs/s0_f8_nbexp.log)")

# Loesbare Richtung: dieselbe Algebra wie Z.klassifiziere, zusaetzlich A12 und der Eigenvektor zu lmax.
c = Z.CV.astype(np.float64); M = m19.astype(np.float64)*Z.W19[None, :]
S0 = M.sum(1); S1 = M @ c
G = np.einsum("nk,ka,kb->nab", 6.0*M, c, c)
Dd = np.where(S0 > 0, 6.0/np.where(S0 > 0, S0, 1), 0.0)
G = G - Dd[:, None, None]*S1[:, :, None]*S1[:, None, :]
k = np.argmin(np.abs(nv), axis=1); h = np.zeros_like(nv); h[np.arange(len(nv)), k] = 1.0
e1 = np.cross(h, nv); e1 /= np.linalg.norm(e1, axis=1)[:, None]; e2 = np.cross(nv, e1)
q = lambda u, Mx, v: np.einsum("na,nab,nb->n", u, Mx, v)
A11, A22, A12 = q(e1, G, e1), q(e2, G, e2), q(e1, G, e2)
Snn, Sn1, Sn2 = q(nv, G, nv), q(e1, G, nv), q(e2, G, nv)
ent = (Snn < 1e-8) | ((Sn1*Sn1 + Sn2*Sn2) <= 1e-6*Snn*(A11 + A22))
sd = np.where(ent, 1.0, Snn)
A11 = np.where(ent, A11, A11 - Sn1*Sn1/sd); A22 = np.where(ent, A22, A22 - Sn2*Sn2/sd); A12 = np.where(ent, A12, A12 - Sn1*Sn2/sd)
th = 0.5*np.arctan2(2*A12, A11 - A22)              # Hauptachse zu lmax der symmetrischen 2x2
dvec = np.cos(th)[:, None]*e1 + np.sin(th)[:, None]*e2

# Lage 1 und Zonen (dieselben Definitionen wie lage1_auswertung.txt)
lage = {int(r['n']): int(r['lage1']) for r in csv.DictReader(open(os.path.join(lauf, "facetten_lage.csv")))}
P = {int(r['n']): r for r in csv.DictReader(l for l in open(os.path.join(lauf, "facetten_persistenz.csv")) if not l.startswith('#'))}
z = n//(Nx*Ny); zg = z.max() - 0.45/dx
nx_, nz_ = np.array([float(P[i]['nx']) for i in n]), np.array([float(P[i]['nz']) for i in n])
rate = np.array([float(P[i]['rate']) for i in n]); yw = np.array([float(P[i]['yw']) for i in n])
rang_csv = np.array([int(P[i]['rang']) for i in n])
if not np.array_equal(rang_csv, rg19): raise SystemExit("Rang aus CSV != Klassifikator")
l1 = np.array([lage[i] for i in n]) == 1
zone = np.where((z < zg) | (nz_ <= 0.3), 'rest', np.where(nx_ < -0.25, 'Scheibe', np.where(nx_ > 0.25, 'Heck', 'Dach')))
schnitt = yw < 0.5*np.abs(nv).sum(1)

U = np.memmap(vtk, dtype=">f4", mode="r", offset=d["off_u"], shape=(Nx*Ny*Nz, 3))
print("\nLage 1, D3Q19-Rang 1: Wirkrichtung des Wandmodells gegen die Stroemung (Momentanfeld t = " + tms + " ms)")
print(f"  {'Zone':8s} {'Schnitt':12s} {'n':>7s} | |cos(d,u_t)| Median  >=0,9  >=0,5 | Wandmodell-Anteil an u_t (Median |cos|^2) | D3Q27 -> Rang 2")
for zn in ('Scheibe', 'Dach', 'Heck', 'rest'):
    for s, lab in ((True, 'zerschnitten'), (False, 'ganz')):
        m = l1 & (rg19 == 1) & (zone == zn) & (schnitt == s)
        idx = np.nonzero(m)[0]
        if idx.size == 0: continue
        u = np.asarray(U[n[idx]]).astype(np.float64); un = (u*nv[idx]).sum(1); ut = u - un[:, None]*nv[idx]
        cs = np.abs((dvec[idx]*ut).sum(1))/np.maximum(np.linalg.norm(ut, axis=1), 1e-30)
        print(f"  {zn:8s} {lab:12s} {idx.size:7d} | {np.median(cs):19.2f} {100*np.mean(cs>=0.9):5.1f}% {100*np.mean(cs>=0.5):5.1f}% |"
              f" {np.median(cs**2):40.2f} | {100*np.mean(rg27[idx]==2):5.1f} %")
print("\nLESART: |cos| = 1 heisst, die eine loesbare Richtung liegt in der Stroemung -- dann fehlt dem Rang 1 praktisch nichts.")
print("|cos| klein heisst, das Wandmodell greift quer zur Stroemung; der Stroemungsanteil laeuft ohne Wandschub.")

# ★ Selbstpruefung der Richtung (Iron Rule: Zwischenergebnis belegen): fuer Rang-1-Zellen muss die quadratische Form
# entlang d gleich lmax sein und quer dazu (n x d) ~ 0, relativ zu lmax.
r1 = rg19 == 1
t = np.stack([A11, A12, A12, A22], 1).reshape(-1, 2, 2)
v = np.stack([np.cos(th), np.sin(th)], 1); w = np.stack([-np.sin(th), np.cos(th)], 1)
ld = np.einsum("na,nab,nb->n", v, t, v); lq = np.einsum("na,nab,nb->n", w, t, w)
tr = A11 + A22; disc = np.maximum(tr*tr - 4*(A11*A22 - A12*A12), 0.0); lmax = 0.5*(tr + np.sqrt(disc))
print(f"\nSelbstpruefung Rang 1 ({int(r1.sum())} Zellen): max |q(d)-lmax|/lmax = {np.max(np.abs(ld[r1]-lmax[r1])/lmax[r1]):.2e},"
      f" max q(quer)/lmax = {np.max(lq[r1]/lmax[r1]):.2e}  (Soll: beide ~0, Rang-1-Schwelle lmin/lmax < 1e-9)")
