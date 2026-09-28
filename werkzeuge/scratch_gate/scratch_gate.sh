#!/usr/bin/env bash
# scratch_gate.sh — 2-Sekunden-Gate gegen die Scratch-Fehlerklasse (Befund 2026-08-26):
# waechst eine Kernel-Schleife ueber IGCs Unroll-Budget, werden laufzeitindizierte
# private Arrays (fhn/fpre/j/c()/w()) speicherheimisch -> private_size>0 im .zeinfo
# -> Faktor ~100 Laufzeit (2 statt 240 MLUPs, "0 GB/s"; g13-g15). Dieses Gate baut den
# AKTUELLEN src/kernel.cpp zur .cl (gen_main.cpp, Defines des Kanal-Referenzfalls),
# kompiliert offline per ocloc (KEIN GPU-Lauf) fuer iGPU und B70 und schlaegt fehl, sobald
# ein Kernel private_size>0 ODER spill_size>0 traegt (seit Rang-1-Remat).
# Legitimes Spill-Wachstum erfordert eine BEWUSSTE Lockerung dieses Gates, nie ein stilles.
#
# ★ 11.09.2026 — GESAMTDECKUNG. Das Gate prueft ab jetzt JEDEN Kernel, nicht nur
# stream_collide. Anlass ist ein Befund der zweiten Agentenrunde: fac_nachbar_ab traegt
# private_size=7296 (= 228 B der c()-Tabelle x 32 Lanes) und ist damit GENAU die
# Fehlerklasse, gegen die dieses Gate gebaut wurde -- unentdeckt, weil hier bis heute
# "stream_collide" fest verdrahtet stand. Ein Waechter, der nur an einer Stelle hinsieht,
# ist kein Waechter.
#
# Aufruf: werkzeuge/scratch_gate/scratch_gate.sh    (beliebiges Arbeitsverzeichnis)
# Exit 0 = sauber, Exit 1 = Scratch ODER Spill zurueck. Referenz 26.08.2026 nachmittags
# (Rang-1-Remat): stream_collide private 0 UND spill 0 in BEIDEN Armen auf BEIDEN Geraeten.
# Historie: vor Unroll-Fix private 4256/8512; vor Remat spill 448/832 (Prod) bzw. 672/1216 (ELIBB).
set -eu
HIER="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HIER/../.." && pwd)"
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT

# ── BEKANNTE, AUSDRUECKLICH ERKLAERTE ABWEICHUNGEN ────────────────────────────────────
# Format: "<kernelname>:<private>:<spill>". Nur exakt diese Werte gelten als bekannt --
# waechst die Zahl, schlaegt das Gate zu. Ein Eintrag hier ist eine SCHULD, kein Freibrief:
# er gehoert entfernt, sobald der Befund behoben ist, und er braucht immer eine Begruendung.
#
#   (leer) — der Eintrag fac_nachbar_ab:7296:0 / :3648:0 wurde am 11.09.2026 behoben und
#   entfernt: der laufzeitindizierte c(ib)-Zugriff in kernel.cpp ist durch eine Mitschrift
#   in der Schleife ersetzt. Das Gate hat den Eintrag selbst als veraltet gemeldet.
#   stream_collide:0:64 (B70 0xe223) / stream_collide:0:224 (iGPU 0x7d67) -- NUR im Arm prod8nahr1q4 (CFD_FAC_R1Q=4, Einspeisung der
#   Zellquelle r1q_einspeisen spaet in apply_facette_imem, wo fhn, fw, t1/t2 und die PINV-Groessen gleichzeitig leben). Eingetragen
#   28.09.2026 abends als BEWUSSTE Lockerung nach Eingrenzung: ohne Einspeisung 0 B; die urspruengliche Form hatte 192/256 B, die
#   paarweise Form (Richtung/Gegenrichtung teilen c.du und c.s) senkt auf 64/224 B. Gemessene Kosten am 8-mm-Fahrzeug B70 mit der
#   192-B-Form: Wanduhr 7:45 gegen 7:40 min (r1q_f8_vr4_7 gegen r1q_f8_b7, je ein Lauf). SCHULD: vor einem Standard-Umstieg auf R1Q
#   die Einspeisung aus apply_facette_imem nach stream_collide (nach dem Aufruf, du als float3 zurueckgeben) verlegen und neu messen.
BEKANNT="stream_collide:0:64 stream_collide:0:224"

g++ -O1 -c "$REPO/src/kernel.cpp" -o "$T/kernel.o"
g++ -O1 "$HIER/gen_main.cpp" "$T/kernel.o" -o "$T/gen"
# ★ 11.09.2026: VIER Arme statt zwei. Ohne den PTRT-Arm prueft das Gate den
# Produktionsstand gar nicht -- der #ifdef PTRT-Block in kernel.cpp bleibt inert,
# solange PTRT nicht definiert ist.
# ★ 12.09.2026 RHO-ARM (TODO 2 Schritt 4). Ohne ihn prueft das Gate den 2-Byte-rho-Stand nicht:
# store_rho/load_rho sind dann die Identitaet und der Rueckleser in store_rho_diag fehlt ganz.
# Zwoelf statt vier Arme; das Gate bleibt damit unter fuenfzehn Sekunden.
"$T/gen" on  on  off off off off "$T/e1p1.cl" >/dev/null
"$T/gen" on  off off off off off "$T/e1p0.cl" >/dev/null
"$T/gen" off on  off off off off "$T/e0p1.cl" >/dev/null
"$T/gen" off off off off off off "$T/e0p0.cl" >/dev/null
"$T/gen" on  on  on  off off off "$T/e1p1r.cl" >/dev/null
"$T/gen" on  off on  off off off "$T/e1p0r.cl" >/dev/null
"$T/gen" off on  on  off off off "$T/e0p1r.cl" >/dev/null
"$T/gen" off off on  off off off "$T/e0p0r.cl" >/dev/null
# ★ 12.09.2026 (Audit-Schleife, Pruefer B): die beiden PRODUKTIONSARME. Nach TODO 2 laeuft die
# Produktion mit CFD_RHO_SPARSAM und CFD_U_SPARSAM, und deren Zweige haengen an stream_collide --
# dem Kernel, an dem sich Scratch entscheidet. Ohne diese zwei Arme prueft das Gate acht Varianten,
# aber nicht die, die gerechnet wird. Kein volles Kreuz (16 Arme): geprueft wird der Produktionspunkt
# ELIBB an, PTRT an, SPARSAM an, beide rho-Formate.
"$T/gen" on  on  off on  off off "$T/e1p1s.cl" >/dev/null
"$T/gen" on  on  on  on  off off "$T/e1p1rs.cl" >/dev/null
# ★ 12.09.2026 U-ARM (TODO 2 Schritt 4 fuer u). Zwei weitere Arme statt eines vollen Kreuzes (32):
# geprueft wird der PRODUKTIONSPUNKT ELIBB an, PTRT an, SPARSAM an -- einmal mit u16 allein und
# einmal mit beiden 2-Byte-Feldern. Das ist die Kombination, die gerechnet wird; ein Gate, das
# acht Varianten prueft und die gefahrene nicht, hat dieses Projekt schon einmal bezahlt.
"$T/gen" on  on  off on  on  off "$T/e1p1su.cl" >/dev/null
"$T/gen" on  on  on  on  on  off "$T/e1p1rsu.cl" >/dev/null
# ★ 15.09.2026 RHO_RAND-ARME (C2b): Produktionspunkt ELIBB an, PTRT an, u16 an, RHO_RAND statt SPARSAM --
# einmal mit 2-Byte-rho (die Produktion) und einmal mit float-rho.
"$T/gen" on  on  on  off on  on  "$T/e1p1ruR.cl" >/dev/null
"$T/gen" on  on  off off on  on  "$T/e1p1uR.cl" >/dev/null
# ★ 15.09.2026 Klemmen S0a -- PRODUKTIONSPARITAET: zwei Arme aus den ECHTEN Defines des 8-mm-Standardlaufs
# (CFD_DUMP_CL=1, Lauf rr_s0_dump_b70 @ b9329a5; Vorspann vor get_opencl_c_code() abgeschnitten). Die Kanal-Arme oben
# kennen weder SGS_FDWAND/SGS_SISM (Nahfeld) noch SPONGE (Fernfeld) -- genau dort liegt der Buchungsort der Stufe 0.
# Die defs-Dateien sind ein SCHNAPPSCHUSS: aendert sich die Produktionszeile, neu dumpen (Anleitung gen_main.cpp).
"$T/gen" datei "$HIER/defs_prod8_nah.txt"  "$T/prod8nah.cl"  >/dev/null
"$T/gen" datei "$HIER/defs_prod8_fern.txt" "$T/prod8fern.cl" >/dev/null
# ★ 15.09.2026 Klemmen S0b: Negativhaken-Arm (KLEMM_HAKEN3 veraendert den Buchungsblock, muss ebenfalls sauber bauen)
"$T/gen" dateih3 "$HIER/defs_prod8_nah.txt" "$T/prod8nahh3.cl" >/dev/null
# ★ 15.09.2026 Klemmen Stufe 1 P1a (KLEMMEN-STUFE1-PLAN.md §7): Positiv-Arme. Defines aus positiv_defines() (kernel.cpp) --
# dieselbe Funktion wie die Emission in lbm.cpp. p1 = Messarm, p2 = anwenden, f = K0 eingeschlossen, h1/h3 = Testhaken.
"$T/gen" datei "$HIER/defs_prod8_nah.txt"  "$T/prod8nahp1.cl"   pos1    >/dev/null
"$T/gen" datei "$HIER/defs_prod8_nah.txt"  "$T/prod8nahp2.cl"   pos2    >/dev/null
"$T/gen" datei "$HIER/defs_prod8_fern.txt" "$T/prod8fernp2.cl"  pos2    >/dev/null
"$T/gen" datei "$HIER/defs_prod8_fern.txt" "$T/prod8fernp1.cl"  pos1    >/dev/null # Pruefbefund P1a NIEDRIG 5: p1 ist nicht scratch-monoton zu p2
"$T/gen" datei "$HIER/defs_prod8_nah.txt"  "$T/prod8nahp2fh1.cl" pos2fh1 >/dev/null
"$T/gen" datei "$HIER/defs_prod8_nah.txt"  "$T/prod8nahp1h3.cl" pos1h3  >/dev/null
# ★ 15.09.2026 Klemmen Z2d: Betragsklemme (U_BETRAG) allein und mit Positiv-Modus 2 (Arm M-CB1)
"$T/gen" datei "$HIER/defs_prod8_nah.txt"  "$T/prod8nahub.cl"    u       >/dev/null
# ★ 16.09.2026 APG-Umbau (PLAN-APG-2026-09-16.md): Vorkernel fac_nachbar_ab mit DDF-Lesen + neuer APG-Zweig -- eigener Scratch-/Spill-Arm,
# einmal mit dem Konstantgradient-Haken (anderer Zweig im Vorkernel). Nur Nahfeld: dort sind die Facetten.
"$T/gen" datei "$HIER/defs_prod8_nah.txt"  "$T/prod8naha.cl"     a       >/dev/null
"$T/gen" datei "$HIER/defs_prod8_nah.txt"  "$T/prod8nahA.cl"     A       >/dev/null
# ★ 22.09.2026 Mozaffari-Formeltausch (CFD_FAC_APG_MOZ): zwei Divisionen + sqrt + siebenfaches Ternaer in apply_facette_imem -- eigener Arm (Pruefbefund B-M5).
"$T/gen" datei "$HIER/defs_prod8_nah.txt"  "$T/prod8nahM.cl"     M       >/dev/null
"$T/gen" datei "$HIER/defs_prod8_fern.txt" "$T/prod8fernub.cl"   u       >/dev/null
"$T/gen" datei "$HIER/defs_prod8_nah.txt"  "$T/prod8nahp2ub.cl"  pos2u   >/dev/null
# ★ 15.09.2026 Klemmen Z2e/Z2f: Bildhuellen-Tor (Arm M-T) und numerische Huelle (Arm M-CB1B2 = pos2 + u + r)
"$T/gen" datei "$HIER/defs_prod8_nah.txt"  "$T/prod8naht.cl"     t       >/dev/null
"$T/gen" datei "$HIER/defs_prod8_nah.txt"  "$T/prod8nahp2ubr.cl" pos2ur  >/dev/null
"$T/gen" datei "$HIER/defs_prod8_fern.txt" "$T/prod8fernp2ubr.cl" pos2ur >/dev/null

# ★ 28.09.2026 R1Q (Audit A, M1 -- Bedingung vor 4 mm): Produktionsdefines des Nahfelds + CFD_FAC_R1Q. r1q1 = Messmodus
# (FAC_R1Q), r1q4 = Anwendung V_R ohne Druckanteil (FAC_R1Q + _AN + _VR + _OHNE_DRUCK). Neu im Kernel: r1q_einspeisen bekommt das
# private fhn per Zeiger, r1q_pinv/a/b bleiben ueber Pass 2 hinweg live. Die defs-Dateien sind defs_prod8_nah.txt + diese Zeilen.
"$T/gen" datei "$HIER/defs_prod8_nah_r1q1.txt" "$T/prod8nahr1q1.cl" >/dev/null
"$T/gen" datei "$HIER/defs_prod8_nah_r1q4.txt" "$T/prod8nahr1q4.cl" >/dev/null

rc=0
neu_bekannt=""
for dev in 0x7d67 0xe223; do
  for arm in e1p1 e1p0 e0p1 e0p0 e1p1r e1p0r e0p1r e0p0r e1p1s e1p1rs e1p1su e1p1rsu e1p1ruR e1p1uR prod8nah prod8fern prod8nahh3 prod8nahp1 prod8nahp2 prod8fernp2 prod8fernp1 prod8nahp2fh1 prod8nahp1h3 prod8nahub prod8fernub prod8nahp2ub prod8naht prod8nahp2ubr prod8fernp2ubr prod8naha prod8nahA prod8nahM prod8nahr1q1 prod8nahr1q4; do
    ausgabe=$("$HIER/igc_offline.sh" "$T/$arm.cl" "$dev" ALLE || true)
    # ★ 11.09.2026: BAUFEHLER IST NICHT SCRATCH. Vorher fiel ein gescheiterter Bau in beide
    # Gates, weil die Zeile dann schlicht kein "private_size=0" enthielt -- das Gate meldete
    # also "Scratch zurueck", wo in Wahrheit drei Defines fehlten. Zwei verschiedene Befunde
    # unter einer Meldung sind schlimmer als gar keine Meldung.
    if echo "$ausgabe" | grep -q "BUILD FEHLGESCHLAGEN"; then
      echo ">>> BAUFEHLER (nicht Scratch!) in $arm/$dev -- Defines der Zwillingsliste gegen lbm.cpp pruefen"
      rc=1; continue
    fi
    n_kernel=0
    while IFS= read -r zeile; do
      case "$zeile" in *": simd="*) ;; *) continue ;; esac
      n_kernel=$((n_kernel+1))
      kn=${zeile%%:*}
      pv=$(echo "$zeile" | grep -oE 'private_size=[0-9]+' | cut -d= -f2)
      sp=$(echo "$zeile" | grep -oE 'spill_size=[0-9]+'   | cut -d= -f2)
      if [ "${pv:-0}" = "0" ] && [ "${sp:-0}" = "0" ]; then continue; fi
      if echo " $BEKANNT " | grep -q " $kn:$pv:$sp "; then
        echo "    bekannt: $kn private=$pv spill=$sp ($arm/$dev) -- siehe BEKANNT-Liste im Kopf"
        neu_bekannt="$neu_bekannt $kn"
        continue
      fi
      echo ">>> SCRATCH/SPILL-GATE VERLETZT: $kn private=$pv spill=$sp ($arm/$dev)"
      rc=1
    done <<< "$ausgabe"
    # ★ Ein Gate, das nichts findet, weil es nichts SIEHT, ist der eigentliche Defekt.
    # Deshalb ist eine leere Kernelliste selbst ein Fehler.
    if [ "$n_kernel" -lt 5 ]; then
      echo ">>> GATE BLIND: nur $n_kernel Kernel im .zeinfo von $arm/$dev -- Auswertung pruefen"
      rc=1
    else
      echo "$arm $dev: $n_kernel Kernel geprueft"
    fi
  done
done

# Eine BEKANNT-Zeile, die nie zutrifft, ist behoben oder falsch -- beides gehoert gemeldet.
for eintrag in $BEKANNT; do
  kn=${eintrag%%:*}
  case " $neu_bekannt " in *" $kn "*) ;; *)
    echo ">>> BEKANNT-LISTE VERALTET: '$eintrag' trifft nirgends mehr -- Zeile aus dem Kopf entfernen"; rc=1 ;;
  esac
done
exit $rc
