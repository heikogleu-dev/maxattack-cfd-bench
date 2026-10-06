#pragma once

#include "defines.hpp"
#include <unordered_map>
#include <array>
#include "opencl.hpp"
#include "graphics.hpp"
#include "units.hpp"
#include "info.hpp"

// ★ TODO 2 Schritt 4 (12.09.2026) -- Wandlung zwischen rho und seinem Speicherwort.
// Die beiden Funktionen MUESSEN zu den Geraetemakros load_rho/store_rho (lbm.cpp) passen; dort steht
// dieselbe Rechnung mit vload_half/vstore_half_rte. Ohne RHO_FP16 sind beide die Identitaet.
//
// WARUM DAS TRAEGT: die Kette float->Wort->float ist ein WERT-FIXPUNKT,
// rho_unpack(rho_pack(rho_unpack(h))) == rho_unpack(h), bitgleich als float32 und auch nach acht
// Umlaeufen. Nachgerechnet 12.09. mit genau diesen Wandlern: 0 Verletzungen ueber die 59.394
// Bitmuster, die in [0,5; 1,5] landen, und ueber die 60.416 in (0,5; 2,0).
// ES IST KEIN WORT-FIXPUNKT -- BERICHTIGT 12.09. (Pruefer A): rho_pack(rho_unpack(w)) == w gilt nur
// fuer 33.791 der 65.536 Woerter. Bei w = 1 etwa ist rho-1 = 1,8e-12, das verschwindet in der
// float32-Aufloesung bei 1,0, und zurueck kommt das Wort 0. Wer hier eine Bitgleichheit von
// SPEICHERWOERTERN annimmt, baut sich beim naechsten Bytevergleich eine Falle. Was traegt, ist
// allein die WERT-Aussage -- und die genuegt fuer beides unten.
// Daran haengen zwei Dinge, die sonst still brechen wuerden:
//   - pruefe_slice_ebene (setup.cpp) behaelt sein "Soll: exakt 0",
//   - apply_velocity_inlet (kernel.cpp), das nichts als rho[n]=rho[m] tut, driftet nicht.
// WORAN ES HAENGT -- BERICHTIGT 12.09. (Pruefagent): hier stand, die Exaktheit haenge an
// RHO_CLAMP_MIN 0.5f und am 2.0f-Tor im Kopplungs-Lift. Das ist zu eng. Sterbenz ((r-1.0f) exakt
// auf [0,5; 2,0]) ist HINREICHEND, aber nicht notwendig: fuer jedes r, das aus load_rho kommen
// kann, ist (r-1) exakt, und (r-1)*2^15 ist dann per Konstruktion ein half-Wort. Der Fixpunkt
// haengt allein daran, dass BEIDE Skalen exakte Zweierpotenzen bleiben -- 3.0517578E-5f ist
// bitgenau 2^-15 und 32768.0f ist 2^15. Wer RHO_CLAMP_MIN senkt, verliert nichts.
//
// UND DESHALB ist load_rho auch gegen -cl-mad-enable unempfindlich -- BERICHTIGT 12.09.
// (Pruefer A): nicht, weil dort "nichts zu runden" waere. Die Addition von 1.0f RUNDET sehr wohl,
// und zwar genau im interessanten Bereich ((1+d) ist erst ab |d| >= 1,22e-4 exakt). Der Grund ist,
// dass h*2^-15 EXAKT ist (Zweierpotenz, kein Unterlauf -- das kleinste Ergebnis ist 1,8e-12).
// Auf einem exakten Produkt liefern fma und mul+add dieselbe einzige Rundung, also dasselbe Bit.
//
// WAS NICHT GILT (Pruefagent 12.09., MITTEL): Host- und Geraetepacker sind NICHT dieselbe Rechnung.
// float_to_half (utilities.hpp) addiert 0x1000 und schneidet ab, rundet also bei Gleichstand VOM
// NULLPUNKT WEG; vstore_half_rte rundet zur geraden Zahl. An 400.000 zufaelligen rho in 1+-5e-4
// gemessen: 12,6 % abweichende Speicherwoerter. Folgenlos ist das nur, WEIL der Host ausschliesslich
// Werte packt, die selbst aus load_rho stammen (rho_pack(1.0f) und Rho_Feld::set in
// lese_yslice_in_host) -- auf diesen 59.394 Werten stimmen beide Packer exakt ueberein.
// Wer eine Hostschreibstelle ergaenzt, die ein rho-Feld SAET, laeuft still gegen diese Bedingung.
// ★ 15.09.2026 RHO_RAND C2c: HOST-ZWILLING der Randschalen-Packung. AUSDRUCKSGLEICH zu rr_idx() in kernel.cpp
// (RHO_RAND-PLAN.md §4): z<2 -> n; z>=Nz-2 -> 2NxNy + n - (Nz-2)NxNy; sonst Ring je z-Schicht (y<2, y>=Ny-2, x<2, x>=Nx-2).
// Innenzellen liefern r1_anzahl (= Papierkorb-Slot). Der Selbsttest in pruefe_rho_rand_c0 beweist die Bijektion auf R1.
inline ulong r1_anzahl(const uint Nx, const uint Ny, const uint Nz) {
	return (Nx>4u&&Ny>4u&&Nz>4u) ? (ulong)Nx*(ulong)Ny*(ulong)Nz-(ulong)(Nx-4u)*(ulong)(Ny-4u)*(ulong)(Nz-4u) : (ulong)Nx*(ulong)Ny*(ulong)Nz;
}
inline ulong rr_idx_host(const ulong n, const uint Nx, const uint Ny, const uint Nz) {
	const ulong a = (ulong)Nx*(ulong)Ny;
	const uint x = (uint)(n%(ulong)Nx), y = (uint)((n/(ulong)Nx)%(ulong)Ny), z = (uint)(n/a);
	if(z<2u) return n;
	if(z+2u>=Nz) return 2ull*a+(n-(ulong)(Nz-2u)*a);
	const ulong r0 = 4ull*a+(ulong)(z-2u)*(4ull*(ulong)Nx+4ull*(ulong)(Ny-4u));
	if(y<2u) return r0+(ulong)x+(ulong)y*(ulong)Nx;
	if(y+2u>=Ny) return r0+2ull*(ulong)Nx+(ulong)x+(ulong)(y+2u-Ny)*(ulong)Nx;
	if(x<2u) return r0+4ull*(ulong)Nx+4ull*(ulong)(y-2u)+(ulong)x;
	if(x+2u>=Nx) return r0+4ull*(ulong)Nx+4ull*(ulong)(y-2u)+(ulong)(x+4u-Nx);
	return r1_anzahl(Nx, Ny, Nz);
}
// ★ 04.10.2026 U_RAND (PLAN-VRAM-URAND-FLAGS-2026-10-04.md B.4): KOPFBELEGUNG des kompakten u-Puffers, EINZIGE Quelle. Die ersten
// KOPF_W uint-Woerter des u-Puffers tragen diese Werte; lbm.cpp emittiert daraus die URK_*-Defines, der Geraetecode liest sie ueber
// ur_k(). Danach die Maske von A (Modus 3) in uint-Woertern, ab DOFF (in velxx-Woertern) die Daten: je Komponente US Slots R1|A|V|Papierkorb.
// (★ 05.10.2026 Korrektur C-N8: Segment C ist entfallen, die N2F-Bloecke liegen in A -- hier stand "Masken (A, C)" und "R1|A|C|V".)
namespace ur_k {
	enum : uint { MAGIC=0u, US=1u, DOFF=2u, R1N=3u, AN=4u, AX0=5u, AY0=6u, AZ0=7u, ANX=8u, ANY=9u, ANZ=10u, AMASK=11u, AMOFF=12u,
		VOFF=13u, VN=14u, VBX0=15u, VBY0=16u, VBZ0=17u, VBNX=18u, VBNY=19u, VBNZ=20u, P=21u,
		KOPF_W=64u }; // ★ U1c: kein Segment C mehr (die N2F-Bloecke liegen in A); V-Box (VBX0..VBNZ) setzt der Host je Leseplan-Schritt
	constexpr uint GUELTIG = 0x55524E44u;     // "URND": Kopf gueltig
	constexpr uint PLATZHALTER = 0x0BAD0BADu; // Platzhalter vor alloc_u_rand (ab U1b); Slot 485 zaehlt jeden Lesezugriff darauf
	constexpr uint REGEL_VX = 64u;            // velxx-Index der Init-Regel im Kopf: Nicht-Solid x,y,z = 64..66, reines Solid 67..69 (uint-Woerter 32..34) -- NUR mit U_FP16 (2-Byte-velxx); Sperre in LBM_Domain::allocate und alloc_u_rand (Pruefbefund M2, 05.10.2026)
}
// ★ 05.10.2026 U_RAND Korrektur M1/N2 (PLAN-VRAM-URAND-FLAGS-2026-10-04.md B.15): Groesse der Ausgabe V aus dem Gitter (EINZIGE Quelle fuer
// ur_v_anlegen UND die VRAM-Vorpruefung im Konstruktor); scheiben != nullptr bekommt die VOLL-Scheibengrenzen.
ulong ur_v_layout(const ulong Nx, const ulong Ny, const ulong Nz, std::vector<ulong>* scheiben);
// Obergrenze des kompakten u-Puffers in Byte VOR dem Zensus (Konstruktor-Vorpruefung); alloc_u_rand vergleicht den echten Puffer dagegen.
ulong ur_vorpruef_bytes(const uint Nx, const uint Ny, const uint Nz, const uint modus);
inline float rho_unpack(const rhoxx w) { // Speicherwort -> rho
#ifdef RHO_FP16
	// ★★ 12.09.2026 (Audit-Schleife, Pruefer A, HOCH): half_to_float ist ausdruecklich "without
	// infinity" -- es bildet ALLE 2048 Woerter mit Exponent 0x1F auf ENDLICHE Floats ab. Ohne die
	// Zeile unten saehe der Host nach einer Dichteexplosion auf dem Geraet eine glatte 3,0
	// (0x7C00 = Geraete-inf -> 3.0, 0xFC00 -> -1.0), und der NaN-Zaehler der Slice-Pruefung
	// (setup.cpp, std::isnan(...)!=std::isnan(...)) waere KONSTRUKTIV NULL. Genau die Bauform,
	// die dieses Projekt jagt: ein Waechter, der nicht feuern kann.
	// vstore_half_rte saettigt ab rho >= 3,0 nach 0x7C00 -- der Fall ist also erreichbar, nicht
	// theoretisch. Deshalb wird das Bitmuster hier ehrlich weitergereicht: Mantisse 0 -> +-inf,
	// sonst NaN, Vorzeichen erhalten. Das kostet den Geraetepfad nichts (reine Hostfunktion).
	if((w&0x7C00u)==0x7C00u) return as_float((uint)(w&0x8000u)<<16 | 0x7F800000u | (uint)(w&0x03FFu)<<13);
	return half_to_float(w)*3.0517578E-5f+1.0f; // NICHT fma: der Fixpunktbeweis und das Geraetemakro rechnen getrennt
#else // RHO_FP16
	return w;
#endif // RHO_FP16
}
inline rhoxx rho_pack(const float r) { // rho -> Speicherwort
#ifdef RHO_FP16
	return float_to_half((r-1.0f)*32768.0f);
#else // RHO_FP16
	return r;
#endif // RHO_FP16
}

// ★ TODO 2 Schritt 4 (12.09.2026) -- Speicherformat von u auf der HOSTSEITE. FP16S OHNE Verschiebung:
// Wort = half(u*2^15), zurueck = Wort*2^-15. Die Begruendung, warum hier KEINE Verschiebung steht,
// obwohl rho eine hat, steht vollstaendig an U_FP16 in defines.hpp -- kurz: u hat keinen Sockel, und
// die Verschiebung wuerde den Wort-Fixpunkt zerstoeren.
//
// DER FIXPUNKT, und er ist staerker als der von rho: 3.0517578E-5f ist bitgenau 2^-15 und 32768.0f
// ist 2^15, beide Multiplikationen runden also nicht, und einen Unterlauf gibt es nicht (das kleinste
// Ergebnis ist 1,8e-12). Damit ist u_pack(u_unpack(w)) == w als BITMUSTER, nicht nur als Wert.
// Mit den Wandlern dieser Datei nachgerechnet: 0 Verletzungen ueber die 63.488 ENDLICHEN Woerter,
// auch nach acht Umlaeufen. Daran haengen pruefe_slice_ebene (Soll exakt 0) und apply_pressure_outlet.
// AUSGENOMMEN sind die 2048 Woerter mit Exponent 0x1F (BERICHTIGT 12.09., Pruefagent HOCH -- hier
// stand "alle 65536"): der Inf/NaN-Durchreicher unten liefert dort +-inf oder NaN, und float_to_half
// saettigt das auf 0x7FFF. 2046 der 2048 verletzen den Fixpunkt. Kein endliches u erreicht sie --
// die Geschwindigkeitsklemme haelt +-0,57735, das Format traegt bis 1,99902.
//
// HOST- UND GERAETEPACKER SIND NICHT DIESELBE RECHNUNG -- und anders als bei rho ist das hier NICHT
// von vornherein folgenlos, weil der Host u an 56 Stellen SAET (Freistrom, Wandgeschwindigkeit,
// Kanalprofil) statt nur zurueckzuschreiben, was er geladen hat. float_to_half addiert 0x1000 und
// schneidet ab (Gleichstand VOM NULLPUNKT WEG), vstore_half_rte rundet zur GERADEN Zahl. Deshalb
// nachgemessen statt angenommen: an 4.000.000 zufaelligen u innerhalb der Geschwindigkeitsklemme
// weichen 267 Speicherwoerter ab, also 0,007 % (Ziehung: rand() ueber +-def_c, Saat 12345 -- ein
// zweiter Pruefer kommt mit mt19937 und uniform_real_distribution auf 0,062 %, weil dessen
// 24-Bit-Raster ueberproportional viele Gleichstaende erzeugt. Die ZAHL haengt am Ziehungsverfahren
// und ist ohne dessen Angabe nicht reproduzierbar; belastbar ist nur die Richtung: die Abweichung
// ist immer genau EIN half-ULP und tritt nur an Gleichstaenden auf). Und die Werte, die der Host WIRKLICH saet, stimmen
// exakt ueberein: 0,0 / 0,05 / 0,075 / 0,1 / 0,125 liefern in beiden Packern dasselbe Wort.
// ★ BERICHTIGT 12.09. abends (Pruefer B und C, unabhaengig): hier stand, pruefe_slice_ebene sei der
// "laufende Nachweis" dieser Deckung. DAS KANN ER NICHT SEIN. Der Gather-Pfad legt dem Hostpacker
// ausschliesslich Werte vor, die selbst aus u_unpack stammen -- also exakt darstellbare half-Werte.
// An denen trifft float_to_half nie eine Rundungsentscheidung, dmax ist per Wort-Fixpunkt exakt 0,
// und ein Gleichstand, an dem sich die Packer unterscheiden, kann dort nicht auftreten. Genau die
// Bauform, die dieser Commit an anderer Stelle als Lehre festhaelt: ein Waechter, der nicht feuern
// kann. Fuer Indexabbildung, Transfer, rho und flags bleibt er wertvoll.
// Was die Deckung WIRKLICH belegen wuerde: die Saatwerte selbst pruefen (unten aufgezaehlt) oder ein
// Wort, das der GERAETECODE schreibt und der Host roh zurueckliest. Nicht gebaut.
inline float u_unpack(const velxx w) { // Speicherwort -> Geschwindigkeitskomponente
#ifdef U_FP16
	// ★★ Derselbe HOCH-Befund wie bei rho_unpack, und er wiegt hier schwerer: half_to_float ist
	// ausdruecklich "without infinity" und bildet ALLE 2048 Woerter mit Exponent 0x1F auf ENDLICHE
	// Floats ab. Ohne die Zeile unten saehe der Host nach einer Geschwindigkeitsexplosion auf dem
	// Geraet eine glatte 2,0 (0x7C00 -> 2.0, 0x7E00 -> 3.0), und die beiden u-Waechter, die es
	// wirklich gibt -- std::isnan in pruefe_slice_ebene und std::isfinite in der Sondenauswertung --
	// waeren KONSTRUKTIV NULL. Genau die Bauform, die dieses Projekt jagt: ein Waechter, der nicht
	// feuern kann. Kostet den Geraetepfad nichts, das hier ist reine Hostarithmetik.
	if((w&0x7C00u)==0x7C00u) return as_float((uint)(w&0x8000u)<<16 | 0x7F800000u | (uint)(w&0x03FFu)<<13);
	return half_to_float(w)*3.0517578E-5f; // NICHT fma: Fixpunktbeweis und Geraetemakro rechnen getrennt
#else // U_FP16
	return w;
#endif // U_FP16
}
inline velxx u_pack(const float v) { // Geschwindigkeitskomponente -> Speicherwort
#ifdef U_FP16
	return float_to_half(v*32768.0f);
#else // U_FP16
	return v;
#endif // U_FP16
}

uint bytes_per_cell_host(); // returns the number of Bytes per cell allocated in host memory
uint bytes_per_cell_device(); // returns the number of Bytes per cell allocated in device memory
const char* vram_quelle(); // welcher Weg den letzten Wert geliefert hat -- Meldungen nennen ihn, statt ihn zu behaupten
ulong vram_frei_gemessen(const ulong kapazitaet_mib=0ull, const uint haken_stufe=0u); // ★ 06.10. ZKS: haken_stufe 1/2 = Vorpruefung Stufe 1/2 (Testhaken CFD_VRAM_HAKEN_STUFE), 0 = alle uebrigen Aufrufer // // ★ 29.08.: freier VRAM GEMESSEN -- Debugfs, sonst Summe ueber alle DRM-Clients aus /proc/*/fdinfo (12.09.); 0 = kein Weg trug;
                           // device.info.memory ist nur die 20/19-Rekonstruktion und sieht den Desktop nicht
uint bandwidth_bytes_per_cell_device(); // returns the bandwidth in Bytes per cell per time step from/to device memory
uint3 resolution(const float3 box_aspect_ratio, const uint memory); // input: simulation box aspect ratio and VRAM occupation in MB, output: grid resolution

string default_filename(const string& path, const string& name, const string& extension, const ulong t); // generate a default filename with timestamp
string default_filename(const string& name, const string& extension, const ulong t); // generate a default filename with timestamp at exe_path/export/

#pragma warning(disable:26812)
enum enum_transfer_field { fi, rho_u_flags, flags, F, phi_massex_flags, gi, T, enum_transfer_field_length };

// C1b: Host-Facette (baue_facetten in setup.cpp fuellt sie, LBM::alloc_facetten laedt sie hoch)
struct Facette {
	float nx, ny, nz, yw; // Normale (ins Fluid), Wandabstand des Zellzentrums zur Ausgleichsebene
	float cx_, cy_, cz_;  // Fit-Schwerpunkt (fuer y_w-Neuberechnung nach der Glaettung, Nachpruefer B3)
	float r21_, r10_;     // Eigenwertverhaeltnisse lmin/lmid (K2) und lmid/lmax (K3) -- fuer die Schwelleneichung
	uint  n_punkte;       // Stuetzpunkte (geschnittene Links) -- Flaechenproxy fuer die Glaettung
	uint  eigene_links;   // davon Links DIESER Zelle (fuer Akkumulator-Hygiene in Stufe 2)
	uchar klasse;         // 0 sauber, sonst Bitmaske K1=1 K2=2 K3=4 K4=8 Orientierung=16 Ueberlauf=32
	uchar achse;          // dominante Achse 0/1/2, Tie-Break: kleinste Achsnummer
	ulong n;              // Zellindex in der Domaene
};

ulong zaehl_takt(); // ★ 11.09.2026 gemeinsamer Zaehltakt fuer Kernel-Gatter UND Host-Sollformeln
bool klemm_bilanz_env(); // ★ 15.09.2026 Klemmen S0b (lbm.cpp)
uint klemm_haken_env();
uint u_klemme_env(); // ★ Z2d (lbm.cpp)
uint tor_huelle_env(); // ★ Z2e
uint rho_huelle_env(); // ★ Z2f
bool rho_huelle_aktiv(); // ★ Audit 16.09.2026 (B2/C-M1): WIRKSAMER Zustand -- Haken 1/3/4 uebersteuern die Emission; Bericht/Wickelschranke/Soll muessen DIESE Funktion lesen, nicht die Umgebung
uint positiv_env(); // ★ 15.09.2026 Klemmen Stufe 1 P1a (lbm.cpp): CFD_POSITIV 0/1/2, CFD_POSITIV_HAKEN 0..3, CFD_POSITIV_FACETTE 0/1
uint positiv_haken_env();
uint positiv_facette_env();
string positiv_defines(const uint modus, const uint haken, const uint facette, const bool fp16s, const unsigned long long N, const uint Nx, const uint Ny); // kernel.cpp, einzige Quelle (auch Scratch-Gate)
uint positiv_stichprobe(const unsigned long long N, const uint Nx, const uint Ny); // kernel.cpp: kleinste Primzahl >= ceil(N / sicheres Gitter), teilt weder Nx noch Ny
unsigned long long positiv_sicheres_gitter(); // kernel.cpp: groesstes Gitter mit belegt sicheren Atomics in fast jeder Zelle je Schritt (Kugel 16 mm)
uint positiv_haken_periode(); // kernel.cpp: Periode P der Haken-1-Zellen (Kernel-Define def_pos_hP und Host-Soll)

// ★ 12.09.2026 (Heiko): SCHRITTBASIERTE SCHALTER FOLGEN u_lat JETZT VON SELBST.
// Bis heute taten sie es ausdruecklich NICHT -- die Begruendung stand an u_lat_schalter in
// setup.cpp: eine stille Umrechnung waere eine weitere Variable im Arm. Heiko hat das am 12.09.
// umgedreht, und zwar mit dem besseren Argument: ein Schalter, den man bei jeder Aenderung von
// Hand nachziehen muss, wird irgendwann vergessen, und dann traegt der Arm ZWEI Aenderungen statt
// einer -- unsichtbar. Die Umrechnung ist jetzt automatisch und LAUT: jeder betroffene Schalter
// meldet seinen alten und seinen neuen Wert und die physikalische Zeit dahinter.
// skal = u_lat(Vorgabe)/u_lat(Lauf). Schritte fuer dieselbe physikalische Zeit skalieren damit,
// weil dt = u_lat*dx/si_u proportional zu u_lat ist. Gilt fuer feine WIE grobe Schritte, weil
// dt_c = ratio*dt_f und ratio unberuehrt bleibt.
double ulat_skal();                     // 1.0, solange u_lat auf der Vorgabe steht
void   ulat_skal_setzen(const double s); // genau einmal, aus u_lat_schalter, VOR dem ersten Leser
// ★ 16.09.2026 (TODO 4a, Heiko 14:58 "Zeiten folgen der Aufloesung"): Schritt-Schalter folgen dx UND u_lat, weil dt = u_lat*dx/si_u.
// Referenzsprosse der schrittbasierten Schalter ist 4 mm (setup.cpp Kopfkommentar: dt = 1e-5 s bei 4 mm, u_lat 0,075, si_u 30).
// Nur die drei CFD_DX-Faelle (fahrzeug, fahrzeug_dd, fernfeld) sind auf 4 mm definiert; Kugel (CFD_KUGEL_DX, eigene Schrittwerte je Zeile),
// Kanal (T aus T_ett) und facetten_test bekommen exakt 1,0. Umgebungsrein wie ulat_skal, weil fahrzeug_dd env_schritte VOR dem dx-Leser ruft.
#define DX_SCHRITT_VORGABE_MM 4.0f
double dx_skal();                      // DX_SCHRITT_VORGABE_MM/CFD_DX in den drei CFD_DX-Faellen, sonst exakt 1.0
void   dx_skal_setzen(const double s); // Ist=Soll aus dem Fall, wie ulat_skal_setzen
double schritt_skal();                 // = ulat_skal()*dx_skal(): DER Faktor fuer env_schritte und zaehl_takt

int sc_simd16_geraet(); // ★ 02.10.2026 Messarm CFD_SC_SIMD16=<OpenCL-Geraete-ID>, -1 = aus (lbm.cpp, streng gelesen)
inline ulong domaenen_bau_zaehler = 0ull; // ★ 05.10.2026 A-N4/B-N4: zaehlt Domaenenbauten (nur Hauptthread; parallel gebaute Mehrdomaenen-Zerlegungen bekommen trotzdem verschiedene Werte, solange die Konstruktoren seriell laufen)
uint zellbasen_modus(); // ★ 05.10.2026 CFD_ZELLBASEN (PLAN-ZELLBASEN-2026-10-05.md): 1 = Byte-Zellbasen in stream_collide (Vorgabe), 0 = alter Pfad (index_f); streng gelesen, global fuer alle Domaenen
uint rand_dispatch_modus(); // ★ 06.10.2026 CFD_RAND_DISPATCH (F1, PLAN-KERNEL-ARCHITEKTUR Rang 1): 1 = boden_eq per Bereichsstart ueber seine z-Lagen (Kernel zeichengleich); 0 = Volllauf ueber N (Vorgabe); streng gelesen, global fuer alle Domaenen
// ★ 04.10.2026 KRAFT-P1 (Heiko-Entscheid, Bauplan Planungsagent): Ergebnis des Kernels kraft_p1_gpu in Gittereinheiten,
// double-Endsumme in fester Gruppenreihenfolge. P1 = Sum 2 w_i (rho_quelle - 1) c_i ueber dieselbe Linkmenge wie F
// (Quelle j[ib] nicht Solid), Lage K = z-Index der Solidzelle, K = 0..7 einzeln. ok = false: nicht gebunden oder t = 0.
struct KP1 {
	double px = 0.0, py = 0.0, pz = 0.0;
	double lx[8] = {}, lz[8] = {};
	unsigned long long links = 0ull, typ_e = 0ull, klemme = 0ull, wandzellen = 0ull;
	unsigned long long links_lage[8] = {};
	bool ok = false;
};
// ★ 05.10.2026 FLAGS4 (PLAN-VRAM-URAND-FLAGS-2026-10-04.md Teil C): flags in 4 Bit auf dem Geraet. Vorkommende Bits sind 0, 1, 6, 7
// (TYPE_S, TYPE_E, TYPE_X, TYPE_Y); TYPE_T/F/I/G (Bits 2..5) gibt es nur unter TEMPERATURE/SURFACE, dort ist FLAGS4 gesperrt.
// Bitumrechnung ohne Tabelle, bijektiv auf die 16 Kombinationen der Bits {0,1,6,7}:
//   enc(f) = (f & 0x03) | ((f >> 4) & 0x0C)      dec(c) = (c & 0x03) | ((c & 0x0C) << 4)
// Speicherform: Zelle n in Byte n>>1, Nibble n&1 (gerade Zelle unten). Little-Endian ist das gleichwertig zu uint-Wort n>>3, Verschiebung 4*(n&7).
inline uchar flags4_enc(const uchar f) { return (uchar)((f&0x03u)|((f>>4u)&0x0Cu)); }
inline uchar flags4_dec(const uchar c) { return (uchar)((c&0x03u)|((c&0x0Cu)<<4u)); }
inline ulong flags4_bytes(const ulong N) { return 4ull*((N+7ull)/8ull); } // gepackte Laenge in Byte (ganze uint-Woerter: die Kernel-Schreiber arbeiten mit atomic_and/atomic_or auf dem Wort)
ulong flags4_packen(const uchar* f, const ulong N, uchar* p); // packt N Zellen nach p (flags4_bytes(N) Byte, Rest 0); liefert die Zahl der Zellen mit fremden Bits (f & 0x3C), Soll 0
void flags4_entpacken(const uchar* p, const ulong N, uchar* f); // Umkehrung
uint flags4_selbsttest(); // F0: Rundreise aller 16 Codes und eines Pseudozufallsfelds (ungerades N); liefert die Zahl der Beanstandungen, Soll 0
// ★ 05.10.2026 FLAGS4 F3 (PLAN-VRAM-URAND-FLAGS-2026-10-04.md C.2): flags je Domaene. Der Host behaelt das VOLLE Bytefeld h (alle
// Hostleser bleiben unveraendert); das Geraet bekommt k(). FLAGS4=0: k() = h, also derselbe Memory wie bisher (Host+Geraet, N Byte).
// FLAGS4=1: h ist nur Host, k() = g ist der gepackte Puffer (4 Bit je Zelle, ganze uint-Woerter) mit einem Host-Schatten zum Packen.
// Dass das hier KEIN Memory ist, ist Absicht: eine vergessene Kernelbindung "…, flags, …" bindet die Klasse als Konstante mit
// sizeof(Flags_Puffer) Byte und scheitert beim Kernelbau mit CL_INVALID_ARG_SIZE (-51) -- laut, vor dem ersten Schritt, auf jedem Geraet.
// Und read_from_device/write_to_device packen/entpacken selbst, so dass keine der rund 40 Synchronisationsstellen still ins Leere laeuft
// (ein reiner Host-Memory machte aus jedem vergessenen Aufruf einen stillen No-Op).
class Flags_Puffer {
private:
	Memory<uchar> h; // volles Bytefeld je Zelle (Host); FLAGS4=0 zugleich der Geraetepuffer
	Memory<uchar> g; // FLAGS4=1: gepackter Geraetepuffer flags4_bytes(N) Byte, Host-Schatten fuer Packen/Entpacken
	Device* dev = nullptr;
	ulong N = 0ull;
	bool vier = false;
	uint haken = 0u; // CFD_FLAGS4_HAKEN=1: beim ERSTEN Hochladen ein Nibble im gepackten Puffer kippen (Soll: Geraeteprobe reisst)
public:
	ulong n_hoch = 0ull, n_runter = 0ull; // Wirkpfad: gepackte Hoch-/Runterladungen
	ulong rundreise_abw = 0ull, fremd = 0ull; // Summen ueber alle Hochladungen, Soll 0
	long haken_zelle = -1l; // gekippte Zelle (Testhaken), sonst -1
	inline Flags_Puffer() {}
	Flags_Puffer(const Flags_Puffer&) = delete;
	Flags_Puffer& operator=(const Flags_Puffer&) = delete;
	void anlegen(Device& device, const ulong N, const bool vier, const uint haken); // lbm.cpp
	inline uchar& operator[](const ulong i) { return h[i]; }
	inline const uchar& operator[](const ulong i) const { return h[i]; }
	inline uchar* data() { return h.data(); }
	inline const uchar* data() const { return h.data(); }
	inline const ulong length() const { return h.length(); }
	inline Memory<uchar>& k() { return vier ? g : h; } // Kernelbindung
	inline Memory<uchar>& host() { return h; } // fuer den Memory_Container der Huelle (nur Hostzugriff)
	inline bool ist_vier() const { return vier; }
	inline ulong geraet_bytes() const { return vier ? g.capacity() : h.capacity(); }
	void read_from_device(); // blockierend; FLAGS4: gepackt lesen (Zero-Copy: finish), dann entpacken
	void write_to_device(const bool blocking=true); // FLAGS4: Queue leeren, packen (Fremdbit- und Rundreisepruefung), BLOCKIEREND schreiben
	inline void enqueue_write_to_device() { write_to_device(false); }
	void geraet_in(std::vector<uchar>& ziel); // Geraeteinhalt entpackt in ziel lesen, OHNE h anzufassen (Geraeteprobe nach initialize)
};
class LBM_Domain {
private:
	uint Nx=1u, Ny=1u, Nz=1u; // (local) lattice dimensions
	uint Dx=1u, Dy=1u, Dz=1u; // lattice domains
	int Ox=0, Oy=0, Oz=0; // lattice domain offset
	ulong t = 0ull; // discrete time step in LBM units

	float nu = 1.0f/6.0f; // kinematic shear viscosity
	float fx=0.0f, fy=0.0f, fz=0.0f; // global force per volume
	float sigma=0.0f; // surface tension coefficient
	float alpha=1.0f, beta=1.0f, T_avg=1.0f; // alpha = thermal diffusion coefficient, beta = (volumetric) thermal expansion coefficient, T_avg = 1 = average temperature
	uint particles_N = 0u;
	float particles_rho = 1.0f;

	Device device; // OpenCL device associated with this LBM domain
	Kernel kernel_initialize; // initialization kernel
	Kernel kernel_stream_collide; // main LBM kernel
	Kernel kernel_update_fields;
	Kernel kernel_boden_eq; // V1-apply_floor_velocity-Port; schreibt NUR fi -- rho/u der Bandzellen zeigen bis zum naechsten update_fields den VOR-Reset-Stand (Sonden/Slices/Kopplungs-Extraktion lesen pre-Reset; XL-3)
	Kernel kernel_einlass_eq; // V1-apply_inlet_velocity-Port; wie boden_eq: schreibt NUR fi, Sonden lesen pre-Reset
	Memory<fpxx> fi; // LBM density distribution functions (DDFs); only exist in device memory
	ulong t_last_update_fields = max_ulong; // optimization to not call kernel_update_fields multiple times if (rho, u, T) are already up-to-date
	// FORK -- Block-Tiling (sparse solid): fi nur fuer aktive Tiles allozieren. VRAM-gegen-Tempo-Regler,
	// physikalisch bit-neutral. Default AUS -> tile_slot bleibt ein 1-Element-Platzhalter, die Makros
	// TS_P/TS_A sind leer und der erzeugte Device-Code ist bit-identisch zu Upstream.
	Memory<uint> tile_slot; // tile_id -> kompakter Slot; 0xFFFFFFFF = tote Tile
	uint sparse_tiles_x = 0u, sparse_tiles_y = 0u, sparse_tiles_z = 0u;
	// Read-once-Kopien der statischen Schalter, im Konstruktor uebernommen. Alles ausserhalb des
	// Konstruktors liest AUSSCHLIESSLICH diese -- sonst aenderte ein Schalterwechsel zwischen zwei
	// Domaenen rueckwirkend das Verhalten der ersten (finalize_sparse_tiles laeuft viel spaeter).
	bool sparse_on = false;
	uint sparse_T = 8u;
	// FORK -- Druck-Auslass. Leer, solange set_pressure_outlet_faces() nicht gerufen wurde; der Kernel
	// bleibt dann default-konstruiert und enqueue_apply_pressure_outlet() ist ein No-op.
	// po_interior wird HOST-seitig bestimmt, nicht device-seitig aus einer Richtung abgeleitet: nur so
	// sind Kanten und Ecken (Zelle liegt auf zwei oder drei Auslassflaechen) sauber loesbar, und nur so
	// laesst sich vorab pruefen, dass jede Innenzelle wirklich Fluid ist und jede Randzelle genau einmal
	// vorkommt. Die frueher gespeicherte Richtung (po_dirs) konnte beides nicht.
	Memory<uint> po_cells;    // Randzellen, jede genau einmal (★ 08.09. uint statt ulong: N < 2^32, spart die Haelfte)
	Memory<uint> po_interior; // zugehoerige echte Innenzelle, aus der extrapoliert wird (★ 08.09. uint)
	Kernel kernel_apply_pressure_outlet;
	uint po_N_active = 0u;
	// FORK -- Geschwindigkeits-Einlass: u bleibt vorgeschrieben (das macht TYPE_E), rho laeuft mit.
	// Spiegelbild des Druck-Auslasses; beide benutzen denselben Sammler collect_boundary_pairs.
	Memory<ulong> vi_cells, vi_interior;
	Kernel kernel_apply_velocity_inlet;
	uint vi_N_active = 0u;
	bool collect_boundary_pairs(const uint face_mask, const string& wofuer, std::vector<ulong>& cells, std::vector<ulong>& interior);
public:
	// ★ oeffentlich seit 2026-08-15: die y+-Messung (messe_yplus, setup.cpp) liest den F-Puffer
	// direkt mit der BBox-Indizierung -- die Huellen-Sicht lbm.F waere die U1-Falle (rechnet mit
	// voller Domaenengroesse, Puffer ist BBox-gross).
	uint fbx0=0u, fby0=0u, fbz0=0u, fbnx=0u, fbny=0u, fbnz=0u; // FORK: aktive F-Bounding-Box dieser Domaene
	uint smx0=0u, smy0=0u, smz0=0u, smnx=0u, smny=0u, smnz=0u; // ★ TODO 2: Schreibmasken-Box dieser Domaene
private:
	float po_rho = 1.0f; // vorgeschriebene Dichte am Auslass (LBM-Einheiten); 1.0 = Referenzdruck
	float po_sigma = 1.0f; // Ankerrate des Flaechenmittels gegen rho_out
	uint po_hart = 0u;     // 1 = alter harter Rand je Zelle (CFD_PO_HART), der Kontrollarm
	Memory<float> po_mean;   // Mittelwert der Dichte ueber die Innenzellen der Auslassebene, je Schritt neu
	Memory<float> po_part;   // Teilsumme je Arbeitsgruppe -- ersetzt die atomare Addition (2026-08-24)
	uint po_groups = 0u;     // Zahl der Arbeitsgruppen = ceil(N_po/WORKGROUP_SIZE)
	Kernel kernel_po_reduce_mean, kernel_po_final_mean;
#ifdef FORCE_FIELD
	Kernel kernel_update_force_field; // calculate forces from fluid on TYPE_S cells
	Kernel kernel_reset_force_field; // reset force field (also on TYPE_S cells)
	Kernel kernel_object_center_of_mass; // calculate center of mass of all cells flagged with flag_marker
	Memory<float> of_part;   // ★ 2026-08-25 Teilsumme (x,y,z) je Arbeitsgruppe -- ersetzt atomic_add_f
	uint of_groups = 0u;     // FESTE Zahl Arbeitsgruppen (Grid-Stride) -> feste Summationsreihenfolge
	Kernel kernel_object_force_final;
	Kernel kernel_object_force; // add up force for all cells flagged with flag_marker
	Kernel kernel_object_force_zband; // FORK Kraft-Zerlegung (CFD_KRAFT_ZBAND): object_force auf das z-Band [z_lo,z_hi)
	Kernel kernel_object_torque; // add up torque around specified rotation_center for all cells flagged with flag_marker
	ulong t_last_force_field = max_ulong; // optimization to not call kernel_update_force_field multiple times if F is already up-to-date
#endif // FORCE_FIELD
#ifdef MOVING_BOUNDARIES
	Kernel kernel_update_moving_boundaries; // mark/unmark cells next to TYPE_S cells with velocity!=0 with TYPE_MS
#endif // MOVING_BOUNDARIES
#ifdef SURFACE
	Kernel kernel_surface_0; // additional kernel for computing mass conservation and mass flux computation
	Kernel kernel_surface_1; // additional kernel for flag handling
	Kernel kernel_surface_2; // additional kernel for flag handling
	Kernel kernel_surface_3; // additional kernel for flag handling and mass conservation
	Memory<float> mass; // fluid mass; phi=mass/rho
	Memory<float> massex; // excess mass; used for mass conservation
#endif // SURFACE
#ifdef TEMPERATURE
	Memory<fpxx> gi; // thermal DDFs
#endif // TEMPERATURE
#ifdef PARTICLES
	Kernel kernel_integrate_particles; // intgegrates particles forward in time and couples particles to fluid
#endif // PARTICLES

	void allocate(Device& device); // allocate all memory for data fields on host and device and set up kernels
	string device_defines(const Device_Info& device_info) const; // returns preprocessor constants for embedding in OpenCL C code
	string u_rand_defines() const; // ★ 04.10.2026 U_RAND: URK_*-Defines aus ur_k (einzige Quelle), dazu U_RAND/UR_MAGIC/UR_REGEL_VX

public:
	// FORK -- Block-Tiling. Statisch, weil das Setup den Schalter setzen muss, BEVOR der LBM-Konstruktor
	// laeuft: bei aktivem Sparse wird fi zunaechst nur als 1-Zell-Platzhalter alloziert (ein spaeteres
	// Free des vollen fi-Buffers bringt den Intel-Treiber mit CL_OUT_OF_RESOURCES zu Fall).
	// finalize_sparse_tiles() legt die echte sparse fi an -- NACH der Voxelisierung, weil erst dann
	// feststeht, welche Tiles voll solid sind.
	// FORK -- F-Bounding-Box: F nur um den Koerper allozieren statt ueber die ganze Domaene.
	// MUSS vor der LBM-Konstruktion gesetzt werden, weil allocate() F sonst auf N legt. Wird nach dem
	// Lesen zurueckgesetzt (read-once), damit eine zweite Domaene nicht versehentlich dieselbe Box erbt.
	static uint s_fbbox[6]; // {x0, y0, z0, nx, ny, nz}; nx==0 -> volle Domaene
	static uint s_smbox[6]; // ★ TODO 2: SCHREIBMASKEN-Box {x0,y0,z0,nx,ny,nz}; nx==0 -> faellt auf die F-BBox zurueck.
	                        // Nahfeld: die Facetten-BBox (dort lesen sgs_fdwand/fac_nachbar_ab).
	                        // Fernfeld: der Fussabdruck des Nahfelds (dort entnimmt extract_plane_macros jeden Grobschritt).
	static void set_force_bbox(const uint x0, const uint y0, const uint z0, const uint nx, const uint ny, const uint nz);
	void set_velocity_inlet_faces(const uint face_mask); // FORK: Geschwindigkeits-Einlass, rho laeuft mit
	void enqueue_apply_velocity_inlet();

	// FORK -- Doppel-Domaene: EIN Streifenpuffer, gross genug fuer die groesste vorkommende Ebene, plus
	// zwei Kernel. Nur belegt, wenn LBM::alloc_coupling_planes() gerufen wurde; sonst bleibt alles unangetastet.
	// Der Puffer haelt 4 floats je Zelle (rho, u_x, u_y, u_z). Grobe Ebenen sind klein -- ein paar hundert kB --,
	// darum genuegt EIN Puffer fuer alle fuenf Flaechen nacheinander.
	// Oeffentlich, weil die Kopplung von der LBM-Ebene aus gefahren wird und nicht von der Domaene.
	Memory<float> coupling_plane;
	ulong coupling_max_plane_cells = 0ull;
	Kernel kernel_extract_plane_macros;
	Memory<uchar> slice_flags; // ★ Slice-Ebenen-Read 2026-08-26: flags-Ebene (1 B/Zelle), Groesse = coupling_max_plane_cells
	Kernel kernel_extract_plane_flags;
	Kernel kernel_drive_boundary_cubic_lift;
	void alloc_coupling_planes(const ulong max_plane_cells); // legt coupling_plane an und bindet beide Kernel
	// ★ 15.09.2026 RHO_RAND C1: Rekonstruktion von rho einer Ebene aus den DDFs. Eigene Puffer (nicht coupling_plane,
	// den die Kopplung jeden Grobschritt beschreibt); nur angelegt, wenn alloc_rho_rek gerufen wird.
	Memory<float> rho_rek_out;   // 4 floats je Ebenenzelle: rho rekonstruiert, rho roh, Klasse, heutiger Pufferwert
	Memory<rhoxx> rho_rek_wort;  // rho rekonstruiert als GERAETE-gepacktes Speicherwort (Wortvergleich gegen den Puffer)
	ulong rho_rek_max = 0ull;
	Kernel kernel_rho_rek_ebene;
	void alloc_rho_rek(const ulong max_plane_cells);
	// ★ 15.09.2026 RHO_RAND C2a: Ausgabe-rho einer Ebene (Nachkollisionssumme, Entscheidung (b)). Eigener Puffer, 1 float je Zelle.
	Memory<float> rho_aus;
	ulong rho_aus_max = 0ull;
	Kernel kernel_rho_ausgabe_ebene;
	void alloc_rho_ausgabe(const ulong max_plane_cells);
	std::vector<ulong> po_zellen_liste() { std::vector<ulong> v; v.reserve(po_N_active); for(uint i=0u; i<po_N_active; i++) v.push_back((ulong)po_cells[i]); return v; } // ★ 15.09. C1-Pruefung: Auslasszellen fuer die Klasse "TYPE_E Auslass"
	void alloc_facetten_domain(const std::vector<Facette>& F, const uint Nx, const uint Ny, const std::unordered_map<ulong,std::array<uchar,18>>* qmap=nullptr, const uint sgs_gdiag=0u, const uint sgs_fdwand=0u, const uint sgs_sism=0u); // sgs_gdiag als PARAMETER statt Statik (02.09.: zwei Statik-Lebensdauer-Fallen hintereinander -- ffc-Parsing und H1-Resetliste nullten s_sgs_gdiag vor alloc; env-getriebener Parameter hat keine Lebensdauer) // C1b: Puffer bauen + binden; qmap = Remesh-q (B1-Stufe 2)

	// ★ P9c N2F-SCHALE (Heiko): near->far-Schalen-Rueckkopplung. Nur belegt, wenn alloc_schale()
	// gerufen wurde (CFD_N2F_SCHALE>0) -- sonst bleibt alles unangetastet (Default bitidentisch).
	// Die Liste traegt DIREKT Zellindizes (fein: Deckungspunkte, grob: Schalenzellen); schale_unear
	// ist der Blend-EINGANG (Host-Upload vom Nahfeld-Blockmittel), schale_uout der Extract-AUSGANG
	// (Nahfeld: Blockmittel; Fernfeld: Waechter-Punktwerte). Getrennte Puffer, damit der Waechter-
	// Extract auf dem Fernfeld das hochgeladene unear nicht ueberschreibt.
	Memory<uint> schale_liste; // ★ 08.09. uint statt ulong (VRAM-Sparmassnahme 3)
	Memory<float> schale_unear, schale_uout; // je 3 float pro Schalenzelle (ux,uy,uz)
	Memory<float> schale_gewicht; // ★ Gradient-Blend: Zellgewicht in [0;1] (Lagen-Rampe innen 1 -> aussen 1/N; x+ skalierbar); wirkt im Kernel als a = alpha*gewicht[gid]
	Kernel kernel_schale_extract, kernel_schale_blend;
	uint schale_n = 0u;          // 0 = nicht alloziert = alle Schale-Aufrufe No-Op
	uint schale_modus = 0u;      // 0 = EQ-Arm (Altverhalten), Bit 0 = FNEQ (Nichtgleichgewichtsanteil erhalten), 2 = IDENT-Debug (exaktes No-Op, Paritaetsbeweis)
	float schale_alpha = 0.0f;   // Konstruktionszeit-Kopie von s_schale_alpha (read-once wie EINLASS_EQ)
	static float s_schale_alpha; // CFD_N2F_SCHALE: Blendfaktor u_neu=(1-a)*u_far+a*u_near; 0 = aus. Setup setzt lbm_f EXPLIZIT 0 (Blend laeuft NUR im Fernfeld).
	bool  schale_paritaet = false;  // Konstruktionszeit-Kopie (read-once)
	static bool s_schale_paritaet; // CFD_N2F_PARITAET: Kernel bekommt alpha EXAKT 0, der Enqueue laeuft aber -- das Torgatter haengt weiter an schale_alpha>0. Ohne diese Trennung schaltet ein alpha=0 den Kernel ganz ab und der Beweis liefe ins Leere (gemessen 2026-08-22: Slot-22-Wirkpfad NULL -> harter Fehler).
	void alloc_schale(const std::vector<ulong>& liste, const std::vector<float>& gewichte, const uint ratio, const uint modus, const bool blendet=true); // ★ 08.09. blendet=false: Blend-Eingang und Gewichte als 1-Element-Dummy (VRAM-Sparmassnahme 4, das Nahfeld blendet nie). Muster alloc_coupling_planes: echte Puffer, Kernel mit echten Puffern; NACH finalize_sparse_tiles rufen (fi-Bindung!)
	void enqueue_schale_blend(); // post-stream Blend; No-Op bei schale_n==0 ODER schale_alpha==0

	// ★★ Daempfungszone -- PRO DOMAENE, und das ist keine Kosmetik. Vorpruefung 2026-08-09:
	// die Zone wurde aus device_defines() direkt per getenv gelesen und traf damit JEDE LBM-Instanz.
	// Im Doppel-Domaenen-Fall waere sie im NAHFELD gelandet, wo zwischen Einlassflaeche und
	// Fahrzeugnase nur 57,5 Zellen liegen -- eine 64-Zellen-Zone haette die Nase um 7 Zellen
	// UEBERDECKT und die Staupunktstroemung durch Faktor 145 bis 751 laufen lassen. Cd waere
	// bedeutungslos gewesen. Auch N=32 rettet das nicht (dann 100 mm vor der Nase, mitten im Stau).
	// Bewusst OHNE Selbstruecksetzung (anders als s_fbbox/s_sparse_tiles_on): das Setup setzt den
	// Wert ausdruecklich VOR JEDEM Konstruktor. Read-once waere hier falsch herum gewesen, denn
	// lbm_f wird ZUERST gebaut -- die Zone haette also genau die falsche Domaene erwischt.
	// ★★ RHO_CLAMP-Zaehler. Heiko 2026-08-09: "rho clamp ist doch auch nur ne Kruecke die man
	// benoetigt wenn der Code falsch ist" -- richtig. Deshalb MUSS messbar sein, ob und wie oft sie
	// greift. Ein Lauf, in dem sie dauernd zuschlaegt, rechnet auf einem verfaelschten Feld und ist
	// KEIN Ergebnis. Ich hatte diesen Waechter in defines.hpp beschrieben und nicht gebaut -- genau
	// der lautlose No-op, den dieses Projekt jagt, in meiner eigenen Klemme.
	static constexpr uint hits_n = 500u; // ★ 06.10.2026 ZKS Pruefbefund M-2: 496 -> 500 (Slots 496..499 Pruefkernel zk_pruef; Legende lbm.cpp). // ★ 06.10.2026 ZKS Z2: 493 -> 496 (Slots 493 Gruppen-Wirkpfad, 494 Zellen-Wirkpfad, 495 Untergruppen-Geraetetest; Legende lbm.cpp). // ★ 05.10.2026 Korrektur A-M3: 492 -> 493 (Slot 492 = ZELLBASEN-Konstantenspiegel; 488/491 bleiben fuer die ZK-Linien, 500..503 fuer M1 frei -- Merge-Hinweis in der Legende lbm.cpp). // ★ 04.10.2026 U_RAND: 484 -> 492 (Slots 484..491, Legende lbm.cpp). // ★ 04.10.2026: 473 -> 484 (Pruefbefunde REK-PI: 473 Nenner-Kollaps der Sekante, 474 nu_s an nu geklemmt, 475..481 Netto-Fluss-Histogramm T/(-rho u_tau^2), 482 Messbesuche, 483 ohne T_aus; Legende lbm.cpp). // ★ 03.10.2026: 440 -> 473 (Slots 440..471 = REK-PI nach PLAN-REK-PI.md §7, 472 = REK-PI w_WM in der Kollision angewandt -- Bauabweichung: Wirkpfad des tau_eff, im Plan ohne Slot; Legende lbm.cpp). // ★ 03.10.2026: 433 -> 440 (Slots 433..439 = R1Q Stufe 5 SPALTE1: kappa-Klassen 433..438, Schattentor 439; Legende lbm.cpp). // ★ 02.10.2026: 432 -> 433 (Slot 432 = Sub-Group-Breite von stream_collide unter SC_SIMD16). ★ 28.09.2026: 408 -> 432 (Rang-1-Querrest CFD_FAC_R1Q, Slots 408..431). ★ 24.09.2026: 384 -> 392 (Doppelterm-Korrektur) -> 400 (Normalanteil 388..392 und Stufe S2 393..397). IN DIESER ZEILE STEHT KEINE FREILISTE MEHR -- ★ Pruefbefund M4 (24.09. nachmittags, drittes Auftreten derselben Klasse): hier stand "Frei sind 398/399", waehrend 398/399 seit demselben Vormittag belegt sind (kernel.cpp, Leere Probe und Widerspruchs-Entscheider) und die Legende bereits 403 als naechsten Slot fuehrte. Wer dem Kommentar folgte, vergab 398/399 ein zweites Mal. ★ Pruefbefund M5: hier stand nach dem zweiten Schritt noch der Kommentar des ersten ("387..391 sind frei") -- eine widerspruechliche Zweitfassung IN DERSELBEN ZEILE wie der Wert, dieselbe Falle wie Pruefbefund 3-E. VERBINDLICH ist allein die Legende in lbm.cpp bei der Allokation. Alter Text: Mit 381..385 (Groessenhistogramm, vier Grenzen statt zwei -- Pruefbefund M4) und 386 (Begleitzaehler zu 378) war der Puffer voll; 387..391 sind frei. Der Kernel indiziert mit LITERALEN und kennt keine Schranke. // ★ 22.09.2026 S-1 (REKONSTRUKTION-PLAN.md §5): 320 -> 384. Der Kernel indiziert rho_clamp_hits[] mit LITERALEN und kennt KEINE Schranke -- ein Zaehler oberhalb der Pufferlaenge schreibt aus dem Puffer heraus, ohne dass irgendetwas meldet. Der Plan braucht 12-14 Slots, frei waren 3 (317..319). ★ 15.09.2026 Klemmen Stufe 1 P1a: Slotzahl an EINER Stelle (Allokation lbm.cpp, Leser setup.cpp)
	Memory<uint> rho_clamp_hits; // Slotzahl steht in hits_n eine Zeile darueber und NUR dort; die Legende in lbm.cpp bei der Allokation ist die einzige Quelle der Vergabe. ★ Hier stand bis 24.09. eine Zweitfassung mit einer festen Zahl -- genau die Falle, die Pruefbefund 3-E und M5 beschreiben. (S-1), 320 seit 15.09.2026 abends (Klemmen Stufe 1 und 2: 271..305 -- Stand 16.09.2026, Pruefbefund N-4; die VERBINDLICHE Slotvergabe steht allein in lbm.cpp bei der Allokation; 288 mit Stufe 0, vorher 224 seit 08.09., davor 128) (Legende: lbm.cpp bei der Allokation) (70 Kraftpfad, 71 reserviert; 30.08.). Legende steht an EINER Stelle: lbm.cpp bei der Allokation (Pruefbefund 3-E: hier stand eine widerspruechliche Zweitfassung, aus der der naechste Slot vergeben worden waere).
	// ★ uint je Domaene: ein pathologischer Lauf (Test B mass 415 Mio = ~10 % von 2^32) kann
	// ueberlaufen. Fuer einen Waechter, der bei >0 ohnehin den Lauf disqualifiziert, vertretbar --
	// aber die ZAHL ist oberhalb einiger Milliarden nicht mehr woertlich zu nehmen.
	static bool s_facetten;  // C1b: Facettenpfad an (CFD_FACETTEN>0)
	static bool s_fac_imem;  // C1b iMEM-Umbau: CFD_FACETTEN=3/4 (Slip-Velocity-BB) statt 1/2 (Paartausch-Kontrollarm)
	static float s_fac_ema;  // EMA-Faktor fuer u_s (CFD_FAC_EMA; 0 = aus; WIDERLEGT in J3 -- filtert die falsche Seite, bleibt als A/B-Arm)
	static float s_fac_pema; // PEMA: beidseitige EINGANGS-Filterung P-quer/u-quer (CFD_FAC_PEMA; Weg A der Analyse)
	// ★★ 06.09.2026 KINEMATISCHES WANDMODELL (Ponsin & Lozano 2025), CFD_FAC_UW.
	// Statt einen Impuls zu LOESEN wird die Wandgeschwindigkeit HINGESCHRIEBEN:
	//   u_w = u_B - u_tau^2*delta_w/(nu+nu_t)   -- der Gradient Wand->Abtastpunkt traegt dann tau_w.
	// Damit entfallen Momentenmatrix, Rang, Schur, Kaskade und Gates ersatzlos; die Ein-Link-Facetten
	// (kipp26 10.620 = ein Drittel, Kugel 2.892 = 21,5 %, 4 mm 504.225) bekommen zum ersten Mal
	// ueberhaupt eine Wandbehandlung, weil die Sperre J.n = 0 bei J || c nur den SOLVE betraf.
	// 0 = aus (bitgleich zum Vorstand) | 1 = Gleichgewichts-nu_t (1+kappa*y+) | 2 = gemessenes nu_t aus fac_wfd
	static uint s_fac_rdiag; // ★ 07.09.2026 Rueckfall-Diagnose (CFD_FAC_RDIAG): Slots 136..154, bitneutral. DIE SLOTVERGABE STEHT AN EINER EINZIGEN STELLE: der Legende in lbm.cpp bei der Allokation. HIER KEINE ZAHL MEHR -- die frueheren Zweitangaben (zuletzt "295", waehrend 295..300 laengst belegt waren) haetten die naechste Vergabe still auf einen benutzten Slot gelegt (Audit-Schleife 16.09.2026, Befund C-H1; genau die Klasse, vor der Pruefbefund 3-E schon einmal gewarnt hat). Historie nur noch als Datum: (Klemmen Stufe 0/1/2, Legende lbm.cpp; bis 15.09. abends: 271; bis 15.09.: 221 -- 216 RHO_RAND R1-Zugriff ausserhalb (Soll 0), 217/218 rho_rek_ebene, 219/220 rho_ausgabe_ebene, 15.09.; 212/213 = u-Huellenwaechter, 214 = Betragstor im Kopplungs-Lift, 215 = dessen Besuchszaehler; berichtigt 12.09., die Legende in lbm.cpp ist die fuehrende) (204..207 rho/u-SPARSAM und 210/211 rho-2-Byte-Bereichswaechter, beide 12.09. -- die Legende an der Allokation in lbm.cpp fuehrt; 188..198 NUT_SKAL-Diskriminator, 199..203 P-TRT seit 10.09. abends: 199 Block besucht, 200 Geistanteil vorhanden, 201 Abzug ungleich null -- diese drei SAETTIGEN bei 4 mm nach 800 Schritten und koennen dabei sogar WICKELN; 202/203 sind die ueber n%1024 ausgeduennte Zweitzaehlung, die nicht saettigt, und 203 prueft zusaetzlich, ob der Abzug die FP16S-Speicherrundung ueberlebt. DER SCHARFE TEST IST 203 GEGEN 202, NICHT 201 GEGEN 200) (Puffer seit 08.09. 224 statt 160; 126/127 SISM, 160-167 van-Driest-D^2-Histogramm als Zeitintegral, 168 VD-Wirkpfad, 169 VD ohne Besuch, 170-185 VD-Letzt-Stichprobe in zwei Baenken) -- die Legende an der Allokation in lbm.cpp (grep "rho_clamp_hits = Memory") ist die fuehrende Fassung
	static uint s_fac_uw;
	static bool s_fac_uw_sn; // A/B: Normalnullung wieder einschalten -- misst den Preis von J.n = 0
	static uint s_fac_masse_alle; // 0 aus | 1 Kompensation ueber ALLE 19 Links | 2 NUR auf f_0 (VERWORFEN 04.09.: Bulk-Mode, f_0<=0) | 3 ARM X: Injektion wie 1, Rueckfall-Entscheid im Schatten wie ALPHA2 // CFD_FAC_MASSE_ALLE (04.09.2026): alpha-Kompensation ueber ALLE 19 Links statt nur ueber die Wandlinks -- hebt das ALPHA2-Downdate auf, OHNE die zellweise Massenerhaltung aufzugeben
	static bool s_fac_satgate; // (a-strich): Klemme -> BB-Rueckfall-Gate (CFD_FAC_SATGATE; Stabilitaetsanalyse G8)
	static uint s_fac_kraft;   // ★ 30.08.: CFD_FAC_KRAFT -- Zellkraft statt Slip: 1 = an Rueckfallzellen, 2 = an allen Facettenzellen (Diskriminator); 0 = aus, bitgleich
	static uint s_boden_eq_n; static uint s_boden_eq_down; static uint s_boden_eq_split; static float s_boden_eq_u; static uint s_boden_eq_abstand; // ★ BODEN_EQ (V1-Port): Fluidzeilen z=1..N post-stream auf u_road-Equilibrium (lokales rho); 0 = aus. Read an der Konstruktion in Member eingefroren.
	static uint s_einlass_eq_n; static float s_einlass_eq_u; // ★ EINLASS_EQ (V1-Port apply_inlet_velocity): Spalten x=1..N post-stream auf u-Equilibrium (lokales rho); 0 = aus. Read-once wie BODEN_EQ.
	static uint s_u_takt; // ★ TODO 2 Schritt 3 (CFD_U_SPARSAM): 0 = aus, sonst ratio (u voll am letzten Substep jedes Grobschritts)
	static uint s_rho_takt; // ★ TODO 2 Schritt 1 (CFD_RHO_SPARSAM): Sample-Kadenz in FEINEN Schritten; 0 = aus (dann ist der Geraetecode zeichengleich zu vorher)
	static uint s_rho_rand; // ★ 15.09.2026 RHO_RAND (CFD_RHO_RAND, RHO_RAND-PLAN.md): 0 = aus, 1 = rho nur in der Domaenen-Randschale R1. fahrzeug_dd NUR Nahfeld (Statik vor dem Fernfeld-Bau genullt) und Kugel-Pruefstand
	static bool s_flags4_pruef; // ★ 05.10.2026 FLAGS4: CFD_FLAGS4 steht in der Zeile (auch =0) -> Geraeteprobe in BEIDEN Armen (MS-Zahl ist der Soll fuer den 4-Bit-Arm). Lesestelle/Nullung wie s_flags4
	static uint s_zk; // ★ 06.10.2026 ZKS (CFD_ZK, PLAN-ZKS-VOLLBAU-2026-10-06.md): 0 = aus (Vorgabe), 1 = Zeilenkompaktierung von fi im Nahfeld. NUR fahrzeug_dd-Nahfeld; Lesestelle setup.cpp neben s_flags4, Fernfeld genullt (Werkzeugfalle 21); im Konstruktor nur eingefroren (zk_on). EIGENE Statik, setzt NIE s_sparse_tiles_on (sonst FLAGS4-Sperre und SPARSE_TILES-Emission)
	static uint s_flags4; // ★ 05.10.2026 FLAGS4 (CFD_FLAGS4, PLAN-VRAM-URAND-FLAGS-2026-10-04.md Teil C): 0 = Byte je Zelle (wie bisher), 1 = 4 Bit je Zelle auf dem Geraet. NUR fahrzeug_dd-Nahfeld; Lesestelle setup.cpp neben s_u_rand, Fernfeld genullt (Werkzeugfalle 21)
	static uint s_u_rand; // ★ 04.10.2026 U_RAND (CFD_U_RAND, PLAN-VRAM-URAND-FLAGS-2026-10-04.md Teil B): 0 = aus, 1 = Pruefstand (A = ganzes Gitter), 2 = Grossbox, 3 = eng (Maske + N2F-Bloecke in A). NUR fahrzeug_dd-Nahfeld; Lesestelle setup.cpp neben s_u_takt, Fernfeld genullt
	static uint s_fac_alpha;
	static bool s_fac_elibb;
	static uint s_sgs_fdwand;  // ★ 02.09. SGS-GEISTERMODEN-FIX (CFD_SGS_FDWAND=1): w an Facettenzellen aus |S|_FD des u-Felds (FD-Kernel, ein Schritt versetzt) statt aus dem Pi-Tensor, den das Wandmodell kontaminiert (B66/B69)
	static uint s_sgs_vandriest; static float s_sgs_vd_aplus; static ulong s_sgs_vd_ab; // ★ 08.09.2026 VAN DRIEST auf der Facetten-Architektur (CFD_SGS_VANDRIEST): 0 aus, 1 MESSARM (D^2 nur gebinnt, w unangetastet -> bitgleich), 2 ANGEWANDT (nu_t <- nu_t*D^2). y+ aus dem WANDMODELL (tw-Laufmittel + y_w), NICHT aus dem lokalen Strain -- damit faellt die V1-Rueckkopplung strukturell weg. A+ = CFD_SGS_VD_APLUS, Default 26 (van Driest 1956). Slots 160..167 D^2-Histogramm, 168 Wirkpfad, 169 Facette ohne Besuch
	static uint s_sgs_sism; static uint s_sgs_sism_T; static ulong s_sgs_sism_ab; // ★ 07.09.2026 SHEAR-IMPROVED SMAGORINSKY (CFD_SGS_SISM=1, Leveque/Toschi/Shao/Bertoglio JFM 570, 2007) im FD-Kernel sgs_fdwand: nu_t = c2*max(0, |S|_FD - |<S>|), <S> = EMA der SECHS S-Komponenten je Facette (fac_sb; die billige Form <|S|> waere ein anderes Modell -- im Zeitmittel nu_t = 0 = WANDFREI). T = EMA-Zeitkonstante in SCHRITTEN (alpha = 1/T erst im Kernel, Muster def_fac_nu: keine Festkomma-Quantisierung), ab = Warmlaufsperre in Schritten (bis dahin klassische FDWAND-Formel WORTGLEICH, EMA laeuft ab 0 mit). Braucht CFD_SGS_FDWAND=1 und Facetten. Wirkpfad Slots 126 (Abzug aktiv) / 127 (Klemme |S|<Sbar); Zeitreihe sism_sbar.csv
	static uint s_sgs_gdiag;   // ★ 31.08. g-DIAGNOSE (CFD_SGS_GDIAG=1): sparser Messkernel ueber die Facettenzellen -- |S|_FD, |S|_Pi, D_WALE, D_Sigma, |Omega| je Zelle akkumuliert; fasst Physik nicht an
	static uint s_fac_messnur; // ★ 30.08. CFD_FAC_MESSNUR: Facetten bauen und MESSEN, im Kernel aber NICHTS anwenden -- BB-Physik mit Facetten-Instrument (Aepfel-mit-Aepfeln-Bezug fuer BB-Vergleiche)
	static float s_fac_rek_leiter; // ★ 24.09.2026 DIAGNOSELEITER (CFD_FAC_REK_LEITER): skaliert die aus dem Wandmodellziel HERGELEITETE Amplitude. NUR unter CFD_FAC_REK=3, Vorgabe 1.0 (dann exakt bitgleich, Multiplikation mit 1.0f ist verlustfrei). ZWECK: trennt "der Speicherpfad quantisiert den Hub weg" von "die Zellquelle traegt die Physik nicht". GEMESSEN 24.09.: 100 % der Amplituden liegen unter dem Messhub 1e-4, 76,7 % im Fach 1e-5..1e-4. Skaliert die Wirkung auf cf mit der Leiter mit, ist die Quantisierung der Begrenzer; skaliert sie nicht, ist die Aktorfamilie unabhaengig davon tot. KEIN Messarm und KEIN Handwert im Ergebnis -- ein Instrument wie die eps-Amplitudenleiter vom August, das danach wieder herausfliegt.
	static uint s_fac_r1q; // ★ 28.09.2026 RANG-1-QUERREST (CFD_FAC_R1Q): 0 aus, 1 messen (rechnen+zaehlen, nichts anwenden), 2 V_Z (nur Modellanteil), 3 V_R (voller Rest (I-M)(Z-P)), 4 V_R ohne isotropen Druckanteil 2(rho-1)(S1.t), 5 wie 4, aber NUR SPALTE 1 (R2' := 0: ersetzt wird nur die Laengsspalte P1-A1 -> Z1, der Quer-BB-Austausch P2 bleibt wie die PINV ihn behandelt; K2 aus UNTERBODEN-R1Q-ANALYSE.md, 03.10.2026). An Lage-1-Zellen mit Rang 1 (Marke -1 in fac_geo[8i+7]) wird der Teil des Wandmodellziels, den der PINV-Zweig nicht aufpraegen kann, als Zellquelle eingespeist.
	static uint s_fac_rekpi; // ★ 03.10.2026 REK-PI (CFD_FAC_REKPI, PLAN-REK-PI.md): 0 aus, 1 messen (rechnen+zaehlen, bitgleich zu 0), 3 (★ 04.10.) Ausweichstufe: nur u setzen, kein Pi-Tausch/tau_eff, 2 anwenden (u_t-Setzung + Pi_tn-Tausch + tau_eff an ALLEN Facettenzellen mit Referenzpunkt; zieht FAC_REK und FAC_REK_R3 nach). In setup.cpp neben s_fac_r1q gesetzt und genullt (Werkzeugfalle 21), NIE im Konstruktor
	static uint s_fac_rek; // ★ 22.09.2026 S0/S1 WANDZELL-REKONSTRUKTION (CFD_FAC_REK): 0 = aus, 1 = Umfang Rang 0 (REKONSTRUKTION-PLAN.md §12). Marke in fac_geo[8i+7], Amplitude in [8i+6] -- beide Laufzeitladungen, KEIN JIT-Define fuer die Amplitude (sonst optimiert IGC die Identitaet weg).
	static uint s_fac_pinv; // ★ 04.09. CFD_FAC_PINV: Moore-Penrose-Pseudoinverse statt achsenparalleler Skalarleiter im gekoppelten Zweig
	static uint s_fac_idx_voll; // ★ 03.09. CFD_FAC_IDX_VOLL: fac_idx wieder als volles uint-Feld (Rueckschalter fuer das A/B gegen die Bitmaske)
	static float s_sgs_nut_skal; // ★ 10.09.2026 DISKRIMINATOR-MESSARM (CFD_SGS_NUT_SKAL, Default 1,0 = aus = bitgleich): nu_t am KLASSISCHEN Modell wird an Facettenzellen mit diesem Faktor skaliert. KEIN Produktionsschalter und KEIN Stellknopf -- der Faktor wird aus dem SISM-Lauf ABGELESEN (Lagenmessung 08.09.: SISM senkt nu_t in Lage 1 um 85,2 %, also Faktor 0,148). Beantwortet, ob SISMs Kraftgewinn Modellphysik ist oder nur die fehlende Wanddaempfung. Wirkpfad Slots 188 (besucht, == Slot 76) / 189 (nu_t > 0) / 190 (w wirklich geaendert), Histogramm nu_t/nu_mol 191..198.
	static uint s_sgs_band; // ★ 08.09. CFD_SGS_BAND: Zahl der zusaetzlichen Wandlagen (0 = aus, 2 = Lage 2, 3 = Lage 2+3)
	static uint s_sgs_band_pi; // ★ 22.09. CFD_SGS_BAND_PI (Plan C): 0 = FD-Modus (Bandkernel liefert Sbar_FD), 1 = Pi-konsistente EMA in stream_collide (kein Bandkernel, kein u-Lesen)
	static uint s_f_liste; // ★ 03.09. CFD_F_LISTE: F nur noch an Wandsolidzellen (Markerliste statt BBox-Vollfeld)
	static uint s_fac_nachbar; // ★ 30.08. CFD_FAC_NACHBAR: Wandmodell-EINGANG aus der zweiten Fluidzelle entlang der Normale (Stufenschatten-Fix, Weg-1 Stufe 3)
	static uint s_fac_kdiag;   // ★ 30.08. KLASSEN-DIAGNOSTIK (CFD_FAC_KDIAG=1): 16 float je Facette akkumuliert (u_t, tw, twe, |P1|, s1, phi1, Rueckfall, Besuche, u_t_abtast, y_abtast, tw_angewandt, besuche_angewandt, 05.09.: Druckrest A, |A|, Ziel B, Geometrie C); Host-Tabelle je Treppenklasse
	static bool s_fac_elibb_pur; // Pur-Arm  // ★ B1/B2 (2026-08-25): ELIBB 18-Link, q aus der Facettenebene
	static float s_fac_qmin;
	static float s_fac_kappa; // Grazing-Guard
	static float s_fac_utkorr; // 3/2-Abtastpunkt-Messarm (Wandmodell-Eingang)
	static float s_fac_qkappe; // Ex-Stabilitaetskappe q>0,5 -- seit MLS-Blende (26.08.) Default 1,0 = aus; Env-Hebel fuer A/Bs
	static uint s_fac_qdiag; // QDIAG-Diagnosearme  // q-Boden (P1-Entscheid): darunter HWBB, mit Zaehler
	static bool s_fac_quergate; // ★ 2026-08-25 Querimpuls-Gate im iMEM-Solve
	static bool s_fac_lsq; // ★ 2026-08-25 kleinste-Quadrate-Rueckfall im iMEM-Solve // J4-Massenkorrektur 0/1/2 (CFD_FAC_ALPHA)
	static float s_fac_apg; // APG-Messarm (PLAN-APG-2026-09-16.md §A): kappa auf y_ab*dp/ds im tw-Ziel (lineare Duennschichtform); 0 = aus (bitgleich). Seit 22.09. schaltet CFD_FAC_APG_MOZ=1 auf die Mozaffari-Form um -- kappa ist dann OHNE Physikwirkung (nur Zaehlerbezug) und MUSS 1 sein (Waechter in setup.cpp). Seit 16.09.: dp/ds aus dem Vorkernel fac_nachbar_ab (grad rho in fac_nb[2..4]), braucht CFD_FAC_NACHBAR=1
	// ★★ 22.09.2026 MOZAFFARI-FORMELTAUSCH (CFD_FAC_APG_MOZ). 0 = heutige lineare Duennschichtform
	// mit harter Klemme [0, 2*tw] (bitgleich), 1 = beschraenkte u_tau-Daempfung nach
	// Mozaffari-Jacob-Sagaut 2024. C und alpha_p0 sind die PAPERWERTE 0,4 / 0,005 -- sie sind
	// Kalibrierkonstanten des Fits (NACA-4412, Ahmed), keine Stellknoepfe dieses Projekts.
	static uint  s_fac_apg_moz; // (hiess kurz s_sgs_apg_moz -- Pruefbefund N3/N11: gehoert zur s_fac_apg-Familie)
	static float s_fac_apg_c, s_fac_apg_ap0;
	static uint s_timer_apg; // ★ 22.09.2026 CFD_TIMER_APG: 0 = aus (bitgleich, kein finish_queue), 1 = Vorkernel fac_apg_ab isoliert messen. SERIALISIERT -- Wanduhr dieses Arms ist KEIN Leistungsmass.
	static uint s_fac_apg_haken; // ★ 16.09.2026 CFD_FAC_APG_HAKEN (3 = analytischer Lineargradient rho = 1 + x/1024 im Vorkernel, Host prueft gx exakt, gz traegt kx): 0 aus, 1 = fac_nb am Ende zuruecklesen und grad-rho-Statistik berichten (Host-Puffer bleibt), 2 = Konstantgradient (1e-3,0,0) im Vorkernel -> Ist=Soll bitgenau
	Memory<float> fac_pu;    // PEMA-Zustand 6 float je Facette
	bool fac_pema_on = false;
	static long s_fac_diagz; // Iron Rule 3: Diagnose-Facette (Zellindex; -1 = aus)
	uint boden_eq_n = 0u; float boden_eq_u = 0.0f; uint boden_eq_down = 0u, boden_eq_split = 0xFFFFFFFFu, boden_eq_abstand = 0u; // Konstruktionszeit-Kopien (BODEN_EQ)
	uint einlass_eq_n = 0u; float einlass_eq_u = 0.0f; // Konstruktionszeit-Kopien (EINLASS_EQ)
	uint rho_takt = 0u;        // Konstruktionszeit-Kopie von s_rho_takt (read-once-Doktrin)
	bool klemm_bilanz_on = false; // ★ 15.09.2026 Klemmen S0b: Konstruktionszeit-Kopie von CFD_KLEMM_BILANZ (Vorgabe 1)
	uint u_klemme = 0u; // ★ Z2d: Konstruktionszeit-Kopie von CFD_U_KLEMME
	uint positiv_modus = 0u; // ★ 15.09.2026 Klemmen Stufe 1 P1a: Konstruktionszeit-Kopie von CFD_POSITIV (0 aus, 1 Messarm, 2 anwenden)
	uint positiv_haken = 0u, positiv_facette = 0u; // ★ Pruefbefund P1b NIEDRIG 6: Konstruktionszeit-Kopien fuer den Bericht
	bool rho_rand_on = false;  // ★ 15.09. Konstruktionszeit-Kopie von s_rho_rand (read-once-Doktrin); das Setup liest DIESEN Wert, nicht die Umgebungsvariable
	ulong rr_N = 0ull;         // ★ 15.09. RHO_RAND C2c: Zellen der Randschale R1 (rho-Puffer = rr_N+1, letzter Slot Papierkorb); 0 ohne RHO_RAND
	uint u_takt = 0u;          // Konstruktionszeit-Kopie von s_u_takt
	ulong u_fenster_n = 0ull, u_fenster_frei = 0ull; // ★ 05.10.2026 (SWEEP-SPZ-ZB4 Slot-206-Waechter): Schritte im Zaehlfenster [zaehl_takt+2, +6) von stream_collide bzw. davon ohne erzwungenes Vollschreiben (felder_voll Bit 1 = 0)
	// ★ 04.10.2026 U_RAND (PLAN-VRAM-URAND-FLAGS-2026-10-04.md Teil B): u kompakt (Kopf + R1|A|V|Papierkorb; ★ C-N8: kein Segment C mehr), voller Hostspiegel separat.
	bool u_rand_on = false;    // Konstruktionszeit-Kopie von s_u_rand
	uint ur_modus_soll = 0u;   // Konstruktionszeit-Kopie des Modus (= s_u_rand); alloc_u_rand baut danach
	uint ur_modus = 0u;        // = CFD_U_RAND: 1 = A ist das ganze Gitter (Pruefstand), 2 = Grossbox (U1d), 3 = eng mit Masken und N2F-Bloecken (U2); 0 = aus
	ulong ur_US = 0ull, ur_R1N = 0ull, ur_AN = 0ull, ur_VOFF = 0ull, ur_VN = 0ull, ur_P = 0ull, ur_DOFF = 0ull, ur_len = 0ull; // Layout (Slots bzw. velxx-Woerter)
	std::vector<uint> ur_kopf;  // Host-Zwilling von Kopf + Masken (uint-Woerter, Laenge ur_DOFF/2)
	Memory<velxx> u_spiegel;    // voller Hostspiegel N x 3 (nur Host); LBM::u bindet ihn unter U_RAND
	ulong ur_idx_host(const ulong n) const; // Zwilling von ur_idx (kernel.cpp)
	void ur_layout(const uint modus); // Modus 1/2/3 (Gitter/Grossbox/eng); fuellt ur_kopf (+Maske) und das Layout
	void ur_kopf_in_puffer();   // ur_kopf -> Kopfwoerter im Hostpuffer von u
	void ur_packen();           // Hostspiegel -> kompakter Hostpuffer (alle gespeicherten Zellen)
	void ur_spiegel_init_nachziehen(); // ★ 05.10.2026 A-N2: Spiegel an gespeicherten TYPE_ONLY_S-Zellen auf 0, wie initialize im Kernel
	void ur_streuen();          // kompakter Hostpuffer -> Hostspiegel (alle gespeicherten Zellen)
	uint ur_selbsttest();       // Bijektion der Slots, Kopf-Gegenlese; liefert Beanstandungen
	ulong ur_init_waechter();   // ★ U1d: ungespeicherte Zellen des Spiegels == Init-Regel (Wortvergleich); liefert die Verletzungen
	inline const velxx* u_host_voll() const { return u_rand_on ? u_spiegel.data() : u.data(); } // Setup-Stand von u (Zensus)
	// ★ U1b: spaete Allokation. Der Konstruktor legt nur einen PLATZHALTER an (Kopf mit Magic ur_k::PLATZHALTER, keine Slots); die
	// echte Groesse steht erst nach Facetten-, Band- und N2F-Listenbau fest. alloc_u_rand legt an und bindet ALLE u-Kernel neu
	// (Muster alloc_f_liste: erst anlegen, dann binden). Freigegeben wird dabei nur der kleine Platzhalter (NEO-Defekt, lbm.cpp).
	bool ur_alloziert = false;
	void ur_platzhalter();
	void alloc_u_rand(const uint modus, const float u_lat);
	uint ur_neu_binden(); // liefert die Zahl der neu gebundenen Kernel
	// ★ U1b: GUELTIGKEITSSTEMPEL des Hostspiegels (Plan B.5). Ein Hostzugriff ausserhalb ist ein harter Fehler (Zugriffssperre,
	// U_Feld::pruefe_zugriff). ur_setup: vor initialize ist der Spiegel die Wahrheit. Danach gilt eine Zelle bei t, wenn
	// VOLL(t) ODER (KOMPAKT(t) und die Zelle hat einen Slot) ODER sie in der gestempelten Ebene/Saeule liegt.
	bool ur_setup = true;
	ulong ur_t_kompakt = ~0ull, ur_t_voll = ~0ull, ur_t_ebene = ~0ull, ur_t_saeule = ~0ull;
	uint ur_ebene_y = 0u, ur_saeule_x = 0u, ur_saeule_y = 0u, ur_saeule_zn = 0u;
	// ★ U1c: AUSGABE V + LESEPLAN (Plan B.5). Der Host sagt VOR dem Grobschritt an, was er danach liest; stream_collide schreibt am
	// LETZTEN Substep die eigenen Werte zusaetzlich nach V. Bits: 4 Ebene y = VY0, 8 VOLL (Scheiben), 16 Saeule.
	uint ur_plan = 0u;          // fuer den naechsten letzten Substep (wird beim Ausfuehren verbraucht)
	bool ur_vbox_aktiv = false; // V-Box im Kopf gesetzt (muss vor dem naechsten stream_collide ohne Plan geleert werden)
	uint ur_v_y = 0u;           // Ebene der Slice-Ausgabe (y = Ny/2, wie lese_yslice_in_host(fNy/2) im Setup)
	uint ur_plan_lauf = 0u;     // im letzten Grobschritt ausgefuehrt, verbraucht von LBM::u_rand_ausgabe
	ulong ur_plan_t = ~0ull;    // t NACH dem ausfuehrenden Schritt
	std::vector<ulong> ur_scheiben; // Scheibengrenzen fuer VOLL (Vielfache der Arbeitsgruppe, je Grenze mitten in der Zeile y = Ny/2)
	uint ur_pruef = 0u, ur_haken = 0u; // CFD_U_RAND_PRUEF (Modus 1: V gegen kompakt), CFD_U_RAND_TESTHAKEN
	ulong ur_soll_489 = 0ull, ur_soll_490 = 0ull; // Host-Soll der Ausgabezaehler
	ulong ur_pruef_n = 0ull, ur_pruef_abw = 0ull, ur_n_voll = 0ull, ur_n_ebene = 0ull, ur_n_saeule = 0ull;
	// ★ 05.10.2026 Korrektur M3/H1 (Plan B.15): V GEGEN KOMPAKT in JEDEM Modus -- an jeder Lesung die gespeicherten Nicht-R1-Zellen des
	// Lesebereichs (Spiegel nach allen Streuungen) gegen den kompakten Puffer; Soll 0. Dazu Kosten und die Kombinationen der Leseplan-Bits.
	ulong ur_vgl_n = 0ull, ur_vgl_abw = 0ull, ur_vgl_bytes = 0ull, ur_vgl_bereiche = 0ull;
	double ur_vgl_s = 0.0;
	ulong ur_n_bits[8] = {0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull}; // Index bits>>2: 1 Ebene, 2 VOLL, 3 VOLL+Ebene, 4 Saeule, 5 Ebene+Saeule, 6 VOLL+Saeule, 7 alle drei
	void ur_v_anlegen();        // V-Groesse und Scheiben aus dem Gitter; setzt ur_VOFF/ur_VN (hinter A)
	void ur_saeule_setzen(const uint x, const uint y, const uint zn); // Wandprofil-Saeule (Host; die V-Box setzt enqueue_stream_collide)
	void ur_vbox(const uint x0, const uint y0, const uint z0, const uint nx, const uint ny, const uint nz); // V-Box in den Kopf (finish + Schreiben)
	void ur_voll_scheiben(const uint fv); // VOLL: stream_collide scheibenweise, V je Scheibe lesen und streuen
	uint felder_voll_h = 3u;   // je Schritt gesetzter Kernelparameter, BITFELD: Bit 0 = rho ueberall, Bit 1 = u ueberall (3 = heutiges Verhalten)
	bool rho_voll_zwang = false; // Host erzwingt Vollschreiben (Abschlusspfad, unregelmaessige Feldlesung)
	long fac_diagz_wert = -1l; // Konstruktionszeit-Kopie von s_fac_diagz (Gross-Audit: Spaet-Lese-Pfad geschlossen)
	Memory<float> fac_diag;  // 19-float-Kettenprotokoll ([16] Selektor, [17] alpha, [18] dp_ds)
	bool fac_diagz_on = false; uint fac_diag_fid = 0xFFFFFFFFu;
	bool fac_elibb_on = false; // ★ B2: ELIBB-Konstruktionszustand (eingefroren wie diagz)
	// ★★ 22.09.2026 CFD_TIMER_APG (REKONSTRUKTION-PLAN.md §10 Hebel 1, Plan-Schritt 2). APG kostet
	// GEMESSEN +17,0 % Zeitschleife (16.09., 4 mm), das Datenvolumen erklaert davon nur 2,6 % --
	// Faktor 6,5 Luecke. Ohne diesen Timer ist jede Zuordnung geraten: der Verlust verteilt sich auf
	// den VORKERNEL fac_apg_ab (133 Streuzugriffe je Facette) und den APG-Zweig IN stream_collide,
	// und nur der Vorkernel ist isoliert messbar. Instanzkopie, weil fahrzeug_dd die Statik vor dem
	// Bau des Fernfelds nullt -- dieselbe Falle wie bei apg_on (Pruefbefund HOCH-1 vom 16.09.).
	uint timer_apg = 0u; double apg_t_summe = 0.0, apg_t_min = 1.0e30, apg_t_max = 0.0; ulong apg_t_n = 0ull;
	// ★ 03.10.2026 CFD_GPU_PROFIL (Leistungsanzeige je GPU). profil_an = Konstruktionszeit-Kopie von device.profil_queue
	// (die Queue traegt CL_QUEUE_PROFILING_ENABLE nur, wenn das Setup Device::profil_anlegen VOR dem Konstruktor setzte,
	// heute ausschliesslich fahrzeug_dd). do_time_step markiert die Schrittgrenzen und sammelt NUR die Launches des
	// Schritts; ausgewertet wird nur an ohnehin synchronisierten Stellen (LBM::run nach info.update, LBM::finish), kein
	// eigenes wait/finish. Je Schritt: Spanne = groesstes END - kleinstes START, Kernelsumme = Summe (END - START).
	bool profil_an = false;
	vector<Event> profil_events; // Events der noch nicht ausgewerteten Schritte
	vector<ulong> profil_grenzen; // Startindex jedes noch nicht ausgewerteten Schritts in profil_events
	ulong profil_schritte = 0ull; // ausgewertete Schritte, kumulativ
	ulong profil_kern_ns = 0ull, profil_spanne_ns = 0ull; // kumulativ, Geraeteuhr in ns
	ulong profil_verletzt = 0ull; // Selbsttest 0 < Kernelsumme <= Spanne verletzt (Schritte)
	ulong profil_fehler = 0ull; // Event mit Fehlerstatus oder Profilwert nicht lesbar (Schritte)
	void profil_schritt_beginnen(); // do_time_step, VOR dem ersten Launch: Grenze markieren, Ziel setzen
	void profil_schritt_beenden(); // do_time_step, NACH dem letzten Launch: Ziel loesen
	void profil_auswerten(); // nur nach einer Barriere rufen; wertet nur Schritte aus, deren Events ALLE CL_COMPLETE sind
	// ★ 23.09.2026 EINZIGE QUELLE der fac_nb-Stride-Erweiterung unter CFD_FAC_REK. Die Formel stand bis heute
	// AUSGESCHRIEBEN an drei Stellen (lbm.cpp Host-Spiegel, lbm.cpp Stride-Waechter, setup.cpp berichte_apg) plus
	// der JIT-Emission -- die vierte fand erst der Pruefagent (H-1), sie haette den ersten APG+REK-Lauf mit rc=1
	// beendet. Wer die Zahl aendert, aendert sie HIER; die Nutzer rechnen damit. Stufe A: 3 (Richtung t_nb),
	// Stufe A2 ab 23.09.: 4 (zusaetzlich der Impuls-Akkumulator Summe rho*du_x je Facette).
	static constexpr ulong nb_rekpi_floats = 4ull; // ★ 04.10.2026: 3 -> 4, roff+8 = Ausfluss-Halbmoment T_aus des Netto-Fluss-Zaehlers (nur an Zaehlschritten geschrieben/gelesen; 4 mm +12,5-13,1 MB, fac_nb 36 B je Facette = 113-118 MB). // ★ 03.10.2026 REK-PI: u_tau, u_WM(y_w), nu_eff je Facette in roff+5..7, HINTER den REK-Floats (fac_nachbar_ab rechnet das Wandgesetz, Spill-Begrenzung in stream_collide). Kosten bei 4 mm: 3,13-3,28 Mio Facetten x 12 B = 37,6-39,4 MB. Die Stride-Formel steht an FUENF Stellen (lbm.cpp Konstruktor, device_defines, alloc-Waechter, VRAM-Schaetzung; setup.cpp APG-Bericht) -- alle tragen diesen Summanden.
	static constexpr ulong nb_rek_floats = 5ull; // ★ 24.09.2026 S2: 4 -> 5. Der fuenfte float haelt die aus dem Wandmodellziel bestimmte Amplitude des naechsten Schritts (roff+4). Kosten bei 4 mm: 3,13 Mio Facetten x 4 B = 12,5 MB.
	Memory<float> fac_nb; Kernel kernel_fac_nachbar; Kernel kernel_fac_apg; bool nachbar_on = false; bool apg_on = false; float apg_kappa = 0.0f; uint apg_haken = 0u; uint apg_moz = 0u; float apg_moz_c = 0.0f, apg_moz_ap0 = 0.0f; ulong nb_stride = 2ull; ulong nb_roff = 2ull; // ★ 23.09. nb_roff: Offset der REK-Felder in fac_nb, EINZIGE Hostquelle (Pruefbefund N7 -- die Formel war an zwei Stellen in setup.cpp dupliziert) // ★ 16.09. HOCH-1 (Pruefagent): APG-Zustand je INSTANZ eingefroren (allocate), weil fahrzeug_dd die Statik vor dem Bau des Fernfelds nullt; // ★ 16.09. kernel_fac_apg: APG-Vorkernel (grad rho in fac_nb[2..4]), nur unter s_fac_apg != 0 gebunden // ★ 03.09. deterministische Nachbarabtastung: (u_t_abt, y_abt) je Facette (2 float) aus eigenem Kernel nach stream_collide, ein Schritt Versatz (fac_wfd-Muster); Konstruktionszustand eingefroren
	Memory<float> fac_wfd; Kernel kernel_sgs_fdwand; bool fdwand_on = false; // ★ Geistermoden-Fix: w je Facettenzelle (1 float), Konstruktionszustand eingefroren (Emission + Platzhalter im ctor); alloc rebindet ueber den env-Parameter
	bool vandriest_on = false; uint vandriest_modus = 0u; ulong vandriest_ab = 0ull; // ★ 08.09. van Driest: Konstruktionszustand eingefroren (Statik-Lebensdauer-Lehre 02.09.)
	Memory<float> fac_sb; bool sism_on = false; uint sism_T = 0u; ulong sism_ab = 0ull; // ★ 07.09. SISM: EMA der 6 S-Komponenten je Facette (6 float), Konstruktionszustand + T/ab als Konstruktionszeit-Kopie eingefroren (read-once-Doktrin wie boden_eq_n; die Statik kann von einer Resetliste genullt werden, BEVOR alloc laeuft -- Lehre 02.09.). Kein Platzhalter im ctor noetig: kernel_sgs_fdwand entsteht selbst erst in alloc_facetten_domain
	Memory<uint> gd_zellen; Memory<float> fac_gd; Kernel kernel_sgs_gdiag; bool gdiag_on = false; // ★ 08.09. gd_zellen uint statt ulong (VRAM). ★ g-Diagnose: fid->Zellindex-Liste, 8-float-Akkumulator je Facette, eigener Kernel (kein Eingriff in stream_collide)
	void sgs_gdiag_gpu(); // Mess-Enqueue an der Chunk-/Sample-Kadenz (run mit finish)
	Memory<float> fac_kd; bool fac_kdiag_on = false; // ★ Klassen-Diagnostik-Akkumulator (16 float je Facette seit 05.09.), nur mit CFD_FAC_KDIAG; Konstruktionszustand eingefroren
	Memory<float> fac_us;    // EMA-Zustand 3 float je Facette (nur gebunden wenn s_fac_ema>0)
	bool fac_ema_on = false;
	static float s_fac_tau;  // 1 = voll, 0 = nur Tausch (CFD_FACETTEN=2)
	static float s_fac_budget;    // CFD_FAC_BUDGET (Facetten-1a-Arm B4t): Skalierung des Tangentialbudgets |s1|<=2ut*k, |s2|<=ut*k. Default 1.0 = bitidentisch. Die +-2ut/+-ut-Budgets sind DESIGN (Gl. 9), nie geeicht -- Planungsagent 2026-08-22.
	static float s_fac_budget_sn; // CFD_FAC_BUDGET_SN (Arm Bsn): Skalierung des sn-Budgets |sn|<=ut*k. Default 1.0 = bitidentisch.
	static float s_fac_isogate;   // CFD_FAC_ISOGATE (09.09.2026): 1 = ISOTROPES Tangentialgate sqrt(s1^2+s2^2) <= 2*budget*ut statt der getrennten Schranken |s1|<=2ut*k UND |s2|<=ut*k. Default 0.0 = bitidentisch. Anlass: gemessen liegen 43,4 % der Gate-Rueckfaelle mit s1_soll/u_t in [0;1] INNERHALB des s1-Budgets -- dort kann nur das s2-Gate mit dem HALBEN Budget gefeuert haben, und die Basis (t1 = Richtung des momentanen u_t, kernel.cpp:2073) rotiert mit der Stroemung. Das Gate misst dann die zufaellige Lage von u_s in einer mitrotierenden Basis, nicht die Groesse des Slips.
	static float s_fac_deteps;    // CFD_FAC_DETEPS (09.09.2026): Faktor K des Rauschbodens im Vollrangtest des gekoppelten Schur-Zweigs. det_eps = K*eps*(G11+G22)*Snnroh/Snn. Default 0.0 = bitidentisch. K=16 ist nachgerechnet: echte Rang-2-Linkmengen (Kante, Ecke) liegen 5-6 Dekaden ueber dem Rauschterm, ebene Mengen fallen sauber in den PINV-Zweig durch.
	bool facetten_on = false, facetten_bound = false; // read-once + Bindungswaechter
	uint fac_param_pos = 0u; ulong fac_N = 0ull;      // Parameterposition in stream_collide, aktive Facetten
	Memory<float> fac_geo;   // AoS 8 float je Facette: nx,ny,nz,yw,fac_a(=1/|n_achse|),achse,[6] Rekonstruktions-Amplitude eps (CFD_FAC_REK; S0 = 0),[7] Rekonstruktions-Marke (1 = Rang 0) -- seit 22.09.2026 BELEGT, der frueher hier notierte Zband-z-Anspruch ist damit hinfaellig, Stride-Umbau vertagt (D9)
	Memory<uint>  fac_idx;   // uint je F-BBox-Zelle: Facettenindex oder 0xFFFFFFFF
	Memory<float> fac_tau;   // Akkumulator 6 float je Facette: [0] Summe tau_w (y+), [1..3] Ist-Wandkraft x/y/z (Cd-Reibung), [4] Delta-m-Leck (iMEM; Paararm 0), [5] Normalkontamination (iMEM; Paararm 0); nur Zellen mit tatsaechlicher Modifikation
	Memory<uint>  fac_tau_n; // Akkumulator: Anzahl Beitraege
	Memory<uchar> fac_q; // ★ B1: q je Link (18 uchar je aktive Facette; 0 = kein Schnitt -> HWBB, sonst q = qb/254, 127 = exakt 0,5)
	// ★ kraft_facetten-GPU-Reduktion: Druckanteil des Cd-Pfads auf dem Geraet statt per Voll-F-Transfer.
	Memory<uint> kf_liste; // ★ 08.09. uint statt ulong; ★ 04.10. KF-FILTER: nur Wandsolidzellen (4 mm 3,74 M statt 62,1 M, ~14 MiB statt 237)  // Markerzellen-Indexliste (Host-Scan-Reihenfolge der F-BBox)
	Memory<float> kf_psum;   // 3 float je Arbeitsgruppe: px,py,pz-Teilsummen (atomikfrei)
	Memory<uint>  kf_pcnt;   // 3 uint je Arbeitsgruppe: voll,proj,unklar
	Kernel kernel_kraft_facetten;
	ulong kf_N=0ull; uchar kf_marker=0u; bool kf_zper=false, kf_bound=false; // Bindungsschluessel (marker,z_per) + Waechter
	ulong kf_verworfen=0ull; // ★ 04.10.2026 KF-FILTER (Pruefbefund M2): Markerzellen ohne Wandkontakt (F = 0), die der Listenbau verwirft -- fuer die Zensus-Ausgaben
	// ★ FORK Kraft-Zerlegung (CFD_KRAFT_ZBAND): zweiter Bindungs-Slot fuer die z-Band-Teilliste (z<zband).
	// Eigener Puffersatz + eigener Kernel -- der Hauptslot bleibt wortgleich unangetastet.
	Memory<uint> kfb_liste; // ★ 08.09. uint statt ulong (derselbe Membersatz wie kf_liste)  // Band-Teilliste (dieselbe Scan-Reihenfolge, Filter z<zband)
	Memory<float> kfb_psum;   // 3 float je Arbeitsgruppe
	Memory<uint>  kfb_pcnt;   // 3 uint je Arbeitsgruppe
	Kernel kernel_kraft_facetten_band;
	ulong kfb_N=0ull; uint kfb_zband=0u; uchar kfb_marker=0u; bool kfb_zper=false, kfb_bound=false; // EIGENE Bindungsschluessel (Pruefagent: der Hauptslot aktualisiert kf_marker/kf_zper VOR dem Bandslot-Vergleich -- geteilte Schluessel waeren ein stiller Stolperdraht)
	void bind_kraft_facetten(const std::vector<ulong>& liste, const uchar marker, const bool z_per, const bool band_slot=false); // Liste hochladen, Kernel binden; band_slot=true -> kfb_*-Satz
	void kraft_facetten_gpu(double& px, double& py, double& pz, ulong& n_voll, ulong& n_proj, ulong& n_unklar, const bool band_slot=false); // Kernel + double-Endsumme
	// ★ 04.10.2026 KRAFT-P1: eigener Kernel ueber den Hauptslot kf_liste (gebunden in bind_kraft_facetten, nur band_slot=false).
	// Feste Gruppenzahl kp1_groups (Grid-Stride, Determinismus wie object_force); je Gruppe 19 float + 12 uint (0,12 MB).
	Memory<float> kp1_f;     // 19 float je Arbeitsgruppe: [0..2] P x/y/z, [3..10] px Lage 0..7, [11..18] pz Lage 0..7
	Memory<uint>  kp1_u;     // 12 uint je Arbeitsgruppe: [0] Links, [1..8] Links je Lage, [9] TYPE_E-Quelle, [10] rho an RHO_CLAMP, [11] Wandzellen mit Link
	Kernel kernel_kraft_p1;
	uint kp1_groups = 1024u;
	bool kp1_bound = false;  // true erst, wenn kernel_kraft_p1 wirklich erzeugt ist (leere Liste: false, kraft_p1_gpu liefert dann Nullen mit ok)
	void kraft_p1_gpu(KP1& k, const uint haken); // haken 1: tt = t statt t-1, 2: Quelle j[i] statt j[ib] (Negativtests des Selbsttests)
	static bool s_sgs_wandfrei;
	static bool s_sgs_guo; // ★ 2026-08-25 Guo-Korrektur des Nichtgleichgewichtsmoments im Smagorinsky // Test B: kein nu_t in Wandzellen (CFD_SGS_WANDFREI)
	static bool s_sgs_diag;
	static ulong s_sgs_diag_ab;  // erster Zeitschritt, ab dem das nu_t-Histogramm zaehlt (Warmlaufsperre)     // P0-Diagnostik: nu_t/nu_0-Dekadenhistogramm, Slots 28..32 (CFD_SGS_DIAG). Default aus = kein Define, kein alloc, kein Zaehlen.
	static bool s_wandfunktion; // Wandfunktions-Bounce-Back nach Han et al. 2021 (CFD_WANDFUNKTION)
	static float s_wf_tau;      // 1 = volle WFB, 0 = nur Free-Slip-Tausch (Zwischenarm)
	static uint s_sponge_n;  // 0 = aus; Zonenbreite in Zellen (CFD_SPONGE_N)
	static float s_sponge_a; // Viskositaetsfaktor am Rand (CFD_SPONGE_A)
	static float s_sponge_wmin; // untere Klemme fuer w in der Zone (CFD_SPONGE_WMIN)
	static bool s_sparse_tiles_on; // CFD_SPARSE_TILES
	static uint s_sparse_T;        // CFD_TILE: 8 = VRAM-lastig (-40 % Tempo, 1,43 GB), 16 = Tempo-lastig (-28 %, 0,77 GB)
	void finalize_sparse_tiles();  // Tiles klassifizieren, sparse fi allozieren, Kernel neu binden

	Memory<rhoxx> rho; // density of every cell -- Speicherwort, NICHT die Dichte: rho_unpack/rho_pack (TODO 2 Schritt 4)
	Memory<velxx> u; // velocity of every cell -- Speicherwort, NICHT die Geschwindigkeit: u_unpack/u_pack (TODO 2 Schritt 4)
	Flags_Puffer flags; // flags of every cell -- ★ 05.10.2026 FLAGS4: Fassade (Hostfeld je Byte, Geraetepuffer ueber flags.k(), siehe Flags_Puffer)
	bool flags4_on = false;    // ★ 05.10.2026 FLAGS4: Konstruktionszeit-Kopie von s_flags4 (Geraetepuffer 4 Bit je Zelle, JIT-Define FLAGS4)
	const ulong bau_nr = ++domaenen_bau_zaehler; // ★ 05.10.2026 Korrektur A-N4/B-N4: Instanzkennung je Domaenenbau (Cache-Schluessel von fac_z_karte statt des Zeigers -- ABA-sicher)
	bool zellbasen_on = false;  // ★ 05.10.2026 ZELLBASEN: Entscheid je Domaene im Konstruktor VOR dem JIT (zellbasen_modus()==1, D3Q19, kein Block-Tiling, sizeof(fpxx)*N < 2^32), JIT-Define ZELLBASEN
	bool zk_on = false;  // ★ 06.10.2026 ZKS: Konstruktionszeit-Kopie von s_zk (vor dem JIT eingefroren)
	uint zk_nxp = 0u;
	uint zk_haken = 0u;  // ★ ZKS: CFD_ZK_HAKEN, im Konstruktor eingefroren (1..4 Zensus-Selbsttests, nur Host; 5 = [495]-Geraetehaken, nur CPU/iGPU)    // ★ ZKS: Nx auf ein Vielfaches der Arbeitsgruppe aufgerundet (Pad-Dispatch-Zeilenlaenge)
	static constexpr uint zk_w = 16u; // ★ ZKS: Lochausrichtung W (Lochanfang auf-, Lochende abgerundet); Lochterm im Kernel je Untergruppe (<= 16 Lanes)
	ulong zk_loecher = 0ull, zk_zeilen_loch = 0ull, zk_n_strich = 0ull; // ★ ZKS Zensus (zk_zensus): Lochzellen, Zeilen mit Loch, N' = N - Loecher
	vector<uint> zk_A, zk_L; // ★ ZKS Z4: Zensus je Zeile (Lochanfang, Lochlaenge), aus zk_zensus
	vector<ulong> zk_P;      // ★ ZKS Z4: exklusives Praefix der Lochlaengen (NR+1)
	ulong zk_S = 0ull;       // ★ ZKS Z4: fi-Stride S (N'+1 auf 64 gerundet), Papierkorb PK = N'
	uint zk_s2_pos = 0u;     // ★ ZKS Z4: Parameterposition von zk_s2 in stream_collide
	Kernel kernel_zk_pruef;  // ★ ZKS M-2: einmaliger Pruefkernel nach initialize
	ulong zk_s2_wert = 0ull; // ★ ZKS: der an stream_collide gesetzte Stride in Byte (fuer den Kopfvergleich in zk_pruef)
	void zk_pruefen();       // ★ ZKS M-2: Pruefkernel starten, Slots 496..499 lesen, Abbruch bei > 0 (vor der Zeitschleife)
	ulong zk_spaet_bytes(const bool mit_facetten) const; // ★ ZKS M-1: Spaetpuffer aus bekannten Groessen (Host-Flags, F-Box, Instanzschalter) statt Handwert 320 MiB
	void zk_finalisieren();  // ★ ZKS Z4: Zeilentabelle fuellen, Vorpruefung Stufe 2, fi kompakt anlegen, alle fi-Kernel neu binden
	// (Z2: Memory<uint> zk_tab -- entfallen, die Tabelle ist seit Z4 tile_slot) ★ ZKS Z2: Zeilentabelle, Kopf [0..15] (0 Magic 'ZKS1', 1 S = fi-Stride, 2 PK, 3 Strecken, 4 NR, 5 W, 6 Nx, 7 Nx_pad) + 16 uint Zeilensatz je Zeile; in Z2 LEER (P = 0, L = 0)
	ulong zk_gid(const ulong n) const { return (n/(ulong)get_Nx())*(ulong)zk_nxp+n%(ulong)get_Nx(); } // ★ ZKS: dichte Zellnummer -> Dispatch-Index (n = N -> Ende des Dispatch-Raums)
	void zk_zensus(); // ★ ZKS Z1: Lochzensus nach der Voxelierung (Chebyshev-2 ganz solid, periodisch, laengste ausgerichtete Strecke je Zeile), Selbsttests, ZK-ZENSUS-Zeile. Geraet bleibt in Z1 dicht
	bool zellbasen_jit = false; // ★ 05.10.2026 ZELLBASEN: aus dem JIT-Text eingefroren (Wirkpfad, Zeile ZELLBASEN im Log)
	bool rand_dispatch_on = false;  // ★ 06.10.2026 F1 RAND-DISPATCH: Konstruktionszeit-Kopie von rand_dispatch_modus()==1 (boden_eq per Bereichsstart, reiner Host-Pfad)
	bool rand_dispatch_haken_on = false; // ★ 06.10.2026 M2-Testhaken CFD_RAND_DISPATCH_HAKEN=1: Bereich um eine z-Ebene kuerzer, die Abnahme muss reissen
	ulong rd_boden_a = 0ull, rd_boden_anz = 0ull; // ★ 06.10.2026 F1: Startbereich, einmal in allocate aus den eingefrorenen boden_eq_* berechnet (dieselben Zahlen fuer Start und Logzeile RAND-DISPATCH)
	ulong boden_eq_stichproben = 0ull; // ★ 06.10.2026 M2: Zahl der enqueue_boden_eq-Aufrufe mit t%zaehl_takt()==0 (Soll-Faktor der Abnahme)
	void boden_eq_abnahme(const string& wo); // ★ 06.10.2026 M2: Ist=Soll Slot 20 + 117 gegen Bandzellen x Stichproben; der Aufrufer liest rho_clamp_hits vorher
	bool flags4_pruef = false; // ★ 05.10.2026 FLAGS4: Konstruktionszeit-Kopie von s_flags4_pruef (CFD_FLAGS4 in der Zeile gesetzt, 0 oder 1): Geraeteprobe nach dem Hochladen und nach initialize
	uint flags4_geraeteprobe(const string& wann, const bool gegen_host); // lbm.cpp: Kernel flags_pruefsumme gegen die Hostrechnung; liefert die Zahl abweichender 256er-Gruppen
#ifdef FORCE_FIELD
	Memory<float> F; // individual force for every cell
	// ★ 03.09.2026 F-MARKERLISTE (CFD_F_LISTE, Default aus). Layout wie fac_idx: je 32 F-BBox-Zellen
	// ein Paar [2b]=Maske (Bit gesetzt = WANDsolidzelle, hat einen F-Slot), [2b+1]=exklusive
	// Praefixsumme. Das LETZTE Wort (Index 2*ceil(FBN/32)) traegt die Slotzahl = den Stride von F,
	// weil der erst nach dem Maskenbau feststeht und deshalb kein Compile-Define sein kann.
	Memory<uint> f_maske;
	bool fac_rek_r3_jit = false; bool fac_rek_s2_jit = false; // ★ 23.09. R3-Arm im uebersetzten Kernel (aus dem JIT-Text, nicht aus der Statik)
	ulong fac_rek_marken = 0ull; float fac_rek_eps = 0.0f; // ★ 22.09. S0: Zahl der gesetzten Marken und die Amplitude -- Vergleichsgroessen fuer die Abnahme
	bool fac_r1q_on = false; uint fac_r1q_stufe = 0u; bool fac_r1q_jit = false; bool fac_r1q_an_jit = false; bool fac_r1q_vr_jit = false; bool fac_r1q_od_jit = false; bool fac_r1q_k2_jit = false; bool sc_simd16_jit = false; ulong sc_spill = ~0ull; ulong sc_private = ~0ull; ulong fac_r1q_marken = 0ull; // ★ 28.09. R1Q: Hostzustand, Kernelzustand aus dem JIT-Text (Mess-/Anwendungsdefine getrennt), Zahl der gesetzten Marken
	bool fac_rekpi_on = false; uint fac_rekpi_stufe = 0u; bool fac_rekpi_jit = false; bool fac_rekpi_an_jit = false; bool fac_rekpi_nur_u_jit = false; // ★ 04.10. Stufe 3 = FAC_REKPI_NUR_U // ★ 03.10. REK-PI: Host eingefroren (Konstruktor) gegen JIT-Text (Kohaerenz in alloc_facetten_domain)
	bool fac_rek_on = false; bool fac_rek_jit = false; // ★ 22.09. S0: Hostzustand und der aus dem JIT-Text gelesene Kernelzustand -- alloc vergleicht sie (Pruefbefund M2 vom selben Tag)
	bool fac_pinv_on = false; // Konstruktionszustand eingefroren
	bool fac_idx_voll_on = false; // Konstruktionszustand eingefroren; true = alte Vollfeld-Bauform von fac_idx
	bool f_liste_on = false; // Konstruktionszustand eingefroren (Statik-Lebensdauer-Lehre 02.09.)
	ulong f_slots = 0ull;    // Zahl der Wandsolidzellen = F-Slots; 0 = Vollfeld-Arm
	uint f_param_sc = 0u, f_param_uf = 0u; // Signaturposition von F in stream_collide bzw. update_fields (fuer den Rebind nach der Neuanlage)
	// ★★ 08.09.2026 SGS-BAND (CFD_SGS_BAND): SISM auf die Wandlagen 2..N statt nur auf die Facettenzellen.
	// Heikos Befund des Tages: die Einzellink-Zellen (16,3 % der wandnahen Zellen, y_w ~ 1,1 statt 0,5)
	// sind geometrisch LAGE 2 und werden von einem Lage-1-Modell nie erreicht. Gemessen faellt die
	// SISM-Absenkung nach aussen kaum ab (4 mm: Lage 1/2/3 = 85,2/80,0/75,9 %), das Band traegt also.
	// KEIN neuer Kernel: derselbe sgs_fdwand-Text wird ein zweites Mal mit den band_*-Puffern gestartet
	// (zweites Kernel-Objekt aus demselben Programm) -- damit entfaellt jedes Klammerfallen-Risiko.
	// Die Bandliste ist DISJUNKT zur Facettenmenge; stream_collide fragt erst fdw_fid, dann band_fid.
	Memory<uint> band_idx;    // Bitmaske+Praefixsumme ueber die F-BBox (Muster fac_idx), nur Lage 2..N
	Memory<uint> band_zellen; // ★ 22.09.2026 (B32): GLOBALER Zellindex n je Bandzelle in fbi-Sortierreihenfolge (uint; Gitter < 2^32 Zellen, Waechter in alloc_sgs_band). Bis 22.09. stand hier fbi -- der Kernel liest n; am Fahrzeug (Teil-BBox) rechnete das Band deshalb an fremden Zellen.
	Memory<float> band_sbar;  // ★ 08.09. NACH dem Kipptest: Sbar (Betrag des zeitgemittelten Scherratentensors) je Bandzelle. Frueher war das ein fertiges w -- der w-ERSATZ kippte am 8-mm-Stressarm bei Schritt 392, auch ohne SISM.
	Memory<float> band_sb;    // EMA der 6 S-Komponenten je Bandzelle (nur unter SISM belegt)
	float nut_skal = 1.0f; // ★ 10.09. Diskriminator-Messarm: Konstruktionszustand, eingefroren wie band_on. Die Abnahme liest IHN, nicht die Umgebungsvariable -- sonst haette sie eine zweite Wahrheitsquelle (Muster pruefe_band_wirkpfad, das D->band_on liest).
	ulong band_N = 0ull; uint band_lagen = 0u; bool band_on = false; bool band_pi_on = false; bool band_pi_jit = false; /* ★ 22.09. A-M2: Kernel-Modus aus dem JIT-Text */ uint band_param_pos = 0u; // Signaturposition von band_idx in stream_collide, in alloc_facetten_domain berechnet (dort sind alle Schalter im Scope)
	ulong band_n_lage[8] = {0,0,0,0,0,0,0,0}; // Zellzahl je Lage, fuer den Bericht und Ist=Soll
	// ★ 22.09.2026 BAND-g-DIAGNOSE (Plan B3(b)): zweite Instanz des sgs_gdiag-Kernels ueber band_zellen -- misst |S|_FD und |S|_Pi an DERSELBEN
	// Bandzelle zur selben Zeit (Pi/FD in Lage 2..N war nie gemessen; der Band-Abzug mischt Pi-|S| mit FD-Sbar). Physikfrei, nur mit CFD_SGS_GDIAG.
	Memory<float> band_gd; Kernel kernel_band_gdiag; std::vector<uchar> band_lage_h; bool band_gdiag_on = false;
	Kernel kernel_sgs_band;
	void alloc_sgs_band(const uchar* flags_host, const uint Nx, const uint Ny, const uint Nz, const uint lagen);
	uint pruefe_rho_rand_c0(const uchar* flags_host, const uint Nx, const uint Ny, const uint Nz, const bool testhaken); // ★ 15.09. RHO_RAND C0: Host-Waechter (R1-Abdeckung aller rho-Leser) + APG-Zensus; liefert die Zahl der Beanstandungen
	// ★ 04.10.2026 U_RAND U0 (PLAN-VRAM-URAND-FLAGS-2026-10-04.md B.8/B.10): Mengenbau und Host-Zensus. Reine Hostpruefung, kein Geraetecode.
	// E = statische u-Lesemenge im Nahfeldinneren (Facetten L1, ihre sgs_fdwand-Achsnachbarn, fac_nachbar_ab-Linkziele, Wandsolid fuer
	// update_force_field, Solidnachbarn kuenftiger TYPE_MS-Zellen); R1 = Randschale Dicke 2; C = N2F-4^3-Bloecke. Liefert die Beanstandungen.
	struct URandMengen {
		std::vector<ulong> e_bits;                 // N Bits, gesetzt = Zelle in E (auch wenn sie zugleich in R1 liegt)
		ulong n_e_aus = 0ull, n_e_r1 = 0ull;       // |E ohne R1|, |E in R1|
		uint ex0=0u, ey0=0u, ez0=0u, enx=0u, eny=0u, enz=0u; // Huelle von E ohne R1 (Ursprung, Masse; leer = 0)
		uint cbx0=0u, cby0=0u, cbz0=0u, cbnx=0u, cbny=0u, cbnz=0u; // Huelle der N2F-Bloecke in BLOCKkoordinaten (Block = (x+2)/4)
		std::vector<ulong> c_bits;                 // Blockmenge ueber die Blockhuelle (1 Bit je Block)
		ulong n_bloecke = 0ull;
		uint gx0=0u, gy0=0u, gz0=0u, gnx=0u, gny=0u, gnz=0u; // Grossbox (Stufe 1a) = Huelle von (E ohne R1) und den Blockzellen
		uint cr = 4u, ch = 2u; // ★ 05.10.2026 Korrektur A-H1: Blockkante = ratio, Fensterversatz = ratio/2 (schale_extract: Fenster [c-ch, c-ch+cr)), Block(x) = (x+ch)/cr
		std::vector<ulong> zentren; // ★ A-H1: Kopie der N2F-Deckungspunkte fuer die Fensterprobe NACH dem Layout (alloc_u_rand), danach geleert
		std::vector<ulong> haken_aus; // ★ 05.10.2026 Laufzeit-Testhaken 6/7/8 (CFD_U_RAND_TESTHAKEN): Zellen, die ur_layout (Modus 3) NICHT speichert -- leer im Normalbetrieb
	};
	URandMengen ur_mengen; // gefuellt von pruefe_u_rand_c0, gelesen vom Speicherbau (ab U1d)
	uint pruefe_u_rand_c0(const uchar* flags_host, const velxx* u_host, const uint Nx, const uint Ny, const uint Nz, const std::vector<ulong>& n2f_zentren, const uint ratio, const float u_lat, const uint testhaken); // ★ 05.10.2026 A-H1: ratio (Blockkante der N2F-Fenster)
	void alloc_f_liste(const uchar* flags_host, const uint Nx, const uint Ny, const uint Nz); // baut Maske+Praefix, legt F neu an, rebindet
	// ★ 04.10.2026 KF-FILTER (Pruefbefund M1): DAS Wandsolid-Praedikat von alloc_f_liste, herausgezogen -- Solid (nicht TYPE_E) mit mindestens einem
	// Nicht-Solid-Nachbarn unter den 18 (D3Q27: 26) Richtungen, z ausserhalb des Gitters zaehlt als Wand. Der kf_liste-Filter nutzt es in BEIDEN
	// Armen (CFD_F_LISTE 0/1), damit die Liste armgleich bleibt; alloc_f_liste ruft dieselbe Funktion (wortgleich per Konstruktion).
	static bool wand_solid_host(const uchar* flags_host, const uint x, const uint y, const uint z, const uint Nx, const uint Ny, const uint Nz);
	// Zelle (als F-BBox-Index) -> F-Slot. AUSDRUCKSGLEICH zu f_slot() in kernel.cpp -- beide Pfade
	// werden von CFD_FAC_GPU_PRUEF zahlenscharf gegeneinander gestellt.
	inline bool f_slot_host(const ulong fbi, ulong& slot) const {
		if(!f_liste_on) { slot = fbi; return true; }
		const ulong ib = 2ull*(fbi>>5);
		const uint maske = f_maske[ib];
		const uint l = (uint)(fbi&31ull);
		if(((maske>>l)&1u)==0u) return false;
		slot = (ulong)f_maske[ib+1ull] + (ulong)__builtin_popcount(maske & ((1u<<l)-1u));
		return true;
	}
	inline ulong f_stride() const { return f_liste_on ? f_slots : (ulong)fbnx*(ulong)fbny*(ulong)fbnz; }
	Memory<float> object_sum; // sum of individual cell data for an object
#endif // FORCE_FIELD
#ifdef SURFACE
	Memory<float> phi; // fill level of every cell
#endif // SURFACE
#ifdef TEMPERATURE
	Memory<float> T; // temperature of every cell
#endif // TEMPERATURE
#ifdef PARTICLES
	Memory<float> particles; // particle positions
#endif // PARTICLES

	Memory<char> transfer_buffer_p, transfer_buffer_m; // transfer buffers for multi-device domain communication, only allocate one set of transfer buffers in plus/minus directions, for all x/y/z transfers
	Kernel kernel_transfer[enum_transfer_field::enum_transfer_field_length][2]; // for each field one extract and one insert kernel
	void allocate_transfer(Device& device); // allocate all memory for multi-device transfer
	ulong get_area(const uint direction);
	void enqueue_transfer_extract_field(Kernel& kernel_transfer_extract_field, const uint direction, const uint bytes_per_cell);
	void enqueue_transfer_insert_field(Kernel& kernel_transfer_insert_field, const uint direction, const uint bytes_per_cell);

	LBM_Domain(const Device_Info& device_info, const uint Nx, const uint Ny, const uint Nz, const uint Dx, const uint Dy, const uint Dz, const int Ox, const int Oy, const int Oz, const float nu, const float fx, const float fy, const float fz, const float sigma, const float alpha, const float beta, const uint particles_N, const float particles_rho); // compiles OpenCL C code and allocates memory

	void enqueue_initialize(); // write all data fields to device and call kernel_initialize
	void enqueue_stream_collide(); // call kernel_stream_collide to perform one LBM time step
	void enqueue_update_fields(); // update fields (rho, u, T) manually
	void enqueue_boden_eq(); // V1-apply_floor_velocity-Port
	void enqueue_einlass_eq(); // V1-apply_inlet_velocity-Port
	void enqueue_apply_pressure_outlet(); // FORK: Druck-Auslass, No-op ohne konfigurierte Flaechen
	void set_pressure_outlet_faces(const uint face_mask, const float rho_out); // FORK: TYPE_E-Zellen der Aussenflaechen sammeln und Kernel bauen
#ifdef SURFACE
	void enqueue_surface_0();
	void enqueue_surface_1();
	void enqueue_surface_2();
	void enqueue_surface_3();
#endif // SURFACE
#ifdef FORCE_FIELD
	void enqueue_update_force_field(); // calculate forces from fluid on TYPE_S cells
	void enqueue_object_center_of_mass(const uchar flag_marker=TYPE_S); // calculate center of mass of all cells flagged with flag_marker
	void enqueue_object_force(const uchar flag_marker=TYPE_S); // add up force for all cells flagged with flag_marker
	void enqueue_object_force_zband(const uchar flag_marker, const uint z_lo, const uint z_hi); // FORK Kraft-Zerlegung: object_force auf das z-Band [z_lo,z_hi); object_sum WIEDERVERWENDET -- strikt sequenziell zu enqueue_object_force aufrufen
	void enqueue_object_torque(const float3& rotation_center, const uchar flag_marker=TYPE_S); // add up torque around specified rotation_center for all cells flagged with flag_marker
#endif // FORCE_FIELD
#ifdef MOVING_BOUNDARIES
	void enqueue_update_moving_boundaries(); // mark/unmark cells next to TYPE_S cells with velocity!=0 with TYPE_MS
#endif // MOVING_BOUNDARIES
#ifdef PARTICLES
	void enqueue_integrate_particles(const uint time_step_multiplicator=1u); // intgegrates particles forward in time and couples particles to fluid
#endif // PARTICLES

	void increment_time_step(const ulong steps=1ull); // increment time step
	void reset_time_step(); // reset time step
	void finish_queue();
	void flush_queue();

	const Device& get_device() const { return device; }
	uint get_Nx() const { return Nx; } // get (local) lattice dimensions in x-direction
	uint get_Ny() const { return Ny; } // get (local) lattice dimensions in y-direction
	uint get_Nz() const { return Nz; } // get (local) lattice dimensions in z-direction
	ulong get_N() const { return (ulong)Nx*(ulong)Ny*(ulong)Nz; } // get (local) number of lattice points
	uint get_Dx() const { return Dx; } // get lattice domains in x-direction
	uint get_Dy() const { return Dy; } // get lattice domains in y-direction
	uint get_Dz() const { return Dz; } // get lattice domains in z-direction
	uint get_D() const { return Dx*Dy*Dz; } // get number of lattice domains
	float get_nu() const { return nu; } // get kinematic shear viscosity
	float get_tau() const { return 3.0f*get_nu()+0.5f; } // get LBM relaxation time
	float get_fx() const { return fx; } // get global froce per volume
	float get_fy() const { return fy; } // get global froce per volume
	float get_fz() const { return fz; } // get global froce per volume
	float get_sigma() const { return sigma; } // get surface tension coefficient
	float get_alpha() const { return alpha; } // get thermal diffusion coefficient
	float get_beta() const { return beta; } // get thermal expansion coefficient
	ulong get_t() const { return t; } // get discrete time step in LBM units
	uint get_velocity_set() const; // get LBM velocity set
	void set_fx(const float fx) { this->fx = fx; } // set global froce per volume
	void set_fy(const float fy) { this->fy = fy; } // set global froce per volume
	void set_fz(const float fz) { this->fz = fz; } // set global froce per volume
	void set_f(const float fx, const float fy, const float fz) { set_fx(fx); set_fy(fy); set_fz(fz); } // set global froce per volume

	void voxelize_mesh_on_device(const Mesh* mesh, const uchar flag=TYPE_S, const float3& rotation_center=float3(0.0f), const float3& linear_velocity=float3(0.0f), const float3& rotational_velocity=float3(0.0f)); // voxelize mesh
	void enqueue_unvoxelize_mesh_on_device(const Mesh* mesh, const uchar flag=TYPE_S); // remove voxelized triangle mesh from LBM grid

#ifdef GRAPHICS
	class Graphics {
	private:
		Kernel kernel_clear; // reset bitmap and zbuffer
		Memory<int> bitmap; // bitmap for rendering
		Memory<int> zbuffer; // z-buffer for rendering
		Memory<float> camera_parameters; // contains camera position, rotation, field of view etc.

		LBM_Domain* lbm = nullptr;
		Kernel kernel_graphics_flags; // render flag lattice with wireframe
		Kernel kernel_graphics_flags_mc; // render flag lattice with marching-cubes
		Kernel kernel_graphics_field; // render a colored velocity vector for each cell
		Kernel kernel_graphics_field_slice; // render one slice of velocity field according to slics settings
		Kernel kernel_graphics_streamline; // render streamlines
		Kernel kernel_graphics_q; // render vorticity (Q-criterion)

#ifdef SURFACE
		const string path_skybox = get_exe_path()+"../skybox/skybox8k.png";
		Image* skybox_image = nullptr;
		Memory<int> skybox; // skybox for free surface raytracing
		Kernel kernel_graphics_rasterize_phi; // rasterize free surface
		Kernel kernel_graphics_raytrace_phi; // raytrace free surface
		Image* get_skybox_image() const { return skybox_image; }
#endif // SURFACE

#ifdef PARTICLES
		Kernel kernel_graphics_particles;
#endif // PARTICLES

		ulong t_last_rendered_frame = max_ulong; // optimization to not call draw_frame() multiple times if camera_parameters and LBM time step are unchanged
		bool update_camera(); // update camera_parameters and return if they are changed from their previous state

	public:
		Graphics() {} // default constructor
		Graphics(LBM_Domain* lbm) {
			this->lbm = lbm;
#ifdef SURFACE
			skybox_image = read_png(path_skybox);
#endif // SURFACE
		}
		Graphics& operator=(const Graphics& graphics) { // copy assignment
			lbm = graphics.lbm;
#ifdef SURFACE
			skybox_image = graphics.get_skybox_image();
#endif // SURFACE
			return *this;
		}
		void allocate(Device& device); // allocate memory for bitmap and zbuffer
		bool enqueue_draw_frame(const int visualization_modes, const int field_mode=0, const int slice_mode=0, const int slice_x=0, const int slice_y=0, const int slice_z=0, const bool visualization_change=true); // main rendering function, calls rendering kernels, returns true if new frame is rendered, false if old frame is returned when camera has not moved
		int* get_bitmap(); // returns pointer to bitmap
		int* get_zbuffer(); // returns pointer to zbuffer
		string device_defines(const Device_Info& device_info) const; // returns preprocessor constants for embedding in OpenCL C code
	}; // Graphics
	Graphics graphics;
#endif // GRAPHICS
}; // LBM_Domain



// FORK Doppel-Domaene: achsen-normale Ebene in Zellkoordinaten einer Domaene.
// axis: 0 = X-normal (Ebene spannt Y,Z), 1 = Y-normal (spannt X,Z), 2 = Z-normal (spannt X,Y).
// extent_a laeuft ueber die ERSTE aufgespannte Achse, extent_b ueber die zweite.
struct PlaneSpec {
	uint3 origin;   // Zellindex der unteren Ecke
	uint extent_a;  // Zellen entlang der ersten aufgespannten Achse
	uint extent_b;  // Zellen entlang der zweiten aufgespannten Achse
	uint axis;      // Normalenachse (0=x, 1=y, 2=z)
};

class LBM {
private:
	uint Nx=1u, Ny=1u, Nz=1u; // (global) lattice dimensions
	uint Dx=1u, Dy=1u, Dz=1u; // lattice domains
	bool initialized = false; // becomes true after LBM::initialize() has been called

	void sanity_checks_constructor(const vector<Device_Info>& device_infos, const uint Nx, const uint Ny, const uint Nz, const uint Dx, const uint Dy, const uint Dz, const float nu, const float fx, const float fy, const float fz, const float sigma, const float alpha, const float beta, const uint particles_N, const float particles_rho); // sanity checks on grid resolution and extension support
	void sanity_checks_initialization(); // sanity checks during initialization on used extensions based on used flags
	void initialize(); // write all data fields to device and call kernel_initialize
	void do_time_step(const bool sync_single_gpu=true); // call kernel_stream_collide to perform one LBM time step; sync_single_gpu=false laesst die Warteschlange offen (fuer run_async)

	void communicate_field(const enum_transfer_field field, const uint bytes_per_cell);

	void communicate_fi();
	void communicate_rho_u_flags();
	void communicate_flags();
#ifdef FORCE_FIELD
	void communicate_F();
#endif // FORCE_FIELD
#ifdef SURFACE
	void communicate_phi_massex_flags();
#endif // SURFACE
#ifdef TEMPERATURE
	void communicate_gi();
	void communicate_T();
#endif // TEMPERATURE
#ifdef PARTICLES
	void communicate_particles();
#endif // PARTICLES

public:
	template<typename T> class Memory_Container { // does not hold any data itsef, just links to LBM_Domain data
	private:
		ulong N = 0ull; // buffer length
		uint d = 1u; // buffer dimensions
		LBM* lbm = nullptr;
		Memory<T>** buffers = nullptr; // host buffers
		string name = "";

		uint Nx=1u, Ny=1u, Nz=1u, Dx=1u, Dy=1u, Dz=1u, D=1u; // auxiliary variables: (local) lattice dimensions, lattice domains, number of domains
		uint NxDx=1u, NyDy=1u, NzDz=1u, Hx=0u, Hy=0u, Hz=0u; // auxiliary variables: number of domains, shortcuts for N_/D_, halo offsets
		ulong NxNy=1ull, local_Nx=1ull, local_Ny=1ull, local_Nz=1ull, local_N=1ull; // auxiliary variables: shortcut for Nx*Ny, size of each domain, number of cells in each domain
		inline void initialize_auxiliary_variables() { // these variables are frequently used in reference() functions, so pre-compute them only once here
			Nx = lbm->get_Nx(); Ny = lbm->get_Ny(); Nz = lbm->get_Nz();
			Dx = lbm->get_Dx(); Dy = lbm->get_Dy(); Dz = lbm->get_Dz();
			D = Dx*Dy*Dz; // number of domains
			NxNy = (ulong)Nx*(ulong)Ny; // shortcut for Nx*Ny
			NxDx=Nx/Dx; NyDy=Ny/Dy; NzDz=Nz/Dz; // shortcuts for N_/D_
			Hx=Dx>1u; Hy=Dy>1u; Hz=Dz>1u; // halo offsets
			local_Nx=(ulong)(NxDx+2u*Hx); local_Ny=(ulong)(NyDy+2u*Hy); local_Nz=(ulong)(NzDz+2u*Hz); // size of each domain
			local_N = local_Nx*local_Ny*local_Nz; // number of cells in each domain
		}
		inline void initialize_auxiliary_pointers() {
			/********/ x = Pointer(this, 0x0u);
			if(d>0x1u) y = Pointer(this, 0x1u);
			if(d>0x2u) z = Pointer(this, 0x2u);
		}
		inline T& reference(const ulong i) { // stitch together domain buffers and make them appear as one single large buffer
			if(D==1u) { // take shortcut for single domain
				return buffers[0]->data()[i]; // array of structures
			} else { // decompose index for multiple domains
				const ulong global_i=i%N, t=global_i%NxNy;
				const uint x=(uint)(t%(ulong)Nx), y=(uint)(t/(ulong)Nx), z=(uint)(global_i/NxNy); // n = x+(y+z*Ny)*Nx
				const uint px=x%NxDx, py=y%NyDy, pz=z%NzDz, dx=x/NxDx, dy=y/NyDy, dz=z/NzDz, domain=dx+(dy+dz*Dy)*Dx; // 3D position within domain and which domain
				const ulong local_i = (ulong)(px+Hx)+((ulong)(py+Hy)+(ulong)(pz+Hz)*local_Ny)*local_Nx; // add halo offsets
				const ulong local_dimension = i/N;
				return buffers[domain]->data()[local_i+local_dimension*local_N]; // array of structures
			}
		}
		inline T& reference(const ulong i, const uint dimension) { // stitch together domain buffers and make them appear as one single large buffer
			if(D==1u) { // take shortcut for single domain
				return buffers[0]->data()[i+(ulong)dimension*N]; // array of structures
			} else { // decompose index for multiple domains
				const ulong global_i=i%N, t=global_i%NxNy;
				const uint x=(uint)(t%(ulong)Nx), y=(uint)(t/(ulong)Nx), z=(uint)(global_i/NxNy); // n = x+(y+z*Ny)*Nx
				const uint px=x%NxDx, py=y%NyDy, pz=z%NzDz, dx=x/NxDx, dy=y/NyDy, dz=z/NzDz, domain=dx+(dy+dz*Dy)*Dx; // 3D position within domain and which domain
				const ulong local_i = (ulong)(px+Hx)+((ulong)(py+Hy)+(ulong)(pz+Hz)*local_Ny)*local_Nx; // add halo offsets
				const ulong local_dimension = max(i/N, (ulong)dimension);
				return buffers[domain]->data()[local_i+local_dimension*local_N]; // array of structures
			}
		}
		inline string vtk_type() const {
			/**/ if constexpr(std::is_same<T, char >::value) return "char" ; else if constexpr(std::is_same<T, uchar >::value) return "unsigned_char" ;
			else if constexpr(std::is_same<T, short>::value) return "short"; else if constexpr(std::is_same<T, ushort>::value) return "unsigned_short";
			else if constexpr(std::is_same<T, int  >::value) return "int"  ; else if constexpr(std::is_same<T, uint  >::value) return "unsigned_int"  ;
			else if constexpr(std::is_same<T, slong>::value) return "long" ; else if constexpr(std::is_same<T, ulong >::value) return "unsigned_long" ;
			else if constexpr(std::is_same<T, float>::value) return "float"; else if constexpr(std::is_same<T, double>::value) return "double"        ;
			else print_error("Error in vtk_type(): Type not supported.");
			return "";
		}
		inline void write_vtk(const string& path, const bool convert_to_si_units=true) { // write binary .vtk file
			float spacing = 1.0f;
			T unit_conversion_factor = (T)1;
			if(convert_to_si_units) {
				spacing = units.si_x(1.0f);
				if(name=="rho") unit_conversion_factor = (T)units.si_rho(1.0f);
				if(name=="u"  ) unit_conversion_factor = (T)units.si_u  (1.0f);
				if(name=="F"  ) unit_conversion_factor = (T)units.si_F  (1.0f);
				if(name=="T"  ) unit_conversion_factor = (T)units.si_T  (1.0f);
			}
			const string filename = create_file_extension(path, ".vtk");
			const float3 origin = spacing*float3(0.5f-0.5f*(float)Nx, 0.5f-0.5f*(float)Ny, 0.5f-0.5f*(float)Nz);
			const string header =
				"# vtk DataFile Version 3.0\nFluidX3D "+filename.substr(filename.rfind('/')+1)+"\nBINARY\nDATASET STRUCTURED_POINTS\n"
				"DIMENSIONS "+to_string(Nx)+" "+to_string(Ny)+" "+to_string(Nz)+"\n"
				"ORIGIN "+to_string(origin.x)+" "+to_string(origin.y)+" "+to_string(origin.z)+"\n"
				"SPACING "+to_string(spacing)+" "+to_string(spacing)+" "+to_string(spacing)+"\n"
				"POINT_DATA "+to_string((ulong)Nx*(ulong)Ny*(ulong)Nz)+"\n"
				"SCALARS data "+vtk_type()+" "+to_string(dimensions())+"\nLOOKUP_TABLE default\n"
			;
			const uint chunk_size_MB = 4u*thread::hardware_concurrency(); // in MB; convert and write data in chunks, to reduce memory footprint and time for large memory allocation
			const ulong chunk_elements = (1048576ull*(ulong)chunk_size_MB)/((ulong)dimensions()*sizeof(T));
			const ulong chunks=length()/chunk_elements, chunk_remainder=length()%chunk_elements;
			T* data = new T[chunk_elements*(ulong)dimensions()];
			create_folder(filename);
			std::ofstream file(filename, std::ios::out|std::ios::binary);
			file.write(header.c_str(), header.length()); // write non-binary file header
			for(ulong c=0u; c<chunks+1ull; c++) { // iterate over all full chunks + last chunk_remainder chunk
				const ulong N = c<chunks ? chunk_elements : chunk_remainder;
				if(N==0ull) break; // chunk_remainder may be 0, then skip last iteration
				parallel_for(N, [&](ulong i) {
					for(uint d=0u; d<dimensions(); d++) { // LBM to SI units, LittleEndian to BigEndian, AoS to SoA
						data[i*(ulong)dimensions()+(ulong)d] = reverse_bytes((T)(unit_conversion_factor*reference(c*chunk_elements+i, d)));
					}
				});
				file.write((char*)data, N*(ulong)dimensions()*sizeof(T)); // write binary data
			}
			file.close();
			delete[] data;
			info.allow_printing.lock();
			print_info("File \""+filename+"\" saved.");
			info.allow_printing.unlock();
		}

	public:
		class Pointer {
		private:
			Memory_Container* memory = nullptr;
			uint dimension = 0u;
		public:
			inline Pointer() {}; // default constructor
			inline Pointer(Memory_Container* memory, const uint dimension) {
				this->memory = memory;
				this->dimension = dimension;
			}
			inline T& operator[](const ulong i) { return memory->reference(i, dimension); }
			inline const T& operator[](const ulong i) const { return memory->reference(i, dimension); }
		};
		Pointer x, y, z; // host buffer auxiliary pointers for multi-dimensional array access (array of structures)

		inline Memory_Container(LBM* lbm, Memory<T>** buffers, const string& name) {
			this->N = lbm->get_N();
			this->d = buffers[0]->dimensions();
			if(this->N*(ulong)this->d==0ull) print_error("Memory size must be larger than 0.");
			this->lbm = lbm;
			this->buffers = buffers;
			this->name = name;
			initialize_auxiliary_variables();
			initialize_auxiliary_pointers();
		}
		inline Memory_Container() {} // default constructor
		inline Memory_Container& operator=(Memory_Container&& memory) noexcept { // move assignment
			this->N = memory.N;
			this->d = memory.d;
			this->lbm = memory.lbm;
			this->buffers = memory.buffers;
			this->name = memory.name;
			initialize_auxiliary_variables();
			initialize_auxiliary_pointers();
			return *this;
		}
		inline void reset(const T value=(T)0) {
			for(uint domain=0u; domain<D; domain++) buffers[domain]->reset(value);
		}
		inline const ulong length() const { return N; }
		inline const uint dimensions() const { return d; }
		inline const ulong range() const { return N*(ulong)d; }
		inline const ulong capacity() const { return N*(ulong)d*sizeof(T); } // returns capacity of the buffer in Byte
		inline T& operator[](const ulong i) { return reference(i); }
		inline const T& operator[](const ulong i) const { return reference(i); }
		inline const T operator()(const ulong i) const { return reference(i); }
		inline const T operator()(const ulong i, const uint dimension) const { return reference(i, dimension); } // array of structures
		inline void read_from_device() {
#ifndef UPDATE_FIELDS
			if(lbm->initialized) for(uint domain=0u; domain<D; domain++) lbm->lbm_domain[domain]->enqueue_update_fields(); // only if simulation has already been initialized: make sure data in device memory is up-to-date
#endif // UPDATE_FIELDS
			for(uint domain=0u; domain<D; domain++) buffers[domain]->enqueue_read_from_device();
			for(uint domain=0u; domain<D; domain++) buffers[domain]->finish_queue();
		}
		inline void write_to_device() {
			for(uint domain=0u; domain<D; domain++) buffers[domain]->enqueue_write_to_device();
			for(uint domain=0u; domain<D; domain++) buffers[domain]->finish_queue();
		}
		inline void write_host_to_vtk(const string& path="", const bool convert_to_si_units=true) { // write binary .vtk file
			write_vtk(default_filename(path, name, ".vtk", lbm->get_t()), convert_to_si_units);
		}
		inline void write_device_to_vtk(const string& path="", const bool convert_to_si_units=true) { // write binary .vtk file
			read_from_device();
			write_host_to_vtk(path, convert_to_si_units);
		}
	};

	// ★ TODO 2 Schritt 4 (12.09.2026) -- rho liegt NICHT mehr als Memory_Container offen.
	// GRUND, und er ist die Hauptfalle dieses Umbaus: Memory_Container::operator[] gibt T& zurueck.
	// Mit T=ushort bleibt JEDE Hostzugriffsstelle nach dem Formatwechsel uebersetzbar und rechnet
	// still falsch -- ushort->float ist keine Verengung und warnt nicht, und eine der Stellen
	// (LBM::lese_yslice_in_host) SCHREIBT. Ein Grep findet, was man sucht; ein geloeschter
	// operator[] findet, was man NICHT sucht. Deshalb hier get/set statt Indizierung: jede
	// vergessene Stelle ist ein Uebersetzungsfehler.
	// ★ 15.09.2026 RHO_RAND C2c (RHO_RAND-C2-PLAN.md §3.2): unter RHO_RAND haelt der Puffer nur die Randschale R1. Der
	// Memory_Container rechnet aber mit N (lbm.hpp, reference()) -- jedes c[n] waere ein Heap-Ueberlauf (Plan K5). Deshalb
	// im RAND-Betrieb: get() liest die zuletzt gesetzte Ausgabe-Ebene (Nachkollisionssumme, float, NIE rho_pack -- der
	// Hostpacker rundet anders), sonst R1 ueber rr_idx_host -- aber nur dort, wo R1 gepflegt wird (TYPE_E oder x >= Nx-2);
	// alles andere ist ein harter Fehler (Host-Zugriffssperre). set() ist gesperrt.
	class Rho_Feld {
	private:
		Memory_Container<rhoxx> c;
		LBM* lbm_ = nullptr; bool rand = false;
		std::vector<float> ebene; uint ebene_achse = 3u, ebene_pos = 0u; ulong ebene_t = ~0ull;
		ulong r1_t = ~0ull; // Zeitschritt des letzten R1-Hostspiegel-Reads (Pruefpass C2c NIEDRIG 3)
		float get_rand(const ulong n); // lbm.cpp
	public:
		ulong n_cache = 0ull, n_r1 = 0ull; // Zugriffe im RAND-Betrieb (Bericht berichte_rho_rand)
		inline Rho_Feld() {}
		inline Rho_Feld(LBM* lbm, Memory<rhoxx>** buffers, const string& name) : c(lbm, buffers, name), lbm_(lbm) {}
		inline Rho_Feld& operator=(Memory_Container<rhoxx>&& m) noexcept { c = std::move(m); return *this; }
		void binde_rand(LBM* l); // lbm.cpp
		inline void setze_ebene(const uint achse, const uint pos, const ulong t, std::vector<float>& w) { ebene.swap(w); ebene_achse = achse; ebene_pos = pos; ebene_t = t; }
		inline float get(const ulong n) { return rand ? get_rand(n) : rho_unpack(c[n]); } // Dichte in Gitter-Einheiten
		inline void set(const ulong n, const float r) { if(rand) print_error("RHO_RAND: Rho_Feld::set ist gesperrt -- der Host packt keine Nachkollisionssumme (lbm.hpp, rho_unpack/rho_pack)."); c[n] = rho_pack(r); }
		void read_from_device(); // lbm.cpp -- RAND: liest den R1-Puffer (Memory liest seine eigene Laenge) und stempelt r1_t
		inline void write_to_device() { c.write_to_device(); }
		inline const ulong length() const { if(rand) print_error("RHO_RAND: Rho_Feld::length() ist unter RAND ohne Bedeutung (Puffer = R1)."); return c.length(); }
	};

	// ★ TODO 2 Schritt 4 (12.09.2026) -- u liegt NICHT mehr als Memory_Container offen.
	// WARUM NICHT DIESELBE BAUFORM WIE Rho_Feld (get/set, geloeschter operator[])? Weil rho SIEBEN
	// Hostzugriffsstellen hat und u HUNDERTNEUNUNDZWANZIG. Einhundertneunundzwanzig Handumbauten sind
	// nicht sicherer als einer -- sie sind hundertfuenfunddreissig Gelegenheiten, einen zu verpatzen.
	// (135 nachgezaehlt: 134 ueber den Stellvertreter plus der eine rohe Domaenenzugriff. Zwei frueher
	// hier stehende Zahlen, 129 und 131, zaehlten setup.cpp nach Vorkommen und lbm.cpp nach ZEILEN.)
	// Deshalb hier ein Stellvertreter je Komponente: "lbm.u.x[n]" bleibt an allen Stellen WOERTLICH
	// stehen und rechnet trotzdem richtig, und ein roher velxx& ist nirgends mehr erreichbar. Die
	// rho-Falle (ushort& wandelt still nach float, ohne Warnung) ist damit KONSTRUKTIV ausgeschlossen
	// statt nur weggegrept -- der Container ist privat, es gibt keinen Weg an ihm vorbei.
	//
	// DREI ZUSAGEN, an denen das haengt, und jede einzelne ist eine bekannte Stellvertreterfalle:
	//  1. GENAU EINE Wandlung, operator float(). Mit einer zweiten (etwa operator double()) wuerden
	//     fabs(), std::isnan() und jede andere ueberladene Gleitkommafunktion SOFORT mehrdeutig --
	//     heute traegt es, weil alle drei Kandidaten dieselbe Wandlung benutzen und dann die
	//     Identitaet die Gleitkomma-Promotion schlaegt.
	//  2. KEIN operator float& und kein operator const float&. Sonst bindet "const float& v = u.x[n]"
	//     an ein Temporary, das am Semikolon stirbt.
	//  3. Die Kopierzuweisung ist GELOESCHT. "u.x[a] = u.x[b]" waere sonst eine stille Neubindung des
	//     Stellvertreters, also ein Schreibvorgang, der nichts schreibt. Heute gibt es keine solche
	//     Stelle; sie ist die kanonische Falle dieser Bauform und deshalb hart gesperrt.
	// NICHT weitergereicht werden write_vtk (vtk_type() liefert "unsigned_short" und der SI-Faktor
	// wuerde auf eine Ganzzahl abgeschnitten), reset(T) und operator[]/operator() -- alle drei geben
	// rohe Speicherwoerter heraus oder schreiben sie.
	class U_Feld {
	private:
		Memory_Container<velxx> c;
		LBM* lbm_ = nullptr; bool rand = false; // ★ 04.10.2026 U_RAND: im RAND-Betrieb ist c der volle HOSTSPIEGEL, der Geraetepuffer ist kompakt
	public:
		// Der Stellvertreter setzt auf den KOMPONENTENZEIGERN auf, die Memory_Container ohnehin
		// oeffentlich fuehrt (Pointer x/y/z). reference() selbst ist dort privat, und das bleibt es --
		// der Umweg ueber die Zeiger kostet nichts und haelt die Kapselung des Containers unangetastet.
		class Komp {
		private:
			Memory_Container<velxx>::Pointer* p = nullptr;
			const U_Feld* f = nullptr; // ★ 04.10.2026 U_RAND U1b: fuer die Zugriffssperre
		public:
			inline Komp() {}
			inline Komp(Memory_Container<velxx>::Pointer* p, const U_Feld* f) : p(p), f(f) {}
			class Ref { // was aus u.x[n] wird: liest ueber u_unpack, schreibt ueber u_pack
			private:
				Memory_Container<velxx>::Pointer* p; ulong i; const U_Feld* f;
			public:
				inline Ref(Memory_Container<velxx>::Pointer* p, const ulong i, const U_Feld* f) : p(p), i(i), f(f) {}
				inline operator float() const { if(f->rand) f->pruefe_zugriff(i); return u_unpack((*p)[i]); } // Zusage 1: GENAU EINE Wandlung. ★ U_RAND: Stempelpruefung
				inline Ref& operator=(const float v) { if(f->rand) f->pruefe_schreiben(i); (*p)[i] = u_pack(v); return *this; } // ★ 05.10.2026 A-N3: Schreibsperre ohne Slot nach initialize
				inline Ref& operator=(const Ref&) = delete; // Zusage 3: keine stille Neubindung
			};
			inline Ref operator[](const ulong i) const { return Ref(p, i, f); }
		};
		Komp x, y, z;
		inline U_Feld() {}
		// ★ 12.09. (Pruefagent, NIEDRIG): nicht kopierbar. Eine Kopie truege die Komp-Zeiger auf den
		// Container der QUELLE -- ein Stellvertreter, der still in ein fremdes Feld schreibt. Heute
		// gibt es keine Kopie; gesperrt wird sie trotzdem, weil sie sonst uebersetzen wuerde.
		U_Feld(const U_Feld&) = delete;
		U_Feld& operator=(const U_Feld&) = delete;
		inline U_Feld(LBM* lbm, Memory<velxx>** buffers, const string& name) : c(lbm, buffers, name) { zeiger_setzen(); }
		inline U_Feld& operator=(Memory_Container<velxx>&& m) noexcept { c = std::move(m); zeiger_setzen(); return *this; }
		inline void read_from_device() { if(rand) read_rand(); else c.read_from_device(); }
		inline void write_to_device() { if(rand) write_rand(); else c.write_to_device(); }
		inline const ulong length() const { return c.length(); }
		void binde_rand(LBM* l); // lbm.cpp -- ★ 04.10.2026 U_RAND
		inline bool ist_rand() const { return rand; }
	private:
		void read_rand();  // lbm.cpp: kompakten Puffer lesen, in den Spiegel streuen
		void write_rand(); // lbm.cpp: Spiegel packen, kompakten Puffer schreiben
		inline void zeiger_setzen() { x = Komp(&c.x, this); y = Komp(&c.y, this); z = Komp(&c.z, this); }
	public:
		void pruefe_zugriff(const ulong n) const; // lbm.cpp -- ★ 04.10.2026 U_RAND U1b: Zugriffssperre (harter Fehler ausserhalb der Stempel)
		void pruefe_schreiben(const ulong n) const; // lbm.cpp -- ★ 05.10.2026 A-N3: Schreibsperre (Zelle ohne Slot nach initialize -> harter Fehler)
	};

	LBM_Domain** lbm_domain; // one LBM domain per GPU

	Rho_Feld rho; // density of every cell -- Zugriff ueber get/set, siehe Rho_Feld
	U_Feld u; // velocity of every cell -- Zugriff ueber u.x/u.y/u.z, siehe U_Feld
	// ★ 05.10.2026 FLAGS4 F3 (PLAN-VRAM-URAND-FLAGS-2026-10-04.md C.2): flags der Huelle. Hostzugriff unveraendert ueber den Container
	// (er zeigt auf das volle Bytefeld je Domaene), die Synchronisation laeuft ueber Flags_Puffer der Domaenen. Der Container selbst
	// darf NICHT mehr lesen/schreiben: unter FLAGS4 hat sein Puffer keinen Geraeteteil, ein Aufruf liefe still ins Leere.
	class Flags_Feld {
	private:
		Memory_Container<uchar> c;
		LBM* lbm_ = nullptr;
	public:
		inline Flags_Feld() {}
		Flags_Feld(const Flags_Feld&) = delete;
		Flags_Feld& operator=(const Flags_Feld&) = delete;
		inline Flags_Feld& operator=(Memory_Container<uchar>&& m) noexcept { c = std::move(m); return *this; }
		inline void binde(LBM* l) { lbm_ = l; }
		inline uchar& operator[](const ulong n) { return c[n]; }
		inline const ulong length() const { return c.length(); }
		void read_from_device();  // lbm.cpp: wie Memory_Container::read_from_device, aber ueber Flags_Puffer je Domaene
		void write_to_device();   // lbm.cpp: dito
	};
	Flags_Feld flags; // flags of every cell -- ★ 05.10.2026 FLAGS4: Hostzugriff wie bisher, Synchronisation ueber die Domaenen-Fassade
#ifdef FORCE_FIELD
	Memory_Container<float> F; // individual force for every cell
#endif // FORCE_FIELD
#ifdef SURFACE
	Memory_Container<float> phi; // fill level of every cell
#endif // SURFACE
#ifdef TEMPERATURE
	Memory_Container<float> T; // temperature of every cell
#endif // TEMPERATURE
#ifdef PARTICLES
	Memory<float>* particles; // particle positions
#endif // PARTICLES

	LBM(const uint Nx, const uint Ny, const uint Nz, const uint Dx, const uint Dy, const uint Dz, const float nu, const float fx=0.0f, const float fy=0.0f, const float fz=0.0f, const float sigma=0.0f, const float alpha=0.0f, const float beta=0.0f, const uint particles_N=0u, const float particles_rho=0.0f); // compiles OpenCL C code and allocates memory
	LBM(const uint Nx, const uint Ny, const uint Nz, const float nu, const float fx=0.0f, const float fy=0.0f, const float fz=0.0f, const float sigma=0.0f, const float alpha=0.0f, const float beta=0.0f, const uint particles_N=0u, const float particles_rho=1.0f); // compiles OpenCL C code and allocates memory
	LBM(const uint Nx, const uint Ny, const uint Nz, const float nu, const uint particles_N, const float particles_rho=1.0f); // compiles OpenCL C code and allocates memory
	LBM(const uint Nx, const uint Ny, const uint Nz, const float nu, const float fx, const float fy, const float fz, const uint particles_N, const float particles_rho=1.0f); // compiles OpenCL C code and allocates memory
	LBM(const uint3 N, const uint Dx, const uint Dy, const uint Dz, const float nu, const float fx=0.0f, const float fy=0.0f, const float fz=0.0f, const float sigma=0.0f, const float alpha=0.0f, const float beta=0.0f, const uint particles_N=0u, const float particles_rho=0.0f); // compiles OpenCL C code and allocates memory
	LBM(const uint3 N, const float nu, const float fx=0.0f, const float fy=0.0f, const float fz=0.0f, const float sigma=0.0f, const float alpha=0.0f, const float beta=0.0f, const uint particles_N=0u, const float particles_rho=1.0f); // compiles OpenCL C code and allocates memory
	LBM(const uint3 N, const float nu, const uint particles_N, const float particles_rho=1.0f); // compiles OpenCL C code and allocates memory
	LBM(const uint3 N, const float nu, const float fx, const float fy, const float fz, const uint particles_N, const float particles_rho=1.0f); // compiles OpenCL C code and allocates memory
	// FORK Doppel-Domaene: explizite Geraetewahl. smart_device_selection() nimmt immer das schnellste Geraet;
	// fuer zwei LBM-Instanzen auf zwei VERSCHIEDENEN GPUs (Fine auf der dGPU, Coarse auf der iGPU) braucht es diesen Weg.
	LBM(const uint3 N, const float nu, const Device_Info& device_info, const float fx=0.0f, const float fy=0.0f, const float fz=0.0f, const float sigma=0.0f, const float alpha=0.0f, const float beta=0.0f, const uint particles_N=0u, const float particles_rho=1.0f);
	~LBM();

	// ★ Zugriff auf die RHO_CLAMP-Zaehler aller Domaenen. Sitzt in LBM_Domain, gebraucht wird er in
	// der Huelle -- ohne diese Zahl ist ein Lauf kein Ergebnis (siehe berichte_dichteklemme).
	void rho_clamp_hits_total(ulong& unten, ulong& oben) {
		unten = 0ull; oben = 0ull;
#ifdef RHO_CLAMP
		for(uint d=0u; d<get_D(); d++) {
			lbm_domain[d]->rho_clamp_hits.read_from_device();
			unten += (ulong)lbm_domain[d]->rho_clamp_hits[0];
			oben  += (ulong)lbm_domain[d]->rho_clamp_hits[1];
		}
#endif // RHO_CLAMP
	}
	void run(const ulong steps=max_ulong, const ulong total_steps=max_ulong); // initializes the LBM simulation (copies data to device and runs initialize kernel), then runs LBM
	// FORK Doppel-Domaene: setzt `steps` Zeitschritte ohne Barriere in die Warteschlange und kehrt sofort zurueck.
	// Der Aufrufer MUSS finish() rufen, bevor er ein Geraetepuffer liest. Erfordert einen vorherigen run() (Initialisierung).
	void run_async(const ulong steps);
	void finish(); // FORK: Barriere ueber alle Warteschlangen dieser LBM-Instanz
	void update_fields(); // update fields (rho, u, T) manually
	void finalize_sparse_tiles(); // FORK: Block-Tiling abschliessen; nach Voxelisierung UND Randbedingungen aufrufen, no-op wenn aus
	void set_pressure_outlet_faces(const uint face_mask, const float rho_out=1.0f); // FORK: Druck-Auslass. Bits: 1=x_min 2=x_max 4=y_min 8=y_max 16=z_min 32=z_max
	void set_velocity_inlet_faces(const uint face_mask); // FORK: Geschwindigkeits-Einlass -- u vorgeschrieben, rho laeuft mit der Innenzelle mit
	void alloc_facetten(const std::vector<Facette>& F, const std::unordered_map<ulong,std::array<uchar,18>>* qmap=nullptr, const uint sgs_gdiag=0u, const uint sgs_fdwand=0u, const uint sgs_sism=0u); // C1b: Einzeldomaene, filtert klasse!=0, laedt hoch, bindet
	// FORK -- Doppel-Domaene (Kopplung grob -> fein). Reihenfolge: einmal alloc_coupling_planes() auf BEIDEN
	// Domaenen, danach je Fernfeld-Schritt extract_plane_macros() auf der groben und drive_boundary_from_coarse()
	// auf der feinen Domaene. Beide erfordern einen vorherigen run() (Kernel brauchen initialisierte Puffer).
	bool plane_fits(const PlaneSpec& plane, const char* who) const; // prueft, dass die Ebene ganz in der Domaene liegt
	void alloc_coupling_planes(const ulong max_plane_cells);
	void extract_plane_macros(const PlaneSpec& plane, std::vector<float>& host_buf, const bool u_verwerfen=false); // liest (rho,u) einer Ebene in host_buf (4 floats/Zelle); ★ 05.10.2026 C-N2: unter U_RAND nur mit u_verwerfen = true
	void rho_rek_ebene(const PlaneSpec& plane, const ulong t_rek, const uint modus, std::vector<float>& out4, std::vector<rhoxx>& worte); // ★ 15.09. RHO_RAND C1/C2a: rho einer Ebene aus den DDFs bei t_rek; modus 0 = Identitaet (t), 1 = Nachkollision (t-1, ohne MS)
	void rho_ausgabe_ebene(const PlaneSpec& plane, const ulong t_aus, const bool zaehlen, std::vector<float>& out); // ★ 15.09. RHO_RAND C2a: Ausgabe-rho (Nachkollisionssumme), t_aus = get_t()-1
	void rho_schicht_in_host(const uint z, const bool zaehlen); // ★ 15.09. RHO_RAND C2c: z-Schicht der Ausgabe in den rho-Cache (VTK)
	ulong rho_aus_gezaehlt_zellen = 0ull, rho_aus_gezaehlt_e = 0ull, rho_aus_ist_219 = 0ull, rho_aus_ist_220 = 0ull; // ★ C2c: Ist=Soll der gezaehlten Ausgabeaufrufe (Slots 219/220)
	ulong rho_aus_rr_verglichen = 0ull, rho_aus_rr_abw = 0ull; // ★ C2c: Geraete-rr_idx gegen Host-rr_idx_host an TYPE_E der gezaehlten Ebene (Soll: verglichen > 0, Abweichungen 0)
	void u_rand_ausgabe(); // ★ 04.10.2026 U_RAND U1c: nach einem Grobschritt mit Leseplan V/R1 lesen, in den Spiegel streuen, Stempel setzen
	void lese_yslice_in_host(const uint y); // ★ Slice-Ebenen-Read 2026-08-26: (rho,u,flags) EINER y-Ebene per Device-Gather in die Host-Arrays streuen (Transportweg-Optimierung, wertgleich)
	void drive_boundary_from_coarse(const PlaneSpec& fine_plane, const std::vector<float>& coarse_face, const uint coarse_a, const uint coarse_b, const uint ratio); // kubischer Lift in die TYPE_E-Randzellen
	// ★ P9c N2F-SCHALE (Heiko): near->far-Schalen-Rueckkopplung. Reihenfolge: alloc_schale() auf
	// BEIDEN Instanzen (fein: Deckungspunkt-Indizes + ratio; grob: Schalen-Indizes, ratio=1), danach
	// je Kopplungsfenster schale_extract_u() auf der feinen (mittel=1: Blockmittel) und
	// schale_upload_unear() auf der groben Instanz. Der Blend selbst laeuft in do_time_step
	// (enqueue_schale_blend, nach einlass_eq) und ist ueber s_schale_alpha nur im Fernfeld scharf.
	// Gradient-Blend: gewichte (je Listenzelle, [0;1]) und modus (0 EQ / Bit 0 FNEQ / 2 IDENT-Debug).
	void alloc_schale(const std::vector<ulong>& liste, const std::vector<float>& gewichte, const uint ratio, const uint modus, const bool blendet=true);
	void schale_extract_u(std::vector<float>& out, const uint mittel); // Kernel-Run + Read des out-Puffers (blockierend)
	void schale_upload_unear(const std::vector<float>& unear); // Host -> schale_unear (Blend-Eingang)
	void reset(); // reset simulation (takes effect in following run() call)
#ifdef FORCE_FIELD
	void update_force_field(); // calculate forces from fluid on TYPE_S cells
	float3 object_center_of_mass(const uchar flag_marker=TYPE_S); // calculate center of mass of all cells flagged with flag_marker
	float3 object_force(const uchar flag_marker=TYPE_S); // add up force for all cells flagged with flag_marker
	float3 object_force_zband(const uchar flag_marker, const uint z_lo, const uint z_hi); // FORK Kraft-Zerlegung (CFD_KRAFT_ZBAND); coordinates() ist domaenenlokal -- nur D=1
	float3 object_torque(const float3& rotation_center, const uchar flag_marker=TYPE_S); // add up torque around specified rotation_center for all cells flagged with flag_marker
#endif // FORCE_FIELD
#ifdef MOVING_BOUNDARIES
	void update_moving_boundaries(); // mark/unmark cells next to TYPE_S cells with velocity!=0 with TYPE_MS
#endif // MOVING_BOUNDARIES
#if defined(PARTICLES)&&!defined(FORCE_FIELD)
	void integrate_particles(const ulong steps=max_ulong, const ulong total_steps=max_ulong, const uint time_step_multiplicator=1u); // intgegrate passive tracer particles forward in time in stationary flow field
#endif // PARTICLES&&!FORCE_FIELD

	uint get_Nx() const { return Nx; } // get (global) lattice dimensions in x-direction
	uint get_Ny() const { return Ny; } // get (global) lattice dimensions in y-direction
	uint get_Nz() const { return Nz; } // get (global) lattice dimensions in z-direction
	ulong get_N() const { return (ulong)Nx*(ulong)Ny*(ulong)Nz; } // get (global) number of lattice points
	uint get_Dx() const { return Dx; } // get lattice domains in x-direction
	uint get_Dy() const { return Dy; } // get lattice domains in y-direction
	uint get_Dz() const { return Dz; } // get lattice domains in z-direction
	uint get_D() const { return Dx*Dy*Dz; } // get number of lattice domains
	float get_nu() const { return lbm_domain[0]->get_nu(); } // get kinematic shear viscosity
	float get_tau() const { return 3.0f*get_nu()+0.5f; } // get LBM relaxation time
	float get_Re_max() const { return 0.57735027f*sqrt((float)(sq(Nx)+sq(Ny)+sq(Nz)))/get_nu(); } // Re < Re_max = c*L_max/nu
	float get_fx() const { return lbm_domain[0]->get_fx(); } // get global froce per volume
	float get_fy() const { return lbm_domain[0]->get_fy(); } // get global froce per volume
	float get_fz() const { return lbm_domain[0]->get_fz(); } // get global froce per volume
	float get_sigma() const { return lbm_domain[0]->get_sigma(); } // get surface tension coefficient
	float get_alpha() const { return lbm_domain[0]->get_alpha(); } // get thermal diffusion coefficient
	float get_beta() const { return lbm_domain[0]->get_beta(); } // get thermal expansion coefficient
	ulong get_t() const { return lbm_domain[0]->get_t(); } // get discrete time step in LBM units
	uint get_velocity_set() const { return lbm_domain[0]->get_velocity_set(); }
	void set_fx(const float fx) { for(uint d=0u; d<get_D(); d++) lbm_domain[d]->set_fx(fx); } // set global froce per volume
	void set_fy(const float fy) { for(uint d=0u; d<get_D(); d++) lbm_domain[d]->set_fy(fy); } // set global froce per volume
	void set_fz(const float fz) { for(uint d=0u; d<get_D(); d++) lbm_domain[d]->set_fz(fz); } // set global froce per volume
	void set_f(const float fx, const float fy, const float fz) { set_fx(fx); set_fy(fy); set_fz(fz); } // set global froce per volume

	void coordinates(const ulong n, uint& x, uint& y, uint& z) const { // disassemble 1D linear index to 3D coordinates (n -> x,y,z)
		const ulong t = n%((ulong)Nx*(ulong)Ny); // n = x+(y+z*Ny)*Nx
		x = (uint)(t%(ulong)Nx);
		y = (uint)(t/(ulong)Nx);
		z = (uint)(n/((ulong)Nx*(ulong)Ny));
	}
	void coordinates(const float3& p, uint& x, uint& y, uint& z) const { // turn 3D position into closest 3D grid coordinates
		const float3 mp = mirror_position(p);
		x = (uint)(mp.x+1.5f*(float)Nx)%Nx;
		y = (uint)(mp.y+1.5f*(float)Ny)%Ny;
		z = (uint)(mp.z+1.5f*(float)Nz)%Nz;
	}
	ulong index(const uint x, const uint y, const uint z) const { // turn 3D coordinates into 1D linear index
		return (ulong)x+((ulong)y+(ulong)z*(ulong)Ny)*(ulong)Nx;
	}
	ulong index(const uint3 xyz) const { // turn 3D coordinates into 1D linear index
		return index(xyz.x, xyz.y, xyz.z);
	}
	ulong index(const float3& p) const { // turn 3D position into closest 1D linear index
		uint x=0u, y=0u, z=0u;
		coordinates(p, x, y, z);
		return index(x, y, z);
	}
	float3 position(const uint x, const uint y, const uint z) const { // returns position in box [-Nx/2, Nx/2] x [-Ny/2, Ny/2] x [-Nz/2, Nz/2]
		return float3((float)x-0.5f*(float)Nx+0.5f, (float)y-0.5f*(float)Ny+0.5f, (float)z-0.5f*(float)Nz+0.5f);
	}
	float3 position(const ulong n) const { // returns position in box [-Nx/2, Nx/2] x [-Ny/2, Ny/2] x [-Nz/2, Nz/2]
		uint x, y, z;
		coordinates(n, x, y, z);
		return position(x, y, z);
	}
	float3 mirror_position(const float3& p) const { // mirror position into periodic boundaries
		float3 r;
		r.x = sign(p.x)*(fmod(fabs(p.x)+0.5f*(float)Nx, (float)Nx)-0.5f*(float)Nx);
		r.y = sign(p.y)*(fmod(fabs(p.y)+0.5f*(float)Ny, (float)Ny)-0.5f*(float)Ny);
		r.z = sign(p.z)*(fmod(fabs(p.z)+0.5f*(float)Nz, (float)Nz)-0.5f*(float)Nz);
		return r;
	}
	float3 size() const { // returns size of box
		return float3((float)Nx, (float)Ny, (float)Nz);
	}
	float3 center() const { // returns center of box
		return float3(0.5f*(float)Nx-0.5f, 0.5f*(float)Ny-0.5f, 0.5f*(float)Nz-0.5f);
	}
	uint smallest_side_length() const {
		return min(min(Nx, Ny), Nz);
	}
	uint largest_side_length() const {
		return max(max(Nx, Ny), Nz);
	}
	float3 relative_position(const uint x, const uint y, const uint z) const { // returns relative position in box [-0.5, 0.5] x [-0.5, 0.5] x [-0.5, 0.5]
		return float3(((float)x+0.5f)/(float)Nx-0.5f, ((float)y+0.5f)/(float)Ny-0.5f, ((float)z+0.5f)/(float)Nz-0.5f);
	}
	float3 relative_position(const ulong n) const { // returns relative position in box [-0.5, 0.5] x [-0.5, 0.5] x [-0.5, 0.5]
		uint x, y, z;
		coordinates(n, x, y, z);
		return relative_position(x, y, z);
	}
	void write_status(const string& path=""); // write LBM status report to a .txt file

	void voxelize_mesh_on_device(const Mesh* mesh, const uchar flag=TYPE_S, const float3& rotation_center=float3(0.0f), const float3& linear_velocity=float3(0.0f), const float3& rotational_velocity=float3(0.0f)); // voxelize mesh
	void unvoxelize_mesh_on_device(const Mesh* mesh, const uchar flag=TYPE_S); // remove voxelized triangle mesh from LBM grid
	void write_mesh_to_vtk(const Mesh* mesh, const string& path="", const bool convert_to_si_units=true) const; // write mesh to binary .vtk file
	void voxelize_stl(const string& path, const float3& center, const float3x3& rotation, const float size=0.0f, const uchar flag=TYPE_S); // read and voxelize binary .stl file
	void voxelize_stl(const string& path, const float3x3& rotation, const float size=0.0f, const uchar flag=TYPE_S); // read and voxelize binary .stl file (place in box center)
	void voxelize_stl(const string& path, const float3& center, const float size=0.0f, const uchar flag=TYPE_S); // read and voxelize binary .stl file (no rotation)
	void voxelize_stl(const string& path, const float size=0.0f, const uchar flag=TYPE_S); // read and voxelize binary .stl file (place in box center, no rotation)

#ifdef GRAPHICS
	class Graphics {
	private:
		LBM* lbm = nullptr;
		std::atomic_int running_encoders = 0;
		uint last_exported_frame = 0u; // for next_frame(...) function
		int last_visualization_modes=0, last_field_mode=0, last_slice_mode=0, last_slice_x=0, last_slice_y=0, last_slice_z=0; // don't render a new frame if the scene hasn't changed since last frame
		void default_settings() {
			visualization_modes |= VIS_FLAG_LATTICE;
#ifdef PARTICLES
			visualization_modes |= VIS_PARTICLES;
#endif // PARTICLES
		}

	public:
		int visualization_modes=0, field_mode=0, slice_mode=0, slice_x=0, slice_y=0, slice_z=0; // field_mode = { 0 (u), 1 (rho), 2 (T) }, slice_mode = { 0 (no slice), 1 (x), 2 (y), 3 (z), 4 (xz), 5 (xyz), 6 (yz), 7 (xy) }, slice_{xyz} = position of slices

		Graphics() {} // default constructor
		Graphics(LBM* lbm) {
			this->lbm = lbm;
			camera.set_zoom(0.5f*(float)fmax(fmax(lbm->get_Nx(), lbm->get_Ny()), lbm->get_Nz()));
			slice_x = (int)lbm->get_Nx()/2;
			slice_y = (int)lbm->get_Ny()/2;
			slice_z = (int)lbm->get_Nz()/2;
			default_settings();
		}
		~Graphics() { // destructor must wait for all encoder threads to finish
			int last_value = running_encoders.load();
			while(last_value>0) {
				const int current_value = running_encoders.load();
				if(last_value!=current_value) {
					print_info("Finishing encoder threads: "+to_string(current_value));
					last_value = current_value;
				}
				sleep(0.016);
			}
		}
		Graphics& operator=(const Graphics& graphics) { // copy assignment
			lbm = graphics.lbm;
			visualization_modes = graphics.visualization_modes;
			field_mode = graphics.field_mode;
			slice_mode = graphics.slice_mode;
			slice_x = graphics.slice_x;
			slice_y = graphics.slice_y;
			slice_z = graphics.slice_z;
			return *this;
		}

		int* draw_frame(); // main rendering function, calls rendering kernels

		void set_camera_centered(const float rx=0.0f, const float ry=0.0f, const float fov=100.0f, const float zoom=1.0f); // set camera centered
		void set_camera_free(const float3& p=float3(0.0f), const float rx=0.0f, const float ry=0.0f, const float fov=100.0f); // set camera free
		bool next_frame(const ulong total_time_steps, const float video_length_seconds); // returns true once simulation time has progressed enough to render the next video frame for a 60fps video of specified length
		void print_frame(); // preview preview of current frame in console
		void write_frame(const string& path="", const string& name="image", const string& extension=".png", bool print_preview=false); // save current frame
		void write_frame(const uint x1, const uint y1, const uint x2, const uint y2, const string& path="", const string& name="image", const string& extension=".png", bool print_preview=false); // save current frame cropped with two corner points (x1,y1) and (x2,y2)
		void write_frame_png(const string& path="", bool print_preview=false); // save current frame as .png file (smallest file size, but slow)
		void write_frame_qoi(const string& path="", bool print_preview=false); // save current frame as .qoi file (small file size, fast)
		void write_frame_bmp(const string& path="", bool print_preview=false); // save current frame as .bmp file (large file size, fast)
		void write_frame_png(const uint x1, const uint y1, const uint x2, const uint y2, const string& path="", bool print_preview=false); // save current frame as .png file (smallest file size, but slow)
		void write_frame_qoi(const uint x1, const uint y1, const uint x2, const uint y2, const string& path="", bool print_preview=false); // save current frame as .qoi file (small file size, fast)
		void write_frame_bmp(const uint x1, const uint y1, const uint x2, const uint y2, const string& path="", bool print_preview=false); // save current frame as .bmp file (large file size, fast)
	}; // Graphics
	Graphics graphics;
#endif // GRAPHICS
}; // LBM

// ★ 03.10.2026 EINZIGE QUELLE der Bytes je Zelle und Schritt fuer die Bandbreitenanzeige (Laufzeile info.cpp, Grafiklabel
// main.cpp, [GPU]-Zeilen setup.cpp). Upstream-KONVENTION (bandwidth_bytes_per_cell_device), keine Messung; unter RHO_RAND
// faellt der rho-Anteil weg (Entscheidung Heiko 7, 15.09.). Wertgleich zum bisherigen Ausdruck in info.cpp/main.cpp.
double bpz_konv(const LBM& lbm); // ★ 05.10.2026 double: FLAGS4 zaehlt flags mit 0,5 B (Pruefbefund FLAGS4 N1). ★ 05.10.2026 C-N11: U_RAND/U_SPARSAM (u-Schreibbytes je Zelle) gehen NICHT ein -- gbps_konv ist eine KONVENTION (Upstream-Bytezahl je Zelle), keine Messgroesse; nur Anzeige
