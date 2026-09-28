#include "lbm.hpp"
#include <atomic> // C2: Zaehler fuer CFD_DUMP_CL-Dateinamen

// ★ F-Null-Read-Gate (Perf-Audit Achse 1, Rang 3): EIN Praedikat fuer Emission, Ansage und
// F-Waechter -- nie dreimal getrennt auswerten (Drift-Schutz). Default AN.
static bool f_nur_solid_an() { const char* e = getenv("CFD_F_NUR_SOLID"); return e==nullptr||e[0]=='\0'||atoi(e)>0; } // leer gesetzt = Default AN (Pruefagent NIEDRIG-2)

Units units; // for unit conversion

#if defined(D2Q9)
const uint velocity_set = 9u;
const uint dimensions = 2u;
const uint transfers = 3u;
#elif defined(D3Q15)
const uint velocity_set = 15u;
const uint dimensions = 3u;
const uint transfers = 5u;
#elif defined(D3Q19)
const uint velocity_set = 19u;
const uint dimensions = 3u;
const uint transfers = 5u;
#elif defined(D3Q27)
const uint velocity_set = 27u;
const uint dimensions = 3u;
const uint transfers = 9u;
#endif // D3Q27

uint bytes_per_cell_host() { // returns the number of Bytes per cell allocated in host memory
	uint bytes_per_cell = 1u+3u*(uint)sizeof(velxx)+(uint)sizeof(rhoxx); // flags, u, rho (★ TODO 2 Schritt 4: rho und u je 4 oder 2 Byte)
#ifdef FORCE_FIELD
	bytes_per_cell += 12u; // F
#endif // FORCE_FIELD
#ifdef SURFACE
	bytes_per_cell += 4u; // phi
#endif // SURFACE
#ifdef TEMPERATURE
	bytes_per_cell += 4u; // T
#endif // TEMPERATURE
	return bytes_per_cell;
}
uint bytes_per_cell_device() { // returns the number of Bytes per cell allocated in device memory
	uint bytes_per_cell = velocity_set*sizeof(fpxx)+1u+3u*(uint)sizeof(velxx)+(uint)sizeof(rhoxx); // fi, flags, u, rho (★ TODO 2 Schritt 4: rho und u je 4 oder 2 Byte)
#ifdef FORCE_FIELD
	bytes_per_cell += 12u; // F
#endif // FORCE_FIELD
#ifdef SURFACE
	bytes_per_cell += 12u; // phi, mass, flags
#endif // SURFACE
#ifdef TEMPERATURE
	bytes_per_cell += 7u*sizeof(fpxx)+4u; // gi, T
#endif // TEMPERATURE
	return bytes_per_cell;
}
uint bandwidth_bytes_per_cell_device() { // returns the bandwidth in Bytes per cell per time step from/to device memory
	uint bandwidth_bytes_per_cell = velocity_set*2u*sizeof(fpxx)+1u; // lattice.set()*2*fi, flags
#ifdef UPDATE_FIELDS
	bandwidth_bytes_per_cell += 3u*(uint)sizeof(velxx)+(uint)sizeof(rhoxx); // u, rho (★ TODO 2 Schritt 4)
#ifdef TEMPERATURE
	bandwidth_bytes_per_cell += 4u; // T
#endif // TEMPERATURE
#endif // UPDATE_FIELDS
#ifdef FORCE_FIELD
	bandwidth_bytes_per_cell += 12u; // F
#endif // FORCE_FIELD
#if defined(MOVING_BOUNDARIES)||defined(SURFACE)||defined(TEMPERATURE)
	bandwidth_bytes_per_cell += (velocity_set-1u)*1u; // neighbor flags have to be loaded
#endif // MOVING_BOUNDARIES, SURFACE or TEMPERATURE
#ifdef SURFACE
	bandwidth_bytes_per_cell += (1u+(2u*velocity_set-1u)*sizeof(fpxx)+8u+(velocity_set-1u)*4u) + 1u + 1u + (4u+velocity_set+4u+4u+4u); // surface_0 (flags, fi, mass, massex), surface_1 (flags), surface_2 (flags), surface_3 (rho, flags, mass, massex, phi)
#endif // SURFACE
#ifdef TEMPERATURE
	bandwidth_bytes_per_cell += 7u*2u*sizeof(fpxx); // 2*gi
#endif // TEMPERATURE
	return bandwidth_bytes_per_cell;
}
uint3 resolution(const float3 box_aspect_ratio, const uint memory) { // input: simulation box aspect ratio and VRAM occupation in MB, output: grid resolution
#ifndef D2Q9
	float memory_required = (box_aspect_ratio.x*box_aspect_ratio.y*box_aspect_ratio.z)*(float)bytes_per_cell_device()/1048576.0f; // in MB
	float scaling = cbrt((float)memory/memory_required);
	return uint3(to_uint(scaling*box_aspect_ratio.x), to_uint(scaling*box_aspect_ratio.y), to_uint(scaling*box_aspect_ratio.z));
#else // D2Q9
	float memory_required = (box_aspect_ratio.x*box_aspect_ratio.y)*(float)bytes_per_cell_device()/1048576.0f; // in MB
	float scaling = sqrt((float)memory/memory_required);
	return uint3(to_uint(scaling*box_aspect_ratio.x), to_uint(scaling*box_aspect_ratio.y), 1u);
#endif // D2Q9
}

string default_filename(const string& path, const string& name, const string& extension, const ulong t) { // generate a default filename with timestamp
	string time = "00000000"+to_string(t);
	time = substring(time, length(time)-9u, 9u);
	return (path=="" ? get_exe_path()+"export/" : path)+create_file_extension((name=="" ? "file" : name)+"-"+time, extension);
}
string default_filename(const string& name, const string& extension, const ulong t) { // generate a default filename with timestamp at exe_path/export/
	return default_filename("", name, extension, t);
}



LBM_Domain::LBM_Domain(const Device_Info& device_info, const uint Nx, const uint Ny, const uint Nz, const uint Dx, const uint Dy, const uint Dz, const int Ox, const int Oy, const int Oz, const float nu, const float fx, const float fy, const float fz, const float sigma, const float alpha, const float beta, const uint particles_N, const float particles_rho) { // constructor with manual device selection and domain offset
	this->Nx = Nx; this->Ny = Ny; this->Nz = Nz;
	this->Dx = Dx; this->Dy = Dy; this->Dz = Dz;
	this->Ox = Ox; this->Oy = Oy; this->Oz = Oz;
	this->nu = nu;
	this->fx = fx; this->fy = fy; this->fz = fz;
	this->sigma = sigma;
	this->alpha = alpha; this->beta = beta;
	this->particles_N = particles_N;
	this->particles_rho = particles_rho;
	// FORK -- F-Bounding-Box HIER aufloesen, nicht erst in allocate(). Der Konstruktor baut den
	// OpenCL-Code (und damit def_FBNX/def_FBN) weiter unten; allocate() laeuft erst DANACH. Stand die
	// Aufloesung in allocate(), wurde def_FBNX = 0 emittiert, f_bbox lieferte ueberall false und
	// SAEMTLICHE Kraefte kamen als exakt null heraus -- auch im Voll-Domaenen-Fall, wo die Box die
	// Identitaet sein muesste. Genau so ist es passiert und so wurde es gefunden.
	if(s_fbbox[3]>0u && s_fbbox[4]>0u && s_fbbox[5]>0u) {
		fbx0=s_fbbox[0]; fby0=s_fbbox[1]; fbz0=s_fbbox[2]; fbnx=s_fbbox[3]; fbny=s_fbbox[4]; fbnz=s_fbbox[5];
	} else { fbx0=0u; fby0=0u; fbz0=0u; fbnx=Nx; fbny=Ny; fbnz=Nz; }
	// ★ TODO 2: die Schreibmasken-Box. Ohne gesetzte Statik ist sie die F-BBox -- damit ist das
	// Nahfeld ohne Zutun richtig versorgt und die Fassung vor dieser Aenderung bleibt erhalten.
	if(s_smbox[3]>0u && s_smbox[4]>0u && s_smbox[5]>0u) {
		smx0=s_smbox[0]; smy0=s_smbox[1]; smz0=s_smbox[2]; smnx=s_smbox[3]; smny=s_smbox[4]; smnz=s_smbox[5];
	} else { smx0=fbx0; smy0=fby0; smz0=fbz0; smnx=fbnx; smny=fbny; smnz=fbnz; }
	// ★ 03.09.2026, TEUER GELERNT: BEIDE folgenden Schalter entscheiden JIT-DEFINES (F_LISTE,
	// FAC_IDX_VOLL) und muessen deshalb hier stehen, nicht in allocate(). Der OpenCL-Quelltext wird
	// bei der ERSTEN Kernel-Erzeugung uebersetzt -- das ist kernel_stream_collide in allocate(),
	// Zeilen VOR dem alten FORCE_FIELD-Block. Dort gesetzt kamen die Defines NIE im Kernel an: der
	// Host baute die kompakte F-Liste, der Kernel rechnete weiter mit def_FBN als Stride und las weit
	// ausserhalb des Puffers -- auf der CPU harmlose Nullen, auf der iGPU nichtdeterministischer
	// Muell mit halbierter Kraft (Befund B80). Dieselbe Falle, die der Kommentar zur F-Bounding-Box
	// vier Zeilen weiter oben bereits beschreibt.
	f_liste_on = s_f_liste>0u;
	band_lagen = s_sgs_band; band_on = s_sgs_band>0u; // ★ 08.09. Konstruktionszustand einfrieren (read-once-Doktrin)
	band_pi_on = band_on&&s_sgs_band_pi>0u&&s_sgs_sism>0u; // ★ 22.09. Plan C: Pi-Modus je Instanz einfrieren; Durchgang 2 N-1: dieselbe Bedingung wie die JIT-Emission (SISM noetig), sonst meldet der Kohaerenzwaechter eine Fehlkonfiguration als H1-Klasse
	nut_skal = s_sgs_nut_skal; // ★ 10.09. dito fuer den Diskriminator-Messarm
	fac_idx_voll_on = s_fac_idx_voll>0u;
	fac_r1q_on = s_facetten&&s_fac_imem&&s_fac_r1q>0u; // ★ 28.09. R1Q: eingefroren wie fac_rek_on
	fac_rek_on = s_facetten&&s_fac_imem&&s_fac_rek>0u; // ★ 22.09. S0: Konstruktionszustand einfrieren, GENAU wie die Geschwister -- die Statik wird in setup.cpp neben s_fac_pinv gesetzt, also VOR dem Konstruktor (Pruefbefund H1 vom 22.09.: eine hier per env_u gefuellte Statik ist zum Einfrieren noch 0)
	fac_pinv_on = s_fac_pinv>0u; // ★ 04.09.: Rang-1-Pseudoinverse statt Skalarleiter (JIT-Define, muss vor der ersten Kernel-Erzeugung stehen)
	for(uint i=0u; i<6u; i++) s_fbbox[i]=0u; // read-once: eine zweite Domaene erbt die Box nicht
	// ★ 2026-08-08, beim Bau der Doppel-Domaene gefunden: das Block-Tiling wurde bis hier ueberall direkt
	// aus den STATISCHEN Schaltern gelesen -- auch in finalize_sparse_tiles(), das erst LANGE nach dem
	// Konstruktor laeuft. Mit zwei Domaenen ist das eine Falle: wer den Schalter zwischen den beiden
	// Konstruktoren umlegt, aendert rueckwirkend das Verhalten der ERSTEN Domaene, deren fi dann als
	// 1-Zellen-Platzhalter stehen bliebe. Deshalb jetzt genauso read-once wie die F-Bounding-Box:
	// hier einmal in Domaenen-Felder uebernehmen, danach lesen ausschliesslich diese.
#ifndef D3Q19
	// ★ Nachpruefer-Befund 2026-08-15: der "D3Q19-Guard" fuer CFD_SGS_WANDFREI war nur ein KOMMENTAR.
	// Der Schalter ist Laufzeit, kein #error kann ihn fangen -- deshalb hart zur Laufzeit: die
	// Wanderkennung benutzt die festen Flaechennachbarn j[1..6], die nur in D3Q19 stimmen.
	if(s_sgs_wandfrei) print_error("CFD_SGS_WANDFREI ist nur fuer D3Q19 gebaut (feste Flaechennachbarn j[1..6]).");
	if(s_wandfunktion) print_error("CFD_WANDFUNKTION ist nur fuer D3Q19 gebaut (feste Diagonalpaare 9/16, 11/18, 15/10, 17/12).");
	if(s_facetten) print_error("CFD_FACETTEN ist nur fuer D3Q19 gebaut (Paartabelle FACETTEN-STUFE2.md).");
#endif // D3Q19
#ifndef FORCE_FIELD
	if(s_facetten) print_error("CFD_FACETTEN braucht FORCE_FIELD (f_bbox-Indexfeld).");
#endif // FORCE_FIELD
	// ★ Pruefbefund F1 (2026-08-25): diese Guards standen erst im #ifndef-D3Q19-, dann im #ifndef-FORCE_FIELD-Block und waren im
	// Produktionsbuild FUNKTIONAL TOT. Jetzt unbedingt. alpha-Sperre entfaellt (Revision W2:
	// die Blende ist rein geometrisch, Additivterm+alpha bleiben und ihre Mathematik gilt exakt).
	if(s_fac_elibb&&(s_fac_ema>0.0f||s_fac_pema>0.0f)) print_error("CFD_FAC_ELIBB mit EMA/PEMA ist nicht definiert (Filter mischen Blende und Additivpfad) -- Messarm rein halten.");
#if defined(FORCE_FIELD)&&!defined(PARTICLES)
	// ★ Ansage-Doktrin F-Null-Read-Gate (Auditor-B B-3-Muster: Default-AN-Verhalten muss im Log stehen).
	if(f_nur_solid_an()) print_info("F-NUR-SOLID aktiv (Default): stream_collide liest F nicht -- F ist an Nicht-Solid-Zellen konstant 0 (F-Waechter prueft das bei initialize()). CFD_F_NUR_SOLID=0 stellt den Upstream-Read her.");
	else print_warning("CFD_F_NUR_SOLID=0: stream_collide liest F an jeder Fluidzelle (Upstream-Pfad, 12 B/Zelle/Schritt in der F-BBox) -- nur fuer A/B-Kontrollarme gedacht.");
#endif
	// ★ 16.09.2026 (PLAN-APG-2026-09-16.md §C, Entscheid E2): die ELIBB x APG-Sperre war eine reine Policy-Sperre aus der Messarm-Zeit
	// (25.08.), keine Wechselwirkung im Code -- die Blende aendert fhn VOR der Abtastung, APG nur das tw-Ziel. ELIBB=1 steht in jeder
	// Produktionszeile, ohne Aufheben gaebe es keinen APG-A/B. ELIBB=2 (PUR) steigt vor dem Wandmodell aus -> APG waere ein stiller No-Op.
	// ★ 16.09. HOCH-1 (Pruefagent): der Haken wird NUR gelesen, wenn diese Instanz APG traegt -- das Fernfeld (fahrzeug_dd) wird nach der
	// Statik-Nullung gebaut und darf weder den Haken tragen noch an "HAKEN ohne APG" sterben (die Ansage-Pruefung steht in setup.cpp neben CFD_FAC_APG).
	s_fac_apg_haken = (s_fac_apg!=0.0f) ? env_u("CFD_FAC_APG_HAKEN", 0u) : 0u;
	if(s_fac_apg_haken>3u) print_error("CFD_FAC_APG_HAKEN kennt nur 0 (aus), 1 (grad-rho-Statistik), 2 (Konstantgradient) und 3 (analytischer Lineargradient).");
	// ★★ 22.09.2026 MOZAFFARI (CFD_FAC_APG_MOZ). Der Zweig haengt an FACETTEN_APG -- ohne APG wird er
	// gar nicht emittiert, ein gesetztes MOZ waere dann ein lautloser No-Op. Und kappa hat unter MOZ
	// KEINE Physikwirkung mehr (nur noch Zaehlerbezug [313..316]/[326]/[327]), deshalb ist jedes kappa != 1
	// eine stille Falschangabe. BEIDE Ansage-Waechter stehen NICHT hier, sondern in setup.cpp je Fall
	// NACH dem Setzen von s_fac_apg (apg_moz_ansage): dieser Konstruktor laeuft auch fuer das APG-freie
	// Fernfeld von fahrzeug_dd durch (Statik genullt) -- dieselbe Falle wie CFD_TIMER_APG/CFD_FAC_APG_HAKEN.
	// Die erste Fassung BEHAUPTETE die Waechter hier und hatte sie nicht (Pruefbefund A-4/B-H1).
	// Hier stehen nur die Pruefungen, die AUSSCHLIESSLICH die Umgebung lesen.
	s_fac_apg_moz = env_u("CFD_FAC_APG_MOZ", 0u);
	if(s_fac_apg_moz>1u) print_error("CFD_FAC_APG_MOZ kennt nur 0 (lineare Form, bitgleich) und 1 (Mozaffari).");
	s_fac_apg_c   = env_f("CFD_FAC_APG_C",   0.4f);
	s_fac_apg_ap0 = env_f("CFD_FAC_APG_AP0", 0.005f);
	if(s_fac_apg_moz==0u&&(getenv("CFD_FAC_APG_C")!=nullptr||getenv("CFD_FAC_APG_AP0")!=nullptr))
		print_error("CFD_FAC_APG_C / CFD_FAC_APG_AP0 ohne CFD_FAC_APG_MOZ=1 -- wirkungslos (Ansage-Doktrin).");
	if(s_fac_apg_moz>0u&&!(s_fac_apg_c>0.0f))
		print_error("CFD_FAC_APG_MOZ=1 mit CFD_FAC_APG_C <= 0: f waere identisch 1, der Tausch ein No-Op.");
	if(s_fac_apg_moz>0u&&s_fac_apg_c>1.0f)
		print_error("CFD_FAC_APG_C > 1: f wuerde negativ, tw = tw_Spalding*f^2 verliert das Vorzeichen und STEIGT ab f = 0 wieder -- die Abbildung waere nicht mehr monoton, das f-Histogramm [318..325] nicht mehr lesbar (Pruefbefund A-7/B-M3). Zulaessig ist 0 < C <= 1, Paperwert 0,4.");
	if(s_fac_apg_moz>0u&&s_fac_apg_c<5e-7f)
		print_error("CFD_FAC_APG_C unter der 6-Stellen-Emissionsquantisierung (to_string 6u) -- wuerde still zu 0.000000f, f identisch 1, No-Op-Arm am Waechter vorbei (Muster s_fac_apg/s_sgs_nut_skal; Pruefbefund A-6).");
	if(s_fac_apg_moz>0u&&!(s_fac_apg_ap0>0.0f))
		print_error("CFD_FAC_APG_AP0 muss > 0 sein -- bei 0 ist f = 1 - C KONSTANT fuer jedes alpha_p > 0 (kein Druckgradientenbezug mehr, pauschales tw*(1-C)^2).");
	if(s_fac_apg_moz>0u&&s_fac_apg_ap0<5e-7f)
		print_error("CFD_FAC_APG_AP0 unter der 6-Stellen-Emissionsquantisierung -- wuerde still zu 0.000000f und damit f = 1 - C konstant (Pruefbefund A-6).");
	// ★★ 22.09.2026 CFD_TIMER_APG (REKONSTRUKTION-PLAN.md §10 Hebel 1 = Plan-Schritt 2).
	// PREIS, ausdruecklich angesagt -- dieselbe Lehre, die CFD_TIMER_FERN am 21.09. einen ganzen Lauf
	// gekostet hat (+101 % Wanduhr, weil der Diagnoseschalter in einer PRODUKTIONSZEILE stand):
	// der Timer setzt zwei finish_queue() um den Vorkernel und serialisiert damit die Pipeline.
	// Die Wanduhr, die MLUPs und der Durchsatz dieses Arms sind KEIN Leistungsmass.
	s_timer_apg = env_u("CFD_TIMER_APG", 0u);
	if(s_timer_apg>1u) print_error("CFD_TIMER_APG kennt nur 0 (aus) und 1 (Vorkernel fac_apg_ab isoliert messen).");
	// ★★ 22.09.2026, EIGENER FEHLER, vom Pruefagenten gefunden: hier stand
	//   if(s_timer_apg>0u&&s_fac_apg==0.0f) print_error(...)
	// Das toetet fahrzeug_dd. Reihenfolge: setup.cpp setzt s_fac_apg, baut lbm_f (Konstruktor laeuft, alles gut),
	// NULLT DANN die Statik fuer die Symmetrie und baut lbm_c -- der Fernfeld-Konstruktor laeuft ERNEUT durch
	// diese Zeile, sieht s_fac_apg==0 und ruft exit(1), bevor ein einziger Zeitschritt lief. Genau der Fehler,
	// den derselbe Baum am 16.09. fuer CFD_FAC_APG_HAKEN schon einmal behoben hat (setup.cpp:7109 sagt es
	// woertlich: "Pruefung seit 16.09. hier statt im Domaenen-Konstruktor, den auch das APG-freie Fernfeld
	// durchlaeuft"). Die Ansage-Pruefung steht jetzt ebenfalls dort, je Fall, NACH dem Setzen von s_fac_apg.
	// Und das Vorbild CFD_TIMER_FERN liegt aus demselben Grund als LOKALE Variable in setup.cpp, nicht als Statik.
	if(s_timer_apg>0u) print_warning("CFD_TIMER_APG=1: DIAGNOSEARM. Zwei finish_queue() um fac_apg_ab serialisieren die Pipeline -- Wanduhr und Durchsatz dieses Laufs sind NICHT mit anderen Armen vergleichbar. Gehoert NIE in eine Produktionszeile (Lehre vom 21.09.2026, CFD_TIMER_FERN, +101 Prozent Wanduhr).");
	if(s_fac_messnur>0u&&s_fac_apg!=0.0f) print_error("CFD_FAC_MESSNUR + CFD_FAC_APG: im Mess-Nur-Modus greift kein tw-Ziel in die Physik, APG waere ein stiller No-Op (Ansage-Doktrin).");
	if(s_fac_apg!=0.0f&&s_fac_nachbar==0u) print_error("CFD_FAC_APG braucht CFD_FAC_NACHBAR=1: dp/ds kommt seit 16.09. aus dem Vorkernel fac_apg_ab (nach fac_nachbar_ab; grad rho in fac_nb[2..4]) und die Korrektur gilt an der Abtasthoehe y_ab.");
	if(s_fac_elibb_pur&&s_fac_apg!=0.0f) print_error("CFD_FAC_ELIBB=2 (PUR) mit APG: der Pur-Arm steigt vor dem Wandmodell aus, APG waere ein stiller No-Op.");
	if(s_fac_elibb&&s_fac_apg!=0.0f) print_info("ELIBB=1 + APG (seit 16.09. zugelassen): die Blende wirkt auf fhn VOR der Abtastung, APG nur auf das tw-Ziel; grad rho kommt aus den DDFs der Achsnachbarn (Vorkernel fac_apg_ab), nicht aus dem geblendeten Eigenwert -- keine Wechselwirkung im Code.");
	if(s_fac_apg_haken==2u) print_warning("CFD_FAC_APG_HAKEN=2: TESTARM -- der Vorkernel schreibt grad rho = (1e-3, 0, 0) an jeder Facette; die Physik dieses Laufs ist entwertet.");
	if(s_fac_apg_haken==3u) print_warning("CFD_FAC_APG_HAKEN=3: TESTARM -- der Vorkernel ersetzt rho durch das analytische Feld 1 + x/1024 und schreibt gz := Nachbarzahl kx; der Host prueft gx == 2^-10 exakt (kx>0) bzw. 0. Die Physik dieses Laufs ist entwertet.");
	if(s_fac_utkorr!=1.0f) print_info("ABTASTPUNKT-MESSARM aktiv: CFD_FAC_UTKORR = "+to_string(s_fac_utkorr,3u)+" auf dem Wandmodell-Eingang (Theorie-Soll 3/2; Ansage-Doktrin).");
	if(s_fac_kappa!=0.4f) print_info("Grazing-Guard geaendert: CFD_FAC_KAPPA = "+to_string(s_fac_kappa,2u)+" (Default 0,4).");
	if(s_fac_qdiag!=0u) print_warning("CFD_FAC_QDIAG = "+to_string((ulong)s_fac_qdiag)+" -- DIAGNOSEARM (2 = nur q<0,5, 3 = nur q>0,5; Arm 1 ist seit K1' ohne Funktion). Kein Messarm fuer Abnahmen.");
	if(s_fac_elibb&&s_fac_imem) print_info("ELIBB 18-Link AKTIV (B2/B3, q>0,5 seit 26.08. als MLS-Blende chi=(2q-1)/(tau0+0,5)): rein geometrische q-Blende (u_W=0) + bestehender Additivterm; q=0,5 ist bitgleich iMEM. Wirkpfad Slot 67 (Blende gesamt), Slot 68 (MLS-q>0,5-Zweig). Blenden-Austausch wird seit B3 in fac_tau[1..5] GEBUCHT; seit dem Buchungsschluss (27.08.) buchen auch die Gate-Rueckfaelle (Slot 69, P-only; offen bleiben nur Slot 9 ut~0 und PEMA-Slot 17 mit reiner Kopfbuchung) -- Reibungspfad und object_force sind damit an allen GATE-Rueckfaellen EIN Bild.");
	// C1b: WFB und Facetten am selben Einfuegepunkt schliessen sich aus -- hart, kein stilles Nacheinander.
	if(s_facetten&&s_wandfunktion) print_error("CFD_FACETTEN und CFD_WANDFUNKTION gleichzeitig ist nicht definiert -- genau einen Pfad waehlen.");
	facetten_on = s_facetten;
#ifndef SUBGRID
	// ★ Audit-Nacharbeit 2: mit abgeschaltetem SUBGRID (Kugel-Validierung!) war der Schalter ein
	// lautloser No-Op -- jetzt harte Abweisung. Die WANDFUNKTION dagegen ist von SUBGRID unabhaengig
	// und wird seit derselben Nacharbeit ausserhalb des SUBGRID-Blocks emittiert.
	if(s_sgs_wandfrei) print_error("CFD_SGS_WANDFREI ohne SUBGRID ist sinnlos (es gaebe kein nu_t zu entfernen).");
	if(s_sgs_diag) print_error("CFD_SGS_DIAG ohne SUBGRID ist sinnlos (es gaebe kein nu_t zu messen).");
	if(s_sgs_fdwand>0u) print_error("CFD_SGS_FDWAND ohne SUBGRID ist sinnlos (der w-Ersatz-Hook liegt im SUBGRID-Block und waere still tot).");
	if(s_sgs_sism>0u) print_error("CFD_SGS_SISM ohne SUBGRID ist sinnlos (der FD-w-Hook liegt im SUBGRID-Block und waere still tot).");
	if(!s_sgs_guo) print_warning("CFD_SGS_GUO=0: Pi^neq OHNE Guo-Korrektur -- die Scherrate ist dort verzerrt, wo die Volumenkraft wirkt (Kontrollarm, nicht die Physik).");
#endif // SUBGRID
	// R2-Nachpruefer: Ansage NACH den harten Abweisern (vorher stand "aktiv" eine Zeile vor dem exit)
	if(s_sgs_wandfrei) print_info("SGS_WANDFREI aktiv: kein nu_t in Zellen mit solidem Flaechennachbarn (Wirkpfad-Zaehler Slot 6, Report am Laufende).");
	// ★ 03.09. (Planungsagent-Befund + Pruefagent-Vorschlag 6): der SGS_DIAG-Block lag im else-Zweig des FDWAND-Lesers und war unter
	// FDWAND an allen Facettenzellen ein stiller No-Op (Bins 35-48 leer). Seit 03.09. nachmittags steht er HINTER dem if/else und
	// misst das finale w -- die Kombination ist jetzt das Instrument fuer "nu_t IST gegen kappa*y+ SOLL" an Wandzellen.
	if(s_sgs_diag&&s_sgs_fdwand>0u) print_info("CFD_SGS_DIAG x CFD_SGS_FDWAND: DIAG misst an Facettenzellen das FD-nu_t aus fac_wfd (Block hinter dem if/else seit 03.09.; vorher stiller No-Op der Bins 35-48).");
	// ★ 07.09.2026 SISM-Waechter (Planungsagent-Vorpruefung): der Sbar-Abzug lebt im FD-Kernel -- ohne FDWAND gibt es den
	// Kernel nicht, ohne Facetten keine Zellenliste; T = 0 waere alpha = inf; ab < 3T laesst den Anfahrtransienten in <S>
	// stehen und die Klemme greift dann flaechig (nu_t = 0 = WANDFREI-Zustand, am 07.09. nach 175 Schritten divergiert).
	if(s_sgs_vandriest>0u&&s_sgs_fdwand==0u) print_error("CFD_SGS_VANDRIEST braucht CFD_SGS_FDWAND=1 -- die Daempfung haengt am FDWAND-Leser (fdw_fid); ohne ihn wird der Block nie erreicht (stiller No-Op).");
	if(s_sgs_vandriest>0u&&s_sgs_sism>0u) print_error("CFD_SGS_VANDRIEST und CFD_SGS_SISM gleichzeitig: ZWEI nu_t-Senker an derselben Zelle, also zwei Variablen in einem Lauf (Iron Rule 1). Genau einen waehlen.");
	if(s_sgs_vandriest>0u&&s_sgs_vd_ab==0ull) print_warning("CFD_SGS_VD_AB = 0: van Driest daempft ab dem ERSTEN Schritt mit dem noch nicht eingeschwungenen tw-Laufmittel (y+ zu klein, D^2 zu klein, Daempfung zu stark) -- genau die Anlaufphase, in der WANDFREI divergierte. Sperre in Schritten setzen -- Empfehlung: die Schrittzahl der Warmlaufphase (CFD_T_WARMUP in Schritten, im Fallbericht ausgewiesen), nicht darunter; kein fester Wert, die Schrittzahl haengt an dx.");
	if(s_sgs_vandriest>0u&&s_sgs_wandfrei) print_error("VANDRIEST x WANDFREI: das WANDFREI-Gate (!sgs_wand) ueberspringt den ganzen FDWAND-Zweig, Slot 168 bliebe 0 -- ein stiller No-Op, der erst nach dem vollen Lauf im Bericht auffiele (Pruefagent 08.09., Befund 4). Genau eines waehlen.");
	if(s_sgs_vandriest==1u) print_info("VAN DRIEST als MESSARM (CFD_SGS_VANDRIEST=1, A+ = "+to_string(s_sgs_vd_aplus,1u)+"): D^2 wird je Facettenzelle gebildet und in die Slots 160..167 gebinnt, w bleibt UNANGETASTET. Der Lauf MUSS bitgleich zum Kontrollarm sein -- damit ist in EINEM Lauf belegt, dass das geraetseitige y+ das Host-yplus_facetten.csv trifft UND dass die Physik unberuehrt bleibt.");
	if(s_sgs_vandriest==2u) print_info("VAN DRIEST ANGEWANDT (CFD_SGS_VANDRIEST=2, A+ = "+to_string(s_sgs_vd_aplus,1u)+"): nu_t <- nu_t*D^2 an Facettenzellen. y+ aus dem WANDMODELL (tw-Laufmittel der Spalding-Kette, y_w aus der Voxelgeometrie) -- NICHT aus dem lokalen Strain wie in V1, wo genau das an der Abloesung nu_t auf 0 trieb. Wirkpfad Slot 168, D^2-Histogramm 160..167, Facetten ohne Besuch 169.");
	if(s_sgs_band>0u&&s_sgs_fdwand==0u) print_error("CFD_SGS_BAND braucht CFD_SGS_FDWAND=1 -- der Bandleser sitzt im else-Zweig des FDWAND-Blocks in stream_collide; ohne ihn waere das Band ein stiller No-Op.");
	if(s_sgs_band==1u) print_error("CFD_SGS_BAND=1 ist sinnlos: Lage 1 IST die Facettenmenge und wird schon von FDWAND/SISM behandelt. Gueltig sind 0 (aus), 2 (Lage 2) oder 3 (Lage 2+3).");
	if(s_sgs_band>0u&&s_sgs_wandfrei) print_error("CFD_SGS_BAND x CFD_SGS_WANDFREI: WANDFREI ueberspringt den ganzen Block, das Band bliebe wirkungslos -- und WANDFREI ist am 8-mm-Fahrzeug nach 175 Schritten divergiert.");
	if(s_sgs_band>0u&&s_sgs_sism==0u) print_error("CFD_SGS_BAND braucht CFD_SGS_SISM=1. Seit dem Umbau vom 08.09. liefert der Bandkernel NUR Sbar; ohne SISM gibt es kein Sbar und der Abzug waere konstant null (stiller No-Op). Die fruehere Fassung ersetzte stattdessen w -- sie kippte am 8-mm-Stressarm bei Schritt 392, auch ohne SISM.");
	if(s_sgs_nut_skal!=1.0f&&s_sgs_fdwand==0u) print_error("CFD_SGS_NUT_SKAL braucht CFD_SGS_FDWAND=1 -- der Skalierzweig sitzt IM FDWAND-Block von stream_collide; ohne ihn ein stiller No-Op.");
	if(s_sgs_nut_skal!=1.0f&&!s_facetten) print_error("CFD_SGS_NUT_SKAL ohne CFD_FACETTEN: keine Facettenzellen, kein Zweig.");
	if(s_sgs_nut_skal!=1.0f&&s_sgs_sism>0u) print_error("CFD_SGS_NUT_SKAL und CFD_SGS_SISM gleichzeitig: ZWEI nu_t-Senker an derselben Zelle, also zwei Variablen in einem Lauf (Iron Rule 1). Der Diskriminator misst gerade GEGEN SISM -- er gehoert in den Arm OHNE SISM.");
	if(s_sgs_nut_skal!=1.0f&&s_sgs_vandriest>1u) print_error("CFD_SGS_NUT_SKAL und CFD_SGS_VANDRIEST=2 gleichzeitig: van Driest senkt nu_t bereits mit D^2, das waeren zwei Senker in einem Lauf.");
	if(s_sgs_nut_skal!=1.0f&&s_sgs_wandfrei) print_error("CFD_SGS_NUT_SKAL x CFD_SGS_WANDFREI: das WANDFREI-Gate ueberspringt den ganzen FDWAND-Zweig -- der Skalierer bliebe wirkungslos, und WANDFREI ist am 8-mm-Fahrzeug nach 175 Schritten divergiert.");
	if(s_sgs_nut_skal!=1.0f&&s_sgs_nut_skal<5e-7f&&s_sgs_nut_skal>0.0f) print_error("CFD_SGS_NUT_SKAL = "+to_string(s_sgs_nut_skal,9u)+" liegt unter 5e-7 und wird von to_string(...,6u) als Festkomma zu 0.000000f emittiert -- nu_t an Wandzellen waere dann EXAKT null, also der WANDFREI-Zustand, und zwar am Waechter unten vorbei (Muster s_fac_ema/s_fac_pema/s_fac_apg). Groesseren Faktor waehlen.");
	if(s_sgs_nut_skal<=0.0f) print_error("CFD_SGS_NUT_SKAL <= 0 setzt nu_t an Wandzellen auf null oder negativ -- das IST der WANDFREI-Zustand, der am 8-mm-Fahrzeug nach 175 Schritten divergierte. Positiv waehlen.");
	if(s_sgs_nut_skal>1.0f) print_warning("CFD_SGS_NUT_SKAL > 1 ERHOEHT die Wanddaempfung. Als Diskriminator gegen SISM ist der Arm nur mit einem Faktor < 1 sinnvoll, und zwar mit dem ABGELESENEN (Lage 1, 4 mm: 0,148).");
	// ★ 22.09. 14:40 Pruefbefund H1: hier stand `s_sgs_band_pi = env_u("CFD_SGS_BAND_PI", 0u)` -- 129 Zeilen NACH `band_pi_on = band_on&&s_sgs_band_pi>0u` (oben). band_pi_on war damit IMMER false:
	// der Host band den FD-Modus (band_sbar der Laenge band_N, FD-Bandkernel gestartet), der Kernel war mit SGS_BAND_PI uebersetzt und schrieb 6 float je Bandzelle -> 6-facher Pufferueberlauf
	// (CPU nichtdeterministisch, B70 CL_OUT_OF_RESOURCES). Die Statik wird jetzt in setup.cpp NEBEN s_sgs_band gesetzt (vor dem Konstruktor), wie jeder andere JIT-relevante Schalter.
	if(s_sgs_band_pi>1u) print_error("CFD_SGS_BAND_PI kennt nur 0 (FD-Modus) und 1 (Pi-konsistente EMA).");
	// ★ 22.09. 14:04 EIGENER FEHLER (Lauf s5_bandpi2_12_8 starb vor dem ersten Schritt): hier stand `if(s_sgs_band_pi>0u&&s_sgs_band==0u) print_error(...)`.
	// fahrzeug_dd nullt s_sgs_band vor dem Fernfeld-Konstruktor, s_sgs_band_pi bleibt 1 -> der Fernfeld-Konstruktor starb -- exakt die CFD_TIMER_APG-Falle von heute frueh
	// (lbm.cpp, Kommentar bei s_timer_apg). Die Ansage-Pruefung steht jetzt in setup.cpp VOR der Fallauswahl (liest nur die Umgebung, laeuft genau einmal).
	if(s_sgs_band_pi>0u&&s_sgs_band>0u) print_info("SGS-BAND im Pi-MODUS (CFD_SGS_BAND_PI=1, Plan C, 22.09.2026): nu_t = c2*max(0, |S_Pi| - |EMA(S_Pi)|) in stream_collide, EMA ueber T = "+to_string((ulong)s_sgs_sism_T)+" Schritte ab 0, Abzug ab Schritt "+to_string(s_sgs_sism_ab)+"; KEIN FD-Bandkernel, KEIN u-Lesen an Lage 2..N. Wirkpfad Slot 186, Klemme Slot 187, Sbar_band(Pi) im Bericht.");
	if(s_sgs_band>3u) print_warning("CFD_SGS_BAND = "+to_string(s_sgs_band)+": mehr als drei Lagen sind ungemessen. Die Lagenmessung reicht bis Lage 6 (Absenkung 85,2/80,0/75,9/72,1/69,1/67,9 % bei 4 mm), aber der Klemm-Verbund ueber viele Lagen ist der WANDFREI-Pfad.");
	if(s_sgs_sism>0u&&s_sgs_fdwand==0u) print_error("CFD_SGS_SISM braucht CFD_SGS_FDWAND=1 -- der Sbar-Abzug lebt im FD-Kernel sgs_fdwand; ohne ihn gaebe es keinen Kernel (stiller No-Op).");
	if(s_sgs_sism>0u&&!s_facetten) print_error("CFD_SGS_SISM ohne CFD_FACETTEN: keine Facettenzellen, kein FD-Kernel.");
	if(s_sgs_sism>0u&&s_sgs_sism_T==0u) print_error("CFD_SGS_SISM_T = 0 (oder ungesetzt): alpha = 1/T waere eine Division durch 0. T in SCHRITTEN angeben und HERLEITEN (Kanal: aus T_ett; Kugel/Fahrzeug: L/U durch dt) -- kein Handwert.");
	if(s_sgs_sism>0u&&s_sgs_sism_T>0u&&s_sgs_sism_ab<3ull*(ulong)s_sgs_sism_T) print_warning("CFD_SGS_SISM_AB < 3*T: <S> traegt am Phasenwechsel noch den Anfahrtransienten -- die Klemme kann flaechig greifen (nu_t = 0 = WANDFREI-Zustand). Der Slot-127-Anteil am Laufende zeigt es.");
	if(s_sgs_sism>0u&&s_sgs_wandfrei) print_warning("SISM x WANDFREI: WANDFREI hat an Wandzellen Vorrang (FDWAND-Leser in stream_collide) -- fac_wfd wird dort nie gelesen; Slot 126 zaehlt im FD-Kernel trotzdem, die Physik ist der WANDFREI-Arm.");
	if(s_sgs_sism>0u) print_info("SHEAR-IMPROVED SMAGORINSKY (CFD_SGS_SISM, 07.09.2026, Leveque et al. 2007) im FD-Kernel: nu_t = c2*max(0, |S|_FD - |<S>|), <S> = EMA ueber T = "+to_string((ulong)s_sgs_sism_T)+" Schritte, klassisch bis Schritt "+to_string(s_sgs_sism_ab)+" (EMA laeuft ab 0 mit, KEIN Warmstart). Wirkpfad Slot 126 (Abzug aktiv) / 127 (Klemme |S|<Sbar, nur Phase 2), Zeitreihe sism_sbar.csv.");
	// ★ 03.09. UTKORR-Ansage fuer NACHBAR. Der No-Op-Waechter (NACHBAR/KDIAG ohne iMEM) sitzt an den LESESTELLEN in setup.cpp: dort
	// wird der Schalter bei fc<3 still auf 0 gesetzt, hier waere er schon unsichtbar (Rauchtest xs_guard_nb_ohne_imem 03.09.: rc=0).
	if(s_fac_nachbar>0u) print_info("NACHBARABTASTUNG aktiv (CFD_FAC_NACHBAR): Wandmodell-Eingang u_t/y_w aus der zweiten Fluidzelle entlang der Normale (Slot 72 angewandt / 73 kein Fluidnachbar / 74 Nachbar still). CFD_FAC_UTKORR = "+to_string(s_fac_utkorr,3u)+" wirkt NUR an Zellen mit Eigenabtastung -- die 3/2-BB-Deflation gilt am Nachbarn nicht (03.09.).");
	if(s_fac_ema>0.0f&&s_fac_ema<5e-7f) print_error("CFD_FAC_EMA > 0 aber unter der Emissionsquantisierung (to_string 6 Stellen) -- der Filter froere still auf dem Warmstart ein.");
	if(s_fac_pema>0.0f&&s_fac_pema<5e-7f) print_error("CFD_FAC_PEMA > 0 aber unter der Emissionsquantisierung -- der Filter froere still ein.");
	if(s_fac_apg!=0.0f&&fabs(s_fac_apg)<5e-7f) print_error("CFD_FAC_APG zu klein fuer die 6-Stellen-Emission -- wuerde still zu 0.000000 (No-Op-Arm)."); // Gross-Audit N
	// ★ 16.09.2026: die Warnung "NICHT bitreproduzierbar" entfaellt -- grad rho kommt aus dem Vorkernel fac_apg_ab (eigener Launch nach stream_collide, In-Order-Queue).
	if(s_fac_pema>0.0f&&s_fac_nachbar>0u) print_error("PEMA + NACHBAR: die gefilterte Kette (kernel.cpp, utb_wm = utb*def_fac_utkorr) rechnet twe aus dem eigenen gefilterten u mit yw -- die Nachbarabtastung waere dort WIRKUNGSLOS, Slot 72 zaehlte trotzdem (Pruefagent 03.09.). Kombination gesperrt.");
	// ★ 16.09.: der folgende Absatz ist HISTORIE -- die RHO_FP16/RHO_SPARSAM/RHO_RAND x APG-Sperren sind entfallen (E5): grad rho kommt aus den DDFs (Vorkernel fac_apg_ab, FP32-Rekonstruktion), nicht aus dem rho-Feld.
	// ★ 12.09.2026 (Audit-Schleife, Pruefer B): APG bildet rho-DIFFERENZEN zwischen Nachbarzellen,
	// Groessenordnung 1e-6..1e-5. Mit rho als FP16S(rho-1) ist der Quantisierungsfehler je Summand
	// |rho-1|*2^-11, bei |rho-1| = 1e-3 also 4,9e-7 -- so gross wie das Signal. Ueberall sonst steht
	// rho als Absolutwert, dort ist das Verhaeltnis 1:4000. APG ist der einzige Verbraucher, den das
	// Format QUALITATIV trifft.
	// Die Sperre steht HIER und nicht im Setup: s_fac_apg wird in drei Setups gesetzt (fahrzeug_dd,
	// kanal, kugel), eine setup-lokale Sperre haette zwei davon offen gelassen. Genau dafuer wurde
	// die APG+PEMA-Sperre darunter schon einmal hierher verlegt (Tiefen-Audit A1-B3).
	// ★ 16.09.2026: die RHO_FP16 x APG-Sperre entfaellt -- rho je Nachbar wird im Vorkernel aus den DDFs gebildet (FP32-Summe), nicht aus dem Halbwort-Puffer gelesen.
	if(s_fac_apg!=0.0f&&s_fac_pema>0.0f) print_error("APG + PEMA: die gefilterte Kette verwirft die APG-Korrektur still -- Kombination gesperrt (Tiefen-Audit A1-B3: Sperre jetzt IM Konstruktor, setup-unabhaengig)."); 
	if(getenv("CFD_SPALDING_IT")&&env_u("CFD_SPALDING_IT",3u)==0u) print_warning("CFD_SPALDING_IT=0 wird auf 1 GEKLEMMT (min 1; Default ohne Env ist 3) -- Gross-Audit N16.");
	if(env_u("CFD_SPALDING_IT", 0u)>0u&&!s_wandfunktion&&!s_facetten) print_warning("CFD_SPALDING_IT wirkt nur mit CFD_WANDFUNKTION oder CFD_FACETTEN -- hier WIRKUNGSLOS (Audit R3).");
#ifndef TRT
	// ★ Audit-Nacharbeit 4: CFD_LAMBDA liegt in der TRT-Emission -- unter SRT (aktueller Build) ist
	// der Schalter TOT. Ein Lambda-A/B liefe bitgleich und ohne jede Meldung; deshalb die Ansage.
	if(getenv("CFD_LAMBDA")!=nullptr) print_warning("CFD_LAMBDA ist gesetzt, aber der Build laeuft mit SRT -- der Schalter ist WIRKUNGSLOS (TRT in defines.hpp aktivieren).");
#endif // TRT
#ifdef UPDATE_FIELDS
	// ★ Audit-Nacharbeit 8: im CFD_REG_BC-Arm liest deriv_reg u[] von Nachbarzellen, waehrend
	// stream_collide unter UPDATE_FIELDS u[] im selben Kernellauf SCHREIBT -- Wettlauf, jedes
	// REG-A/B waere nicht bitreproduzierbar. Default ist aus; wer ihn zieht, wird gewarnt.
#ifdef REGULARIZED_BOUNDARIES
	if(getenv("CFD_REG_BC")!=nullptr&&atoi(getenv("CFD_REG_BC"))>0) print_warning("CFD_REG_BC=1 unter UPDATE_FIELDS: deriv_reg liest u[] im Wettlauf mit dem Schreiber -- Ergebnisse sind NICHT bitreproduzierbar (Audit-Befund 8).");
#ifdef U_FP16
	// ★ TODO 2 Schritt 4 (12.09.2026) -- ANSAGE, kein Abbruch: der regularisierte Rand ist der einzige
	// GEMESSENE u-Leser mit zweistelligem Quantisierungsfehler.
	// EINGEORDNET 12.09. abends (Pruefagent, MITTEL): dieselbe Arithmetik 0,5*(up_-um_) auf
	// quantisierten Nachbarn steht auch in sgs_gdiag und sgs_fdwand -- und die BEIDEN
	// laufen, waehrend deriv_reg an CFD_REG_BC haengt und aus ist. Gemessen wurde davon sgs_fdwand
	// (identische Arithmetik wie sgs_gdiag): 0,018 % auf |S|_FD und auf nu_t, also 150-fach
	// unempfindlicher als der Rand. Grund ist die umgekehrte Paarung -- an der Facettenzelle ist |u|
	// klein (Faktor 7 gegen den Einlassrand) und der Gradient gross (Faktor 75), und 92,6 % der
	// Facettenzellen haben mindestens einen Solidnachbarn, der als exakte 0 eingeht und nicht rundet.
	// NICHT GEMESSEN ist fac_nachbar_ab. Es bildet KEINE Differenz (es liest einen einzigen Nachbarn) --
	// der Verstaerker dort ist eine AUSLOESCHUNG: die Tangentialprojektion ut = u - (u.n)n zieht gleich
	// grosse Terme voneinander ab, der Quantisierungsfehler skaliert aber mit |u|, nicht mit |u_t|. Wo
	// die Stroemung fast wandnormal steht -- Stufenschatten, Abloesekanten, also genau die Faelle, fuer
	// die CFD_FAC_NACHBAR gebaut ist -- entscheidet die Rundung mit ueber das Tor ut2 > 1e-6.
	// Und die 0,018 % oben gelten fuer den klassischen FDWAND-Zweig, wo nu_t ~ |S|_FD. Im SISM-Zweig
	// ist nu_t ~ max(0, |S|_FD - Sbar); der absolute Fehler bleibt, steht dann aber auf der Differenz
	// statt auf dem Betrag, und der relative waechst um denselben Faktor, um den SISM abzieht.
	// Beides ist die offene Luecke, und beides braucht einen Lauf, keine Nachrechnung. Am 4-mm-Feld bei 501 ms voll nachgerechnet
	// (alle 3.290.677 TYPE_E-Zellen, alle neun Ableitungen): auf f_neq stehen 2,69 % relativer RMS,
	// Median je Zelle 3,8 %, p90 28 %. Die beiden x-Flaechen tragen 1,94 % des Signals und 53 % des
	// Fehlers -- dort ist |u| am groessten und der Gradient am kleinsten, die schlechteste Paarung.
	// Der zentrale Zweig von deriv_reg feuert dabei NIE (0 von 3,29 Mio Zellen): TYPE_E liegt auf den
	// Domaenenflaechen, tangentiale Nachbarn sind selbst TYPE_E und der aeussere schlaegt periodisch
	// auf die Gegenflaeche um. 99,80 % der Randzellen bekommen genau eine einseitige Ableitung.
	// Gegenmittel, falls dieser Arm je gefahren wird: u in der Randschale der Dicke 2 in float32
	// halten. deriv_reg liest genau sieben Zellen (n und j[1..6]), die Schale sind 7.684.695 Zellen
	// = 1,48 % des Nahfelds = 43,97 MiB Mehrbedarf, also 1,48 % des Gewinns von 2971 MiB. Sie nimmt
	// 100 % des Fehlers weg, und zwar strukturell statt statistisch. NICHT GEBAUT, weil der Arm aus ist.
	if(getenv("CFD_REG_BC")!=nullptr&&atoi(getenv("CFD_REG_BC"))>0) print_warning("CFD_REG_BC=1 unter U_FP16: deriv_reg bildet Differenzen quantisierter Nachbarn am Rand, wo |u| am groessten und der Gradient am kleinsten ist. Voll gemessen am 4-mm-Feld: 2,69 % relativer RMS auf f_neq, Median je Zelle 3,8 %, p90 28 %; die beiden x-Flaechen tragen 53 % davon. Entweder werkzeuge/u_format.sh FP32 oder die Randschale bauen (43,97 MiB, Begruendung im Quelltext).");
#endif // U_FP16
#endif // REGULARIZED_BOUNDARIES
#endif // UPDATE_FIELDS
#ifndef REGULARIZED_BOUNDARIES
	// R2: die Race-Warnung oben haengt an UPDATE_FIELDS -- ohne einkompilierten REG-Arm waere sie
	// eine Warnung ueber Code, den es nicht gibt. Stattdessen die Wirkungslos-Ansage.
	if(getenv("CFD_REG_BC")!=nullptr&&atoi(getenv("CFD_REG_BC"))>0) print_warning("CFD_REG_BC ist gesetzt, aber REGULARIZED_BOUNDARIES ist nicht einkompiliert -- der Schalter ist WIRKUNGSLOS.");
#endif // REGULARIZED_BOUNDARIES
	// ★ Daempfungszone: Sperre und Wirksamkeitsmeldung.
	if(s_sponge_n>0u) {
		// def_Nx/def_Ny/def_Nz im Sponge-Block sind DOMAENENmasse inklusive Halo -- mit mehreren
		// Domaenen rampte die Zone an jeder inneren Domaenengrenze. Bis das auf globale Koordinaten
		// umgestellt ist, wird das hart verweigert statt lautlos falsch gerechnet.
		if(get_D()!=1u) print_error("Daempfungszone (CFD_SPONGE_N) ist nur fuer eine Domaene gebaut; mit D>1 rampte sie an jeder inneren Domaenengrenze.");
		// ★ Nachpruefer-Befund: fuer n gab es eine Schranke, fuer a und wmin keine. CFD_SPONGE_WMIN=2.5
		// erzwaenge ueber fmax ein w>2, also NEGATIVE effektive Viskositaet und sofortige Explosion;
		// ein negatives a machte nu_s<0. Beides jetzt hart abgewiesen statt lautlos gerechnet.
		if(!(s_sponge_wmin>0.0f && s_sponge_wmin<2.0f)) print_error("CFD_SPONGE_WMIN muss echt zwischen 0 und 2 liegen (w>=2 bedeutet negative Viskositaet).");
		if(s_sponge_a<1.0f) print_error("CFD_SPONGE_A unter 1 wuerde die Viskositaet in der Zone SENKEN statt anheben (negativ sogar nu<0).");
		// Was die Zone laminar bewirkt, VOR dem Lauf hinschreiben -- eine Klemme, die still zuschlaegt,
		// waere genau der lautlose No-op, den dieses Projekt sonst jagt.
		const float nu_rand = nu*s_sponge_a, tau_rand = 0.5f+3.0f*nu_rand, w_rand = 1.0f/tau_rand;
		print_info("Daempfungszone: "+to_string(s_sponge_n)+" Zellen, Faktor "+to_string(s_sponge_a,1u)+" am Rand (quadratische Rampe, Boden ausgenommen).");
		print_info("  laminar am Rand: nu "+to_string(nu,8u)+" -> "+to_string(nu_rand,6u)+", tau "+to_string(tau_rand,4u)+", w "+to_string(w_rand,4u)+" (Klemme bei w = "+to_string(s_sponge_wmin,3u)+")");
		if(w_rand<s_sponge_wmin) print_warning("Die Klemme greift schon LAMINAR (w = "+to_string(w_rand,4u)+" unter "+to_string(s_sponge_wmin,3u)+") -- die Zone ist schwaecher als eingestellt. Faktor senken oder CFD_SPONGE_WMIN bewusst absenken.");
	}
	sparse_on = s_sparse_tiles_on;
	sparse_T  = s_sparse_T;
	s_sparse_tiles_on = false; s_sparse_T = 8u; // read-once beidseitig (Gross-Audit N13: T erbte sonst)
	string opencl_c_code;
#ifdef GRAPHICS
	graphics = Graphics(this);
	opencl_c_code = device_defines(device_info)+graphics.device_defines(device_info)+get_opencl_c_code();
#else // GRAPHICS
	opencl_c_code = device_defines(device_info)+get_opencl_c_code();
#endif // GRAPHICS
	fac_rek_jit = opencl_c_code.find("#define FAC_REK\n")!=string::npos; // ★ 23.09. Pruefbefund M5: das \n MUSS mit -- "#define FAC_REK" ist Teilstring von "#define FAC_REK_R3", ein reiner R3-Text haette hier faelschlich "MIT FAC_REK" gemeldet (messwerte/_bb-Teilstringfalle vom 03.09.)
	fac_rek_r3_jit = opencl_c_code.find("#define FAC_REK_R3")!=string::npos;
	fac_r1q_jit = opencl_c_code.find("#define FAC_R1Q\n")!=string::npos; // ★ 28.09. R1Q: mit \n, sonst ist FAC_R1Q Teilstring von FAC_R1Q_AN (Lehre M5 vom 23.09.)
	fac_r1q_an_jit = opencl_c_code.find("#define FAC_R1Q_AN")!=string::npos;
	fac_r1q_vr_jit = opencl_c_code.find("#define FAC_R1Q_VR")!=string::npos; // ★ 28.09. V_R
	fac_r1q_od_jit = opencl_c_code.find("#define FAC_R1Q_OHNE_DRUCK")!=string::npos; // ★ 28.09. V_R ohne Druckanteil (R1Q=4)
	fac_rek_s2_jit = opencl_c_code.find("#define FAC_REK_S2")!=string::npos; // ★ 24.09. S2: eigener Spiegel. Suchtext OHNE abschliessendes \n, sonst traegt er dieselbe Teilstring-Falle wie fac_rek_jit (Pruefbefund 23.09.). // ★ 23.09. R3: eigener Spiegel, sonst traegt der Arm-2-Lauf keinen Wirkpfadbeleg fuer das Gate. Gelesen in setup.cpp: pruefe_rek_wirkpfad (Wirkpfadwaechter) UND der K2-Zweig des Cd-Pfads sowie der Bilanzblock (★ 24.09., Pruefbefund N3: hier stand "NUR in pruefe_rek_wirkpfad"). Der Kohaerenzvergleich gegen fac_rek_on in alloc_facetten_domain betrifft fac_rek_jit, nicht dieses Feld. // ★ 22.09. S0, Lehre M2: Kernelzustand aus dem JIT-Text einfrieren; alloc_facetten_domain vergleicht ihn gegen fac_rek_on
	band_pi_jit = opencl_c_code.find("#define SGS_BAND_PI")!=string::npos; // ★ 22.09. Pruefbefund A-M2: Kernel-Modus aus dem JIT-Text einfrieren, alloc_sgs_band vergleicht ihn mit dem Host-Modus (H1-Klasse: Host FD, Kernel Pi -> 6x Ueberlauf; war nur ueber Symptome zu finden). Durchgang 2 N-2: HINTER dem #endif, damit auch ein GRAPHICS-Bau ihn setzt.
	// ★ C2 (aus V1 portiert, 2026-08-15): CFD_DUMP_DEFINES druckt die tatsaechlich emittierte
	// Define-Liste (schliesst die Fehlerklasse "Host-Define != OpenCL-Define" -- ein #define in
	// defines.hpp wirkt NUR host-seitig, ein Kernel-#ifdef braucht die Emission ZUSAETZLICH).
	// CFD_DUMP_CL schreibt den kompletten OpenCL-Quelltext je Domaene -- DAS Werkzeug fuer den
	// Kontrollarm-Identitaetsnachweis (Diff zweier Dumps).
	if(env_on("CFD_DUMP_DEFINES")) {
		const string d = device_defines(device_info);
		uint n=0u; string out=""; size_t pos=0ull;
		while(true) {
			const size_t a=d.find("#define", pos); if(a==string::npos) break;
			const size_t b=d.find('\n', a);
			out += "   "+d.substr(a, (b==string::npos ? d.length() : b)-a)+"\n"; n++;
			if(b==string::npos) break; pos=b+1ull;
		}
		print_info("CFD_DUMP_DEFINES: "+to_string(n)+" Defines an OpenCL emittiert:");
		print(out);
	}
	{ // ★ TODO 2 Schritt 4 (12.09.2026) -- IST=SOLL auf den emittierten Quelltext, KEIN Grep im Kopf.
		// Die Falle, gegen die das steht: Kernel::link_parameter reicht nur die cl::Buffer weiter
		// (opencl.hpp), der Typ ist dort vollstaendig geloescht. Ein vergessenes "global float* rho"
		// in irgendeinem Kernel liest dann zwei halbe Dichten als einen float -- Groessenordnung 1e38
		// oder 1e-38, und unter -cl-finite-math-only ohne jede Diagnose. Weil get_opencl_c_code()
		// alle Leerzeichen durch Zeilenumbrueche ersetzt (kernel.hpp), wird hier auf die UMGEBROCHENE
		// Form gezaehlt; wer das uebersieht, baut sich einen Waechter, der immer 0 findet.
		// Die 18 verbliebenen Stellen sind AUSSCHLIESSLICH SURFACE (5) und GRAPHICS (13); beide sind
		// in diesem Bau aus, und defines.hpp schliesst sie unter RHO_FP16 hart aus. Aendert jemand
		// die Zahl, ist das eine bewusste Entscheidung und diese Zeile gehoert mitgeaendert.
		// ★ 14 -> 13 am 12.09. abends: store_rho_diag ist entfallen (der Rueckleser war ein No-Op,
		// siehe die Begruendung an store_rho in kernel.cpp). Der Waechter hat den Wegfall SELBST
		// gemeldet und den Lauf angehalten -- so soll es sein.
		// ★ 13 -> 14 am 15.09.: rho_rek_ebene (RHO_RAND C1) liest rho an TYPE_E und als heutigen Pufferwert.
		// ★ 14 -> 15 am 15.09.: rho_ausgabe_ebene (RHO_RAND C2a) liest rho an TYPE_E.
		const string muster = "global\nfloat*\nrho", muster_t = "global\nrhoxx*\nrho";
		uint n_float=0u, n_t=0u;
		for(size_t i=opencl_c_code.find(muster); i!=string::npos; i=opencl_c_code.find(muster, i+1ull)) n_float++;
		for(size_t i=opencl_c_code.find(muster_t); i!=string::npos; i=opencl_c_code.find(muster_t, i+1ull)) n_t++;
		if(n_float!=18u||n_t!=14u) print_error("rho-Typ-Zensus im OpenCL-Quelltext (Soll rhoxx 14 seit 16.09.: der rho-Parameter von apply_facette_imem entfaellt, APG liest grad rho aus fac_nb; davor 15): "+to_string(n_float)+" x \"global float* rho\" (Soll 18, alle in SURFACE/GRAPHICS) und "
			+to_string(n_t)+" x \"global rhoxx* rho\" (Soll 14; 13 -> 14 -> 15 am 15.09., 15 -> 14 am 16.09. (apply_facette_imem ohne rho-Parameter): rho_rek_ebene, rho_ausgabe_ebene, RHO_RAND C1/C2a). Ein rho-Kernel ist nicht auf rhoxx umgestellt oder es ist einer dazugekommen -- bei 2-Byte-rho waere das ein stiller Faktor-1e38-Fehler, kein Absturz.");
	}
	{ // ★ TODO 2 Schritt 4 (12.09.2026) -- derselbe Zensus fuer u, und er braucht ein SCHAERFERES Muster.
		// Der rho-Zensus zaehlt Teilstrings. Bei u faengt das mehr, als es soll: der Kernel fuehrt einen
		// zweiten Puffer namens "unear", und "global\nfloat*\nu" steckt in "global\nfloat*\nunear"
		// drin. Der naive Zensus haette also 22 statt 21 gezaehlt und in JEDEM Lauf falsch Alarm
		// geschlagen -- gefunden, bevor die Zahl eingetragen war, weil sie am emittierten Quelltext
		// abgelesen und nicht geschaetzt wurde.
		// Deshalb: gezaehlt wird nur, wenn hinter dem "u" KEIN Bezeichnerzeichen mehr steht. Die drei
		// vorkommenden Fortsetzungen sind "," (20x seit 15.09., rho_rek_ebene), ")" (1x -- calculate_Q, dort ist u der letzte
		// Parameter) und der Zeilenumbruch (1x -- graphics_q, dessen Parameterliste per Splice
		// weitergeht). [Zuordnung berichtigt 12.09. abends, Pruefer C; die Zahlen stimmten.]
		// SOLL 21 x float: SURFACE 4 (average_neighbors_non_gas/_fluid, surface_0, surface_2),
		// PARTICLES 1 (integrate_particles), GRAPHICS 13, und DREI ungegatete Hilfsfunktionen --
		// closest_u (wird NIRGENDS aufgerufen, toter Code), interpolate_u (nur von integrate_particles)
		// und calculate_Q (nur von graphics_q_field). Alle drei werden immer mituebersetzt.
		// SOLL 20 x velxx (19 -> 20 am 15.09.: rho_rek_ebene, RHO_RAND C1): die aktiven Traeger. Aendert jemand eine der Zahlen, ist das eine bewusste
		// Entscheidung und diese Zeile gehoert mitgeaendert.
		auto zaehle_wortgenau = [&](const string& m) {
			uint n=0u;
			for(size_t i=opencl_c_code.find(m); i!=string::npos; i=opencl_c_code.find(m, i+1ull)) {
				const size_t j = i+m.length();
				if(j>=opencl_c_code.length()) { n++; continue; }
				const char c = opencl_c_code[j];
				if(!((c>='a'&&c<='z')||(c>='A'&&c<='Z')||(c>='0'&&c<='9')||c=='_')) n++;
			}
			return n;
		};
		const uint n_uf = zaehle_wortgenau("global\nfloat*\nu"), n_uv = zaehle_wortgenau("global\nvelxx*\nu");
		if(n_uf!=21u||n_uv!=20u) print_error("u-Typ-Zensus im OpenCL-Quelltext: "+to_string(n_uf)+" x \"global float* u\" (Soll 21, alle in SURFACE/PARTICLES/GRAPHICS oder totem Code) und "
			+to_string(n_uv)+" x \"global velxx* u\" (Soll 20; 19 -> 20 am 15.09.: rho_rek_ebene, RHO_RAND C1). Ein u-Kernel ist nicht auf velxx umgestellt oder es ist einer dazugekommen -- bei 2-Byte-u waere das ein stiller Fehler, kein Absturz: der Kernel laese zwei halbe Geschwindigkeiten als eine.");
	}
	// ★★ 15.09.2026 ABSTURZSPERRE TESTHAKEN (Heiko: "sicherstellen, dass so ein CL_OUT_OF_RESOURCES nicht versehentlich nochmal passiert").
	// Anlass: CFD_KLEMM_HAKEN=1 im 8-mm-Fahrzeug (65,6 Mio Nahfeldzellen) auf der B70 -- nahezu JEDE Zelle nimmt je Schritt den
	// Trefferzweig mit mehreren Atomics, stream_collide lief ueber das Treiber-Zeitlimit, "xe ... device wedged", sgs_fdwand
	// CL_OUT_OF_RESOURCES, der Desktop auf derselben Karte hing bis zum Neustart (logs/kl_s0d_dd8_h1_b70.log, journalctl 15:49:34).
	// Grenze = das GROESSTE Gitter, an dem ein solcher Haken nachweislich lief: Kugel 16 mm, 340x171x105 = 6 104 700 Zellen
	// (kl_s0b/kl_s0d_ku_h1_b70, rc 0) -- ein Messwert, kein geschaetztes Zeitlimit. Groesser nur auf Geraet 0 (CPU, kein Watchdog).
	{ // ★ 15.09.2026 Klemmen Z2e/Z2f: Sperren VOR dem Kernelbau
	  if(rho_huelle_env()>0u) {
#if !defined(FP16S)||!defined(RHO_FP16)||!defined(RHO_CLAMP)
		print_error("CFD_RHO_HUELLE braucht FP16S, RHO_FP16 und RHO_CLAMP: die numerische Huelle ist die des Halbworts (Plan §2.2).");
#endif
		if(positiv_env()!=2u||u_klemme_env()!=1u) print_error("CFD_RHO_HUELLE=1 (Arm B2) nur mit CFD_POSITIV=2 und CFD_U_KLEMME=1: erst Positivitaet und Betragsklemme machen die Konsistenzhuelle numerisch entbehrlich (Plan §2.2).");
		if(tor_huelle_env()>0u) print_error("CFD_RHO_HUELLE und CFD_TOR_HUELLE schliessen sich aus: unter der numerischen Huelle gibt es keine Bildhuelle (Plan §2.3).");
		if(klemm_haken_env()==1u||klemm_haken_env()==3u||klemm_haken_env()==4u) print_warning("CFD_KLEMM_HAKEN 1/3/4 uebersteuert CFD_RHO_HUELLE (Testband 1,001/1,002): RHO_HUELLE und def_rho_kons_* werden GAR NICHT emittiert, [298]/[299] existieren in diesem Bau nicht (Audit 16.09.2026, Befund B2/C-M1).");
		if(env_u("CFD_RHO_REK_PRUEF", 0u)>0u) print_error("CFD_RHO_HUELLE mit CFD_RHO_REK_PRUEF: die Host-Rekonstruktionspruefung rechnet mit 0,5/1,5.");
		print_warning("CFD_RHO_HUELLE=1: Dichteklemme an der NUMERISCHEN Huelle (Stabilitaetsversuch B2) -- die Physik dieses Laufs aendert sich; Host-Diagnosen mit 0,5/1,5 (N2F-Band) bleiben unveraendert.");
	  }
	  if(tor_huelle_env()>0u&&klemm_haken_env()==5u) print_error("CFD_TOR_HUELLE mit CFD_KLEMM_HAKEN=5: der Haken verengt die Bildhuelle auf 1 +- 1/32768 -- das Tor wuerde fast alles verwerfen.");
	  if(u_klemme_env()>0u) {
#if !defined(SRT)||!defined(RHO_CLAMP)
		print_error("CFD_U_KLEMME=1 ist nur unter SRT + RHO_CLAMP emittiert -- hier waere der Schalter ein stiller No-Op (Pruefbefund Z2d N5).");
#endif
	  }
	  if(klemm_haken_env()==5u) print_warning("CFD_KLEMM_HAKEN=5: TESTARM -- Bildhuelle fuer Slot 300 kuenstlich auf 1 +- 1/32768 verengt (Soll [300] > 0, [270] = 0, Physik unveraendert).");
	  if(tor_huelle_env()>0u&&!klemm_bilanz_env()) print_warning("CFD_TOR_HUELLE=1 ohne Klemm-Instrument (CFD_KLEMM_BILANZ=0): das Soll [270] = 0 ist hier nicht pruefbar (Pruefbefund Z2f NIEDRIG 1).");
#ifndef SRT
	  if(tor_huelle_env()>0u) print_warning("CFD_TOR_HUELLE=1 in einem Nicht-SRT-Bau: KLEMM_BILANZ und damit [270]/[300]/[301]/[302] werden nur unter SRT emittiert -- das Tor wird verengt, aber KEIN Zaehler und kein Konstantenspiegel belegt es (Audit 16.09.2026, Befund A-N4/B8).");
#endif
	  if(tor_huelle_env()>0u&&(klemm_haken_env()==1u||klemm_haken_env()==3u||klemm_haken_env()==4u)) print_warning("CFD_TOR_HUELLE mit CFD_KLEMM_HAKEN 1/3/4: die Testklemme haelt rho im Band 1,001/1,002, also LIEGT jedes rho in der Bildhuelle -- [270] = 0 und [210] = 0 sind dann trivial erfuellt und beweisen nichts. Der Wirkpfadbeleg ist in diesem Fall allein der Konstantenspiegel [301]/[302] (Audit 16.09.2026, Befund B4).");
	  if(tor_huelle_env()>0u) print_info("CFD_TOR_HUELLE=1: Lift-rho-Tor und Waechter [210] aus der Bildhuelle 1 +- Lambda^2 (RHO_CLAMP_MAX-1) -- Soll [270] = 0.");
	}
	{ // ★ 15.09.2026 Klemmen Stufe 1: Sperren des Positivitaetsbegrenzers VOR Kernelbau und Grossallokation (Pruefbefund P1a NIEDRIG 2)
	  const uint pm_ = positiv_env();
	  if(pm_==0u&&(positiv_haken_env()>0u||positiv_facette_env()>0u)) print_error("CFD_POSITIV_HAKEN / CFD_POSITIV_FACETTE ohne CFD_POSITIV > 0 -- wirkungslos (Ansage-Doktrin).");
	  if(pm_>0u) {
#if !defined(SRT)||!defined(D3Q19)||defined(FP16C)||!defined(RHO_CLAMP)
		print_error("CFD_POSITIV ist nur fuer SRT + D3Q19 + FP16S/FP32 + RHO_CLAMP gebaut (TRT nicht gebaut, Momente ausgeschrieben fuer D3Q19, FP16C ohne tau-Herleitung, Klassen und Leser haengen am Klemm-Instrument) -- hier waere der Schalter ein stiller No-Op.");
#endif
		if(zaehl_takt()<3ull) print_error("CFD_POSITIV braucht einen Zaehltakt >= 3 (Zaehlschritte t%takt == 2, Pruefpunkt takt+2/+3) -- hier "+to_string(zaehl_takt())+" (Pruefbefund P1b NIEDRIG 5).");
		if(!klemm_bilanz_env()) print_error("CFD_POSITIV braucht das Klemm-Messinstrument (CFD_KLEMM_BILANZ=1, Vorgabe): Klassen K0..K4, Fensterleser und Bericht haengen daran.");
		if(pm_==2u) print_warning("CFD_POSITIV=2: Positivitaetsbegrenzer WENDET AN (Projektionsform, Masse und Impuls erhalten) -- die Physik dieses Laufs aendert sich"+string(positiv_facette_env()>0u ? ", Facettenzellen K0 eingeschlossen." : ", Facettenzellen K0 ausgenommen (E2)."));
		if(pm_==1u&&positiv_facette_env()>0u) print_warning("CFD_POSITIV_FACETTE wirkt nur in Modus 2 -- im Messarm zaehlt K0 ohnehin als \"wuerde begrenzen\" (Ansage-Doktrin).");
		const uint ph0_ = positiv_haken_env();
		if(ph0_>0u) print_warning("CFD_POSITIV_HAKEN="+to_string(ph0_)+": TESTARM -- "+string(ph0_==1u ? "masse- und impulsfreie Stoerung an jeder "+to_string(positiv_haken_periode())+". reinen Fluidzelle ausserhalb der Randschale bei t = zaehl_takt+2 (aendert die Physik)" : (ph0_==2u ? "tau_i = 1,2 w_i: jede Zelle Kandidat und machtlos (Felder im Messarm bitgleich)" : "Haken-1-Stoerung UND Klassenzaehlung fuer n%7 == 0 uebersprungen (aendert die Physik), Soll: genau die Klassen-Beanstandung"))+".");
	  }
	}
	{ const uint kh_ = klemm_haken_env(); const uint ph_ = positiv_haken_env();
	  const bool cpu_ = device_info.is_cpu;
	  if((kh_>0u||ph_>0u)&&get_N()>positiv_sicheres_gitter()&&!cpu_) print_error("ABSTURZSPERRE: CFD_KLEMM_HAKEN="+to_string(kh_)+" / CFD_POSITIV_HAKEN="+to_string(ph_)+" auf einem GPU-Gitter mit "+to_string(get_N())+" Zellen (> positiv_sicheres_gitter() = "+to_string((ulong)positiv_sicheres_gitter())+", groesstes belegt sicheres Hakengitter). Atomics in fast jeder Zelle je Schritt haben am 15.09. die B70 lahmgelegt (device wedged). Haken nur an der Kugel <= 16 mm oder auf Geraet 0 (CPU)."); }
	if(env_on("CFD_DUMP_CL")) {
		static std::atomic<uint> dump_nr(0u); // je Domaene eine Datei, sonst ueberschreibt die zweite die erste
		const string pfad = "/tmp/fx3d_kernel_dump_"+to_string(dump_nr++)+".cl";
		std::ofstream f(pfad); f<<opencl_c_code; f.close();
		print_info("[CFD_DUMP_CL] OpenCL-Quelltext -> "+pfad+" ("+to_string((uint)opencl_c_code.size())+" Bytes)");
	}
	this->device = Device(device_info, opencl_c_code);
	print_info("Allocating memory. This may take a few seconds.");
	allocate(device); // lbm first
#ifdef GRAPHICS
	graphics.allocate(device); // graphics after lbm
#endif // GRAPHICS
}

// FORK -- F-Bounding-Box und Block-Tiling: Voreinstellungen. Beide default aus/voll = bit-identisch
// zu Upstream. (Die Ueberschrift nannte frueher nur das Block-Tiling, direkt ueber s_fbbox.)
uint LBM_Domain::s_fbbox[6] = {0u,0u,0u,0u,0u,0u};
void LBM_Domain::set_force_bbox(const uint x0, const uint y0, const uint z0, const uint nx, const uint ny, const uint nz) {
	s_fbbox[0]=x0; s_fbbox[1]=y0; s_fbbox[2]=z0; s_fbbox[3]=nx; s_fbbox[4]=ny; s_fbbox[5]=nz;
}
uint LBM_Domain::s_sponge_n = 0u;
float LBM_Domain::s_sponge_a = 3000.0f;
float LBM_Domain::s_wf_tau = 1.0f;
bool LBM_Domain::s_wandfunktion = false;
bool LBM_Domain::s_facetten = false;
bool LBM_Domain::s_fac_imem = false;
float LBM_Domain::s_fac_ema = 0.0f;
float LBM_Domain::s_fac_pema = 0.0f;
uint LBM_Domain::s_fac_masse_alle = 0u;
uint LBM_Domain::s_fac_uw = 0u;
bool LBM_Domain::s_fac_uw_sn = false;
bool LBM_Domain::s_fac_satgate = false;
uint LBM_Domain::s_boden_eq_n = 0u;
uint LBM_Domain::s_boden_eq_down = 0u;
uint LBM_Domain::s_boden_eq_split = 0xFFFFFFFFu;
float LBM_Domain::s_boden_eq_u = -1.0f; // ★ Pruefbefund B7 (12.09.2026): hier stand 0.075f -- die ZWEITE
// verdrahtete Kopie der Gittergeschwindigkeit. Solange u_lat eine Konstante war, war der Wert
// zufaellig richtig; seit CFD_U_LAT existiert, liesse ein Fall, der "s_boden_eq_u = u_lat" vergisst,
// die Fahrbahn mit 0,075 laufen, waehrend die Stroemung auf dem neuen u_lat laeuft -- plausibel
// aussehend und damit unentdeckbar. Sentinel < 0 plus Waechter in enqueue_boden_eq: LAUT statt still.
// (Alle sechs Faelle setzen ihn heute vor der Konstruktion; der Waechter feuert also nie. Genau so
// soll ein Waechter aussehen.)
uint LBM_Domain::s_boden_eq_abstand = 0u; // Heiko 2026-08-20: reifennahe Aussparung (Chebyshev-Abstand zu TYPE_S, Boden ausgenommen) // Setup kann eigenes u_lat durchreichen (XL-Audit B6)
uint LBM_Domain::s_einlass_eq_n = 0u; // ★ EINLASS_EQ (V1-Port apply_inlet_velocity): Spaltenzahl x=1..N hinter dem Einlass; 0 = aus
uint LBM_Domain::s_smbox[6] = {0u,0u,0u,0u,0u,0u}; // ★ TODO 2: Schreibmasken-Box; 0 -> F-BBox
uint LBM_Domain::s_u_takt = 0u;     // ★ TODO 2 Schritt 3 (CFD_U_SPARSAM): 0 = aus, sonst ratio
uint LBM_Domain::s_rho_takt = 0u;   // ★ TODO 2 Schritt 1 (CFD_RHO_SPARSAM): 0 = aus, sonst Sample-Kadenz in FEINEN Schritten
uint LBM_Domain::s_rho_rand = 0u;   // ★ 15.09.2026 RHO_RAND (CFD_RHO_RAND): 0 = aus; nur Nahfeld
float LBM_Domain::s_einlass_eq_u = -1.0f; // Setup reicht sein u_lat durch (Konvention wie s_boden_eq_u); Sentinel wie dort, Pruefbefund B7
bool LBM_Domain::s_schale_paritaet = false; // CFD_N2F_PARITAET (Beweisarm, s. lbm.hpp)
float LBM_Domain::s_schale_alpha = 0.0f; // ★ P9c N2F-SCHALE: Blendfaktor der near->far-Rueckkopplung; 0 = aus. Read-once wie EINLASS_EQ; Setup setzt lbm_f EXPLIZIT 0.
uint LBM_Domain::s_fac_alpha = 0u;
bool LBM_Domain::s_fac_elibb = false;
uint LBM_Domain::s_sgs_fdwand = 0u; // ★ 02.09. Geistermoden-Fix
uint LBM_Domain::s_sgs_gdiag = 0u; // ★ 31.08. g-Diagnose (CFD_SGS_GDIAG)
uint LBM_Domain::s_fac_messnur = 0u; // ★ 30.08. Mess-Nur-Modus (BB-Physik, Facetten-Instrument)
float LBM_Domain::s_fac_rek_leiter = 1.0f; // ★ 24.09. CFD_FAC_REK_LEITER (Diagnoseleiter, Vorgabe 1.0 = bitgleich)
uint LBM_Domain::s_fac_r1q = 0u; // ★ 28.09. CFD_FAC_R1Q (Rang-1-Querrest)
uint LBM_Domain::s_fac_rek = 0u; // ★ 22.09. CFD_FAC_REK (S0/S1 Wandzell-Rekonstruktion)
uint LBM_Domain::s_fac_pinv = 0u; // ★ 04.09. CFD_FAC_PINV: Rang-1-Pseudoinverse im gekoppelten Zweig
uint LBM_Domain::s_fac_idx_voll = 0u; // ★ 03.09. Rueckschalter auf die fac_idx-Vollfeldform (A/B gegen die Bitmaske)
float LBM_Domain::s_sgs_nut_skal = 1.0f; // ★ 10.09. Diskriminator-Messarm, 1,0 = aus = bitgleich
uint LBM_Domain::s_sgs_band = 0u; // ★ 08.09. CFD_SGS_BAND
uint LBM_Domain::s_sgs_band_pi = 0u; // ★ 22.09. CFD_SGS_BAND_PI (Plan C)
uint LBM_Domain::s_f_liste = 0u; // ★ 03.09. CFD_F_LISTE: F nur an Wandsolidzellen
uint LBM_Domain::s_fac_nachbar = 0u; // ★ 30.08. Nachbarabtastung des Wandmodell-Eingangs
uint LBM_Domain::s_fac_kdiag = 0u; // ★ 30.08. Klassen-Diagnostik (CFD_FAC_KDIAG)
bool LBM_Domain::s_fac_elibb_pur = false; // Pur-Arm (Isolationsmessung) // ★ B1/B2 (2026-08-25): ELIBB 18-Link, q aus der Facettenebene
float LBM_Domain::s_fac_qmin = 0.1f;
float LBM_Domain::s_fac_kappa = 0.4f;
float LBM_Domain::s_fac_utkorr = 1.0f; // 3/2-Abtastpunkt-Messarm
float LBM_Domain::s_fac_qkappe = 1.0f; // Ex-Stabilitaetskappe des q>0,5-Zweigs: mit der MLS-Blende (Baustein 1, 26.08.) obsolet -- Default 1,0 = keine Kappung; Env-Hebel CFD_FAC_QKAPPE bleibt fuer A/Bs
uint LBM_Domain::s_fac_qdiag = 0u; // ★ QDIAG-Diagnosearme (Injektionsjagd 2026-08-25)  // q-Boden (P1-Entscheid): darunter HWBB, mit Zaehler
uint LBM_Domain::s_fac_rdiag = 0u; // ★ 07.09. Rueckfall-Diagnose (CFD_FAC_RDIAG), reine Zaehler
uint LBM_Domain::s_fac_kraft = 0u;

// ★ 11.09.2026 GEMEINSAMER ZAEHLTAKT (CFD_ZAEHL_TAKT, Default 100).
// Die Wirkpfadzaehler feuern alle t%100 Schritte. Das ist Diagnostik, keine Physik -- aber es
// ist teuer: E5 hat heute belegt, dass allein das Ausduennen EINES Blocks (P-TRT) von 100 auf
// 1000 zwoelf Sekunden bringt, 2,9 % der Wanduhr am 8-mm-Fahrzeug. Zaehler sind Atomics auf
// einen gemeinsamen Puffer und serialisieren; das ist eine andere Kostenart als Arithmetik und
// die einzige, die heute zweimal wirklich durchgeschlagen hat.
// DER TAKT STEHT AN GENAU EINER STELLE, weil Kernel und Host ihn BEIDE brauchen: der Kernel
// zum Zaehlen, der Host fuer die Ist=Soll-Abnahmen, die die erwartete Zahl aus dem Raster
// ausrechnen (Slots 7/20/21/22/76/186). Liefe beides auseinander, meldeten die Abnahmen
// reihenweise Falschalarm -- genau das hat der Kernel-Pruefer heute fuer E5 vorhergesagt.
// Funktionslokales static: unabhaengig von der Initialisierungsreihenfolge der Statiken.
// ★ 12.09.2026: die globale Schrittskalierung. Bewusst KEIN static-in-function-Initialisierer mit
// Nebenwirkung -- ulat_skal_setzen laeuft aus u_lat_schalter, also VOR jeder Domaenenkonstruktion
// und damit vor jedem Leser. Der Waechter in ulat_skal_setzen faengt die Umkehrung.
// ★ BERICHTIGT 12.09.2026 abends, und der eigene Waechter hat es gefangen: die erste Fassung
// war ein SETZER, den u_lat_schalter aufrufen musste. Beim ersten 4-mm-Start brach der Lauf mit
// "die Schrittskalierung wurde gesetzt, NACHDEM sie schon gelesen wurde" ab -- im dd-Fall liest
// irgendein Pfad frueher als der Setzer laeuft. Eine Reihenfolge zu reparieren, die man nicht
// sieht, ist die schlechtere Loesung: die Skalierung ist eine REINE FUNKTION DER UMGEBUNG und
// wird deshalb jetzt beim ersten Zugriff selbst gebildet. Damit gibt es keine Reihenfolge mehr,
// die falsch sein koennte. Der Setzer bleibt als Ist=Soll-Pruefung bestehen.
static double ulat_skal_aus_umgebung() {
	// ★ DIE RUNDUNG MUSS MITGEHEN: setup.cpp fuehrt u_lat als FLOAT (U_LAT_VORGABE ist 0.075f und
	// 1/N wird beim Zuweisen auf float gerundet). Rechnete es hier in double, unterschieden sich
	// die beiden Wege um 1,6e-8 relativ -- und genau daran ist der Ist=Soll-Waechter beim ersten
	// Selbsttest abgebrochen, obwohl BEIDE Zahlen gleich AUSSAHEN (0.6000001 gegen 0.6000001).
	// Also: exakt dieselben Rundungen wie drueben, und die Schranke danach grosszuegig.
	const double vorgabe = (double)0.075f;
	float u = 0.075f;
	if(const char* v = getenv("CFD_U_LAT")) { const float x = (float)atof(v); if(x>0.0f) u = x; }
	if(const char* v = getenv("CFD_SCHRITTE_PRO_ZELLE")) { const double n_ = atof(v); if(n_>0.0) u = (float)(1.0/n_); }
	return vorgabe/(double)u;
}
static double g_ulat_skal = 0.0; // 0 = noch nicht gebildet
double ulat_skal() { if(g_ulat_skal==0.0) g_ulat_skal = ulat_skal_aus_umgebung(); return g_ulat_skal; }
// Ist=Soll: u_lat_schalter kennt den wirklich gefahrenen u_lat und prueft damit, dass die aus der
// Umgebung gebildete Skalierung dieselbe ist. Weichen sie ab, hat eine Sonderbehandlung im Fall
// (etwa der Kanal, der CFD_U_LAT ausdruecklich NICHT anwendet) einen anderen u_lat gewaehlt --
// dann waeren die Schritt-Schalter gegen eine andere Geschwindigkeit umgerechnet als gerechnet wird.
void ulat_skal_setzen(const double s) {
	if(!(s>0.0)) print_error("ulat_skal_setzen("+to_string((float)s,7u)+"): die Skalierung muss positiv sein.");
	// 1e-6 und nicht 1e-9: der Vergleich soll eine ANDERE Geschwindigkeit fangen, nicht die
	// letzte float-Stelle. (Und dieser Kommentar steht UEBER der Zeile, nicht dahinter -- als
	// Trailer hat er beim ersten Anlauf den halben Ausdruck gefressen.)
	if(fabs(s-ulat_skal())>1e-6*fmax(1.0, fabs(s))) print_error("Schrittskalierung Ist != Soll: aus der Umgebung "+to_string((float)ulat_skal(),7u)+", aus dem gefahrenen u_lat "+to_string((float)s,7u)+". Die Schritt-Schalter waeren gegen eine andere Gittergeschwindigkeit umgerechnet als gerechnet wird.");
}
// ★ 16.09.2026 (TODO 4a): dx-Faktor der Schritt-Schalter, umgebungsrein (CFD_CASE + CFD_DX), Muster ulat_skal.
static double dx_skal_aus_umgebung() {
	const char* c = getenv("CFD_CASE");
	const bool dx_fall = c!=nullptr&&(string(c)=="fahrzeug_dd"||string(c)=="fahrzeug"||string(c)=="fernfeld"); // die drei CFD_DX-Leser in setup.cpp
	if(!dx_fall) return 1.0; // kugel: CFD_KUGEL_DX mit eigenen Schrittwerten je Zeile; kanal: T aus T_ett; facetten_test: kein Leser
	float dx = 4.0f; if(const char* v = getenv("CFD_DX")) { const float x = (float)atof(v); if(x>0.0f) dx = x; } // dieselbe float-Rundung wie env_f
	return (double)DX_SCHRITT_VORGABE_MM/(double)dx;
}
static double g_dx_skal = 0.0; // 0 = noch nicht gebildet
double dx_skal() { if(g_dx_skal==0.0) g_dx_skal = dx_skal_aus_umgebung(); return g_dx_skal; }
void dx_skal_setzen(const double s) {
	if(!std::isfinite(s)||!(s>0.0)) print_error("dx_skal_setzen("+to_string((float)s,7u)+"): die Skalierung muss endlich und positiv sein (CFD_DX = 0 gaebe inf)."); // ★ Pruefagent NIEDRIG
	if(fabs(s-dx_skal())>1e-6*fmax(1.0, fabs(s))) print_error("dx-Schrittskalierung Ist != Soll: aus der Umgebung "+to_string((float)dx_skal(),7u)+", aus dem gefahrenen dx "+to_string((float)s,7u)+". Die Schritt-Schalter waeren gegen eine andere Sprosse umgerechnet als gerechnet wird.");
}
double schritt_skal() { return ulat_skal()*dx_skal(); }
// Z2a: die Huelle 2,1 (Slot 210, u-Wickelschranke in setup.cpp klemm_lesen) haengt NICHT an diesen Makros, sondern am TYPE_E-Bereichswaechter.
static_assert(RHO_CLAMP_MIN==0.5f&&RHO_CLAMP_MAX==1.5f, "Z2a: die hergeleiteten RHO_CLAMP-Grenzen muessen bitgleich 0,5f/1,5f sein (Kontrollarm).");
// Der Zaehltakt ist ein SCHRITT-Schalter und skaliert deshalb mit. Ohne das laege die
// Wirkpfad-Zaehlung bei geaendertem u_lat an einer anderen physikalischen Zeit als in der Vorgabe.
ulong zaehl_takt() { const long long r = llround((double)max(1u, env_u("CFD_ZAEHL_TAKT", 100u))*schritt_skal()); /* ★ 16.09.: u_lat x dx, EIN Faktor fuer jeden Schritt-Schalter */ static const ulong t = (ulong)(r<1ll ? 1ll : r); return t; }
// ★ 15.09.2026 Klemmen S0b (KLEMMEN-STUFE0-PLAN.md): CFD_KLEMM_BILANZ = Messinstrument der beiden Zustandsklemmen (Vorgabe 1 = an,
// 0 = aus -- nur fuer den Wanduhr-A/B). CFD_KLEMM_HAKEN = Negativ-/Positivtests, NUR Testarme (aendern die Physik):
// 1 = RHO_CLAMP auf 1,001/1,002 (Treffer fast ueberall), 2 = def_c = 0,05 (u-Klemme fast ueberall), 3 = wie 1 und
// die rho-Klassenzaehlung fuer n%7==0 uebersprungen (Soll: genau eine Ist!=Soll-Beanstandung),
// 4 = wie 1, zusaetzlich Host-Wickelschranke 2^16 (S0c, Soll: MEHRDEUTIG-Warnung),
// 5 = Bildhuelle (und unter RHO_HUELLE auch die Konsistenzhuelle) kuenstlich auf 1 +- 1/32768 verengt -- POSITIVTEST der Zaehler
//     [300] bzw. [298]/[299], Physik unveraendert (Z2b/Z2e). ★ Audit 16.09.2026, Befund C-M4: 4 und 5 standen hier nicht, obwohl
//     klemm_haken_env() sie seit dem 15.09. annimmt -- eine Schalterlegende, die zwei von fuenf Stellungen verschweigt.
bool klemm_bilanz_env() { return env_u("CFD_KLEMM_BILANZ", 1u)>0u; }
uint klemm_haken_env() { const uint h = env_u("CFD_KLEMM_HAKEN", 0u); if(h>5u) print_error("CFD_KLEMM_HAKEN kennt nur 0..5."); return h; } // 4 = wie 1, zusaetzlich Host-Wickelschranke 2^16 (S0c, Soll: MEHRDEUTIG-Warnung)
// ★ 15.09.2026 Klemmen Stufe 1 P1a (KLEMMEN-STUFE1-PLAN.md): CFD_POSITIV = Positivitaetsbegrenzer in Projektionsform, EIN Schalter fuer beide
// Domaenen (E5). 0 = aus (Vorgabe), 1 = Messarm (s gerechnet und gezaehlt, nicht angewandt -- Felder bitgleich), 2 = anwenden.
// CFD_POSITIV_HAKEN (nur Testarme): 1 = Stoerung an K4-Zellen der Kugel (Soll Eimer [0,25;0,5)), 2 = tau_i -> 1,2*w_i (jede Zelle Kandidat),
// 3 = Klassenzaehlung fuer n%7 == 0 uebersprungen (Soll: genau eine Beanstandung). CFD_POSITIV_FACETTE: 0 = K0 in Modus 2 ausgenommen (E2), 1 = eingeschlossen.
// ★ 15.09.2026 Klemmen Z2d (KLEMMEN-STUFE2-PLAN.md §2.1, E6): CFD_U_KLEMME 0 = Komponentenklemme |u_a| <= c_s (Vorgabe, FluidX3D), 1 = Betragsklemme |u|^2 <= c_s^2.
// ★ 15.09.2026 Klemmen Z2e/Z2f (KLEMMEN-STUFE2-PLAN.md §2.2/§2.3): CFD_TOR_HUELLE=1 Lift-rho-Tor und Waechter 210 aus der Bildhuelle;
// CFD_RHO_HUELLE=1 Zustandsklemme an der NUMERISCHEN Huelle [20/32768; 1+65504/32768] (Arm B2), Konsistenzhuelle 0,5/1,5 nur gezaehlt (298/299).
uint tor_huelle_env() { const uint k = env_u("CFD_TOR_HUELLE", 0u); if(k>1u) print_error("CFD_TOR_HUELLE kennt nur 0 und 1."); return k; }
uint rho_huelle_env() { const uint k = env_u("CFD_RHO_HUELLE", 0u); if(k>1u) print_error("CFD_RHO_HUELLE kennt nur 0 und 1."); return k; }
bool rho_huelle_aktiv() { // ★ Audit-Schleife 16.09.2026, Befund B2/C-M1: EINE Quelle fuer den WIRKSAMEN Zustand statt der blossen Umgebung.
	// Die Emission unten gibt CFD_KLEMM_HAKEN 1/3/4 den Vorrang: dann wird weder RHO_HUELLE noch def_rho_kons_* emittiert, und die
	// Zaehler [298]/[299] existieren im Binary NICHT. Der Bericht entschied vorher an rho_huelle_env() und druckte "0/0" fuer Zaehler,
	// die es nicht gab -- genau die Phantom-Null, die dieses Projekt als harten Fehler fuehrt. Emission, Bericht, Wickelschranke und
	// Haken-5-Soll lesen seitdem DIESE Funktion.
	return rho_huelle_env()>0u&&!(klemm_haken_env()==1u||klemm_haken_env()==3u||klemm_haken_env()==4u);
}
uint u_klemme_env() { const uint k = env_u("CFD_U_KLEMME", 0u); if(k>1u) print_error("CFD_U_KLEMME kennt nur 0 (Komponente) und 1 (Betrag)."); return k; }
uint positiv_env() { const uint m = env_u("CFD_POSITIV", 0u); if(m>2u) print_error("CFD_POSITIV kennt nur 0 (aus), 1 (Messarm) und 2 (anwenden)."); return m; }
uint positiv_haken_env() { const uint h = env_u("CFD_POSITIV_HAKEN", 0u); if(h>3u) print_error("CFD_POSITIV_HAKEN kennt nur 0..3."); return h; }
uint positiv_facette_env() { const uint f = env_u("CFD_POSITIV_FACETTE", 0u); if(f>1u) print_error("CFD_POSITIV_FACETTE kennt nur 0 (K0 ausgenommen) und 1 (eingeschlossen)."); return f; }


// ★ 11.09.2026 SPALDING-TABELLE (CFD_SPALDING_TAB, Default AUS).
// wf_spalding_uplus loest X*S(X)=Y mit DREI Newton-Schritten ohne Konvergenzabfrage. Der
// Kopfkommentar in kernel.cpp sagt selbst: tau_w-Fehler -0,44 % bei Y~2400, -4,4 % bei Y=1e4,
// "bei hohem Re_tau Iterationszahl erhoehen" -- bei 4 mm ist der Fall eingetreten und die Zahl
// stand weiter auf drei. Eigene Messung (200k Punkte log-gleich ueber den gemessenen Bereich
// Y = 1,52 .. 2,19e4, gegen Bisektion in double):
//     Newton it=3 (heute) : max 4,364 %   p99 4,235 %
//     Newton it=8         : max 0,0001 %
//     Tabelle 512 / 2 kB  : max 0,0035 %  p99 0,0033 %
// Die Tabelle ist also rund 1250-mal genauer als der heutige Stand UND billiger (kein exp/log
// je Iteration). Gestuetzt wird sie auf log(Y) gleichverteilt, abgelegt wird log(u+), gelesen
// linear interpoliert -- beide Achsen logarithmisch, dort ist die Kurve fast gerade.
//
// BAUFORM: __constant im Dateibereich, NICHT als privates Array und NICHT als Kernelparameter.
//  - privates Array mit Laufzeitindex ist die Scratch-Falle (Faktor ~100, siehe scratch_gate).
//  - ein Kernelparameter waere ein Signatur-Splice, und den hat dieser Fork zweimal bezahlt
//    (R()-Klammerfalle). Gegenprobe vor dem Bau: eine Minimalkernel-Probe mit genau dieser
//    Bauform ergab auf der B70 private_size=0, spill_size=0.
// Die Emission laeuft ueber device_defines und damit NICHT durch get_opencl_c_code(), die
// Leerzeichen durch Zeilenumbrueche ersetzt -- die Werteliste ist davon unberuehrt.
static double spald_S(const double X) { const double kap=0.41, emkB=0.104874; const double kX=kap*X, e=exp(kX);
	return X + emkB*(e-1.0-kX-0.5*kX*kX-kX*kX*kX/6.0); }
static string spalding_tabelle() {
	// ★ 11.09.2026 nach der Messung auf DEFAULT AN gestellt (Heiko: "nur was messbar positiv
	// ist wird uebernommen"). Belegt am 8-mm-Fahrzeug gegen o8_it8 (acht Newton-Schritte,
	// praktisch auskonvergiert), sechs 50-ms-Fenster, cd_reib als Zeiger:
	//   Newton it=3 : Abstand -0,00077, SECHSMAL negativ -- ein systematischer Versatz
	//   Tabelle     : Abstand -0,00024, Vorzeichen wechselnd -- 69 % des Versatzes weg
	// KEIN Tempogewinn: 402 s gegen 403 s bei 4 s Streuung. Die -4,57 % Instruktionen aus
	// der Analyse schlagen NICHT auf die Wanduhr durch, weil der Facettenpfad nur 0,6 % der
	// Zellen betrifft. Das ist zum zweiten Mal an einem Tag dieselbe Lehre: eine
	// Instruktionszahl ist kein Laufzeitmass. Die Tabelle bleibt trotzdem, weil sie einen
	// GEMESSENEN systematischen Fehler beseitigt und nichts kostet.
	// CFD_SPALDING_TAB=0 stellt den alten Newton-Pfad her, bitgleich geprueft (o8_tab0
	// gegen o8_e5: 28 von 28 Dateien identisch).
	if(env_u("CFD_SPALDING_TAB", 1u)==0u) return (string)"";
	const uint N = 512u; const double Ylo = 1e-1, Yhi = 1e6;
	const double l0 = log(Ylo), dl = (log(Yhi)-l0)/(double)(N-1u);
	string r = "\n	#define SPALDING_TAB";
	r += "\n	#define def_spald_l0 "+to_string((float)l0, 8u)+"f";
	r += "\n	#define def_spald_invdl "+to_string((float)(1.0/dl), 8u)+"f";
	r += "\n	#define def_spald_max "+to_string((float)(N-2u), 1u)+"f";
	r += "\n__constant float def_spald_tab["+to_string(N)+"]={";
	for(uint i=0u; i<N; i++) { // Bisektion in double, dieselbe Klemme X<=100 wie im Kernel
		const double Y = exp(l0+(double)i*dl); double lo=1e-12, hi=100.0;
		for(uint k=0u; k<200u; k++) { const double m=0.5*(lo+hi); if(m*spald_S(m)<Y) lo=m; else hi=m; }
		r += (i>0u ? "," : "")+to_string((float)log(0.5*(lo+hi)), 9u)+"f";
	}
	r += "};";
	return r;
}
 // ★ 30.08. Zellkraft statt Slip (CFD_FAC_KRAFT)
bool LBM_Domain::s_fac_quergate = false; // ★ 2026-08-25 CFD_FAC_QUERGATE: BB belassen, wenn der Querrest die Wandschubspannung uebersteigt
bool LBM_Domain::s_fac_lsq = false; // ★ 2026-08-25 Default AUS nach Pruefbefund 4-A/4-B: das ist eine
// MODELLAENDERUNG, kein Numerikfix. LSQ gewichtet t1 (Stroemungsrichtung, Ziel = Spalding-tau_w, die
// eigentliche Messgroesse) und t2 (Ziel 0, eine blosse Modellannahme) GLEICH -- fuer ein Wandmodell die
// falsche Gewichtung. Ausserdem bricht sie die SATGATE-Invariante "iMEM wirkt nur, wenn es sein Ziel im
// Budget EXAKT erreichen kann": LSQ erreicht es prinzipiell nie exakt, das Gate laesst sie trotzdem
// durch. Braucht einen eigenen Messarm mit eigener Begruendung, nicht den Rang eines Defaults.
float LBM_Domain::s_fac_apg = 0.0f;
uint LBM_Domain::s_fac_apg_haken = 0u; // ★ 16.09.2026 CFD_FAC_APG_HAKEN (gelesen im Konstruktor)
uint  LBM_Domain::s_fac_apg_moz = 0u;   // ★ 22.09.2026 CFD_FAC_APG_MOZ
float LBM_Domain::s_fac_apg_c   = 0.4f;  // Paperwert C
float LBM_Domain::s_fac_apg_ap0 = 0.005f;// Paperwert alpha_p0
uint LBM_Domain::s_timer_apg = 0u; // ★ 22.09.2026 CFD_TIMER_APG (gelesen im Konstruktor)
long LBM_Domain::s_fac_diagz = -1l;
float LBM_Domain::s_fac_tau = 1.0f;
float LBM_Domain::s_fac_budget = 1.0f;    // CFD_FAC_BUDGET (1a-B4t), Default bitidentisch
float LBM_Domain::s_fac_budget_sn = 1.0f; // CFD_FAC_BUDGET_SN (1a-Bsn), Default bitidentisch
float LBM_Domain::s_fac_isogate = 0.0f;   // CFD_FAC_ISOGATE (09.09.), Default bitidentisch
float LBM_Domain::s_fac_deteps = 0.0f;    // CFD_FAC_DETEPS (09.09.), Default bitidentisch
bool LBM_Domain::s_sgs_wandfrei = false;
bool LBM_Domain::s_sgs_guo = true; // ★ 2026-08-25 Default AN: das ist die richtige Physik, CFD_SGS_GUO=0 ist der Kontrollarm
bool LBM_Domain::s_sgs_diag = false;
ulong LBM_Domain::s_sgs_diag_ab = 0ull;
uint LBM_Domain::s_sgs_vandriest = 0u; float LBM_Domain::s_sgs_vd_aplus = 26.0f; ulong LBM_Domain::s_sgs_vd_ab = 0ull; // ★ 08.09. van Driest
uint LBM_Domain::s_sgs_sism = 0u; uint LBM_Domain::s_sgs_sism_T = 0u; ulong LBM_Domain::s_sgs_sism_ab = 0ull; // ★ 07.09. SISM (T/ab in Schritten)
float LBM_Domain::s_sponge_wmin = 0.5f;
bool LBM_Domain::s_sparse_tiles_on = false;
uint LBM_Domain::s_sparse_T = 8u;

void LBM_Domain::allocate(Device& device) {
	const ulong N = get_N();
	// Bei aktivem Sparse zunaechst nur ein 1-Zell-Platzhalter: finalize_sparse_tiles() legt die echte
	// sparse fi an. Grund ist kein Geschmack, sondern ein Treiberdefekt -- das Freigeben eines bereits
	// allozierten 19-GB-fi-Buffers bringt den Intel-NEO mit CL_OUT_OF_RESOURCES zu Fall. Das Move-Assign
	// in finalize gibt so nur den Platzhalter frei, was trivial ist.
	fi = Memory<fpxx>(device, sparse_on ? 1ull : N, velocity_set, false);
	// ★ 15.09.2026 RHO_RAND C2c: rho_rand_on MUSS vor der rho-Allokation stehen (stand hinter ihr, C2-Plan Falle 3).
	rho_rand_on = s_rho_rand>0u;
	rr_N = rho_rand_on ? r1_anzahl((uint)get_Nx(), (uint)get_Ny(), (uint)get_Nz()) : 0ull;
	if(rho_rand_on) { // Konstruktor-Sperren, setup-unabhaengig
		if(Dx*Dy*Dz>1u) print_error("RHO_RAND: die Domaene ist Teil einer Mehrdomaenen-Zerlegung -- der Halo-Transfer liest rho am Domaenenschnitt mit vollen Indizes.");
		if(get_Nx()<5u||get_Ny()<5u||get_Nz()<5u) print_error("RHO_RAND: eine Gitterkante < 5 -- die Randschalen-Packung (rr_idx) ist dort nicht definiert.");
		// ★ 16.09.2026: RHO_RAND x APG-Sperre entfaellt (Vorkernel liest DDFs).
#if defined(SURFACE) || defined(GRAPHICS)
		print_error("RHO_RAND x SURFACE/GRAPHICS: deren rho-Leser greifen auf das volle Feld zu.");
#endif
	}
	rho = Memory<rhoxx>(device, rho_rand_on ? rr_N+1ull : N, 1u, true, true, rho_pack(1.0f)); // ★ TODO 2 Schritt 4: Speicherwort fuer rho=1 (ohne RHO_FP16 1.0f, mit RHO_FP16 0x0000). ★ C2c: unter RHO_RAND nur R1 + 1 Papierkorb-Slot
	u = Memory<velxx>(device, N, 3u); // ★ TODO 2 Schritt 4: Speicherwort, 4 oder 2 Byte je Komponente
	flags = Memory<uchar>(device, N);
	if(sparse_on) { // Tile-Raster aufspannen; der Inhalt kommt erst in finalize_sparse_tiles()
		sparse_tiles_x = ((uint)get_Nx()+sparse_T-1u)/sparse_T;
		sparse_tiles_y = ((uint)get_Ny()+sparse_T-1u)/sparse_T;
		sparse_tiles_z = ((uint)get_Nz()+sparse_T-1u)/sparse_T;
		tile_slot = Memory<uint>(device, (ulong)sparse_tiles_x*sparse_tiles_y*sparse_tiles_z);
	} else {
		tile_slot = Memory<uint>(device, 1ull); // Platzhalter, wird nie gelesen (TS_A ist leer)
	}
	kernel_initialize = Kernel(device, N, "initialize", fi, rho, u, flags);
	// 23 Slots (Legende R3 nachgezogen, massgeblich ist lbm.hpp; [22] N2F-SCHALE-Blend-Wirkpfad t%100 (P9c); [21] EINLASS_EQ-Wirkpfad t%100; [20] BODEN_EQ-Wirkpfad t%100): [0,1] RHO_CLAMP, [2] WFB-Wirkpfad
	// (t%100), [3] tau-Klemme, [4] u_t~0-Skips, [5] Ein-Zellen-Spalt, [6] SGS-Wirkpfad (t%100),
	// [7] Facetten-Wirkpfad (t%100), [8] Facetten-Klemmen (BEIDE, gegatet t%100 seit R3),
	// [9] Facetten-Skips (gegatet t%100 seit R3), [10] reserviert, [11] ohne offenes Paar (t%100).
	// Achtung uint: 3/4/5 zaehlen jeden Schritt
	// und koennten bei ~1e9+ Ereignissen ueberlaufen -- Ist!=Soll faellt im Report auf, aber wer
	// Slots erweitert, gate sie. Vergroesserung statt neuem Puffer: haengt schon an stream_collide,
	// keine Signaturaenderung, Kontrollarm bleibt bitgleich (neue Slots nur unter #ifdef-Emission).
	rho_clamp_hits = Memory<uint>(device, (ulong)hits_n); // ★ 320->384 (hits_n) am 22.09.2026, S-1 der Wandzell-Rekonstruktion: 3 freie Slots (317..319) gegen 12-14 gebrauchte; Kosten: Geraetepuffer +256 B JE DOMAENE (4 B je Slot, 0 MB im VRAM-Budget), Hoststack je KlemmBilanz-Instanz +1280 B (Slot = 4+8+8 = 20 B; Arrays 6400 -> 7680 B, sizeof(KlemmBilanz) 7520 -> 8800; ★ 24.09.2026 hits_n 384 -> 392, die Arrays sind damit 7840 B -- die Allokation liest hits_n, diese Zahlen sind nur die Erlaeuterung B), im dd-Fall zwei Instanzen = +2560 B gegen 8 MiB Threadstack. [Pruefagent 22.09., Befunde 4/5: hier stand zuerst "256 B gesamt" und "9,2 kB" -- 9,2 kB waeren 384*24 B, der Slot kostet aber 20 B.] ★ 288->320 (hits_n) am 15.09.2026 abends: Klemmen Stufe 1 belegt 271..294 (KLEMMEN-STUFE1-PLAN.md §4). 224->288 am 15.09.2026: Klemmen Stufe 0 belegt 221..270 (KLEMMEN-STUFE0-PLAN.md §6). 128->160 am 06.09.: [123] u_w-Wirkpfad, [124] untere Klemme (= reines BB), [125] obere Klemme -- ACHTUNG, KEIN echter Nullbeweis: duw >= 0 immer, also ist uw > ut_ab konstruktiv unmoeglich und der Zaehler kann nie feuern (Pruefbefund 06.09.), [126] SISM angewandt = Sbar-Abzug aktiv (Phase 2, t%100, saettigend, gezaehlt im FD-Kernel sgs_fdwand seit 07.09. abends -- vorher frei), [127] SISM-Klemme |S|_FD < Sbar (nur Phase 2, t%100, saettigend; 127/126 = Klemmanteil: 0 % = EMA tot, 100 % = WANDFREI-Zustand, dazwischen Physik), [128..135] u_w/u_B-Histogramm. // 80->96 am 04.09.: [80] Rang-1-Pseudoinverse angewandt (im Bericht, pruefe_kaskade). [81..91] ZIELERFUELLUNG 04.09.: [81] gestichprobte Besuche (Nenner), [82] angewandt ohne Ziel (twe=0), [83..91] NEUN Eimer fuer r = phi1/(-def_fac_tau*twe) je Besuch, Grenzen -10/-1/0/0,5/0,9/1,1/2/10 (nach dem Erstlauf von sieben auf neun geschaerft). [92] Wirkpfad CFD_FAC_MASSE_ALLE. [93] Modus 2: f_0<=0 nach Kompensation. [94] ARM X: Schatten sagt Rueckfall, roh haette angewandt (X zwingt auf BB). [95] ARM X: roh Rueckfall, Schatten haette angewandt. KREUZTABELLE Gate x Solve-Zweig (04.09. abends, Spalte = Zweig der REALEN Kaskade vor dem Gate, 1=[78] 2=[79] 3=[12] 4=[14]/[80]): [96..99] Gate[10] x Zweig | [100..103] Gate[16] x Zweig | [104..107] Gate[64] x Zweig | [108..111] ARM X [94] x roher Zweig | [112..115] ARM X [95] x roher Zweig | [116] ARM X [95] mit rohem Rang-0 | [117] BODEN_EQ_ABSTAND-Aussparungen (S5). [118..122] REST-DRUCKTERM 05.09., SAETTIGEND: |2*(rho-1)*(S1.t1)|/Ziel, Grenzen 0,001/0,01/0,1/1 -- misst, was vom hydrostatischen Term NACH der Stoerform im Ziel VERBLEIBT. NUR die t1-Komponente (im Zielerfuellungsblock als b1_ze = S1.t1 NEU gebildet -- in der Basis, gegen die geloest wurde; unter PEMA ist das NICHT B1o), denn nur die geht in R1 ein; 49..53 misst dagegen den weggenommenen Term als VOLLEN Tangentialbetrag. [123..125] u_w (s. o.), [126..127] SISM (s. o.). SAETTIGEND (>=0xF0000000): 67-71, 76, 78-95, 108-116, 118-122, 126-127; WICKELND: 7-18, 27, 64, 65, 96-107, 117 -- das Soll an Slot 7 wird mod 2^32 geprueft (D1, bewusst nicht angeglichen). Die frueher hier angekuendigte Kreuztabelle "Gate-Rueckfall nach Solve-Zweig" IST DAMIT GEBAUT. // 72->80 am 02.09.: Slots 20-71 sind luecklos belegt (30-48 SGS_DIAG-Bins ueber BERECHNETE Indizes 30u+b/35u+bw/40u+bw/45u+..., die ein Literal-Grep nicht sieht -- zweimal bezahlte Lektion B-3/B70); neue Zaehler ab 72 // [70] KRAFTPFAD (CFD_FAC_KRAFT, saettigend; Soll Modus 1: == [69]) | [71] Kraftzellen im Anlauf t<100, UNGEGATET (saettigend) | [72..74] NACHBAR angewandt/kein-Fluid/still | [75] MESSNUR-Wirkpfad | [76] FDWAND angewandt | [77] F-Listen-Wirkpfadwaechter (kernel.cpp:912, Soll 0) | [78/79] exakte Solve-Zweige (siehe oben) --  KORRIGIERT 05.09.: hier stand faelschlich "[77..79] frei", alle drei sind belegt // // [67] ELIBB-Wirkpfad beide Zweige (saettigend) | [68] MLS-q>0,5-Zweig allein (saettigend, Audit 26.08.) | [69] Rueckfall-Buchung P-only (saettigend, Buchungsschluss 27.08.; Soll = 13+15+64 +10+16 unter SATGATE) // ★ LEGENDE, Stand 2026-08-27 (Pruefbefund 3-E: die alte war in sich widerspruechlich)
	// [0..1] RHO_CLAMP unten/oben (UNGEGATET, saettigend -- BERICHTIGT 15.09., hier stand t%100) | [2..5] Wandfunktion | [6] SGS_WANDFREI | [7..19] Facetten/iMEM
	// [20] BODEN_EQ | [21] EINLASS_EQ | [22] N2F-SCHALE | [23..24] N2F-Paritaet | [25..26] Paarungsbeweis
	// [27] Slot-13-Split | [28] Geschwindigkeitsklemme | [29] SPONGE | [30..34] nu_t/nu_0 Dekaden
	// [35..39] nu_t/nu_0 wandnaechste Lage | [40..44] davon anliegend | [45..48] oberer Schwanz
	// [49..53] Stoerform-Offset |2(S1.t)|/Ziel | [54..58] |P|/Ziel | [59] Bewegtwand-Term
	// [60..63] Guo-Korrektur, rel. Aenderung von |Pi^neq| | [64] Quergate (CFD_FAC_QUERGATE) | [65] LSQ-Rueckfall
	// [136..140] RDIAG |2(rho-1)(S1.t1)|/|Ziel| an RUECKFALLbesuchen | [141]/[142] dessen Vorzeichen + / -
	// [143] RDIAG Nenner (Rueckfallbesuche mit Ziel>0) | [144..148] s1_soll/u_t bei G11>0 (Gate-Rueckfall)
	// [149] G11roh ~ 0 (c_1 parallel n) | [150..154] s1_soll/u_t bei G11==0 (Rang 0) -- HYPOTHETISCH, s. kernel.cpp
	// [BERICHTIGT 15.09.2026: nicht ALLE -- 0/1/28/29/66 und 221 ff. sind ungegatet] ALLE Ereignis-Slots sind t%100-Stichproben; 49..58 und 60..63 zusaetzlich hash-ausgeduennt (jede 64.). RDIAG (136..154) ist NICHT ausgeduennt.
	// [126..127] SISM Wirkpfad/Klemme | [160..167] VAN DRIEST D^2-Histogramm, Zeitintegral aller Zaehlslots ab CFD_SGS_VD_AB
	// [168] VD Wirkpfad (= Summe 160..167) | [169] VD Facettenzelle ohne tw-Besuch | [170..185] VD Letzt-Stichprobe: zwei Baenke
	// [186] SGS-BAND Wirkpfad (Bandzelle behandelt) | [187] SGS-BAND Klemme (Sbar >= |S|, nu_t = 0). [HISTORISCH, siehe unten] HISTORISCHER STAND, NICHT MEHR GUELTIG (die verbindliche Angabe steht unten am Ende dieser Legende; Audit-Nachpruefung 16.09.2026, Befund H3): damals naechster freier Slot 204 (188..198 NUT_SKAL, 199..203 P-TRT; Puffer 224 seit 08.09.) [BERICHTIGT 10.09. nachts -- hier stand 188].
	// a 8 Eimer, Bank (t/100)&1 wird gezaehlt, die andere im selben Slot genullt -- nach dem Lauf traegt Bank (L/100)&1 genau den
	// letzten Slot L. [HISTORISCH bis 15.09. abends] HISTORISCHER STAND, NICHT MEHR GUELTIG (verbindlich ist die Angabe am Ende dieser Legende): damals naechster freier Slot 271 (Puffer 288 seit 15.09.; 221..270 = Klemmen Stufe 0, Layout KLEMMEN-STUFE0-PLAN.md §6: 221..225 rho-Treffer K0..K4, 226..235 Sum-q rho unten/oben, 236..241 Dekaden |drho|, 242..246 u-Treffer, 247..256 Sum-q jx +/-, 257..262 Dekaden |du|, 263/264 Sum-q jz +/-, 265 Kappung, 266..268 BODEN/EINLASS_EQ, 269/270 schale_blend/Lift-rho-Tor; Sum-q-Slots wickeln ABSICHTLICH mod 2^32). Bis 15.09.: 221 (216 = RHO_RAND R1-Zugriff ausserhalb der Schale, ungegatet, Soll 0, 217/218 = rho_rek_ebene Besuche/TYPE_E, 219/220 = rho_ausgabe_ebene Besuche/TYPE_E nur im gezaehlten Aufruf, 15.09.; 204..207 = rho/u-SPARSAM, 12.09.; 208/209 BEWUSST FREI GELASSEN als Luecke; 210 = rho ausserhalb 0,25..4,0 an der TYPE_E-Lesestelle, UNGEGATET, Soll 0 -- faengt den Fall, dass ein Kernel den 2-Byte-rho-Puffer als float liest; 211 = Besuche derselben Stelle an EINEM Schritt, Soll > 0, sonst hat 210 keine Abdeckung. 212 = |u| >= 1,0 oder nicht-endlich an derselben TYPE_E-Lesestelle, UNGEGATET, Soll 0 -- faengt bei u NICHT die Typverwechslung (das kann nur der Typ-Zensus), sondern die SAETTIGUNG des Halbworts ab |u| = 1,99902; 213 = Besuche dazu an EINEM Schritt, Soll > 0; 214 = Betragstor im Kopplungs-Lift (Invariantenzusicherung, konstruktiv unerreichbar: Klemme 0,57735 x Lift-Gewichte 1,5625 = 0,9021 < 1,0), Soll 0; 215 = Besuche des Lift-Schreibpfads, ohne die die Null in 214 nichts beweist. 216..217 waren am 12.09. kurzzeitig rho-Quantisierungs-Dekaden (HISTORISCH -- seit 15.09. traegt 217 die Besuche von rho_rek_ebene, 216 traegt seit C2b den R1-Zugriff ausserhalb der Schale): der Rueckleser im schreibenden Kernel wurde vom Geraeteuebersetzer wegoptimiert, siehe die Begruendung an store_rho in kernel.cpp; Puffer 224). [BERICHTIGT 10.09. nachts -- hier stand 186 bei Puffer 192, eine dritte, dritte-Groesse-Fassung; die Legende widersprach sich an drei Stellen] Alle VD-Slots nur unter #ifdef SGS_VANDRIEST (Kontrollarm bitgleich).
	// Klemmen Stufe 1 (KLEMMEN-STUFE1-PLAN.md §4, Puffer 320 seit 15.09. abends; Nachtrag P1b/P1c: 272..288, 290..294 und 291 nur an Zaehlschritten t%zaehl_takt == 2 UND Stichprobenzellen n%def_pos_sub == 0, 285 dito, 271/289 je genau ein Schritt in allen Zellen; 294 kappt 292 und 293): [271] Besuche am Pruefpunkt t == zaehl_takt+2 | [272] Kandidaten (ein f*_i + w_i < tau_i, Nicht-E)
	// | [273..277] s < 1 je Klasse K0..K4 (Modus 1: wuerde begrenzen) | [278] machtlos, KONSERVATIV (irgendein B_i < tau_i; Obermenge von unloesbar, Pruefbefund P1b NIEDRIG 1, Plan E3) | [279] davon f_eq_i + w_i < 0 | [280..284] s-Eimer [0;0,25) [0,25;0,5)
	// [0,5;0,75) [0,75;0,95) [0,95;1] (s = 1 durch Rundung moeglich, Modus 2 wendet dann nichts an) | [285] nach load_f negativ (Nicht-E) | [286] Kandidat und rho-Klemme | [287] Kandidat und u-Klemme | [288] H1-Zellen im Eimer [0,25;0,5)
	// | [289] Nachladeprobe t == zaehl_takt+3 | [290] Haken: Selbstpruefung Sum(f**-f*), Sum c(f**-f*) ueber Toleranz | [291] TYPE_E-Kandidaten (f_eq_i + w_i < tau_i) | [292]/[293] Sum-q (1-s), Sum-q Sum|df_i|
	// (Festkomma, wickeln ABSICHTLICH mod 2^32) | [294] Kappung zu 293. Klemmen Z2b: [295] u-Komponentenhuelle |u_a| >= c_s vor der Klemme (Soll = [28] unter der Komponentenklemme)
	// | [296] u-Betragshuelle |u|^2 >= c_s^2 | [297] 296 ohne 295 (Diagonalluecke) | [298]/[299] (nur CFD_RHO_HUELLE) rho unter/ueber der Konsistenzhuelle 0,5/1,5 | [300] Lift-rho ausserhalb der GESCHLOSSENEN Bildhuelle [0,21875; 1,78125], Soll 0 (Haken 5: > 0). Audit 16.09.2026 (B1): [301]/[302] KONSTANTENSPIEGEL der uebersetzten Torgrenzen def_tor_gate_lo/hi als Festkomma (S = def_klemm_s), Ist=Soll gegen die Host-Rechnung -- der Wirkpfadbeleg fuer CFD_TOR_HUELLE, das sonst nur eine Null vorzuweisen hatte. [303] (nur CFD_POSITIV_FACETTE) K0-Facettenzelle WIRKLICH begrenzt, Stichprobe wie [273] -- Wirkpfadbeleg des Schalters (Befund B3; [273] allein zaehlt in beiden Stellungen gleich). [304]/[305] KONSTANTENSPIEGEL der uebersetzten Waechterhuelle def_w210_lo/hi (Befund M3, zweite Haelfte von CFD_TOR_HUELLE), ein Schritt je Zaehltakt. APG 16.09.2026 (PLAN-APG-2026-09-16.md §A5, alle nur an Zaehlschritten, saettigend): [306] Vorkernel-Besuche (Soll = [7]) | [307] entartet (eine Achse ohne Fluidnachbar) | [308] APG-Zweig besucht (Soll = [7]-[9]) | [309]/[310] Klemme unten 0 / oben 2*tw (Summe = [19]) | [311]/[312] dp/ds > 0 (APG) / < 0 (FPG) | [313..316] Autoritaet |kappa*y_ab*dp/ds|/tw in <0,1 / 0,1-0,5 / 0,5-1 / >=1. MOZAFFARI 22.09.2026 (nur unter FACETTEN_APG_MOZ, alle an Zaehlschritten, saettigend): [317] Entartung (u_tau <= 0 oder alpha_p nicht endlich -> f := 1 erzwungen, Soll 0) | [318..325] f-HISTOGRAMM, Grenzen f >= 0,99 / 0,95 / 0,90 / 0,80 / 0,70 / 0,65 / 0,62 / darunter -- in alpha_p umgerechnet (C = 0,4, a0 = 0,005): 1,282e-4 / 7,143e-4 / 1,667e-3 / 5,000e-3 (= a0, natuerliche Mitte) / 1,500e-2 / 3,500e-2 / 9,500e-2. [318] enthaelt auch den gesamten FPG-Ast (f konstruktiv 1) UND die Besuche mit dp/ds exakt 0. Bezug ist das GEKLEMMTE Spalding-tw (Klemme tw_max im Spalding-Block), tw_neu = tw*f^2. ABNAHME: Summe [317..325] == [308] (jeder Besuch genau ein Fach; NICHT [311]+[312], die zaehlen dp/ds == 0 nicht mit -- Pruefbefund A-5/B-M1); [318] < [308], sonst ist der Tausch ein No-Op; [317] == 0 | [326]/[327] SCHATTEN der entfallenen Klemme: korr > tw bzw. -korr > tw (geklemmtes Spalding-tw), also genau die Bedingung, die unter der linearen Form [309]/[310] gezaehlt hat. ABNAHME unter MOZ: [19] = [309] = [310] = 0 UND |[326]+[327] - [316]| <= 1e-3*[316]+16 (symmetrisch: Strikt-/Nichtstrikt-Kante |korr| == tw und GPU-Divisionsrundung 2,5 ulp in [316]; Pruefbefund A-10/B-N1/2A-M1). Slot 8 (Endklemme tw*faca > tw_max) feuert unter MOZ NUR bei faca > 1 (tw <= tw_Spalding <= tw_max), also seltener als linear (dort tw bis 2*tw_Spalding) -- kein Widerspruch zu "keine Klemme mehr", die Endklemme ist die alte Stabilitaetsklemme (2A-N1). REKONSTRUKTION S0 (22.09.2026, CFD_FAC_REK): [328] Wirkpfad (Block erreicht, gegattert) | [329] WIRKUNG |du|/|u| > 1e-6 (Soll in S0 EXAKT 0) | [330] relativer Rundungszaehler (Soll 0 in der Delta-Form; Ausschlag = falsche Formelform gebaut) | [331] RESERVIERT fuer die R3-Kreuztabelle (ab S2). Inkrement am 23.09. ENTFERNT: es stand im selben Gate wie [328], war konstruktiv identisch, kostete ein Atomic je Besuch und taeuschte eine Pruefung vor, die nie feuern konnte (Audit-Schleife, drei Auditoren unabhaengig). [332] S1b MASSENNEUTRALITAET: |rho(f+Df) - rho(f)| > 1e-6*rho. Die Delta-Form erhaelt die Masse ANALYTISCH exakt (Sum w_i = 1, Sum w_i c_i c_i = c_s^2 I heben den -1,5(s.du)-Term genau auf), ★ 23.09. BERICHTIGT: die Probe rechnet INLINE mit derselben Summenreihenfolge wie klemm_rho_roh(), also rho VOR der Dichteklemme -- ein AUFRUF laege hier oberhalb der Definition und ist unter OpenCL C99 der sofortige Lauftod (BERICHTIGT 23.09. abends: der alte Text nannte den Aufruf, wer ihn nachzieht baut error -11 ein). Vorher standen zwei Aufrufe von calculate_rho_u gegeneinander, und DIE KLEMMT auf [0,5;1,5] (kernel.cpp:1257) -- bei einer Zelle dauerhaft ausserhalb war die Differenz EXAKT 0, egal was die Rekonstruktion mit der Masse tat (1,2-1,8 Mrd Klemmtreffer je Lauf, nicht hypothetisch). Schranke 1e-6 -> 5e-7. Numerisch bleibt der Boden beim ulp der Schlussaddition rho += 1.0f, also 5,96e-8 bis 1,19e-7 -- NICHT 1e-9, wie hier zuerst stand (Pruefbefund 23.09.). Die Schranke 1e-6*rho traegt damit rund 8- bis 17-fache Luft; wer sie schaerft, erzeugt Fehlalarme. ★ BERICHTIGT 23.09. abends: die Schranke im Kernel ist 5e-7, nicht 1e-6, und traegt damit 4,2- bis 8,4-fache Luft ueber dem ulp-Boden, nicht die hier zuerst genannten 8 bis 17. Soll 0 in JEDER Stufe. | [333] S1b ZWEITES MOMENT (xy): Summe c_x c_y Df_i gegen rho*(u_x du_y + du_x u_y + du_x du_y). NOETIG, weil [330] und [332] gemeinsam BLIND fuer einen Fehler im s-Vektor sind: Summe_i c_i w_i (c_i.du)(c_i.s) = 0 und Summe_i w_i [4,5(c.du)(c.s) - 1,5(du.s)] = 0 gelten fuer JEDES s, der s-Term wird erst im zweiten Moment sichtbar. Ohne [333] wuerden s = 2u, s = u+du oder das voellige Weglassen der quadratischen Terme unentdeckt durchlaufen. | [334] BEGLEITZAEHLER zu [333]: wie oft war |mxy_soll| <= 1e-7, die Probe an dieser Facette also LEER? [333] hat einen absoluten Boden; wo das Sollmoment darunter liegt, behauptet sie nichts. Ohne [334] liest man 333 = 0 als Beleg, wo gar nichts geprueft wurde. | [335] KONSTANTENSPIEGEL der Rekonstruktion: der UEBERSETZTE Kernel schreibt 0x5245464B ('REFK'), sobald der FAC_REK-Block im Geraetecode steht. Noetig, weil der alte Kohaerenzwaechter (fac_rek_jit gegen fac_rek_on) TAUTOLOGISCH war -- beide Seiten lasen dasselbe eingefrorene Feld, er konnte nie ausloesen (Audit-Schleife 23.09., Auditor B). Dieser Slot belegt den uebersetzten Kernel, nicht den JIT-Text. | [336..343] HISTOGRAMM der Hubrichtung t1.x ueber die Marken, Grenzen -0,5/-0,1/0/0,1/0,5/0,9/0,99. Beantwortet, ob die 10620 Huebe in x gleichsinnig zeigen. Haeufung nahe +1 heisst: die Betragssumme IST die Vektorsumme, und die gemessene Ausbeute von rund 3 % hat eine ANDERE Ursache. Streuung um 0 heisst: die wirksame Amplitude ist eps*<t1.x>, und jede auf eps normierte Amplitudenleiter ist falsch skaliert. | [344..351] HISTOGRAMM von rhon ueber die Marken, Grenzen 0,55/0,7/0,85/0,95/1,05/1,2/1,45. Der Block injiziert rhon*du mit dem GEKLEMMTEN rhon (Untergrenze 0,5); an **85 %** der Facettenbesuche steht es auf der Klemme und daempft dort den Hub um 43 % (GEMESSEN 23.09., Commit b4158aa, 53 195 580 Besuche -- die hier zuerst notierten 6,5 % waren eine Schaetzung aus den globalen Klemmzaehlern und galten fuer ALLE Facettenzellen, nicht fuer die markierten). Ohne diese Zahl ist die Amplitude des Arms unbekannt. Beide sind BITNEUTRAL: kein Zugriff auf fhn, nur atomic_inc an Zaehlschritten. | [352..359] HISTOGRAMM der NACHBARRICHTUNG t_nb.x, gleiche Eimergrenzen wie [336..343]. Die Nachbarabtastung liest das fertige u-Feld einer zweiten Fluidzelle entlang der Normale -- eine Zelle, die die ELIBB-Blende NICHT angefasst hat. Gegen [336..343] (lokale Richtung, aus der zu 85 % geklemmten Zelle) ist das der direkte Vergleich. | [360..367] HISTOGRAMM von cos zwischen lokaler und Nachbarrichtung, Grenzen -0,5/0/0,3/0,6/0,8/0,95/0,999. PFLICHT, nicht Beiwerk: zeigt der Hub kuenftig entlang t_nb, sieht das Wandmodell (Basis t1 bleibt LOKAL, kernel.cpp) davon nur cos*eps, und sin*eps landet im t2-Kanal -- also im Querrest, den das Modell zu null setzen will. | [368] kein Fluidnachbar (Rueckfall auf die lokale Richtung), Gegenprobe zu Slot 73. | [369] BAUPROBE der geschriebenen Richtung: ||t_nb|-1| > 1e-5 ODER |t_nb.n| > 1e-3 (die Tangentialschranke ist 1e-3, BERICHTIGT 23.09. abends -- hier stand 1e-5, Faktor 100 gegen den Kernel). Soll 0. Ohne sie ist ein Stride-Versatz nicht von "die Richtung streut halt" zu unterscheiden. Alle vier sind BITNEUTRAL (nur atomic_inc an Zaehlschritten). R3 23.09.2026 (nur unter CFD_FAC_REK=2, alle an Zaehlschritten, saettigend): [370] markierte Besuche AM GATE (Nenner) | [331] davon ZUSAETZLICH in den Rueckfall gezwungen -- die eigentliche Wirkung von R3; frueher "reserviert", jetzt belegt | [380] davon ohnehin schon Rueckfall (Gate war dort ein No-Op). ABNAHME [331]+[380] == [370], jeder Besuch genau ein Fach. | [371] BAUREIHENFOLGE-PROBE (NICHT der Nullbeweis des Tors -- berichtigt 23.09. spaet; unter der erlaubten Schalterschnittmenge konstruktiv 0, den Nullbeweis liefert der Hashvergleich Arm 1/Arm 2): Marke UND der Additivterm u_s != 0 nach dem Solve, Soll EXAKT 0 -- 370/331/380 zaehlen die Absicht, 371 das Ergebnis. Der Zaehler sass zuerst an pass2_an und war dort TAUTOLOGISCH (Nachpruefung H4). | [372] Normalprobe der Buchung: |du.n| > 1e-3*|du| (RELATIV -- die absolute Schranke prueft bei kleinem eps nichts, Nachpruefung M2), Soll 0 (du steht konstruktiv tangential; ein Normalanteil ginge in den Druckpfad und waere dort doppelt). Die Buchung selbst ist KEIN Zaehler, sondern fwx/y/z -= rho*du vor fac_tau_acc[1..3] -- ihr Beleg ist die Bilanzserie a2_bilanz (23.09.) und K2. [373..377] HISTOGRAMM von |G11roh| an den Marken (Grenzen 1e-6/1e-4/1e-2/1), belegt 23.09. abends. ENTSCHEIDET den Doppelterm: P1 wird aus fhn gebildet, und fhn traegt an den Marken schon das Df -- unter R3 ist phi1 = P1, die Wandkraft enthaelt den Wandlink-Anteil also ein ZWEITES Mal, zusaetzlich zur Buchung fw -= rho*du. Zur fuehrenden Ordnung ist der Zusatzterm rho*eps*G11roh. Liegt das Histogramm nicht im untersten Fach, traegt die Wandkraft den Term doppelt. ★ 24.09.2026 ERLEDIGT: er WIRD korrigiert (exakt, siehe 381..385); dieser Absatz ist keine Handlungsanweisung mehr, sondern die Messung der Groesse. GEMESSEN am kipp26 24.09.: nur 0,38 % der Marken haben |G11roh| < 1e-6 -- der Kanal traegt den Term also SEHR WOHL, die frueher hier und in kernel.cpp stehende Entwarnung war unbelegt. DOPPELTERM-KORREKTUR 24.09.2026 (nur unter CFD_FAC_REK=2, an Zaehlschritten, saettigend): [378] ANWENDUNGSPROBE -- fw wird VOR und NACH den drei Korrekturzeilen auf t1 projiziert und die Differenz gegen rek_dp1 gehalten; Soll EXAKT 0. Faengt drei Fehler, die kein Zaehler auf rek_gate fangen kann: Korrektur hinter fac_tau_acc (toter Code), Korrektur ausserhalb der FAC_REK_R3-Insel, Vorzeichen verkehrt. Sie faengt NICHT ein falsches Vorzeichen in der HERLEITUNG von DP1 -- dagegen hilft nur der Abstand cd_bericht.csv/forces.csv, siehe die R3-ANSAGE in setup.cpp. | [379] t2-KANAL: |DP2| > 0,1*|DP1| an der Marke. Beantwortet empirisch, ob die fuehrende Ordnung rho*eps*G11roh (nur t1) gereicht haette; sie haette rho*eps*G12roh unterschlagen, an ebener Wand exakt 0, an schiefen Linkmengen bis sqrt(G11roh*G22roh). Deshalb ist die EXAKTE Form gebaut: Df wird je Wandlink in der Momentenschleife nachgebildet und auf t1/t2 projiziert, ohne Ordnungsargument. | [381..385] HISTOGRAMM der Korrekturgroesse |DP|/(rho*|du|), Grenzen 0,01/0,1/0,5/1,0 (★ 24.09. Pruefbefund M4: es waren zwei Grenzen, das oberste Fach damit OFFEN bei 0,1 -- die Groesse erreicht aber 2,0, das Fach konnte "ein Zehntel der Quelle" nicht von "doppelt so gross, Reibungspfad gedreht" unterscheiden). ABNAHME: Summe [381..385] == [370]. | [386] BEGLEITZAEHLER zu [378] (Klasse von [334]): wie oft dominiert der Ausloeschungsboden 4,8e-7*|fw|_1 den relativen Teil der Toleranz? Wo er das tut, prueft [378] nichts und seine Null ist kein Beleg -- |fw.t1| ist um Dekaden groesser als |DP1|. Liegt alles im untersten Fach, war der Doppelterm an dieser Geometrie vernachlaessigbar und die Korrektur ein GEMESSENER No-Op. WAS [378] NICHT FAENGT (ausdruecklich, damit seine Null nicht ueberlesen wird): er vergleicht die ANWENDUNG gegen rek_dp1, nicht rek_dp1 gegen die Injektion. Eine Aenderung des Linkgates in der Momentenschleife, die rek_dp1 ueber eine andere Menge als P1 akkumuliert, bleibt unsichtbar; ebenso ein Vorzeichenfehler in der HERLEITUNG von DP1. Und wenn jemand die Korrekturzeilen UND den 378-Block GEMEINSAM hinter fac_tau_acc schiebt, bleibt er gruen -- er misst eine lokale Invariante, nicht die Lage zum Verbraucher (Pruefbefund M3, 24.09.). | [387] BLINDHEITSSCHRANKE von [378]: |DP2| > 1000*|DP1| an der Marke. Die Toleranz von 378 faengt ein verkehrtes Vorzeichen bis Verhaeltnis rund 1e4 und toten Code bis rund 1e3 (eigener float-Versuch 24.09., 200 000 Saetze je Stufe); darueber ist sie blind. [379] zaehlt ab 0,1 und sagt darueber nichts. Steht [387] auf 0, ist die Null von [378] ueber den gefahrenen Wertebereich belastbar. (Historie: am 23.09. stand hier einmal 373, widerspruechlich zur Zeile darueber, die 373..377 belegt -- wer der Legende folgte, haette 373 ein zweites Mal vergeben. Die Zeile, auf die sich diese Berichtigung bezog, ist am 24.09. ersetzt worden; der Hinweis bleibt als Warnung stehen, weil dieselbe Falle bei jeder Neuvergabe droht.) ★ 24.09.2026 BERICHTIGT (Pruefbefund M-7): hier stand bis eben ein zweites Mal "Puffer hits_n = 384 seit 22.09.2026" -- eine widerspruechliche Zweitfassung IN DERSELBEN ZEILE, aus der der naechste Slot falsch vergeben worden waere. Das ist wortwoertlich Pruefbefund 3-E, eine Ebene hoeher. | [388..392] HISTOGRAMM |DP_n|/(rho*|du|), Grenzen 0,01/0,05/0,15/0,3 -- der NORMALANTEIL des Wandlink-Flusses der Rekonstruktion. NICHT dasselbe wie [372]: der prueft |du.n| und ist konstruktiv 0, weil du tangential steht. DP_n verschwindet nur an der EBENEN Wand; an einer Einzellink-Zelle ist c.n von null verschieden, und 92 % der Rang-0-Facetten haben genau einen Link. Gebaut aus dem S2-Planungsschritt (Befund H1, 24.09.). ABNAHME: Summe [388..392] == [370]. WAS ER MISST UND WAS NICHT: allein die GROESSE von DP_n. Ob dieser Anteil im Druckpfad ankommt und ob er dort ungebucht ist, ist damit NICHT beantwortet -- das braucht eine eigene Messung. STUFE S2 24.09.2026 (nur unter CFD_FAC_REK=3, JIT-Define FAC_REK_S2, an Zaehlschritten): [397] WIRKPFAD und Nenner der vier folgenden -- Besuche des Bestimmungsblocks. Soll > 0; steht er auf 0, bleibt die Amplitude auf ihrem Anfangswert 0 und der Arm ist ein stiller No-Op. | [393] SCHRITTFLUKTUATION der Amplitude ★ BERICHTIGT 24.09. nachmittags (Pruefagent H3): hier stand "KONVERGENZ des Lag-1-Kreises". Das galt fuer das ALTE Ziel mit P1 (e_{n+1} = e* - G11roh*e_n). Seit S1 haengt die Amplitude nur noch an twe und rhon DESSELBEN Schritts, es gibt keine Rueckkopplung mehr und also auch nichts zu kontrahieren. Der Zaehler misst jetzt die Schrittschwankung von twe/rhon und damit DIREKT den Lag-1-Fehler (gemessen 63,8 %: an zwei Dritteln der Marken weicht die angewandte Amplitude um mehr als 5 % ab). KEIN Konvergenzbefund. Alter Text: |d1_neu - d1_alt| > 5 % von |d1_neu|. Die Amplitude stammt aus dem VORSCHRITT, weil P1 erst nach der Momentenschleife feststeht. Faellt dieser Zaehler nicht, schwingt sie (Periode-2-Mode, im Plan als real gefuehrt) -- dann muss die Injektion hinter die Momentenschleife. | [394] rhon auf der unteren Klemme. VORZEICHENUMKEHR gegen die frueheren Stufen: rho steht in S2 im NENNER, die Klemme VERSTAERKT die Amplitude statt sie zu daempfen. | [395] ★ NUR ZUSAMMEN MIT SLOT 8 DEUTBAR (24.09. nachmittags, Pruefagent H1/H2): unter S1 ist die Schranke STRUKTURELL redundant, weil |s2_r1| = def_fac_tau*twe <= def_fac_tau*tw_max per fmin gilt. Die Null ist ein STOLPERDRAHT, keine bestandene Messung -- sie feuert nur, wenn jemand die tw_max-Klemme entfernt (wie beim MOZ-Tausch schon einmal geschehen). Slot 8 sagt, ob der Klemmpfad ueberhaupt lief; ohne ihn ist [395] = 0 ein Zufallsbefund. Bis 24.09. nachmittags wurde ausserdem NACH der Division verglichen (fabs(s2_d1)<=0.5f*ut), was an der twe-Klemme um 1-2 ulp ueberschoss und die Zelle STILL auf Bounce-Back warf -- der damals gemessene [400] = 9 ist genau diese Kante. Alter Text: Schranke 0,5*u_t gegriffen -> Rueckfall auf Bounce-Back (SATGATE-Logik, nicht klemmen: Befund G8, die geklemmte Anwendung hat einen vorzeichen-definiten Bias). Hohe Quote heisst, die Amplitude kommt von der SCHRANKE statt vom Wandmodell -- der Knopf waere nur umbenannt. | [396] RICHTUNG der Schrittaenderung (s2_neu > s2_alt). ★ ERSETZT 24.09. nachmittags (Pruefagent H1): hier stand "R1 > 0, die Wand beschleunigt, erwartet rund 40 %". Unter S1 ist s2_r1 = -def_fac_tau*twe mit twe >= 0 lueckenlos und def_fac_tau in {0;1} (drei Zuweisungen, alle Literale, ein CFD_FAC_TAU gibt es nicht) -- der Zaehler KONNTE nicht feuern, die 40 % waren nicht verfehlt sondern unmoeglich, und der Gegenphasentest verglich 0 % gegen 0 %. Dieselbe Klasse wie Slot 124/125 und Slot 331. JETZT: bei einer Periode-2-Mode geht [396]/[397] gegen 100 % und [401]/[402] gegen 0 % (oder umgekehrt), bei reiner Fluktuation liegen BEIDE bei rund 50 %. | [397] WIRKPFAD des Bestimmungsblocks und NENNER der vier S2-Prozentzahlen (★ Pruefbefund MITTEL-3: er fehlte hier, obwohl die M1-Abnahme auf ihm steht). ABNAHME: [397] == [370], strikt -- gleiches Gate, kein return dazwischen. | [398] LEERE PROBE: markierter Besuch mit rho*|du| == 0. Unter S2 ist die Amplitude je Facette UND Schritt 0 -- beim ersten Schritt immer, und nach jedem Schranken-Rueckfall. Die Histogramme 381..385 und 388..392 legen solche Besuche stumm ins unterste Fach; ohne diesen Zaehler liest sich das als "vernachlaessigbar" (Klasse [334]/[386]). | [399] WIDERSPRUCHS-ENTSCHEIDER: |rek_dp1 - rho*du*G11roh| > 20 %. Beide sind dieselbe Groesse in fuehrender Ordnung und je Wandlink vorzeichengleich, es gibt keine Ausloeschung. Gebaut, weil sich [373..377] und [381..385] im S2-Probelauf widersprachen (G11roh >= 1e-2 an 92 %, |DP|/(rho|du|) >= 1e-2 an 21 %) -- und die H1-Korrektur (P1 - rek_dp1) steht genau auf dieser Gleichheit. | [400..402] GEGENPHASE zu [395]/[396], abgetastet bei t%def_zaehl_takt==1 statt ==0, mit [402] als eigenem Nenner. GRUND (Pruefbefund HOCH-2, 24.09.): def_zaehl_takt ist gerade, alle Zaehlschritte sind gerade, und die dort gelesene Amplitude stammt vom Schritt DAVOR, also von einem ungeraden. Eine Periode-2-Mode -- vom Plan ausdruecklich als real gefuehrt -- ist damit konstruktiv unsichtbar und liest sich als einseitiges Vorzeichen (gemessen 99,7 %), als Schranke die nie greift und als dauerhafte Schrittaenderung. Weichen die Paritaeten um mehr als 20 Prozentpunkte ab, schwingt die Amplitude und KEINE Kraftzahl des Arms ist deutbar. [403..407] AMPLITUDENGROESSE, ★ NEU 24.09. nachmittags (Pruefagent H4). Faecher von |s2_neu| an den projekteigenen Schwellen: [403] < 1e-6, [404] 1e-6..1e-5, [405] 1e-5..1e-4, [406] 1e-4..1e-3, [407] >= 1e-3. Nenner ist [397] (gleiches Gate). GRUND: unter S2 ist CFD_FAC_REK_EPS zwingend 0, deshalb schweigen BEIDE Groessenwaechter in setup.cpp (harte Sperre 1e-6 "der Hub ueberlebt store_f nicht", Warnung 1e-4 "darunter messen die Zaehler registerseitig") -- sie haengen an rek_eps_b>0. Fuer die LAUFZEITamplitude gab es keinen einzigen Waechter, und die Herleitung gibt |du| = twe/rhon ~ 4,5e-6..1,35e-5, also 7- bis 22-fach UNTER dem Messhub. Liegt die Masse in [403]/[404], arbeitet der Arm im stillen Band und KEINE seiner gruenen Zahlen belegt etwas ausserhalb der Register. DER KERNEL INDIZIERT DIESE FUENF BERECHNET (s2_f) -- ein Literal-Grep nach hits[404] findet sie NICHT, dieselbe Falle wie die SGS_DIAG-Faecher 30..48. (Stand 24.09.: Puffer 408, mit 403..407 voll -- ★ 28.09. ueberholt, siehe die R1Q-Zeile darunter: hits_n = 432.) DIESE LEGENDE IST DIE EINZIGE QUELLE DER SLOTVERGABE.
	// RANG-1-QUERREST 28.09.2026 (nur unter CFD_FAC_R1Q, JIT-Defines FAC_R1Q / FAC_R1Q_AN, alle an Zaehlschritten, saettigend): [408] markierte Besuche am Anwendungspunkt (Nenner; Marke -1 in fac_geo[8i+7] = Lage 1 und statischer Rang 1) | [409] Quelle berechnet (PINV-Zweig angewandt, pass2_an) | [410] PINV mit Rueckfall | [411] Zweig 3 (entkoppelter Skalarzweig) | [412] Zweig 1/2 (Vollrang) | [413] sonst -- ABNAHME Summe [409..413] == [408] STRIKT | [414] UNmarkiert im PINV-Zweig (nicht bedient) | [415..420] q = 1-a^2-b^2, Grenzen 0,1/0,3/0,5/0,7/0,9, Summe == [409] | [421..425] |p_perp|/|Z1| mit p_perp = (I-Gt/tr)P, Grenzen 0,1/1/10/100, [426] Ziel Z1 == 0, Summe [421..426] == [409] | R1Q=4 (FAC_R1Q_OHNE_DRUCK): V_R mit R' = R + 2(rho-1)(S1.t) -- nur der Reibungsanteil des BB wird ersetzt. [427] |du| > 0,5*ut: unter V_Z (R1Q=2) Stolperdraht, Soll 0; unter V_R (R1Q=3) das TOR (Quelle nicht angewandt, BB bleibt -- wie SATGATE, Schranke wie tw_max) | [428] Impulsprobe Sum c Df != rho*du (aus den Inkrementen, Toleranz 1e-3 relativ) | [429] Massenprobe Sum Df != 0 (aus den Inkrementen, 1e-3 von rho*|du|) | [430] Buchungsprobe dfw != -m | [431] Normalanteil |du.n| > 1e-3|du|. 428..430 nur unter FAC_R1Q_AN. Puffer hits_n = 432 seit 28.09.2026, VOLL.
	kernel_stream_collide = Kernel(device, N, "stream_collide", fi, rho, u, flags, t, fx, fy, fz, felder_voll_h, rho_clamp_hits); // ★ TODO 2: rho_voll HINTER fz, damit set_parameters(4u, t, fx, fy, fz, rho_voll) zusammenhaengend bleibt; absolute Indizes gibt es nur fuer 0 und 4..7
	kernel_update_fields = Kernel(device, N, "update_fields", fi, rho, u, flags, t, fx, fy, fz);
	kernel_boden_eq = Kernel(device, N, "boden_eq", fi, flags, t, 0.0f, 0u, 0u, 0u, 0u, rho_clamp_hits); // Parameter t/u/nz/nz_down/x_split/abstand je Enqueue
	boden_eq_n = s_boden_eq_n; boden_eq_u = s_boden_eq_u; boden_eq_down = s_boden_eq_down; boden_eq_split = s_boden_eq_split; boden_eq_abstand = s_boden_eq_abstand; // u_road = u_lat-Projektkonvention; Konstruktionszeit-Kopie (read-once-Doktrin)
	kernel_einlass_eq = Kernel(device, N, "einlass_eq", fi, flags, t, 0.0f, 0u, rho_clamp_hits); // ★ EINLASS_EQ (V1-Port apply_inlet_velocity): Parameter t/u/nx je Enqueue
	einlass_eq_n = s_einlass_eq_n; einlass_eq_u = s_einlass_eq_u; // Konstruktionszeit-Kopie (read-once-Doktrin)
	rho_takt = s_rho_takt; // ★ TODO 2: Konstruktionszeit-Kopie wie die uebrigen (read-once-Doktrin)
#ifdef SRT
	klemm_bilanz_on = klemm_bilanz_env(); // ★ 15.09.2026 Klemmen S0b: Konstruktionszeit-Kopie, dieselbe Quelle wie die Emission (nur SRT, siehe Emission)
#else
	klemm_bilanz_on = false;
	if(klemm_bilanz_env()) print_warning("CFD_KLEMM_BILANZ ist an, das Messinstrument ist aber nur fuer SRT gebaut -- hier AUS (Ansage-Doktrin, Pruefpass S0b-2).");
#endif
	if(klemm_haken_env()>0u) print_warning("CFD_KLEMM_HAKEN="+to_string(klemm_haken_env())+": TESTARM -- "+string(klemm_haken_env()==2u ? "u-Klemme def_c = 0,05" : klemm_haken_env()==5u ? "Bildhuelle fuer Slot 300 verengt (Physik unveraendert)" : (klemm_haken_env()==4u ? "RHO_CLAMP 1,001/1,002 und Host-Wickelschranke 2^16 (Soll: MEHRDEUTIG)" : "RHO_CLAMP 1,001/1,002"))+string(klemm_haken_env()==3u&&klemm_bilanz_on ? " und rho-Klassenzaehlung fuer n%7==0 uebersprungen (Soll: Abnahme verletzt; im dd-Fall je Domaene eine Warnung)" : "")+". Die Physik dieses Laufs ist KEIN Ergebnis"+string(klemm_haken_env()==5u ? " -- AUSNAHME Haken 5: nur Zaehlerhuellen verengt, Physik unveraendert." : "."));
	if((klemm_haken_env()==3u||klemm_haken_env()==4u||klemm_haken_env()==5u)&&!klemm_bilanz_on) print_error("CFD_KLEMM_HAKEN="+to_string(klemm_haken_env())+" ist ein Negativtest des Messinstruments, das hier AUS ist (CFD_KLEMM_BILANZ=0 oder nicht SRT) -- er liefe still ins Leere (Pruefpass S0c-2 N-d).");
	if((klemm_haken_env()==1u||klemm_haken_env()==3u||klemm_haken_env()==4u)&&env_u("CFD_RHO_REK_PRUEF", 0u)>0u) print_error("CFD_KLEMM_HAKEN 1/3/4 mit CFD_RHO_REK_PRUEF: die Host-Rekonstruktionspruefung rechnet mit RHO_CLAMP 0,5/1,5 -- nicht kombinierbar (Pruefpass S0b).");
	u_klemme = u_klemme_env(); if(u_klemme>0u) print_info("CFD_U_KLEMME=1: u-Klemme als BETRAG |u|^2 <= c_s^2 (Z2d) statt je Komponente -- Slot 28 zaehlt die Betragshuelle.");
	positiv_modus = positiv_env(); // ★ 15.09.2026 Klemmen Stufe 1 P1a: Konstruktionszeit-Kopie, dieselbe Quelle wie die Emission (Sperren seit P1b VOR dem Kernelbau, Pruefbefund P1a NIEDRIG 2)
	positiv_haken = positiv_env()>0u ? positiv_haken_env() : 0u; positiv_facette = positiv_env()>0u ? positiv_facette_env() : 0u;
	if(positiv_modus==1u) print_info("CFD_POSITIV=1: Positivitaetsbegrenzer (Projektionsform) als MESSARM -- s wird an den Zaehlschritten t%"+to_string(zaehl_takt())+" == 2 gerechnet und gezaehlt, die Felder bleiben bitgleich.");
	if(positiv_modus==2u) print_info("CFD_POSITIV=2: Positivitaetsbegrenzer wird JEDEN Schritt geprueft und angewandt; gezaehlt wird an den Zaehlschritten t%"+to_string(zaehl_takt())+" == 2 (Stichprobe).");
	// rho_rand_on steht seit C2c VOR der rho-Allokation (allocate), nicht mehr hier.
	u_takt = s_u_takt;     // ★ TODO 2 Schritt 3: dito
	schale_paritaet = s_schale_paritaet; // Beweisarm: Kernel-alpha 0, Enqueue laeuft (read-once)
	schale_alpha = s_schale_alpha; // ★ P9c N2F-SCHALE: Konstruktionszeit-Kopie (read-once-Doktrin); die Kernel entstehen erst in alloc_schale (Indexlisten-Groesse steht erst nach dem Listenbau fest)

#ifdef FORCE_FIELD
	// FORK -- F-BBox: die Box wurde bereits im Konstruktor aufgeloest (sie muss vor device_defines()
	// feststehen). Hier wird sie nur noch benutzt.
	const ulong F_N = (ulong)fbnx*(ulong)fbny*(ulong)fbnz;
	if(F_N<(ulong)get_N()) print_info("F-BBox: F auf "+to_string(fbnx)+"x"+to_string(fbny)+"x"+to_string(fbnz)
		+" statt "+to_string((ulong)get_N())+" Zellen -> "+to_string((float)(((ulong)get_N()-F_N)*12ull)/1e9f,2u)+" GB gespart");
	// ★ FORK 03.09.2026, VERSUCHT UND ZURUECKGENOMMEN: F ohne Host-Spiegel anzulegen (Memory<float>(device,
	// F_N, 3u, false), fi-Bauform) spart am 4-mm-Nahfeld 1.832 MiB SYSTEM-RAM -- aber nur System-RAM, kein
	// VRAM. Der Versuch endete im Segfault (rc=139, Rauchtest xz_sparsam_cpu 03.09.), und zwar an zwei
	// Stellen, die BEIDE den Host-Spiegel brauchen: LBM::initialize() laedt F per enqueue_write_to_device
	// hoch (lbm.cpp, weiter unten), und der F-Waechter davor prueft die F-NUR-SOLID-Praemisse auf dem Host.
	// Sauber waere: bei !F_host stattdessen kernel_reset_force_field starten UND den Waechter mit Ansage
	// ueberspringen. Das sind zwei weitere Eingriffe fuer einen Posten, der auf der B70 kein einziges MB
	// VRAM bringt -- und der Waechter, den es kostet, ist ein Sicherheitsnetz. Zurueckgestellt.
	// ★ 03.09.2026 F-MARKERLISTE (CFD_F_LISTE). Der Maskenpuffer hat seine ENDGUELTIGE Groesse schon
	// hier -- sie haengt nur an der F-BBox, die der Konstruktor bereits aufgeloest hat. Nur F selbst
	// wird im Listenarm als Platzhalter angelegt und in alloc_f_liste ersetzt (fac_idx-Muster): erst
	// so faellt der Spitzenverbrauch, statt nur der Dauerverbrauch.
	// (f_liste_on und fac_idx_voll_on stehen jetzt im KONSTRUKTOR -- siehe die Begruendung dort.)
	f_maske = Memory<uint>(device, 2ull*((F_N+31ull)/32ull)+1ull);
	for(ulong i=0ull; i<f_maske.length(); i++) f_maske[i]=0u;
	f_maske[2ull*((F_N+31ull)/32ull)] = (uint)F_N; // Stride-Vorbelegung; im Vollfeld-Arm nie gelesen
	f_maske.write_to_device();
	if(f_liste_on) {
		F = Memory<float>(device, 1ull, 3u); // Platzhalter -- alloc_f_liste legt ihn in Slotgroesse neu an
		print_info("F-MARKERLISTE (CFD_F_LISTE, 03.09.): F wird nur fuer WANDsolidzellen alloziert. Bis alloc_f_liste laeuft, steht hier ein 1-Element-Platzhalter.");
	} else {
		F = Memory<float>(device, F_N, 3u);
	}
	object_sum = Memory<float>(device, 1u, 4u); // x, y, z, cell count
	f_param_sc = kernel_stream_collide.get_number_of_parameters();
	kernel_stream_collide.add_parameters(F, f_maske);
	f_param_uf = kernel_update_fields.get_number_of_parameters();
	kernel_update_fields.add_parameters(F, f_maske);
	kernel_update_force_field = Kernel(device, N, "update_force_field", fi, flags, t, F, f_maske, u, rho_clamp_hits); // ★ 2026-08-25 u + hits fuer den Bewegtwand-Term (Slot 59)
	kernel_reset_force_field = Kernel(device, N, "reset_force_field", F, f_maske, rho_clamp_hits);
	kernel_object_center_of_mass = Kernel(device, N, "object_center_of_mass", flags, (uchar)0u, object_sum);
	of_groups = 1024u; // ★ 2026-08-25 feste Gittergroesse statt N -- Determinismus, s. Kernel-Kommentar
	of_part = Memory<float>(device, 3ull*(ulong)of_groups);
	kernel_object_force = Kernel(device, (ulong)of_groups*(ulong)WORKGROUP_SIZE, "object_force", F, f_maske, flags, (uchar)0u, of_part);
	kernel_object_force_zband = Kernel(device, (ulong)of_groups*(ulong)WORKGROUP_SIZE, "object_force_zband", F, f_maske, flags, (uchar)0u, 0u, 0u, of_part); // FORK Kraft-Zerlegung: Aufrufe sequenziell
	kernel_object_force_final = Kernel(device, (ulong)WORKGROUP_SIZE, "object_force_final", of_part, of_groups, object_sum);
	kernel_object_torque = Kernel(device, N, "object_torque", F, f_maske, flags, (uchar)0u, 0.0f, 0.0f, 0.0f, object_sum);
#endif // FORCE_FIELD

#ifdef MOVING_BOUNDARIES
	kernel_update_moving_boundaries = Kernel(device, N, "update_moving_boundaries", u, flags);
#endif // MOVING_BOUNDARIES

#ifdef SURFACE
	phi = Memory<float>(device, N);
	mass = Memory<float>(device, N, 1u, false);
	massex = Memory<float>(device, N, 1u, false);
	kernel_initialize.add_parameters(mass, massex, phi);
	kernel_stream_collide.add_parameters(mass);
	kernel_surface_0 = Kernel(device, N, "surface_0", fi, rho, u, flags, mass, massex, phi, t, fx, fy, fz);
	kernel_surface_1 = Kernel(device, N, "surface_1", flags);
	kernel_surface_2 = Kernel(device, N, "surface_2", fi, rho, u, flags, t);
	kernel_surface_3 = Kernel(device, N, "surface_3", rho, flags, mass, massex, phi);
#endif // SURFACE

#ifdef TEMPERATURE
	gi = Memory<fpxx>(device, N, 7u, false);
	T = Memory<float>(device, N, 1u, true, true, 1.0f);
	kernel_initialize.add_parameters(gi, T);
	kernel_stream_collide.add_parameters(gi, T);
	kernel_update_fields.add_parameters(gi, T);
#endif // TEMPERATURE

#ifdef PARTICLES
	particles = Memory<float>(device, (ulong)particles_N, 3u);
	kernel_integrate_particles = Kernel(device, (ulong)particles_N, "integrate_particles", particles, u, flags, 1.0f);
#ifdef FORCE_FIELD
	kernel_integrate_particles.add_parameters(F, fx, fy, fz);
#endif // FORCE_FIELD
#endif // PARTICLES

	// ★ C1b Stufe 2 (F2): Facetten-Puffer als 1-Element-Platzhalter binden -- die echten Groessen
	// stehen erst nach Voxelisierung + baue_facetten() fest; bind_facetten() ersetzt sie per
	// set_parameters an fac_param_pos (Muster finalize_sparse_tiles). MUSS vor dem TS_P-Block stehen.
	if(facetten_on) {
		fac_ema_on = s_fac_imem&&s_fac_ema>0.0f;
		fac_geo   = Memory<float>(device, 8ull);
		fac_idx   = Memory<uint>(device, 2ull); // Bitmaske+Praefixsumme: ein Block (Maske 0 = keine Facette)
		fac_tau   = Memory<float>(device, 1ull);
		fac_tau_n = Memory<uint>(device, 1ull);
		fac_param_pos = kernel_stream_collide.get_number_of_parameters();
		kernel_stream_collide.add_parameters(fac_geo, fac_idx, fac_tau, fac_tau_n);
		if(fac_ema_on) { fac_us = Memory<float>(device, 3ull); kernel_stream_collide.add_parameters(fac_us); } // Signatur-Paritaet mit #ifdef FACETTEN_EMA
		fac_pema_on = s_fac_imem&&s_fac_pema>0.0f;
		if(fac_pema_on) { fac_pu = Memory<float>(device, 6ull); kernel_stream_collide.add_parameters(fac_pu); }
		fac_diagz_on = s_fac_imem&&s_fac_diagz>=0l; fac_diagz_wert = s_fac_diagz; // Gross-Audit: Konstruktionswert einfrieren -- alloc_facetten liest sonst die Statik der falschen Instanz
		if(fac_diagz_on) { fac_diag = Memory<float>(device, 19ull); kernel_stream_collide.add_parameters(fac_diag); } // 19: [17] alpha, [18] dp_ds
		fac_elibb_on = s_fac_imem&&s_fac_elibb; // ★ B2: Konstruktionszustand einfrieren (dieselbe Lektion wie diagz)
		if(fac_elibb_on) { fac_q = Memory<uchar>(device, 18ull); kernel_stream_collide.add_parameters(fac_q); } // Platzhalter; alloc_facetten_domain baut und rebindet
		fac_kdiag_on = s_fac_imem&&s_fac_kdiag>0u; // ★ Klassen-Diagnostik: Konstruktionszustand einfrieren (Signaturposition = nach fac_q)
		if(fac_kdiag_on) { fac_kd = Memory<float>(device, 16ull); kernel_stream_collide.add_parameters(fac_kd); } // 16 seit 05.09.: [12..15] vorzeichenbehafteter Druckrest A/|A|/B/C (12 seit 04.09.: [10]/[11] = tw und Besuche NUR ueber angewandte Besuche)
		nachbar_on = s_fac_imem&&s_fac_nachbar>0u; // ★ 03.09. deterministische Nachbarabtastung: Konstruktionszustand einfrieren (Emission haengt an derselben Statik; Signaturposition = nach fac_kd, VOR fac_wfd)
		apg_on = nachbar_on&&s_facetten&&s_fac_apg!=0.0f; apg_kappa = apg_on ? s_fac_apg : 0.0f; apg_haken = apg_on ? s_fac_apg_haken : 0u; apg_moz = apg_on ? s_fac_apg_moz : 0u; apg_moz_c = apg_moz ? s_fac_apg_c : 0.0f; apg_moz_ap0 = apg_moz ? s_fac_apg_ap0 : 0.0f; // ★ 22.09. MOZ wie kappa/haken an die INSTANZ gebunden (Bericht gatet auf den Instanzzustand)
		nb_stride = apg_on ? 5ull : 2ull;
		nb_roff = apg_on ? 5ull : 2ull;
		// ★ 23.09. Stufe A: drei Floats fuer die Nachbar-RICHTUNG, nur wenn die Rekonstruktion laeuft.
		// Die Formel MUSS zeichengleich zu der in device_defines() sein -- am 22.09. kostete genau diese
		// Divergenz einen 2,5-fachen Pufferueberlauf auf der GPU. Eigene Zeile, kein Kommentar davor.
		if(fac_rek_on) nb_stride += nb_rek_floats;
		timer_apg = apg_on ? s_timer_apg : 0u; // ★ 22.09.2026: timer_apg genau wie apg_haken an die INSTANZ gebunden -- das Fernfeld traegt kein APG und darf den Timer nicht tragen.
		// ★★ 22.09.2026, EIGENER FEHLER, vom Pruefagenten gefunden: der Kommentar oben stand zuerst MITTEN in
		// dieser Zeile, VOR der nb_stride-Zuweisung. Die Zuweisung lag damit vollstaendig im //-Kommentar,
		// nb_stride blieb auf dem Header-Default 2ull -- waehrend der JIT weiter def_nb_stride 5ul emittiert
		// (lbm.cpp, Emissionsblock). fac_nb waere mit 2*aktiv Floats alloziert worden, die Kernel schreiben und
		// lesen aber bei 5*gid: ab gid >= 0,4*aktiv jeder Zugriff AUSSERHALB des Puffers, 2,5-facher Ueberlauf
		// ohne Schranke auf der GPU. Das traf AUCH den Timer-AUS-Pfad, also jeden APG-Lauf. Genau die Falle,
		// die der verschluckte Kommentar selbst beschreibt, und genau die, die im Gedaechtnis steht
		// ("// mitten in einer Zeile frisst Code"). Deshalb steht hier jede Zuweisung auf einer EIGENEN Zeile. // ★ 16.09. HOCH-1 (Pruefagent): APG-Zustand je Instanz einfrieren -- fahrzeug_dd nullt die Statik VOR dem Bau von lbm_c, alloc_facetten/Launches/Bericht lasen sie danach (JIT-Stride 5 gegen Host-Stride 2 = stiller Ueberlauf, kein Kernel gebunden, kein Bericht). Bedingung = Emission von FACETTEN_APG und def_nb_stride.
		if(nachbar_on) { fac_nb = Memory<float>(device, 2ull); kernel_stream_collide.add_parameters(fac_nb); } // Platzhalter; alloc_facetten_domain baut und rebindet
		fdwand_on = s_sgs_fdwand>0u; // ★ Geistermoden-Fix: Konstruktionszustand einfrieren (Emission haengt an derselben Statik; Signaturposition = nach fac_kd)
		if(fdwand_on) { fac_wfd = Memory<float>(device, 1ull); kernel_stream_collide.add_parameters(fac_wfd); }
		if(band_on) { band_idx = Memory<uint>(device, 2ull); band_sbar = Memory<float>(device, 1ull); kernel_stream_collide.add_parameters(band_idx, band_sbar); } // ★ 08.09. SGS-BAND: Platzhalter NACH fac_wfd (Signaturposition); alloc_sgs_band baut und rebindet // Platzhalter; alloc_facetten_domain baut und rebindet -- der KOHAERENZ-WAECHTER dort verhindert, dass der Platzhalter je gelesen wird
		vandriest_on = fdwand_on&&s_sgs_vandriest>0u; vandriest_modus = s_sgs_vandriest; vandriest_ab = s_sgs_vd_ab; // ★ 08.09. van Driest: Konstruktionszustand einfrieren (Emission haengt an derselben Statik)
		sism_on = fdwand_on&&s_sgs_sism>0u; sism_T = s_sgs_sism_T; sism_ab = s_sgs_sism_ab; // ★ 07.09. SISM: Konstruktionszustand + T/ab einfrieren (Emission haengt an derselben Statik). Puffer fac_sb und die drei Kernelargumente entstehen in alloc_facetten_domain -- KEIN Platzhalter noetig, weil kernel_sgs_fdwand selbst erst dort gebaut wird
	}

	// FORK -- Block-Tiling: tile_slot ist per TS_P der LETZTE Parameter jedes fi-Kernels, muss also NACH
	// allen anderen add_parameters angehaengt werden. Bei ausgeschaltetem Sparse ist TS_P leer, dann darf
	// hier auch nichts gebunden werden -- sonst stimmt die Parameterzahl nicht mehr mit der Device-Seite
	// ueberein, und das ist genau die Fehlerklasse, die still falsche Ergebnisse produziert.
	if(sparse_on) {
		// Multi-GPU + Sparse ist nicht validiert: die transfer_*_fi-Kernel bekaemen tile_slot hier nicht
		// gebunden. Lieber laut abbrechen als still falsch rechnen.
		if(get_D()>1u) print_error("Block-Tiling (CFD_SPARSE_TILES) ist nur fuer eine einzelne GPU validiert, hier laufen "+to_string(get_D())+" Domaenen.");
		kernel_initialize.add_parameters(tile_slot);
		kernel_stream_collide.add_parameters(tile_slot);
		kernel_update_fields.add_parameters(tile_slot);
		kernel_boden_eq.add_parameters(tile_slot); // XL-Audit B1 (Pruefagent R2: NICHT unter FORCE_FIELD -- TS_P haengt nur an SPARSE_TILES)
		kernel_einlass_eq.add_parameters(tile_slot); // EINLASS_EQ: dieselbe Lektion (TS_P haengt NUR an SPARSE_TILES)
#ifdef FORCE_FIELD
		kernel_update_force_field.add_parameters(tile_slot);
#endif // FORCE_FIELD
	}

	if(get_D()>1u&&(fbnx!=Nx||fbny!=Ny||fbnz!=Nz)) print_error("F-BBox + Multi-GPU ist NICHT gebaut (transfer_F/graphics indizieren F voll-domaenig -- OOB)."); // ★ Tiefen-Audit B1: vorher pruefte er die read-once-GENULLTE Statik = toter Code; jetzt die aufgeloesten Member
	if(get_D()>1u) allocate_transfer(device);
}

void LBM_Domain::enqueue_apply_pressure_outlet() { // FORK: Druck-Auslass
	if(po_N_active==0u) return; // nicht konfiguriert -> nichts zu tun (kein Leerlauf-Dispatch)
	// Reihenfolge ist tragend: Teilsummen bilden, dann in Indexordnung zusammenfassen, dann anwenden.
	// Alle drei auf derselben in-order-Warteschlange, also ohne zusaetzliche Barriere. po_clear_mean
	// entfiel mit dem Umbau -- po_final_mean schreibt mit "=", und jeder Teilsummen-Slot wird jeden
	// Schritt ueberschrieben (2026-08-24).
	kernel_po_reduce_mean.enqueue_run();
	kernel_po_final_mean.enqueue_run();
	kernel_apply_pressure_outlet.enqueue_run();
}

void LBM_Domain::alloc_coupling_planes(const ulong max_plane_cells) { // FORK: Doppel-Domaene
	if(max_plane_cells==0ull) { print_error("alloc_coupling_planes mit 0 Zellen."); return; }
	coupling_max_plane_cells = max_plane_cells;
	coupling_plane = Memory<float>(device, max_plane_cells*4ull, 1u);
	// Ebenen-Parameter werden bei jedem Aufruf neu gesetzt; hier stehen Platzhalter, damit die
	// Kernel-Objekte ueberhaupt mit der richtigen Signatur entstehen.
	kernel_extract_plane_macros = Kernel(device, max_plane_cells, "extract_plane_macros",
		rho, u, coupling_plane, 0u, 0u, 0u, 0u, 1u, 1u);
	kernel_drive_boundary_cubic_lift = Kernel(device, max_plane_cells, "drive_boundary_cubic_lift",
		rho, u, flags, coupling_plane, 0u, 0u, 0u, 0u, 1u, 1u, 1u, 1u, 4u, rho_clamp_hits); // hits ANGEHAENGT (Slot 214, Betragstor auf u) -- set_parameters(4u, ...) bleibt davon unberuehrt
	// ★ Slice-Ebenen-Read 2026-08-26 (Hausmuster: Puffer anlegen und Kernel MIT echten Puffern
	// erzeugen -- kein Platzhalter-Bind-spaeter, die DIAGZ-Use-after-free-Klasse).
	slice_flags = Memory<uchar>(device, max_plane_cells, 1u);
	kernel_extract_plane_flags = Kernel(device, max_plane_cells, "extract_plane_flags",
		flags, slice_flags, 0u, 0u, 0u, 0u, 1u, 1u);
	print_info("Kopplungspuffer: "+to_string(max_plane_cells)+" Zellen a 4 floats = "
		+to_string((float)(max_plane_cells*16ull)/1048576.0f,2u)+" MB auf "+device.info.name+".");
}

// ★ 15.09.2026 RHO_RAND C1 (RHO_RAND-PLAN.md §5). Hausmuster: Puffer anlegen und Kernel MIT echten Puffern erzeugen.
void LBM_Domain::alloc_rho_rek(const ulong max_plane_cells) {
	if(max_plane_cells==0ull) { print_error("alloc_rho_rek mit 0 Zellen."); return; }
	if(rho_rand_on) { print_error("alloc_rho_rek unter RHO_RAND: der Pruefkernel braucht den vollen rho-Puffer (Pruefarm nur ohne RHO_RAND, C2-Plan §4.2)."); return; }
	if(rho_rek_max>=max_plane_cells) return; // schon gross genug
	if(rho_rek_max>0ull) { print_error("alloc_rho_rek: Vergroesserung eines gebundenen Puffers ist die Use-after-free-Klasse -- einmal gross genug anlegen."); return; }
	rho_rek_max = max_plane_cells;
	rho_rek_out  = Memory<float>(device, max_plane_cells*4ull, 1u);
	rho_rek_wort = Memory<rhoxx>(device, max_plane_cells, 1u);
	kernel_rho_rek_ebene = Kernel(device, max_plane_cells, "rho_rek_ebene", fi, rho, u, flags, t, rho_rek_out, rho_rek_wort, 0u, 0u, 0u, 0u, 1u, 1u, 0u, rho_clamp_hits); // ★ C2a: modus an Index 13, hits 14, tile_slot 15
	if(sparse_on) kernel_rho_rek_ebene.add_parameters(tile_slot); // Guard wie im Kernel (TS_P haengt an SPARSE_TILES, Falle 8)
	print_info("rho-Rekonstruktion (RHO_RAND C1): Puffer fuer "+to_string(max_plane_cells)+" Ebenenzellen = "
		+to_string((float)(max_plane_cells*(16ull+(ulong)sizeof(rhoxx)))/1.0e6f,2u)+" MB auf "+device.info.name+".");
}

// ★ P9c N2F-SCHALE (Muster alloc_coupling_planes): Puffer anlegen und die Kernel MIT ECHTEN
// Puffern erzeugen -- kein Platzhalter-Bind-später (die DIAGZ-Use-after-free-Klasse). MUSS nach
// finalize_sparse_tiles laufen (fi ist dann final gebunden; im dd-Fall hat das Grobgitter ohnehin
// kein Tiling, und das Setup ruft alloc erst nach run(0)).
void LBM_Domain::alloc_schale(const std::vector<ulong>& liste, const std::vector<float>& gewichte, const uint ratio, const uint modus, const bool blendet) {
	const ulong n = (ulong)liste.size();
	if(n==0ull) { print_error("alloc_schale mit leerer Liste."); return; }
	if(n>0x55555555ull) { print_error("alloc_schale: Liste ueberschreitet 2^32/3 Zellen -- die 3u*gid-Indexprodukte in schale_extract/schale_blend (kernel.cpp) wickeln in 32 Bit VOR der 2^32-Grenze (Kernel-Pruefer 2026-08-22 abends). Praktisch unerreichbar, aber der Guard deckt jetzt seine eigene Arithmetik."); return; }
	if(ratio==0u) { print_error("alloc_schale: ratio=0 (Blockmittel-Fenster waere leer)."); return; }
	if((ulong)gewichte.size()!=n) { print_error("alloc_schale: gewichte ("+to_string((ulong)gewichte.size())+") passt nicht zur Liste ("+to_string(n)+") -- der Kernel laese daneben."); return; }
	for(ulong i=0ull; i<n; i++) if(!(gewichte[i]>=0.0f&&gewichte[i]<=1.0f)) { print_error("alloc_schale: gewicht["+to_string(i)+"] = "+to_string(gewichte[i],6u)+" liegt nicht in [0;1] (NaN faengt die Negativform mit)."); return; }
	if(modus>2u) { print_error("alloc_schale: modus = "+to_string(modus)+" (gueltig: 0 EQ, 1 FNEQ, 2 IDENT-Debug)."); return; }
	schale_n = (uint)n;
	schale_modus = modus;
	schale_liste = Memory<uint>(device, n);
	if(get_N()>0xFFFFFFFFull) print_error("alloc_schale: Gitter ueberschreitet 2^32 Zellen -- schale_liste ist seit 08.09. uint (VRAM).");
	for(ulong i=0ull; i<n; i++) schale_liste[i] = (uint)liste[i];
	schale_liste.write_to_device();
	// ★ 11.09.2026 HOST-SPIEGEL FREIGEBEN (VRAM-Audit). Dieser Puffer wird EINMAL gefuellt,
	// hochgeladen und danach host-seitig nie wieder angefasst -- weder per read_from_device()
	// noch per Index. delete_host_buffer() ist seit 03.09. entschaerft (Aux-Zeiger, Double-Free,
	// Zero-Copy-Waechter), wurde aber nirgends gerufen.
	// DIE BEDINGUNG IST EINSEITIG UND DESHALB SICHER: is_zero_copy verlangt uses_ram, also
	// schliesst !uses_ram Zero-Copy aus. Auf der iGPU IST der Host-Puffer der Geraetespeicher,
	// dort wuerde die Freigabe die laufende Rechnung lautlos zerstoeren -- deshalb nur dGPU.
	if(!device.info.uses_ram) schale_liste.delete_host_buffer();
	// ★ 08.09. VRAM-Sparmassnahme 4: blendet=false (Nahfeld) legt Blend-Eingang und Gewichte als 1-Element-Dummy an.
	// Der Blend-Kernel wird trotzdem gebaut -- Signatur und Bindungsreihenfolge bleiben unveraendert; er wird im
	// Nahfeld nie enqueued (setup.cpp blendet nur lbm_c).
	if(!blendet&&schale_alpha!=0.0f) print_error("alloc_schale mit blendet=false auf einer Domaene mit alpha = "+to_string(schale_alpha,3u)+" != 0: der Blend-Kernel liest unear[3*gid+2] UNBEDINGT und laese weit ueber einen 1-Element-Puffer hinaus (Pruefagent-Befund B2, 08.09.).");
	schale_unear = Memory<float>(device, blendet ? 3ull*n : 1ull); // Blend-Eingang (Host-Upload); Ctor-Nullinit -> vor dem ersten Upload waere unear 0, deshalb macht das Setup einen 1-Outer-Vorlauf wie bei der Hinkopplung
	schale_uout  = Memory<float>(device, 3ull*n); // Extract-Ausgang (getrennt, damit der Waechter-Extract unear nicht ueberschreibt)
	schale_gewicht = Memory<float>(device, blendet ? n : 1ull); // Gradient-Blend: Zellgewichte (Lagen-Rampe), wirken als a = alpha*gewicht[gid]
	if(blendet) for(ulong i=0ull; i<n; i++) schale_gewicht[i] = gewichte[i];
	schale_gewicht.write_to_device();
	kernel_schale_extract = Kernel(device, n, "schale_extract", u, flags, schale_liste, (uint)n, ratio, 1u, schale_uout); // mittel (Pos. 5) je Enqueue
	kernel_schale_blend   = Kernel(device, n, "schale_blend", fi, flags, t, 0.0f, schale_liste, (uint)n, schale_unear, schale_gewicht, modus, rho_clamp_hits); // t/alpha (Pos. 2/3) je Enqueue; gewicht+modus VOR diag (Plan-Vorgabe)
	if(sparse_on) kernel_schale_blend.add_parameters(tile_slot); // TS_P haengt NUR an SPARSE_TILES (XL-Audit-B1-Lektion); der Blend laeuft zwar nur im Fernfeld (ohne Tiling), aber die Signatur muss zur Emission der Domaene passen
	print_info("N2F-Schale: "+to_string(n)+" Zellen a 2x3+1 floats + Indexliste = "
		+to_string((float)(n*(blendet?32ull:16ull))/1048576.0f,2u)+" MB auf "+device.info.name+" (alpha dieser Domaene: "+to_string(schale_alpha,3u)+", modus "+to_string(modus)+(modus==2u?" IDENT-Debug":modus==1u?" FNEQ":" EQ")+").");
}

void LBM_Domain::enqueue_schale_blend() { // ★ P9c: post-stream Schalen-Blend (nach einlass_eq)
	// No-Op-Doppelgate: schale_n==0 = nie alloziert; schale_alpha==0 = alloziert, aber nur als
	// Extract-Seite (lbm_f traegt eine Deckungspunkt-Liste, darf aber NIE blenden -- Slot-22-Soll nah==0).
	if(schale_n==0u||schale_alpha==0.0f) return;
	kernel_schale_blend.set_parameters(2u, t, schale_paritaet ? 0.0f : schale_alpha).enqueue_run(); // Paritaetsarm: alpha exakt 0, aber der Kernel LAEUFT (sonst waere der Beweis ein No-Op)
}

// ★ C1b Stufe 2: Facettendaten der Domaene bauen, hochladen, Kernel neu binden (FACETTEN-STUFE2.md F1/F2).
// Filtert klasse!=0 (markierte Zellen behalten reinen BB); fac_a = 1/|n_achse| host-berechnet (R2).
// ★ 29.08. (Heiko: "so dass wir wirklich wissen, wieviel Reserve wir immer noch haben und was
// wir wirklich nutzen"). device_info.memory ist eine REKONSTRUKTION: opencl.hpp:170 rechnet NEOs
// 95-%-Deckel mit 20/19 heraus, der Treiber meldet weniger. Und der Desktop haengt an derselben
// Karte, ohne dass memory_used davon etwas sieht. Der einzige belastbare Wert steht im
// DRM-Debugfs.
//
// ★ 12.09.2026 (Heiko: "den echten freien VRAM messen"): ZWEITER WEG, weil der erste an den
// Rechten scheitert. /sys/kernel/debug ist root-only, sudo ohne Passwort gibt es hier nicht --
// der Debugfs-Weg hat auf dieser Maschine noch NIE einen Wert geliefert, jedes Log sagt
// "gemessener Frei-Wert NICHT lesbar". Der zweite Weg ist /proc/<pid>/fdinfo/<fd> des
// DRM-Geraets: der xe-Treiber schreibt dort "drm-total-vram0: N KiB" je DRM-Client, und das
// ist fuer alle Prozesse DESSELBEN Benutzers lesbar -- also auch fuer gnome-shell und den Editor.
//
// ★★ DAS WIDERSPRICHT EINER FRUEHEREN NOTIZ, und der Widerspruch gehoert benannt statt aufgeloest:
// hier stand bis heute "NICHT ueber /proc/*/fdinfo -- der unterzaehlt grob (23.08.: Faktor sieben)".
// Am 12.09.2026 gemessen, waehrend ein 8-mm-Lauf lief: fdinfo meldet fuer FluidX3D 3265 MiB, die
// Rekonstruktion im selben Lauf 3162 MB (= 3015 MiB). fdinfo zaehlt also MEHR, nicht sieben Mal
// weniger. Erklaerbar ist das damit, dass die Karte seit dem Treiberwechsel unter xe laeuft
// (/dev/dri/renderD129, pdev 0000:04:00.0) und die vram0-Zeilen dort ueberhaupt erst existieren;
// unter i915 gibt es sie nicht, und genau das war am 23.08. vermutlich der Fall. NACHGEPRUEFT ist
// das NICHT -- es ist die plausible Erklaerung, nicht der Beweis. Wer die alte Zahl reproduzieren
// will, braucht den Treiberstand von damals.
//
// ZWEI FALLEN, beide beim Bau getroffen:
//  1. Mehrere Dateideskriptoren desselben Clients tragen DIESELBE Zahl. Ohne Deduplizierung ueber
//     drm-client-id kam gnome-shell fuenfmal vor und die Summe war mehr als doppelt so gross.
//  2. Die Kapazitaet aus device.info.memory ist selbst die 20/19-Rekonstruktion. Die Differenz
//     "Kapazitaet minus gemessene Summe" traegt deren Fehler also mit -- sie ist trotzdem die
//     bessere Zahl, weil sie den Desktop-Anteil (am 12.09. 1331 MiB) ueberhaupt erst sieht.
// Liefert freie MiB, oder 0 wenn keiner der beiden Wege trug.
#include <filesystem>
#include <sstream>
#include <set>
// ★ 12.09.2026: WELCHER WEG getragen hat, steht hier -- die Meldungen behaupteten sonst
// "DRM-Debugfs", auch wenn der Wert aus fdinfo kam. Im ersten Selbsttest genau so passiert.
static const char* g_vram_quelle = "keiner";
const char* vram_quelle() { return g_vram_quelle; }
ulong vram_frei_gemessen(const ulong kapazitaet_mib) {
	g_vram_quelle = "keiner";
	for(const string& pfad : {string("/sys/kernel/debug/dri/0/tile0/vram_mm"), string("/sys/kernel/debug/dri/0/i915_gem_objects")}) {
		std::ifstream f(pfad);
		if(!f) continue;
		string z;
		while(std::getline(f, z)) { // Zeile der Form "free: 12345678 KiB" bzw. "...: N B"
			const size_t p_ = z.find("free");
			if(p_==string::npos) continue;
			ulong wert=0ull; bool ziffer=false;
			for(size_t i=p_; i<z.size(); i++) {
				if(z[i]>='0'&&z[i]<='9') { wert = wert*10ull + (ulong)(z[i]-'0'); ziffer=true; }
				else if(ziffer) break;
			}
			if(!ziffer) continue;
			g_vram_quelle = "DRM-Debugfs";
			if(z.find("KiB")!=string::npos||z.find("kB")!=string::npos) return wert/1024ull;
			if(z.find("MiB")!=string::npos||z.find("MB")!=string::npos) return wert;
			return wert/1048576ull; // Bytes
		}
	}
	// ── ZWEITER WEG: Summe ueber alle DRM-Clients derselben Karte, aus /proc/*/fdinfo ──────────
	if(kapazitaet_mib==0ull) return 0ull; // ohne Kapazitaet laesst sich aus einer Belegung kein Freiwert bilden
	string pdev; // die PCI-Adresse UNSERER Karte, aus den eigenen Deskriptoren
	auto feld = [](const string& t, const string& schluessel) -> string {
		const size_t a = t.find(schluessel); if(a==string::npos) return string();
		const size_t z = t.find('\n', a); const string zeile = t.substr(a+schluessel.length(), z-a-schluessel.length());
		size_t b=0ull; while(b<zeile.length()&&(zeile[b]==' '||zeile[b]=='\t')) b++;
		size_t e=b; while(e<zeile.length()&&zeile[e]!=' '&&zeile[e]!='\t'&&zeile[e]!='\r') e++;
		return zeile.substr(b, e-b);
	};
	auto lies = [](const string& pfad) -> string {
		std::ifstream f(pfad); if(!f) return string();
		std::stringstream ss; ss << f.rdbuf(); return ss.str();
	};
	for(const auto& e : std::filesystem::directory_iterator("/proc/self/fdinfo", std::filesystem::directory_options::skip_permission_denied)) {
		const string t = lies(e.path().string());
		if(t.find("drm-total-vram0:")==string::npos) continue;
		pdev = feld(t, "drm-pdev:"); if(!pdev.empty()) break;
	}
	if(pdev.empty()) return 0ull; // keine Karte mit vram0-Zeilen -- iGPU-Lauf oder alter Treiber
	std::set<string> gesehen; ulong summe_kib = 0ull;
	std::error_code ec;
	for(const auto& pe : std::filesystem::directory_iterator("/proc", std::filesystem::directory_options::skip_permission_denied, ec)) {
		const string pid = pe.path().filename().string();
		if(pid.empty()||pid[0]<'0'||pid[0]>'9') continue;
		std::error_code ec2;
		for(const auto& fe : std::filesystem::directory_iterator(pe.path()/"fdinfo", std::filesystem::directory_options::skip_permission_denied, ec2)) {
			const string t = lies(fe.path().string());
			if(t.find("drm-total-vram0:")==string::npos||t.find(pdev)==string::npos) continue;
			const string cid = feld(t, "drm-client-id:");
			if(cid.empty()||!gesehen.insert(cid).second) continue; // je Client EINMAL (Falle 1)
			summe_kib += (ulong)atoll(feld(t, "drm-total-vram0:").c_str());
		}
		if(ec2) continue;
	}
	if(summe_kib==0ull) return 0ull;
	const ulong belegt_mib = summe_kib/1024ull;
	g_vram_quelle = "Summe ueber alle DRM-Clients (/proc/*/fdinfo)";
	return kapazitaet_mib>belegt_mib ? kapazitaet_mib-belegt_mib : 0ull;

	return 0ull;
}

// ★ 03.09.2026 F-MARKERLISTE -- Maske bauen, F in Slotgroesse neu anlegen, alle Leser rebinden.
// Wird aus dem Setup gerufen, NACHDEM die Geometrie steht und BEVOR initialize() laeuft (dasselbe
// Fenster wie alloc_facetten_domain). Das Praedikat ist wortgleich zu update_force_field: solid UND
// mindestens ein Nicht-Solid unter den 18 D3Q19-Links.
// DIE MASKE IST ABSICHTLICH EINE OBERMENGE. Der Host wickelt x/y periodisch und ueberspringt z
// ausserhalb des Gitters, der Kernel wickelt neighbors() in allen Richtungen. Eine Zelle, die der
// Kernel schreibt und der Host nicht in der Maske haette, waere ein STILLER Kraftverlust -- deshalb
// nimmt der Host im Zweifel MEHR auf: jede Solidzelle, die in IRGENDEINER Auslegung einen
// Nicht-Solid-Nachbarn haette, bekommt einen Slot. Ein paar ungenutzte Slots kosten nichts.
// Die Gegenrichtung faengt der Wirkpfad-Zaehler Slot 77 (store3_F ohne Slot), Soll 0 am Laufende.
// ★★ 08.09.2026 SGS-BAND: Zellenliste der Wandlagen 2..lagen, DISJUNKT zur Facettenmenge.
// Die Lagen entstehen EINMAL hier auf dem Host per 6-Flaechen-Dilatation ab der Facettenmenge --
// der Kernel bekommt keine Lagendefinition, sondern eine Liste. Damit gilt "Lage 1 = Facettenmenge"
// per Konstruktion, und die drei widerspruechlichen Lage-1-Begriffe des Codes (fdw_fid, 18-Link-Solid,
// 6-Flaechen-Dilatation) koennen nicht mehr auseinanderlaufen. Voraussetzung: fac_idx ist gebaut
// (die Facettenmaske wird hier als Host-Spiegel gelesen).
void LBM_Domain::alloc_sgs_band(const uchar* flags_host, const uint Nx, const uint Ny, const uint Nz, const uint lagen) {
	if(!band_on||lagen<2u) return;
	if(fac_idx.length()<2ull) { print_error("alloc_sgs_band vor alloc_facetten_domain -- die Facettenmaske fac_idx ist noch nicht gebaut, die Bandlagen haetten keinen Anker."); return; }
	if((ulong)Nx*(ulong)Ny*(ulong)Nz>0xFFFFFFFFull) { print_error("alloc_sgs_band: Gitter ueberschreitet 2^32 Zellen -- band_zellen traegt den globalen Zellindex n als uint (seit 22.09., B32)."); return; }
	const ulong FN=(ulong)fbnx*(ulong)fbny*(ulong)fbnz, FNB=(FN+31ull)/32ull;
	// Zwei Bitfelder: "schon vergeben" (Facette oder eine fruehere Lage) und "aktuelle Front".
	std::vector<uint> vergeben((size_t)FNB,0u), front((size_t)FNB,0u), neu_((size_t)FNB,0u);
	auto bit=[&](std::vector<uint>& v, const ulong fbi){ return (v[(size_t)(fbi>>5)]>>(uint)(fbi&31ull))&1u; };
	auto setz=[&](std::vector<uint>& v, const ulong fbi){ v[(size_t)(fbi>>5)] |= 1u<<(uint)(fbi&31ull); };
	ulong n_fac=0ull;
	for(ulong b=0ull; b<FNB; b++) { const uint m=fac_idx[2ull*b]; vergeben[(size_t)b]=m; front[(size_t)b]=m; n_fac+=(ulong)__builtin_popcount(m); }
	static const int FZ6[6][3]={{1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1}};
	std::vector<ulong> liste; std::vector<uint> lage_von; liste.reserve((size_t)(2ull*n_fac));
	for(uint L=2u; L<=lagen; L++) {
		for(size_t b=0; b<(size_t)FNB; b++) neu_[b]=0u;
		ulong n_L=0ull;
		for(uint zb=0u; zb<fbnz; zb++) for(uint yb=0u; yb<fbny; yb++) for(uint xb=0u; xb<fbnx; xb++) {
			const ulong fbi=(ulong)xb+((ulong)yb+(ulong)zb*(ulong)fbny)*(ulong)fbnx;
			if(!bit(front,fbi)) continue; // nur von der aktuellen Front aus dilatieren
			for(uint i=0u; i<6u; i++) {
				const int zn=(int)(fbz0+zb)+FZ6[i][2]; if(zn<0||zn>=(int)Nz) continue; // z NICHT wickeln (Muster alloc_f_liste)
				const uint xn=(uint)((((int)(fbx0+xb)+FZ6[i][0])%(int)Nx+(int)Nx)%(int)Nx);
				const uint yn=(uint)((((int)(fby0+yb)+FZ6[i][1])%(int)Ny+(int)Ny)%(int)Ny);
				if(xn<fbx0||xn>=fbx0+fbnx||yn<fby0||yn>=fby0+fbny||(uint)zn<fbz0||(uint)zn>=fbz0+fbnz) continue; // ausserhalb der F-BBox: keine Bandzelle
				const ulong nn=(ulong)xn+((ulong)yn+(ulong)zn*(ulong)Ny)*(ulong)Nx;
				if((flags_host[nn]&(TYPE_S|TYPE_E))!=0u) continue; // nur echtes Fluid
				const ulong fbn=(ulong)(xn-fbx0)+((ulong)(yn-fby0)+(ulong)((uint)zn-fbz0)*(ulong)fbny)*(ulong)fbnx;
				if(bit(vergeben,fbn)||bit(neu_,fbn)) continue; // schon Facette oder fruehere/gleiche Lage
				setz(neu_,fbn); liste.push_back(fbn); lage_von.push_back(L); n_L++;
			}
		}
		band_n_lage[L<8u?L:7u]=n_L;
		for(size_t b=0; b<(size_t)FNB; b++) { vergeben[b]|=neu_[b]; front[b]=neu_[b]; }
		if(n_L==0ull) { print_warning("alloc_sgs_band: Lage "+to_string(L)+" ist LEER -- die Dilatation findet keine weiteren Fluidzellen (Geometrie zu duenn?)."); break; }
	}
	band_N=(ulong)liste.size();
	if(band_N==0ull) { print_error("alloc_sgs_band: Bandliste leer, obwohl CFD_SGS_BAND >= 2 -- stiller No-Op."); band_on=false; return; }
	// ★ 11.09.2026 SPEICHERWAECHTER (VRAM-Audit, Befund G3). Diese Funktion pruefte den freien
	// Speicher GAR NICHT, waehrend die unmittelbar benachbarte alloc_facetten_domain es tut.
	// Die Bandpuffer fielen damit ungeprueft NACH dem Facettenwaechter -- bei 4 mm sind das
	// 118,8 MB, die in keinem Waechter und in keinem Reserveposten standen. Wortgleich zum
	// Vorbild aufgebaut, damit beide Meldungen gleich zu lesen sind.
	{	const ulong mb_band = (8ull*FNB + (band_pi_on ? 4ull*band_N+4ull : 8ull*band_N) + (sism_on ? 24ull*band_N : 4ull) + (gdiag_on ? 32ull*band_N : 0ull)) / 1048576ull; // ★ 22.09. + band_gd (8 float je Bandzelle) unter CFD_SGS_GDIAG
		const ulong belegt = (ulong)device.info.memory_used, kapazitaet = (ulong)device.info.memory;
		const ulong frei_gemessen = vram_frei_gemessen(kapazitaet);
		const ulong frei = frei_gemessen>0ull ? frei_gemessen : (kapazitaet>belegt ? kapazitaet-belegt : 0ull);
		const ulong mindest = (ulong)env_u("CFD_VRAM_MIN_FREI_MB", 1024u);
		print_info("SGS-BAND: "+to_string(band_N)+" Bandzellen, Puffer "+to_string(mb_band)+" MB | belegt "
			+to_string(belegt)+" MB von "+to_string(kapazitaet)+" MB"
			+(frei_gemessen>0ull ? string(", GEMESSEN frei "+to_string(frei_gemessen)+" MB")
			                     : string(", gemessener Frei-Wert NICHT lesbar")));
		if(!device.info.uses_ram && mb_band+mindest > frei)
			print_error("SGS-Bandpuffer passen nicht: "+to_string(mb_band)+" MB noetig, "+to_string(frei)
				+" MB frei, Mindestluft "+to_string(mindest)+" MB (CFD_VRAM_MIN_FREI_MB). "
				+to_string(band_N)+" Bandzellen. Lagenzahl senken (CFD_SGS_BAND) oder die Mindestluft bewusst herabsetzen.");
	}
	// Maske+Praefixsumme wie fac_idx: der Kernel rechnet aus fbi den Listenindex, ohne ein Feld je Zelle.
	band_idx=Memory<uint>(device,2ull*FNB);
	for(ulong i=0ull; i<2ull*FNB; i++) band_idx[i]=0u;
	for(const ulong fbi : liste) band_idx[2ull*(fbi>>5)] |= 1u<<(uint)(fbi&31ull);
	{	ulong lauf=0ull;
		for(ulong b=0ull; b<FNB; b++) { band_idx[2ull*b+1ull]=(uint)lauf; lauf+=(ulong)__builtin_popcount(band_idx[2ull*b]); }
		if(lauf!=band_N) { print_error("alloc_sgs_band: Praefixsumme "+to_string(lauf)+" != Bandzellen "+to_string(band_N)+" -- doppelte fbi in der Liste (die Nummerierung im Kernel waere verschoben)."); band_on=false; return; } }
	band_idx.write_to_device();
	// Die Liste MUSS in derselben Reihenfolge stehen, in der der Kernel sie aus der Maske nummeriert
	// (Scan ueber fbi aufsteigend), sonst zeigt band_sbar[bid] auf die falsche Zelle.
	std::vector<ulong> sortiert(liste); std::sort(sortiert.begin(), sortiert.end());
	if(gdiag_on) { // ★ 22.09. Band-g-Diagnose: Lage je Bandzelle in fbi-Sortierreihenfolge (bisher ging lage_von beim Sortieren verloren); nur unter GDIAG (Pruefbefund N3)
		std::vector<std::pair<ulong,uint>> paare(liste.size()); for(size_t i=0; i<liste.size(); i++) paare[i]=std::make_pair(liste[i], lage_von[i]);
		std::sort(paare.begin(), paare.end()); band_lage_h.assign(paare.size(), 0u);
		for(size_t i=0; i<paare.size(); i++) { if(paare[i].first!=sortiert[i]) { print_error("alloc_sgs_band: Lagenzuordnung und Sortierung laufen auseinander (Index "+to_string((ulong)i)+")."); band_on=false; return; } band_lage_h[i]=(uchar)std::min(255u, paare[i].second); }
	}
	// ★★ 22.09.2026 DEFEKT BEHOBEN (Tagesprotokoll B32, seit 2330bd5 am 08.09.): hier stand `band_zellen[i]=(uint)sortiert[i]`,
	// also der F-BBOX-Index fbi. Der Kernel sgs_fdwand liest gd_zellen[gid] aber als GLOBALEN Zellindex n und ruft
	// neighbors(n, j) -- genau wie fuer die Facettenliste, die n traegt (gd_zellen[k]=f.n). Am Fahrzeug (F-BBox 564x239x155 im
	// Gitter 961x349x249) rechnete der Bandkernel damit Sbar an Zellen mit n < 20,9 M (z-Index 0..62, Strasse/Raeder) und lieferte
	// es ueber band_sbar[gid] an die echten Bandzellen (Dach z ~150). Im Kanal ist die F-BBox das ganze Gitter (fbi == n), deshalb
	// bestand jeder Rauchtest. Alle Band-Befunde seit 08.09. sind als Bandmessungen ungueltig. Die REIHENFOLGE bleibt die der
	// fbi-Sortierung (= bid-Nummerierung der Maske band_idx in stream_collide); gespeichert wird je Position der globale Index n.
	band_zellen=Memory<uint>(device,band_N);
	{	ulong n_verletzt=0ull, n_ausserhalb=0ull;
		for(ulong i=0ull; i<band_N; i++) {
			const ulong fbi=sortiert[i];
			const uint xb=(uint)(fbi%(ulong)fbnx), yb=(uint)((fbi/(ulong)fbnx)%(ulong)fbny), zb=(uint)(fbi/((ulong)fbnx*(ulong)fbny));
			const uint x=fbx0+xb, y=fby0+yb, z=fbz0+zb;
			if(x>=Nx||y>=Ny||z>=Nz) { n_ausserhalb++; band_zellen[i]=0u; continue; }
			const ulong n=(ulong)x+((ulong)y+(ulong)z*(ulong)Ny)*(ulong)Nx;
			if((flags_host[n]&(TYPE_S|TYPE_E))!=0u) n_verletzt++; // SELBSTPRUEFUNG (Iron Rule 3): jede Bandzelle muss echtes Fluid sein -- mit dem fbi-Fehler waeren es Strassen-/Radzellen gewesen
			band_zellen[i]=(uint)n;
		}
		if(n_ausserhalb>0ull||n_verletzt>0ull) { print_error("alloc_sgs_band: Bandliste nach fbi->n-Umrechnung fehlerhaft -- "+to_string(n_ausserhalb)+" Zellen ausserhalb des Gitters, "+to_string(n_verletzt)+" Solid/E-Zellen von "+to_string(band_N)+" (Soll 0/0). F-BBox-Ursprung "+to_string(fbx0)+"/"+to_string(fby0)+"/"+to_string(fbz0)+", Groesse "+to_string(fbnx)+"x"+to_string(fbny)+"x"+to_string(fbnz)+"."); band_on=false; return; }
		print_info("SGS-BAND Liste: "+to_string(band_N)+" Bandzellen fbi -> n umgerechnet (F-BBox-Ursprung "+to_string(fbx0)+"/"+to_string(fby0)+"/"+to_string(fbz0)+", "+to_string(fbnx)+"x"+to_string(fbny)+"x"+to_string(fbnz)+" im Gitter "+to_string(Nx)+"x"+to_string(Ny)+"x"+to_string(Nz)+"); Selbstpruefung: 0 Solid/E-Zellen, 0 ausserhalb"+(fbx0==0u&&fby0==0u&&fbz0==0u&&fbnx==Nx&&fbny==Ny ? string(" -- F-BBox = Gitter, fbi == n (dieser Fall haette den Defekt B32 NICHT gezeigt)") : string(" -- F-BBox ist eine Teilbox, fbi != n (hier wirkte der Defekt B32)"))+".");
	}
	band_zellen.write_to_device();
	if(band_pi_jit!=band_pi_on) { print_error("alloc_sgs_band: KOHAERENZ Host/Kernel verletzt -- Kernel-Text "+string(band_pi_jit?"MIT":"OHNE")+" SGS_BAND_PI, Host band_pi_on = "+to_string(band_pi_on?1u:0u)+". Genau das war Pruefbefund H1 (22.09.): Host bindet band_sbar (N float), der Kernel schreibt 6 float je Bandzelle."); band_on=false; return; } // ★ 22.09. Pruefbefund A-M2
	band_sbar=Memory<float>(device, band_pi_on ? 1ull : band_N); // Start 0 = kein Abzug; im Pi-Modus ungenutzt (Pruefbefund A-N4: 4*band_N B toter Speicher, 4 mm ~10 MiB) -> Platzhalter
	band_sbar.write_to_device();
	band_sb=Memory<float>(device, sism_on ? 6ull*band_N : 1ull);
	band_sb.write_to_device();
	// ZWEITES Kernel-Objekt aus DEMSELBEN Programmtext "sgs_fdwand" -- kein neuer Kernel, keine neue
	// Signatur, damit auch keine R()-Klammerfalle. Der Kernel kennt nur "Liste rein, w raus"; welche
	// Zellen in der Liste stehen, entscheidet allein der Host.
	// ★ Rebind NACH dem Move-Assignment der echten Puffer (fac_wfd-Lehre: der Rebind VOR dem Neubau band
	// den gleich darauf zerstoerten Platzhalter, CL -52 beim ersten Enqueue). Die Position stammt aus
	// alloc_facetten_domain, wo alle Schalter im Scope sind.
	if(band_param_pos==0u) { print_error("alloc_sgs_band: band_param_pos ist 0 -- alloc_facetten_domain lief nicht oder FDWAND war aus. Das Band braucht den FDWAND-Zweig in stream_collide."); band_on=false; return; }
	if(band_pi_on) { // ★ 22.09. Plan C: stream_collide bekommt den EMA-ZUSTAND band_sb (6 float je Bandzelle) an der Sbar-Position; kein FD-Bandkernel
		if(!sism_on||band_sb.length()<6ull*band_N) { print_error("alloc_sgs_band: Pi-Modus braucht den SISM-EMA-Puffer band_sb (6 float je Bandzelle) -- CFD_SGS_SISM=1 fehlt oder Puffer zu klein."); band_on=false; return; }
		kernel_stream_collide.set_parameters(band_param_pos, band_idx, band_sb);
	} else {
	kernel_stream_collide.set_parameters(band_param_pos, band_idx, band_sbar);
	kernel_sgs_band = Kernel(device, band_N, "sgs_fdwand", u, flags, band_zellen, (uint)band_N, band_sbar);
	if(sism_on) kernel_sgs_band.add_parameters(t, band_sb, rho_clamp_hits, 1u); // sbar_out = 1: der Bandkernel liefert Sbar, kein w
	if(sparse_on) kernel_sgs_band.add_parameters(tile_slot); // ★ 22.09. Pruefbefund M1: TS_P ist der letzte Parameter von sgs_fdwand -- der Lage-1-Kernel bekam ihn (B-7-Lehre), der Bandkernel nicht; mit CFD_SPARSE_TILES waere der Band-Launch mit CL_INVALID_KERNEL_ARGS gestorben (bisher nie kombiniert)
	}
	if(gdiag_on) { // ★ 22.09.2026 BAND-g-DIAGNOSE: zweite Instanz desselben Kernels ueber die Bandliste (Liste traegt n, s. o.). Kein Kernel-, kein JIT-Text geaendert.
		band_gd = Memory<float>(device, 8ull*band_N);
		for(ulong q8=0ull; q8<8ull*band_N; q8++) band_gd[q8]=0.0f;
		band_gd.write_to_device();
		kernel_band_gdiag = Kernel(device, band_N, "sgs_gdiag", fi, u, flags, band_zellen, (uint)band_N, band_gd, t, fx, fy, fz, s_sgs_guo?1u:0u);
		if(sparse_on) kernel_band_gdiag.add_parameters(tile_slot); // TS_P zuletzt, wie bei kernel_sgs_gdiag (B-7)
		band_gdiag_on = true;
		print_info("BAND-g-DIAGNOSE (CFD_SGS_GDIAG x CFD_SGS_BAND): "+to_string(band_N)+" Bandzellen, "+to_string((ulong)(32ull*band_N/1048576ull))+" MB -- misst |S|_FD, |S|_Pi, D_WALE, D_Sigma, |Omega| je Bandzelle (Lage 2.."+to_string(lagen)+"); Physik unangetastet.");
	}
	string je; for(uint L=2u; L<=lagen&&L<8u; L++) je += (L>2u?" + ":"")+to_string(band_n_lage[L])+" (Lage "+to_string(L)+")";
	print_info("SGS-BAND gebunden: "+to_string(band_N)+" Bandzellen = "+je+"; Speicher "
		+to_string((float)(band_N*(sism_on?32ull:8ull)+2ull*FNB*4ull)/1048576.0f,1u)+" MB (Liste+w+EMA+Maske) auf "+device.info.name
		+". Lage 1 (Facetten, "+to_string(n_fac)+" Zellen) bleibt UNVERAENDERT -- eigene Puffer, eigener Launch.");
}

// ★ 15.09.2026 RHO_RAND, Commit C0 (RHO_RAND-PLAN.md §7/§9). Reine HOST-Pruefung: kein Kernel, kein
// Geraetepuffer, keine Aenderung am Rechenweg. Sie beantwortet vor dem Umbau zwei Fragen am echten Gitter:
//  (1) WAECHTER: Liegt jede Zelle, deren rho-Puffer nach dem Umbau noch gelesen wird, in der Randschale
//      R1 (Dicke 2 an allen sechs Flaechen)? Das sind die TYPE_E-Zellen (stream_collide, Lift, Verify),
//      die po_interior-Zellen (po_reduce_mean, apply_pressure_outlet) und die vi_interior-Zellen
//      (apply_velocity_inlet). po/vi muessen zusaetzlich in der Schreibmaske x >= Nx-2 liegen -- das ist
//      AUSDRUCKSGLEICH die RHO_SPARSAM-Maske in stream_collide (kernel.cpp: (n%Nx)+2 >= Nx).
//  (2) ZENSUS fuer die APG-Linie: welche Zellen liest APG (alle 18 D3Q19-Nachbarn j[ia] einer
//      Facettenzelle, uebersprungen wird nur reines TYPE_S, kernel.cpp apply_facette_imem), und wie viele
//      davon liegen NICHT in Lage 1 + Lage 2 (Heikos Vorschlag "rho-Region auf den SGS-Bandzellen")?
//      Lage 2 wird hier UNABHAENGIG von CFD_SGS_BAND nach genau den Regeln von alloc_sgs_band gebildet
//      (6-Flaechen-Dilatation, nur Fluid, F-BBox-beschnitten, x/y gewickelt, z nicht) -- ist das Band
//      gebaut, muss |L2| mit band_n_lage[2] uebereinstimmen (Ist=Soll).
// Beanstandungen werden gesammelt (print_warning) und als Zahl zurueckgegeben; abgebrochen wird im Setup
// EINMAL am Ende, damit man alle Befunde auf einmal sieht (Muster Kopplungspruefung).
// testhaken: zaehlt eine kuenstliche TYPE_E-Zelle in der Domaenenmitte als Innenzelle -- der Negativtest,
// der beweist, dass Waechter (1) ueberhaupt ausloesen kann. Die Flags werden dabei NICHT veraendert.
uint LBM_Domain::pruefe_rho_rand_c0(const uchar* flags_host, const uint Nx, const uint Ny, const uint Nz, const bool testhaken) {
	uint bad = 0u;
	const ulong NxNy = (ulong)Nx*(ulong)Ny, NN = NxNy*(ulong)Nz;
	auto in_r1 = [&](const ulong n) {
		const uint x = (uint)(n%(ulong)Nx), y = (uint)((n/(ulong)Nx)%(ulong)Ny), z = (uint)(n/NxNy);
		return x<2u||x+2u>=Nx||y<2u||y+2u>=Ny||z<2u||z+2u>=Nz;
	};
	auto in_maske = [&](const ulong n) { return (uint)(n%(ulong)Nx)+2u>=Nx; };
	const ulong r1_N = (Nx>4u&&Ny>4u&&Nz>4u) ? NN-(ulong)(Nx-4u)*(ulong)(Ny-4u)*(ulong)(Nz-4u) : NN;

	// ---- (0) SELBSTTEST der beiden Praedikate an konstruierten Zellen (Pruefbefund M1: der Testhaken allein
	// bewies nur den Meldeweg, nicht die Klassifizierung). Mitte der y/z-Ebene, x an beiden Seiten der Grenze.
	if(Nx>=6u&&Ny>=6u&&Nz>=6u) {
		const ulong yz = ((ulong)(Ny/2u)+(ulong)(Nz/2u)*(ulong)Ny)*(ulong)Nx;
		const bool ok = in_r1(yz+1ull) && !in_r1(yz+2ull) && !in_r1(yz+(ulong)(Nx-3u)) && in_r1(yz+(ulong)(Nx-2u))
		             && !in_maske(yz+(ulong)(Nx-3u)) && in_maske(yz+(ulong)(Nx-2u)) && !in_maske(yz+1ull);
		if(!ok) { print_warning("RHO_RAND-SELBSTTEST: in_r1/in_maske klassifizieren die Grenzzellen x=1/2 bzw. x=Nx-3/Nx-2 falsch -- alle Waechter unten waeren wertlos."); bad++; }
	}
	// ---- (1a) TYPE_E-Zellen. Die Schleife laeuft ueber Koordinaten (keine Division je Zelle) und zaehlt
	// nebenbei die R1-Zellen -- das ist der Ist=Soll-Beleg des Koordinatentests gegen die geschlossene Formel.
	ulong n_e = 0ull, n_e_innen = 0ull, erste_innen = 0xFFFFFFFFFFFFFFFFull, n_r1_gezaehlt = 0ull, n_e_praedikat = 0ull;
	for(uint z=0u; z<Nz; z++) for(uint y=0u; y<Ny; y++) {
		const bool rand_yz = y<2u||y+2u>=Ny||z<2u||z+2u>=Nz;
		const ulong n0 = ((ulong)y+(ulong)z*(ulong)Ny)*(ulong)Nx;
		for(uint x=0u; x<Nx; x++) {
			const bool r = rand_yz||x<2u||x+2u>=Nx;
			if(r) n_r1_gezaehlt++;
			const ulong n = n0+(ulong)x;
			if((flags_host[n]&TYPE_E)==0u) continue;
			n_e++;
			if(in_r1(n)!=r) n_e_praedikat++; // Lambda und Koordinatentest muessen an jeder TYPE_E-Zelle uebereinstimmen
			if(!r) { if(n_e_innen==0ull) erste_innen = n; n_e_innen++; }
		}
	}
	{	// ★ 15.09. C2c: Host-Zwilling rr_idx_host ist auf R1 eine Bijektion auf [0, r1_N) und liefert innen den Papierkorb r1_N
		// (Kernel-rr_idx ist ausdrucksgleich; Geraet gegen Host belegt der Kopplungs-Verify, der ueber den Host-Index liest).
		std::vector<ulong> belegt((size_t)((r1_N+63ull)/64ull), 0ull);
		ulong n_doppelt = 0ull, n_ausserhalb = 0ull, n_innen_falsch = 0ull;
		for(uint z=0u; z<Nz; z++) for(uint y=0u; y<Ny; y++) for(uint x=0u; x<Nx; x++) {
			const ulong n = (ulong)x+((ulong)y+(ulong)z*(ulong)Ny)*(ulong)Nx;
			const ulong ri = rr_idx_host(n, Nx, Ny, Nz);
			const bool r = x<2u||x+2u>=Nx||y<2u||y+2u>=Ny||z<2u||z+2u>=Nz;
			if(!r) { if(ri!=r1_N) n_innen_falsch++; continue; }
			if(ri>=r1_N) { n_ausserhalb++; continue; }
			ulong& wort = belegt[(size_t)(ri>>6)]; const ulong bit = 1ull<<(uint)(ri&63ull);
			if(wort&bit) n_doppelt++; else wort |= bit;
		}
		ulong n_belegt = 0ull; for(const ulong w : belegt) n_belegt += (ulong)__builtin_popcountll(w);
		if(n_doppelt>0ull||n_ausserhalb>0ull||n_innen_falsch>0ull||n_belegt!=r1_N) { print_warning("RHO_RAND-SELBSTTEST rr_idx_host: doppelt "+to_string(n_doppelt)+", ausserhalb "+to_string(n_ausserhalb)+", Innenzelle ohne Papierkorb "+to_string(n_innen_falsch)+", belegt "+to_string(n_belegt)+" von "+to_string(r1_N)+" (Soll 0/0/0/alle)."); bad++; }
		else print_info("RHO_RAND C0 Selbsttest rr_idx_host: Bijektion auf R1 ("+to_string(n_belegt)+" Zellen), innen Papierkorb");
	}
	if(n_r1_gezaehlt!=r1_N||n_e_praedikat>0ull) { print_warning("RHO_RAND-SELBSTTEST: R1 gezaehlt "+to_string(n_r1_gezaehlt)+" gegen Formel "+to_string(r1_N)+", Praedikat-Abweichungen an TYPE_E-Zellen "+to_string(n_e_praedikat)+" (Soll gleich / 0)."); bad++; }
	else print_info("RHO_RAND C0 Selbsttest: R1 gezaehlt = Formel ("+to_string(n_r1_gezaehlt)+"), Praedikat an allen TYPE_E-Zellen gleich"+string(Nx>=6u&&Ny>=6u&&Nz>=6u ? ", Grenzzellen bestanden" : ", Grenzzellen UEBERSPRUNGEN (Kante < 6)")); // Pruefpass 2, Hinweis 1: ein bestandener Test muss sichtbar sein
	if(testhaken) { // laeuft DURCH das Praedikat: ein in_r1, das immer true liefert, faellt hier auf
		const ulong mitte = (ulong)(Nx/2u)+(ulong)(Ny/2u)*(ulong)Nx+(ulong)(Nz/2u)*NxNy;
		print_warning("RHO_RAND-TESTHAKEN (CFD_RHO_RAND_TESTHAKEN=1): Zelle "+to_string(mitte)+" in der Domaenenmitte wird als TYPE_E-Zelle behandelt (Flags unveraendert). Soll: Waechter (1a) schlaegt an.");
		if(!in_r1(mitte)) { if(n_e_innen==0ull) erste_innen = mitte; n_e_innen++; }
		else { print_warning("RHO_RAND-TESTHAKEN: in_r1 haelt die Domaenenmitte fuer eine Randzelle -- Praedikat defekt."); bad++; }
		if(in_maske(mitte)) { print_warning("RHO_RAND-TESTHAKEN: in_maske haelt die Domaenenmitte fuer beschrieben -- Praedikat defekt."); bad++; }
	}
	if(n_e_innen>0ull) {
		const uint x = (uint)(erste_innen%(ulong)Nx), y = (uint)((erste_innen/(ulong)Nx)%(ulong)Ny), z = (uint)(erste_innen/NxNy);
		print_warning("RHO_RAND-Waechter (1a): "+to_string(n_e_innen)+" TYPE_E-Zelle(n) liegen AUSSERHALB der Randschale R1, erste bei ("+to_string(x)+","+to_string(y)+","+to_string(z)+"). stream_collide liest dort rho aus dem Puffer, den es unter RHO_RAND nicht mehr gibt.");
		bad++;
	}
	// ---- (1b) po_interior: in R1 UND in der Schreibmaske
	ulong po_aus_r1 = 0ull, po_aus_maske = 0ull;
	for(uint i=0u; i<po_N_active; i++) { const ulong m = (ulong)po_interior[i]; if(!in_r1(m)) po_aus_r1++; if(!in_maske(m)) po_aus_maske++; }
	if(po_aus_r1>0ull||po_aus_maske>0ull) {
		print_warning("RHO_RAND-Waechter (1b): von "+to_string(po_N_active)+" po_interior-Zellen liegen "+to_string(po_aus_r1)+" ausserhalb R1 und "+to_string(po_aus_maske)+" ausserhalb der Schreibmaske x >= Nx-2. po_reduce_mean und apply_pressure_outlet laesen dort veraltetes rho.");
		bad++;
	}
	// ---- (1c) vi_interior: dito (im Nahfeld heute leer)
	ulong vi_aus_r1 = 0ull, vi_aus_maske = 0ull;
	for(uint i=0u; i<vi_N_active; i++) { const ulong m = vi_interior[i]; if(!in_r1(m)) vi_aus_r1++; if(!in_maske(m)) vi_aus_maske++; }
	if(vi_aus_r1>0ull||vi_aus_maske>0ull) {
		print_warning("RHO_RAND-Waechter (1c): von "+to_string(vi_N_active)+" vi_interior-Zellen liegen "+to_string(vi_aus_r1)+" ausserhalb R1 und "+to_string(vi_aus_maske)+" ausserhalb der Schreibmaske x >= Nx-2. apply_velocity_inlet laese dort veraltetes rho; ein Einlass an x- braucht eine erweiterte Maske.");
		bad++;
	}
	// Je Kennzahl eine kurze Zeile (Pruefbefund N6: der Konsolenumbruch macht Zahlen am Ende langer Zeilen fuer grep unsichtbar).
	print_info("RHO_RAND C0 R1-Zellen: "+to_string(r1_N)+" von "+to_string(NN));
	print_info("RHO_RAND C0 R1-Speicher 2 B: "+to_string((float)(2ull*r1_N)/1.0e6f,2u)+" MB = "+to_string((float)(2ull*r1_N)/1048576.0f,2u)+" MiB");
	print_info("RHO_RAND C0 TYPE_E: "+to_string(n_e)+", ausserhalb R1: "+to_string(n_e_innen)+" (Soll 0)");
	print_info("RHO_RAND C0 po_interior: "+to_string(po_N_active)+", ausserhalb R1/Maske: "+to_string(po_aus_r1)+"/"+to_string(po_aus_maske)+" (Soll 0/0)");
	print_info("RHO_RAND C0 vi_interior: "+to_string(vi_N_active)+", ausserhalb R1/Maske: "+to_string(vi_aus_r1)+"/"+to_string(vi_aus_maske)+" (Soll 0/0)");

	// ---- (2) APG-Zensus
	const ulong FN = (ulong)fbnx*(ulong)fbny*(ulong)fbnz;
	if(fac_N==0ull||FN==0ull) { print_info("RHO_RAND C0 Zensus: keine aktiven Facetten in dieser Domaene -- APG-Zensus entfaellt."); return bad; }
	auto ist_fac = [&](const ulong fbi) {
		return fac_idx_voll_on ? fac_idx[fbi]!=0xFFFFFFFFu : ((fac_idx[2ull*(fbi>>5)]>>(uint)(fbi&31ull))&1u)!=0u;
	};
	// Rechenbox = F-BBox plus eine Zelle Rand (APG-Nachbarn einer Facette am BBox-Rand liegen dort),
	// auf die Domaene beschnitten. Bitfelder nur ueber diese Box, nicht ueber die Domaene.
	const uint bx0 = fbx0>0u ? fbx0-1u : 0u, by0 = fby0>0u ? fby0-1u : 0u, bz0 = fbz0>0u ? fbz0-1u : 0u;
	const uint bx1 = min(Nx, fbx0+fbnx+1u), by1 = min(Ny, fby0+fbny+1u), bz1 = min(Nz, fbz0+fbnz+1u);
	const uint bnx = bx1-bx0, bny = by1-by0, bnz = bz1-bz0;
	const ulong BN = (ulong)bnx*(ulong)bny*(ulong)bnz, BW = (BN+63ull)/64ull;
	std::vector<ulong> b_l1((size_t)BW,0ull), b_l2((size_t)BW,0ull), b_n18((size_t)BW,0ull);
	auto bget = [&](const std::vector<ulong>& v, const ulong i) { return ((v[(size_t)(i>>6)]>>(uint)(i&63ull))&1ull)!=0ull; };
	auto bset = [&](std::vector<ulong>& v, const ulong i) { v[(size_t)(i>>6)] |= 1ull<<(uint)(i&63ull); };
	auto bidx = [&](const uint x, const uint y, const uint z) { return (ulong)(x-bx0)+((ulong)(y-by0)+(ulong)(z-bz0)*(ulong)bny)*(ulong)bnx; };
	// Lage 1
	ulong n_l1 = 0ull;
	for(uint zb=0u; zb<fbnz; zb++) for(uint yb=0u; yb<fbny; yb++) for(uint xb=0u; xb<fbnx; xb++) {
		const ulong fbi = (ulong)xb+((ulong)yb+(ulong)zb*(ulong)fbny)*(ulong)fbnx;
		if(!ist_fac(fbi)) continue;
		bset(b_l1, bidx(fbx0+xb, fby0+yb, fbz0+zb)); n_l1++;
	}
	static const int FZ6[6][3] = {{1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1}};
	static const int D18[18][3] = {{1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1},{1,1,0},{-1,-1,0},{1,0,1},{-1,0,-1},{0,1,1},{0,-1,-1},{1,-1,0},{-1,1,0},{1,0,-1},{-1,0,1},{0,1,-1},{0,-1,1}};
	ulong n_l2 = 0ull, n_n18 = 0ull, n_n18_aus_box = 0ull, n_n18_e = 0ull;
	for(uint z=fbz0; z<fbz0+fbnz; z++) for(uint y=fby0; y<fby0+fbny; y++) for(uint x=fbx0; x<fbx0+fbnx; x++) {
		if(!bget(b_l1, bidx(x,y,z))) continue;
		// Lage 2: Regeln wie alloc_sgs_band
		for(uint i=0u; i<6u; i++) {
			const int zn = (int)z+FZ6[i][2]; if(zn<0||zn>=(int)Nz) continue;
			const uint xn = (uint)((((int)x+FZ6[i][0])%(int)Nx+(int)Nx)%(int)Nx);
			const uint yn = (uint)((((int)y+FZ6[i][1])%(int)Ny+(int)Ny)%(int)Ny);
			if(xn<fbx0||xn>=fbx0+fbnx||yn<fby0||yn>=fby0+fbny||(uint)zn<fbz0||(uint)zn>=fbz0+fbnz) continue;
			const ulong nn = (ulong)xn+((ulong)yn+(ulong)zn*(ulong)Ny)*(ulong)Nx;
			if((flags_host[nn]&(TYPE_S|TYPE_E))!=0u) continue;
			const ulong bi = bidx(xn, yn, (uint)zn);
			if(bget(b_l1, bi)||bget(b_l2, bi)) continue;
			bset(b_l2, bi); n_l2++;
		}
		// APG-Lesemenge: 18 Nachbarn, periodisch wie neighbors() im Kernel, nur reines TYPE_S entfaellt
		for(uint i=0u; i<18u; i++) {
			const uint xn = (uint)((((int)x+D18[i][0])%(int)Nx+(int)Nx)%(int)Nx);
			const uint yn = (uint)((((int)y+D18[i][1])%(int)Ny+(int)Ny)%(int)Ny);
			const uint zn = (uint)((((int)z+D18[i][2])%(int)Nz+(int)Nz)%(int)Nz);
			const ulong nn = (ulong)xn+((ulong)yn+(ulong)zn*(ulong)Ny)*(ulong)Nx;
			if((flags_host[nn]&(TYPE_S|TYPE_E))==TYPE_S) continue;
			if(xn<bx0||xn>=bx1||yn<by0||yn>=by1||zn<bz0||zn>=bz1) { n_n18_aus_box++; continue; } // gewickelt oder ausserhalb der Rechenbox
			const ulong bi = bidx(xn, yn, zn);
			if(bget(b_n18, bi)) continue;
			bset(b_n18, bi); n_n18++;
			if((flags_host[nn]&TYPE_E)!=0u) n_n18_e++;
		}
	}
	ulong n_n18_ohne_l12 = 0ull, n_n18_ohne_l12_kante = 0ull;
	for(ulong w=0ull; w<BW; w++) {
		const ulong rest = b_n18[(size_t)w]&~(b_l1[(size_t)w]|b_l2[(size_t)w]);
		n_n18_ohne_l12 += (ulong)__builtin_popcountll(rest);
	}
	// Die fehlenden Zellen einordnen: haben sie einen Flaechennachbarn in L1 (dann waeren sie L2 -- darf nicht
	// vorkommen, ausser am F-BBox-Rand) oder nur Kanten-/Diagonalkontakt?
	for(uint z=bz0; z<bz1; z++) for(uint y=by0; y<by1; y++) for(uint x=bx0; x<bx1; x++) {
		const ulong bi = bidx(x,y,z);
		if(!bget(b_n18, bi)||bget(b_l1, bi)||bget(b_l2, bi)) continue;
		if((flags_host[(ulong)x+((ulong)y+(ulong)z*(ulong)Ny)*(ulong)Nx]&TYPE_E)!=0u) continue; // TYPE_E ist nie L2 und oben schon als Plausibilitaet gemeldet (Pruefpass 2, Hinweis 2)
		bool flaeche = false;
		for(uint i=0u; i<6u&&!flaeche; i++) {
			const int xn=(int)x+FZ6[i][0], yn=(int)y+FZ6[i][1], zn=(int)z+FZ6[i][2];
			if(xn<(int)bx0||xn>=(int)bx1||yn<(int)by0||yn>=(int)by1||zn<(int)bz0||zn>=(int)bz1) continue;
			if(bget(b_l1, bidx((uint)xn,(uint)yn,(uint)zn))) flaeche = true;
		}
		if(!flaeche) n_n18_ohne_l12_kante++;
	}
	// PLAUSIBILITAET, keine RHO_RAND-Verletzung (Pruefbefund N4): TYPE_E liegt in R1. Es waere aber ein Zeichen fuer einen zu engen Nahkasten.
	if(n_n18_e>0ull) print_warning("RHO_RAND C0 Zensus (Plausibilitaet): "+to_string(n_n18_e)+" APG-gelesene Zellen sind TYPE_E -- Facetten liegen am Domaenenrand, der Nahkasten ist sehr eng.");
	// Zellen der Lesemenge MIT Flaechenkontakt zu L1, die trotzdem nicht in L1+L2 liegen, duerfte es nicht geben
	// (sie waeren Lage 2), ausser am F-BBox-Rand -- die Box ist aber um M=4 aufgeweitet. Soll 0 (Pruefbefund M2c).
	const ulong n_n18_ohne_l12_flaeche = n_n18_ohne_l12-n_n18_ohne_l12_kante;
	if(n_n18_ohne_l12_flaeche>0ull) { print_warning("RHO_RAND C0 Zensus: "+to_string(n_n18_ohne_l12_flaeche)+" Zellen der APG-Lesemenge haben Flaechenkontakt zu L1, liegen aber nicht in L1+L2 -- die L2-Nachbildung oder der F-BBox-Beschnitt stimmt nicht."); bad++; }
	if(n_n18_aus_box>0ull) { print_warning("RHO_RAND C0 Zensus: "+to_string(n_n18_aus_box)+" APG-Lesungen (Mehrfachzaehlung) fallen ausserhalb der Rechenbox F-BBox+1 oder werden periodisch gewickelt -- der Zensus waere dort unvollstaendig."); bad++; }
	if(band_on&&band_lagen>=2u) {
		if(band_n_lage[2]!=n_l2) { print_warning("RHO_RAND C0 Zensus: |L2| = "+to_string(n_l2)+" weicht vom gebauten SGS-Band ab (band_n_lage[2] = "+to_string(band_n_lage[2])+") -- die Nachbildung von alloc_sgs_band ist nicht ausdrucksgleich (oder CFD_FAC_IDX_VOLL=1: alloc_sgs_band liest fac_idx immer als Bitmaske)."); bad++; }
		else print_info("RHO_RAND C0 L2 Ist=Soll: "+to_string(n_l2)+" = band_n_lage[2]");
	} else print_info("RHO_RAND C0 L2 NICHT gegen alloc_sgs_band geprueft (Band nicht gebaut) -- Beleg nur aus einem Lauf mit CFD_SGS_BAND=2");
	// Regressionsschutz: alloc_facetten_domain prueft dieselbe Gleichheit schon hart (Pruefbefund N1).
	if(n_l1!=fac_N) { print_warning("RHO_RAND C0 Zensus: |L1| = "+to_string(n_l1)+" aus fac_idx, aber fac_N = "+to_string(fac_N)+"."); bad++; }
	// "Obermenge": auf dem Geraet fallen TYPE_MS-Facetten und Fruehausstiege (u_t ~ 0, MESSNUR, ELIBB pur) weg (Pruefbefund N3).
	// Fuer die Auslegung einer Speicherregion ist die statische Obermenge die richtige Zahl.
	print_info("RHO_RAND C0 Zensus L1 (Facetten): "+to_string(n_l1)+" (fac_N "+to_string(fac_N)+")");
	print_info("RHO_RAND C0 Zensus L2 (6-Flaechen-Dilatation): "+to_string(n_l2));
	print_info("RHO_RAND C0 Zensus APG-Lesemenge (statische Obermenge, N18(L1) ohne TYPE_S): "+to_string(n_n18));
	print_info("RHO_RAND C0 Zensus davon NICHT in L1+L2: "+to_string(n_n18_ohne_l12)+" (nur Kanten-/Diagonalkontakt: "+to_string(n_n18_ohne_l12_kante)+", mit Flaechenkontakt: "+to_string(n_n18_ohne_l12_flaeche)+", Soll 0)");
	print_info("RHO_RAND C0 Zensus Speicher APG-Region: "+to_string((float)(4ull*n_n18)/1.0e6f,2u)+" MB FP32 / "+to_string((float)(2ull*n_n18)/1.0e6f,2u)+" MB FP16");
	print_info("RHO_RAND C0 Zensus Rechenbox "+to_string(bnx)+"x"+to_string(bny)+"x"+to_string(bnz)+", Bitfelder "+to_string((float)(3ull*8ull*BW)/1.0e6f,1u)+" MB Host");
	return bad;
}

void LBM_Domain::alloc_f_liste(const uchar* flags_host, const uint Nx, const uint Ny, const uint Nz) {
	if(!f_liste_on) return;
#if defined(PARTICLES)||defined(GRAPHICS)
	print_error("CFD_F_LISTE ist mit PARTICLES/GRAPHICS nicht verdrahtet -- integrate_particles/graphics_flags hielten den F-Platzhalter (Diff-Pruefung D7).");
#endif
	const ulong FN = (ulong)fbnx*(ulong)fbny*(ulong)fbnz;
	const ulong FNB = (FN+31ull)/32ull;
#ifndef D3Q27
	static const int FZ18[18][3] = {{1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1},
		{1,1,0},{-1,-1,0},{1,0,1},{-1,0,-1},{0,1,1},{0,-1,-1},{1,-1,0},{-1,1,0},{1,0,-1},{-1,0,1},{0,1,-1},{0,-1,1}};
	const uint n_fz = 18u;
#else
	// ★ 17.09.2026 (Planungsagent D3Q27, Risiko 1): der Kernel prueft in update_force_field ALLE 26 Nachbarn
	// (kernel.cpp has_fluid_neighbor). Mit nur 18 Richtungen bekaeme eine Wandzelle, deren einziger Fluidnachbar
	// eine Ecke ist, keinen Slot -- store3_F verwirft dann ihre Kraft still (Slot 77). Deshalb hier die 8 Ecken dazu.
	static const int FZ18[26][3] = {{1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1},
		{1,1,0},{-1,-1,0},{1,0,1},{-1,0,-1},{0,1,1},{0,-1,-1},{1,-1,0},{-1,1,0},{1,0,-1},{-1,0,1},{0,1,-1},{0,-1,1},
		{1,1,1},{-1,-1,-1},{1,1,-1},{-1,-1,1},{1,-1,1},{-1,1,-1},{-1,1,1},{1,-1,-1}};
	const uint n_fz = 26u;
#endif
	for(ulong i=0ull; i<2ull*FNB; i++) f_maske[i]=0u;
	ulong n_solid=0ull, n_wand=0ull;
	for(uint zb=0u; zb<fbnz; zb++) for(uint yb=0u; yb<fbny; yb++) for(uint xb=0u; xb<fbnx; xb++) {
		const uint x=fbx0+xb, y=fby0+yb, z=fbz0+zb;
		const ulong n = (ulong)x+((ulong)y+(ulong)z*(ulong)Ny)*(ulong)Nx;
		if((flags_host[n]&(TYPE_S|TYPE_E))!=TYPE_S) continue; // Host-Maske fuer TYPE_BO (device-seitig 0x03)
		n_solid++;
		bool wand=false;
		for(uint i=0u; i<n_fz&&!wand; i++) {
			const int zn0=(int)z+FZ18[i][2];
			// OBERMENGE: liegt der Nachbar ausserhalb des Gitters in z, gilt die Zelle als Wandzelle
			// (der Kernel wickelt dort und koennte einen Nicht-Solid treffen -- wir raten zugunsten
			// eines Slots, nie dagegen).
			if(zn0<0||zn0>=(int)Nz) { wand=true; break; }
			const uint xn=(uint)((((int)x+FZ18[i][0])%(int)Nx+(int)Nx)%(int)Nx);
			const uint yn=(uint)((((int)y+FZ18[i][1])%(int)Ny+(int)Ny)%(int)Ny);
			const ulong nn=(ulong)xn+((ulong)yn+(ulong)zn0*(ulong)Ny)*(ulong)Nx;
			if((flags_host[nn]&(TYPE_S|TYPE_E))!=TYPE_S) wand=true;
		}
		if(!wand) continue;
		n_wand++;
		const ulong fbi=(ulong)xb+((ulong)yb+(ulong)zb*(ulong)fbny)*(ulong)fbnx;
		f_maske[2ull*(fbi>>5)] |= 1u<<(uint)(fbi&31ull);
	}
	{	ulong lauf=0ull;
		for(ulong b=0ull; b<FNB; b++) { f_maske[2ull*b+1ull]=(uint)lauf; lauf += (ulong)__builtin_popcount(f_maske[2ull*b]); }
		if(lauf!=n_wand) { print_error("alloc_f_liste: Maske traegt "+to_string(lauf)+" Bits, gezaehlt wurden "+to_string(n_wand)+" Wandsolidzellen."); return; }
		f_slots = lauf;
	}
	if(f_slots==0ull) { print_error("alloc_f_liste: keine einzige Wandsolidzelle in der F-BBox -- das Setup hat keinen Koerper, oder die Box sitzt falsch."); return; }
	if(f_slots>=0xFFFFFFFFull) { print_error("alloc_f_liste: mehr Slots als der uint-Praefix traegt."); return; }
	f_maske[2ull*FNB] = (uint)f_slots; // der Stride, den F_STRIDE im Kernel liest
	f_maske.write_to_device();
	// ★ ABNAHME DES STRIDE-TRANSPORTS: der Kernel liest den Stride aus f_maske[2*ceil(def_FBN/32)].
	// Stimmt dieser Wert nicht, liest load3_F fuer Fy/Fz AUSSERHALB des Puffers -- das faellt auf der
	// CPU als harmlose Null auf und auf der GPU als nichtdeterministischer Muell. Deshalb hier
	// zurueckgelesen und geprueft, nicht angenommen.
	{	f_maske.read_from_device();
		const uint zurueck = f_maske[2ull*FNB];
		if((ulong)zurueck!=f_slots) print_error("F-MARKERLISTE: Stride-Transport gescheitert -- geschrieben "+to_string(f_slots)+", vom Geraet zurueckgelesen "+to_string((ulong)zurueck)+". load3_F wuerde fuer Fy/Fz ausserhalb des Puffers lesen.");
		else print_info("F-MARKERLISTE: Stride-Transport geprueft, Geraet meldet "+to_string((ulong)zurueck)+" Slots zurueck.");
	}
	// F in Slotgroesse NEU anlegen und ueberall rebinden (Use-after-free-Lehre: erst anlegen, dann binden)
	F = Memory<float>(device, f_slots, 3u);
	for(ulong i=0ull; i<3ull*f_slots; i++) F[i]=0.0f;
	F.write_to_device();
	kernel_stream_collide.set_parameters(f_param_sc, F);
	kernel_update_fields.set_parameters(f_param_uf, F);
	kernel_update_force_field.set_parameters(3u, F);
	kernel_reset_force_field.set_parameters(0u, F);
	kernel_object_force.set_parameters(0u, F);
	kernel_object_force_zband.set_parameters(0u, F);
	kernel_object_torque.set_parameters(0u, F);
	const ulong b_alt = 12ull*FN, b_neu = 12ull*f_slots + 4ull*(2ull*FNB+1ull);
	print_info("F-MARKERLISTE: "+to_string(n_wand)+" Wandsolidzellen von "+to_string(n_solid)+" Solidzellen in "
		+to_string(FN)+" BBox-Zellen ("+to_string((float)(100.0*(double)n_wand/(double)FN),3u)+" %); F "
		+to_string((ulong)(b_alt/1048576ull))+" -> "+to_string((ulong)(b_neu/1048576ull))+" MiB inkl. Maske, gespart "
		+to_string((ulong)((b_alt>b_neu?b_alt-b_neu:0ull)/1048576ull))+" MiB (VRAM UND System-RAM). Wirkpfad-Waechter: Slot 77 muss am Laufende 0 sein.");
}
void LBM_Domain::alloc_facetten_domain(const std::vector<Facette>& F, const uint Nx, const uint Ny, const std::unordered_map<ulong,std::array<uchar,18>>* qmap, const uint sgs_gdiag, const uint sgs_fdwand, const uint sgs_sism) {
	if((sgs_fdwand>0u)!=fdwand_on) print_error("SGS_FDWAND-Konfigurationsbruch: env-Parameter ("+to_string((ulong)sgs_fdwand)+") und Konstruktionszustand ("+string(fdwand_on?"an":"aus")+") widersprechen sich -- Emission haengt am Konstruktionszustand, Puffer am Parameter; beide muessen aus DEMSELBEN CFD_SGS_FDWAND stammen (Statik-Lebensdauer-Lehre 02.09.).");
	if((sgs_sism>0u)!=sism_on) print_error("SGS_SISM-Konfigurationsbruch: env-Parameter ("+to_string((ulong)sgs_sism)+") und Konstruktionszustand ("+string(sism_on?"an":"aus")+") widersprechen sich -- Emission haengt am Konstruktionszustand, Puffer am Parameter; beide muessen aus DEMSELBEN CFD_SGS_SISM stammen (Statik-Lebensdauer-Lehre 02.09.). Haeufigste Ursache: SISM ohne FDWAND oder ohne Facetten."); // ★ 07.09. SISM
	if(!facetten_on) { print_error("alloc_facetten_domain ohne CFD_FACETTEN."); return; }
	const ulong FN = (ulong)fbnx*(ulong)fbny*(ulong)fbnz;
	if(FN==0ull) { print_error("alloc_facetten_domain: F-BBox ist leer."); return; }
	ulong aktiv=0ull, ausgeschlossen=0ull;
	for(const Facette& f : F) { if(f.klasse==0u) aktiv++; else ausgeschlossen++; }
	if(aktiv==0ull) { print_error("alloc_facetten_domain: keine aktive Facette (alle markiert?)."); return; }
	if(aktiv>=0xFFFFFFFFull) { print_error("alloc_facetten_domain: Facettenzahl kollidiert mit dem NIL-Sentinel."); return; }
	// ★ ZWEITE STUFE der Speicherpruefung (29.08.). Die Konstruktor-Vorpruefung kennt die
	// Facettenzahl noch nicht -- hier steht sie. Das ist der letzte grosse Posten, und ohne
	// diese Stufe faellt ein zu grosses Gitter erst nach zehn Minuten Aufbau auf.
	{	const ulong bytes_fac = (fac_idx_voll_on ? 4ull*FN : 8ull*((FN+31ull)/32ull)) // fac_idx: Vollfeld (4 B je Zelle) oder Bitmaske+Praefixsumme (8 B je 32 Zellen, 03.09.)
		                      + (8ull+6ull)*4ull*aktiv  // fac_geo + fac_tau
		                      + 4ull*aktiv              // fac_tau_n
		                      + (fac_elibb_on ? 18ull*aktiv : 0ull)  // fac_q
		                      + (fac_kdiag_on ? 64ull*aktiv : 0ull)  // fac_kd (Klassen-Diagnostik, 16 float seit 05.09. -- Vorschaetzung MUSS mitziehen, sonst ist der VRAM-Waechter um 16 B/Facette blind)
		                      + (nachbar_on ? 4ull*nb_stride*aktiv : 0ull)     // fac_nb (deterministische Nachbarabtastung, 2 float; unter APG 5 float -- Vorschaetzung MUSS mitziehen, KDIAG-Lehre)
		                      + (sgs_gdiag>0u ? 36ull*aktiv : 0ull)  // gd_zellen (4 B seit 08.09.) + fac_gd (32 B) der g-Diagnose
		                      + (sgs_fdwand>0u ? (sgs_gdiag>0u?4ull:8ull)*aktiv : 0ull)  // fac_wfd (4 B) + gd_zellen (4 B seit 08.09.), falls nicht schon von gdiag gebaut
		                      + (sgs_sism>0u ? 24ull*aktiv : 0ull);   // ★ 07.09. fac_sb (6 float) der SISM-EMA -- Vorschaetzung MUSS mitziehen (KDIAG-Lehre: sonst ist der VRAM-Waechter um 24 B/Facette blind)
		const ulong mb_fac = bytes_fac/1048576ull;
		const ulong frei_gemessen = device.info.uses_ram ? 0ull : vram_frei_gemessen((ulong)device.info.memory);
		const ulong belegt = (ulong)device.info.memory_used;
		const ulong kapazitaet = (ulong)device.info.memory;
		print_info("SPEICHER-IST vor den Facettenpuffern: belegt "+to_string(belegt)+" MB von "
			+to_string(kapazitaet)+" MB (rekonstruiert)"
			+(frei_gemessen>0ull ? string(", GEMESSEN frei "+to_string(frei_gemessen)+" MB ("+string(vram_quelle())+")")
			                     : string(", gemessener Frei-Wert NICHT lesbar -- Debugfs braucht Rechte"))
			+" | Facettenpuffer "+to_string(mb_fac)+" MB fuer "+to_string(aktiv)+" aktive Facetten");
		// Gegen den GEMESSENEN Wert pruefen, wenn er da ist -- sonst gegen die Rekonstruktion.
		const ulong frei = frei_gemessen>0ull ? frei_gemessen : (kapazitaet>belegt ? kapazitaet-belegt : 0ull);
		const ulong mindest = (ulong)env_u("CFD_VRAM_MIN_FREI_MB", 1024u); // Heiko 29.08.: 1,0-1,5 GB Restluft sind legitim
		if(!device.info.uses_ram && mb_fac+mindest > frei)
			print_error("Facettenpuffer passen nicht: "+to_string(mb_fac)+" MB noetig, "+to_string(frei)
				+" MB frei, Mindestluft "+to_string(mindest)+" MB (CFD_VRAM_MIN_FREI_MB). "
				+to_string(aktiv)+" aktive Facetten bei F-BBox "+to_string(fbnx)+"x"+to_string(fbny)+"x"+to_string(fbnz)
				+". Gitter verkleinern, Facettenzahl senken oder die Mindestluft bewusst herabsetzen.");
	}
	fac_geo   = Memory<float>(device, 8ull*aktiv);
	const ulong FNB = (FN+31ull)/32ull; // Zahl der 32er-Bloecke der F-BBox
	// ★ 03.09.2026: Bitmaske+Praefixsumme; CFD_FAC_IDX_VOLL=1 legt die alte Vollfeldform an (A/B-Arm).
	fac_idx   = Memory<uint>(device, fac_idx_voll_on ? FN : 2ull*FNB);
	fac_tau   = Memory<float>(device, 6ull*aktiv); // Layout: [6k]=tw, [6k+1..3]=Wandkraft, [6k+4]=Delta-m, [6k+5]=Normalkontamination (iMEM-Umbau)
	fac_tau_n = Memory<uint>(device, aktiv);
	if(fac_idx_voll_on) { for(ulong i=0ull; i<FN; i++) fac_idx[i] = 0xFFFFFFFFu; } // Vollfeldform: NIL-Sentinel wie vor dem 03.09.
	else { for(ulong i=0ull; i<2ull*FNB; i++) fac_idx[i] = 0u; } // Maske 0 = keine aktive Facette; Basen kommen im zweiten Durchgang
	ulong k=0ull, fbi_vor=0ull;
	for(const Facette& f : F) {
		if(f.klasse!=0u) continue;
		const float na = (f.achse==0u) ? fabsf(f.nx) : (f.achse==1u) ? fabsf(f.ny) : fabsf(f.nz);
		fac_geo[8ull*k+0ull]=f.nx; fac_geo[8ull*k+1ull]=f.ny; fac_geo[8ull*k+2ull]=f.nz;
		fac_geo[8ull*k+3ull]=f.yw;
		fac_geo[8ull*k+4ull]=1.0f/fmax(na, 0.57735027f); // Flaechenfaktor, Kappe sqrt(3) (|n_a|>=1/sqrt(3))
		fac_geo[8ull*k+5ull]=(float)f.achse;
		fac_geo[8ull*k+6ull]=0.0f; fac_geo[8ull*k+7ull]=0.0f;
		for(ulong q6=0ull; q6<6ull; q6++) fac_tau[6ull*k+q6]=0.0f;
		fac_tau_n[k]=0u;
		// Zellindex -> F-BBox-Index (dieselbe Formel wie f_bbox im Kernel)
		const uint x=(uint)(f.n%(ulong)Nx), y=(uint)((f.n/(ulong)Nx)%(ulong)Ny), z=(uint)(f.n/((ulong)Nx*(ulong)Ny));
		if(x<fbx0||y<fby0||z<fbz0||x>=fbx0+fbnx||y>=fby0+fbny||z>=fbz0+fbnz) { print_error("Facette ausserhalb der F-BBox -- set_force_bbox deckt die Wandzellen nicht."); return; }
		const ulong fbi=(ulong)(x-fbx0)+((ulong)(y-fby0)+(ulong)(z-fbz0)*(ulong)fbny)*(ulong)fbnx;
		// ★ Invarianten-Waechter (Audit-Entwarnung 2026-08-26): ALLE fac_tau-Buchungen sind
		// nicht-atomare += und racefrei NUR wegen 1 Zelle = 1 Facette. Wuerde eine spaetere
		// Aenderung zwei Facetten auf eine Zelle legen, kaeme das Race STILL -- hier hart abfangen.
		// ★ 03.09.2026 ZWEITE, STAERKERE INVARIANTE: fbi muss STRENG MONOTON wachsen. Nur dann ist
		// der Rang einer Zelle in fbi-Ordnung gleich dem Zaehlerstand k -- und genau das ist die
		// Voraussetzung dafuer, dass die popcount-Nummerierung im Kernel dieselben fids liefert wie
		// dieser Zaehler. Sie gilt heute (setup.cpp baut F in aufsteigender Zellindex-Ordnung, und
		// fbi ist innerhalb der Box ordnungsgleich zum Zellindex), aber sie ist eine ANNAHME UEBER
		// FREMDEN CODE: eine spaetere Umsortierung von F wuerde alle fid-indizierten Puffer STILL
		// verschieben. Deshalb steht hier ein Waechter und kein Kommentar.
		if(k>0ull&&fbi==fbi_vor) { print_error("alloc_facetten_domain: Zelle traegt zwei Facetten -- die racefrei-Invariante (1 Zelle = 1 Facette) waere verletzt."); return; }
		if(k>0ull&&fbi<fbi_vor) { print_error("alloc_facetten_domain: F ist nicht nach F-BBox-Index sortiert (fbi "+to_string(fbi)+" nach "+to_string(fbi_vor)+") -- die popcount-Nummerierung im Kernel waere NICHT mehr gleich dieser Zaehlerreihenfolge, alle fid-indizierten Puffer wuerden still verschoben."); return; }
		fbi_vor = fbi;
		if(fac_idx_voll_on) fac_idx[fbi]=(uint)k; else fac_idx[2ull*(fbi>>5)] |= 1u<<(uint)(fbi&31ull);
		k++;
	}
	// ★ ZWEITER DURCHGANG: exklusive Praefixsumme der Bloecke. fid = base + popcount(Maske unterhalb
	// der eigenen Lane); der Kernel-Helfer fac_fid() rechnet genau das. Abnahme: die Summe ueber alle
	// Bloecke MUSS die Facettenzahl treffen -- sonst passen Maske und Zaehlerstand nicht zusammen.
	if(!fac_idx_voll_on) {	ulong lauf=0ull;
		for(ulong b=0ull; b<FNB; b++) { fac_idx[2ull*b+1ull]=(uint)lauf; lauf += (ulong)__builtin_popcount(fac_idx[2ull*b]); }
		if(lauf!=aktiv) { print_error("alloc_facetten_domain: Bitmaske traegt "+to_string(lauf)+" gesetzte Bits, aber "+to_string(aktiv)+" aktive Facetten wurden gezaehlt -- Maske und fid-Nummerierung sind auseinander."); return; }
		print_info("fac_idx als Bitmaske+Praefixsumme: "+to_string((ulong)((8ull*FNB)/1048576ull))+" MB fuer "+to_string(FN)+" F-BBox-Zellen (frueher "+to_string((ulong)((4ull*FN)/1048576ull))+" MB), Belegung "+to_string((double)aktiv*100.0/(double)FN,3u)+" %");
	} else print_warning("CFD_FAC_IDX_VOLL=1: fac_idx laeuft in der ALTEN Vollfeldform ("+to_string((ulong)((4ull*FN)/1048576ull))+" MB statt "+to_string((ulong)((8ull*FNB)/1048576ull))+" MB). Deklarierter A/B-Arm gegen die Bitmaske -- Ergebnisse muessen BITGLEICH sein, nur der Speicher unterscheidet sich.");
	fac_N = aktiv;
	const bool diagz_gebaut = fac_diagz_on; // Audit 2/3: Rebind haengt am KONSTRUKTIONS-Zustand, nicht an der (potentiell umgesetzten) Statik
	if(fac_diagz_on) { // Iron Rule 3: Diagnose-Facette per Zellindex waehlen (CFD_FAC_DIAGZ = n)
		fac_diag = Memory<float>(device, 19ull); // [17] alpha, [18] dp_ds; Selektor bleibt [16]
		for(ulong q=0ull;q<19ull;q++) fac_diag[q]=0.0f;
		fac_diag[16] = -1.0f; ulong k2=0ull;
		for(const Facette& f : F) { if(f.klasse!=0u) { continue; } if(f.n==(ulong)fac_diagz_wert) { fac_diag[16]=(float)k2; fac_diag_fid=(uint)k2; } k2++; }
		if(fac_diag[16]<0.0f) { fac_diagz_on=false; print_warning("CFD_FAC_DIAGZ: Zelle "+to_string((ulong)fac_diagz_wert)+" traegt keine AKTIVE Facette -- Diagnose HART AUS."); }
		// ★ Nachpruefer Stufe-3: auch im Hart-Aus-Fall REBINDEN -- das Move-Assignment hat den als
		// Kernel-Arg gebundenen Platzhalter zerstoert (Use-after-free auf der iGPU-Zero-Copy);
		// der neue Puffer traegt den -1-Sentinel, der Kernelvergleich matcht nie.
		else print_info("Diagnose-Facette: Zelle "+to_string((ulong)fac_diagz_wert)+" -> fid "+to_string((ulong)fac_diag_fid));
		fac_diag.write_to_device();
	}
	if(fac_elibb_on) { // ★★ B1 (K2-LOESUNGSENTSCHEID, 2026-08-25): q je Link aus der ZELLEIGENEN
		// Facettenebene -- Stufe 1 des Entscheids ("die Facette ist die Wand", Heikos Weg).
		// q_d = y_w / (-n . c_d) = Schnitt der Ebene mit dem Link, in Bruchteilen der Linklaenge.
		// Kodierung uchar: 0 = kein Schnitt in (0,1] (Link bleibt implizites HWBB), sonst
		// q = qb/254 -- 127 ist EXAKT 0,5 (254*0,5 = 127, float-exakt; Entscheid: uchar/254
		// statt V1s 4-Bit/14, Restquantisierung 1/508 statt 1/28 Linklaenge).
		// D3Q19-Richtungstabelle WOERTLICH wie der Kernel (c() Spalten, kernel.cpp D3Q19-Zweig).
		static const int CX19[19]={0,1,-1,0,0,0,0,1,-1,1,-1,0,0,1,-1,1,-1,0,0};
		static const int CY19[19]={0,0,0,1,-1,0,0,1,-1,0,0,1,-1,-1,1,0,0,1,-1};
		static const int CZ19[19]={0,0,0,0,0,1,-1,0,0,1,-1,1,-1,0,0,-1,1,-1,1};
		fac_q = Memory<uchar>(device, 18ull*aktiv);
		ulong nq_schnitt=0ull, nq_boden=0ull, nq_klemme1=0ull; ulong hist[15]={0}; // hist: qb/17 grob (0..14)
		// ★★ B1-STUFE 2 (2026-08-25 abends): q aus der GEGLAETTETEN REMESH-FLAECHE hat VORRANG.
		// Die zelleigene PCA-Ebene ist auf Kruemmung als q-Quelle WIDERLEGT (Kugel-Gate RMS 0,67
		// statt <=0,143; auch gefiltert 0,32-0,51). Sie bleibt RUECKFALL ohne Remesh (Kanal: exakt).
		// Grazing-Guard und q-Boden gelten fuer BEIDE Quellen; die Guard-RICHTUNG kommt aus der
		// PCA-Normale (die Richtung ist robust -- nur ihre Distanz war es nicht).
		ulong nq_remesh=0ull, nq_ebene=0ull, nq_ohne_map=0ull, nq_kappe=0ull;
		std::vector<ulong> fac_zelle; fac_zelle.reserve(aktiv);
		for(const Facette& f2 : F) if(f2.klasse==0u) fac_zelle.push_back(f2.n);
		for(ulong kq=0ull; kq<aktiv; kq++) {
			const float nx=fac_geo[8ull*kq], ny=fac_geo[8ull*kq+1ull], nz=fac_geo[8ull*kq+2ull], yw=fac_geo[8ull*kq+3ull];
			const std::array<uchar,18>* qm = nullptr;
			if(qmap!=nullptr) { auto it=qmap->find(fac_zelle[kq]); if(it!=qmap->end()) qm=&it->second; else nq_ohne_map++; }
			for(uint d=1u; d<19u; d++) {
				const float ndc = nx*(float)CX19[d]+ny*(float)CY19[d]+nz*(float)CZ19[d];
				uchar qb=0u;
				const float clen = sqrtf((float)(CX19[d]*CX19[d]+CY19[d]*CY19[d]+CZ19[d]*CZ19[d]));
				if(ndc<-(float)s_fac_kappa*clen) { // ★ GRAZING-GUARD (EIGENES Interim mit offenem Abloese-Soll, eingefuehrt als K1'-Begleiter, gilt unter MLS weiter): nur Links mit -n.c_hat >= kappa (Default 0,4) interpolieren -- streifende Links sind schlecht konditioniert (q = y_w/kleiner Nenner) UND die groessten Tangentialtraeger: die Injektions-Ratsche der Kugel. Darunter: BB (qb=0).
					const float sq = yw/(-ndc); // Bruchteil der Linklaenge
					// ★ QDIAG (2026-08-25, Kugel-Falsifikation): Hypothesen-Arme fuer die Injektionsjagd.
					// 1 = q>1-Klemme AUS (sq>1 -> BB), 2 = nur q<0,5-Zweig (q>0,5 -> Identitaet),
					// 3 = nur q>0,5-Zweig (q<0,5 -> Identitaet). 0 = normal. NUR Diagnose.
					const uint qd = s_fac_qdiag;
					float sqq = -1.0f; // Quellenwahl: Remesh (Stufe 2) VOR Ebene (Rueckfall)
					if(qm!=nullptr) { const uchar rq=(*qm)[d-1u]; if(rq>0u) { sqq=(float)rq*(1.0f/254.0f); nq_remesh++; } }
					else if(sq>0.0f&&sq<=1.0f) { sqq=sq; nq_ebene++; }
					else if(sq>1.0f&&sq<=1.5f) { nq_klemme1++; } // Ebenen-q>1 -> BB (nur ohne Remesh relevant)
					if(sqq>0.0f) {
						float sqe = sqq;
						if(sqe<(float)s_fac_qmin) { sqe=0.5f; nq_boden++; } // q-Boden (P1-Entscheid)
						// ★★ EX-STABILITAETSKAPPE (historisch: Kernel-Audit Befund 1, 25.08. -- der K1'-Zweig
						// war bei kohaerentem q >= 0,75 instabil, die Kappe 0,65 selbst bei ~25k Schritten,
						// s. Wissensspeicher k1instabilitaet). Mit der MLS-Blende (Baustein 1, 26.08.) ist
						// der q>0,5-Zweig bis q=1 stabil -> Default 1,0 = KEINE Kappung: nq_kappe bleibt
						// dann konstruktiv 0 (beide Quellen liefern hier sqe<=1; Ebenen-q>1 faengt die
						// nq_klemme1-Stufe oben ab) -- nq_kappe>0 im Startprotokoll heisst also: Kappe
						// per CFD_FAC_QKAPPE<1 aktiv gesetzt. Env-Hebel bleibt fuer A/Bs erhalten.
						if(sqe>(float)s_fac_qkappe) { fac_q[18ull*kq+(ulong)(d-1u)]=0u; nq_kappe++; continue; }
						if(qd==2u&&sqe>0.5f) sqe=0.5f; // Arm 2: q>0,5 -> Identitaet
						if(qd==3u&&sqe<0.5f) sqe=0.5f; // Arm 3: q<0,5 -> Identitaet
						qb=(uchar)fmin(fmax((float)(int)(sqe*254.0f+0.5f),1.0f),254.0f);
						nq_schnitt++;
					} // ★ q>1 -> BB-RUECKFALL statt Klemme (QDIAG=1 mass das kostenneutral; eingefuehrt als K1'-Begleiter, gilt unter MLS weiter -- Muell-q bleibt Muell-q)
					// sq>1,5: Ebene weit weg -- Link bleibt HWBB (qb=0), kein Zaehler (normaler Fall der Stufenrueckseite)
				}
				fac_q[18ull*kq+(ulong)(d-1u)] = qb;
				if(qb>0u) hist[qb/17u]++;
			}
		}
		fac_q.write_to_device();
		// ★ NACHGEHOLTE B1-ABNAHME (2026-08-25 abends): fac_q + Zellkoordinaten als CSV, damit
		// das Kugel-Gate "Upload-q gegen analytisches q" offline pruefbar ist. Die Leiter lief
		// heute OHNE dieses Gate -- Prozessfehler, im Befundbuch. Nur bei CFD_FAC_QDUMP=1.
		if(getenv("CFD_FAC_QDUMP")) {
			const string qdp = get_exe_path()+"../export/fac_q_dump_"+(getenv("CFD_RUN_NAME")?string(getenv("CFD_RUN_NAME")):string("lauf"))+"_D"+to_string((ulong)Nx)+".csv"; // ★ Host-Audit Befund 4: Run+Domaenen-Suffix statt Kollision
			FILE* fq = fopen(qdp.c_str(), "w");
			if(fq==nullptr) print_warning("fac_q-Dump: "+qdp+" nicht schreibbar.");
			if(fq) {
				fprintf(fq, "# fid,x,y,z,nx,ny,nz,yw,qb1..qb18 (qb/254 = q; 0 = kein Schnitt)\n");
				ulong kq2=0ull;
				for(const Facette& f : F) { if(f.klasse!=0u) continue;
					const uint xx=(uint)(f.n%(ulong)Nx), yy=(uint)((f.n/(ulong)Nx)%(ulong)Ny), zz=(uint)(f.n/((ulong)Nx*(ulong)Ny));
					fprintf(fq, "%lu,%u,%u,%u,%.6f,%.6f,%.6f,%.6f", kq2, xx, yy, zz,
						fac_geo[8ull*kq2], fac_geo[8ull*kq2+1ull], fac_geo[8ull*kq2+2ull], fac_geo[8ull*kq2+3ull]);
					for(uint d=1u; d<19u; d++) fprintf(fq, ",%u", (uint)fac_q[18ull*kq2+(ulong)(d-1u)]);
					fprintf(fq, "\n"); kq2++;
				}
				fclose(fq); print_info("fac_q-Dump: "+qdp+" ("+to_string((ulong)aktiv)+" Facetten).");
			}
		}
		string hs=""; for(uint hb=0u; hb<15u; hb++) hs+=to_string(hist[hb])+(hb<14u?" ":"");
		print_info("ELIBB fac_q: "+to_string(nq_schnitt)+" geschnittene Links auf "+to_string(aktiv)+" Facetten; QUELLE: Remesh "+to_string(nq_remesh)+", Ebenen-Rueckfall "+to_string(nq_ebene)+", ohne Map-Treffer "+to_string(nq_ohne_map)+" Facetten; q-Boden->0,5: "+to_string(nq_boden)+", Ebenen-q>1->BB: "+to_string(nq_klemme1)+", q>Kappe->BB: "+to_string(nq_kappe)+" (CFD_FAC_QKAPPE "+to_string(s_fac_qkappe,2u)+")");
		if(qmap!=nullptr&&nq_remesh==0ull) print_error("ELIBB Stufe 2: Remesh-Map uebergeben, aber NULL Remesh-q verwendet -- lautloser No-Op der Stufe 2.");
		print_info("  q-Histogramm (Bins von 17/254, 0-basiert): "+hs+"  -- kipp0-Gate: ALLES muss im Bin 7 (q=0,5) liegen");
	}
	// ★ 22.09.2026 S0 (Lehre M2 vom selben Tag): Host- und Kernelzustand muessen denselben Modus meinen. Am 22.09. kostete die
	// umgekehrte Reihenfolge (Host FD, Kernel Pi) einen 6-fachen Pufferueberlauf, der nur ueber Symptome zu finden war.
	if(fac_r1q_jit!=fac_r1q_on) print_error("alloc_facetten_domain: KOHAERENZ Host/Kernel verletzt -- Kernel-Text "+string(fac_r1q_jit?"MIT":"OHNE")+" FAC_R1Q, Host fac_r1q_on = "+to_string(fac_r1q_on?1u:0u)+"."); // ★ 28.09. R1Q
	if(fac_rek_jit!=fac_rek_on) print_error("alloc_facetten_domain: KOHAERENZ Host/Kernel verletzt -- Kernel-Text "+string(fac_rek_jit?"MIT":"OHNE")+" FAC_REK, Host fac_rek_on = "+to_string(fac_rek_on?1u:0u)+".");
	fac_geo.write_to_device(); fac_idx.write_to_device(); fac_tau.write_to_device(); fac_tau_n.write_to_device();
	kernel_stream_collide.set_parameters(fac_param_pos, fac_geo, fac_idx, fac_tau, fac_tau_n);
	if(fac_ema_on) { fac_us = Memory<float>(device, 3ull*aktiv); for(ulong q3=0ull;q3<3ull*aktiv;q3++) fac_us[q3]=0.0f; fac_us.write_to_device(); kernel_stream_collide.set_parameters(fac_param_pos+4u, fac_us); }
	if(fac_pema_on) { fac_pu = Memory<float>(device, 6ull*aktiv); for(ulong q6=0ull;q6<6ull*aktiv;q6++) fac_pu[q6]=0.0f; fac_pu.write_to_device(); kernel_stream_collide.set_parameters(fac_param_pos+(fac_ema_on?5u:4u), fac_pu); }
	if(diagz_gebaut&&fac_diag.length()>=19ull) kernel_stream_collide.set_parameters(fac_param_pos+4u+(fac_ema_on?1u:0u)+(fac_pema_on?1u:0u), fac_diag); // unkonditional bei DIAGZ-Emission (auch Hart-Aus: Sentinel-Puffer statt zerstoertem Platzhalter)
	if(fac_elibb_on) kernel_stream_collide.set_parameters(fac_param_pos+4u+(fac_ema_on?1u:0u)+(fac_pema_on?1u:0u)+(diagz_gebaut?1u:0u), fac_q); // ★ B2: Rebind des in alloc gebauten fac_q (Signaturposition = nach diagz)
	if(fac_kdiag_on) { fac_kd = Memory<float>(device, 16ull*aktiv); for(ulong q8=0ull;q8<16ull*aktiv;q8++) fac_kd[q8]=0.0f; fac_kd.write_to_device(); kernel_stream_collide.set_parameters(fac_param_pos+4u+(fac_ema_on?1u:0u)+(fac_pema_on?1u:0u)+(diagz_gebaut?1u:0u)+(fac_elibb_on?1u:0u), fac_kd); print_info("Klassen-Diagnostik (CFD_FAC_KDIAG): fac_kd "+to_string((ulong)(64ull*aktiv/1048576ull))+" MB, 16 float je Facette (Text sagte bis 05.09. \"10\" -- war schon bei 12 falsch), Tabelle je Treppenklasse am Laufende."); } // ★ Rebind nach fac_q
	if(sgs_gdiag>0u||sgs_fdwand>0u||nachbar_on) { // ★ Liste fid->Zellindex wird von g-Diagnose, Geistermoden-Fix UND Nachbarabtastung (03.09.) gebraucht
		gd_zellen = Memory<uint>(device, aktiv);
		if(get_N()>0xFFFFFFFFull) print_error("gd_zellen ist seit 08.09. uint (VRAM) -- bei N > 2^32 wuerde jeder Zellindex still abgeschnitten. Der Waechter darueber prueft die FACETTENzahl, nicht N (Pruefagent-Befund B1).");
		{ ulong k=0ull; for(const Facette& f : F) { if(f.klasse!=0u) continue; gd_zellen[k++]=(uint)f.n; } }
		gd_zellen.write_to_device();
		// ★ 11.09.2026 HOST-SPIEGEL FREIGEBEN (VRAM-Audit). Dieser Puffer wird EINMAL gefuellt,
		// hochgeladen und danach host-seitig nie wieder angefasst -- weder per read_from_device()
		// noch per Index. delete_host_buffer() ist seit 03.09. entschaerft (Aux-Zeiger, Double-Free,
		// Zero-Copy-Waechter), wurde aber nirgends gerufen.
		// DIE BEDINGUNG IST EINSEITIG UND DESHALB SICHER: is_zero_copy verlangt uses_ram, also
		// schliesst !uses_ram Zero-Copy aus. Auf der iGPU IST der Host-Puffer der Geraetespeicher,
		// dort wuerde die Freigabe die laufende Rechnung lautlos zerstoeren -- deshalb nur dGPU.
		if(!device.info.uses_ram) gd_zellen.delete_host_buffer();
	}
	if(nachbar_on) { // ★ 03.09. DETERMINISTISCHE NACHBARABTASTUNG: Puffer bauen, Kernel binden, stream_collide-Rebind (fac_wfd-Muster, B70-bewiesen)
		const ulong nbs = nb_stride; // ★ 16.09. HOCH-1: Instanzwert, eingefroren in allocate() // ★ 16.09. Stride = def_nb_stride der Emission (2, unter APG 5: grad rho in [2..4])
		// ★★ 22.09.2026 STRIDE-WAECHTER VOR DEM PUFFERBAU (Pruefagent, Befund 1). Der vorhandene Waechter
		// (setup.cpp, berichte_apg: st!=5ull) feuert erst am LAUFENDE -- er ist ein Nachruf, kein Schutz: der
		// ganze Lauf hat dann schon auf korruptem Speicher gerechnet. Hier kostet die Pruefung nichts und
		// greift, BEVOR der Puffer entsteht. Anlass ist ein eigener Fehler von heute: ein Kommentar hatte
		// "nb_stride = apg_on ? 5ull : 2ull;" verschluckt, der Puffer waere mit 2*aktiv Floats entstanden,
		// waehrend der JIT def_nb_stride 5ul emittiert -- ab gid >= 0,4*aktiv jeder Zugriff ausserhalb des
		// Puffers, 2,5-facher Ueberlauf ohne Schranke auf der GPU (Wedge-Klasse auf der B70).
		// ★ 23.09. Stufe A: die Erwartung traegt jetzt auch die drei Richtungsfloats. Wer sie hier vergisst,
		// baut genau den Ueberlauf vom 22.09. nach -- deshalb steht die Formel an BEIDEN Stellen ausgeschrieben.
		const ulong nbs_soll = (apg_on ? 5ull : 2ull) + (fac_rek_on ? nb_rek_floats : 0ull);
		// ★ 23.09. abends, Pruefbefund M3/M5: nb_roff war unbewacht, waehrend nb_stride einen Waechter hat.
		// ★ 24.09. BERICHTIGT, Pruefbefund M1: der Kommentar behauptete hier, der Waechter schuetze die
		// SCHREIBseite des Impuls-Akkumulators. Das trifft NICHT zu -- die Schreibseite benutzt
		// def_nb_roff aus den STATIKEN (device_defines), nicht dieses Instanzfeld. Was er wirklich
		// faengt: eine verschluckte Zuweisungszeile, nach der nb_roff auf dem Header-Default 2ull
		// staende -- genau die Klasse, die am 22.09. den 2,5-fachen Pufferueberlauf gekostet hat.
		// Ein Auseinanderlaufen von Host und JIT bliebe unentdeckt; dafuer gibt es den Stride-Waechter
		// darunter und Slot 369 auf der Leseseite von t_nb.
		const ulong roff_soll = apg_on ? 5ull : 2ull;
		if(nb_roff != roff_soll) print_error("fac_nb-Offset inkonsistent: nb_roff = "+to_string(nb_roff)+", erwartet "+to_string(roff_soll)+" (apg_on = "+string(apg_on?"true":"false")+"). Der Impuls-Akkumulator wuerde in die APG-Gradienten schreiben.");
		if(nbs != nbs_soll) print_error("fac_nb-Stride inkonsistent: nb_stride = "+to_string(nbs)
			+", erwartet "+to_string(nbs_soll)+" (apg_on = "+string(apg_on?"true":"false")+"). Der Puffer wuerde nicht zu den Kernelzugriffen passen.");
		fac_nb = Memory<float>(device, nbs*aktiv);
		for(ulong q=0ull;q<aktiv;q++) { fac_nb[nbs*q]=-1.0f; fac_nb[nbs*q+1ull]=0.0f; for(ulong g=2ull; g<nbs; g++) fac_nb[nbs*q+g]=0.0f; } // Init = "kein Wert" -> Eigenzelle (zaehlt als Slot 73), grad rho 0; enqueue_initialize fuellt vor dem ersten Schritt
		fac_nb.write_to_device();
		// ★ 11.09.2026 HOST-SPIEGEL FREIGEBEN (VRAM-Audit). Dieser Puffer wird EINMAL gefuellt,
		// hochgeladen und danach host-seitig nie wieder angefasst -- weder per read_from_device()
		// noch per Index. delete_host_buffer() ist seit 03.09. entschaerft (Aux-Zeiger, Double-Free,
		// Zero-Copy-Waechter), wurde aber nirgends gerufen.
		// DIE BEDINGUNG IST EINSEITIG UND DESHALB SICHER: is_zero_copy verlangt uses_ram, also
		// schliesst !uses_ram Zero-Copy aus. Auf der iGPU IST der Host-Puffer der Geraetespeicher,
		// dort wuerde die Freigabe die laufende Rechnung lautlos zerstoeren -- deshalb nur dGPU.
		// ★★ 23.09.2026, GERAETEFEHLER aus dem A2-Bau, auf der B70 gefunden und auf der iGPU UNSICHTBAR.
		// uses_ram ist auf der iGPU true (gemeinsamer Speicher, Spiegel bleibt), auf der diskreten B70
		// false -- dort wurde der Host-Spiegel freigegeben, und das Ruecklesen des Impuls-Akkumulators
		// (setup.cpp, Fensterschnappschuss und K2-Block) lief in den Waechter "read_from_device() auf
		// einem Puffer, dessen Host-Spiegel freigegeben wurde". Der Lauf starb nach 48 s bei 25 %.
		// LEHRE: die iGPU ist ein Stellvertreter fuer BITGLEICHHEIT, nicht fuer das Speichermodell.
		// Kosten des Spiegels (System-RAM, NICHT VRAM): nb_stride*fac_N*4 B, bei 4 mm mit 3,13 Mio
		// Facetten 113 MB unter APG+REK, 75 MB ohne APG -- und nur unter CFD_FAC_REK.
		if(!device.info.uses_ram&&apg_haken==0u&&!fac_rek_on) fac_nb.delete_host_buffer(); // ★ 16.09.: unter CFD_FAC_APG_HAKEN bleibt der Host-Spiegel -- der Bericht liest grad rho zurueck
		kernel_fac_nachbar = Kernel(device, aktiv, "fac_nachbar_ab", u, flags, fac_geo, gd_zellen, (uint)aktiv, fac_nb);
		if(apg_on) { // ★ 16.09. APG-Vorkernel: eigener Kernel (Gate-Befund Spill), liest die DDFs, schreibt grad rho nach fac_nb[2..4]; t (Position 5) wird je Schritt nachgesetzt
			kernel_fac_apg = Kernel(device, aktiv, "fac_apg_ab", flags, gd_zellen, (uint)aktiv, fac_nb, fi, t, rho_clamp_hits);
			if(sparse_on) kernel_fac_apg.add_parameters(tile_slot);
			print_info("APG-VORKERNEL fac_apg_ab gebunden (16.09.): grad rho aus den DDFs der 6 Achsnachbarn je Facette, "+to_string((float)(12ull*aktiv)/1048576.0f,1u)+" MB in fac_nb[2..4], Launch je Schritt nach fac_nachbar_ab.");
		}
		if(sparse_on) kernel_fac_nachbar.add_parameters(tile_slot); // B-7-Lehre: TS_P haengt an SPARSE_TILES
		{ const uint nbix=fac_param_pos+4u+(fac_ema_on?1u:0u)+(fac_pema_on?1u:0u)+(diagz_gebaut?1u:0u)+(fac_elibb_on?1u:0u)+(fac_kdiag_on?1u:0u);
		  kernel_stream_collide.set_parameters(nbix, fac_nb); } // Rebind NACH dem Neubau (Platzhalter-Lektion wie fac_wfd)
		print_info("NACHBARABTASTUNG deterministisch (03.09.): Kernel fac_nachbar_ab je Schritt nach stream_collide liest das FERTIGE u-Feld und schreibt (u_t_abt, y_abt) fuer "+to_string(aktiv)+" Facetten; apply_facette_imem liest den Vorschritt (ein Schritt Versatz wie fac_wfd). Der fruehere Direktzugriff u[nb] im selben Kernel war gemessen nicht bitreproduzierbar (xu_det_mit_a/b, 03.09.).");
	}
	if(sgs_fdwand>0u) { // ★ GEISTERMODEN-FIX (02.09.): fac_wfd bauen, FD-Kernel binden, stream_collide-Rebind unten
		fac_wfd = Memory<float>(device, aktiv);
		for(ulong q=0ull;q<aktiv;q++) fac_wfd[q]=1.0f/get_tau(); // = def_w // Init = molekulares w (erster Schritt ohne nu_t an Wandzellen -- dokumentiert harmlos)
		fac_wfd.write_to_device();
		// ★ 11.09.2026 HOST-SPIEGEL FREIGEBEN (VRAM-Audit). Dieser Puffer wird EINMAL gefuellt,
		// hochgeladen und danach host-seitig nie wieder angefasst -- weder per read_from_device()
		// noch per Index. delete_host_buffer() ist seit 03.09. entschaerft (Aux-Zeiger, Double-Free,
		// Zero-Copy-Waechter), wurde aber nirgends gerufen.
		// DIE BEDINGUNG IST EINSEITIG UND DESHALB SICHER: is_zero_copy verlangt uses_ram, also
		// schliesst !uses_ram Zero-Copy aus. Auf der iGPU IST der Host-Puffer der Geraetespeicher,
		// dort wuerde die Freigabe die laufende Rechnung lautlos zerstoeren -- deshalb nur dGPU.
		if(!device.info.uses_ram) fac_wfd.delete_host_buffer();
		kernel_sgs_fdwand = Kernel(device, aktiv, "sgs_fdwand", u, flags, gd_zellen, (uint)aktiv, fac_wfd);
		if(sism_on) { // ★ 07.09. SISM: EMA-Puffer (6 float je Facette, Start 0 -- KEIN Warmstart mit S, der liefert nu_t = 0 im ersten Schritt), t + Zaehler als Kernelargumente. Reihenfolge = Kernel-Signatur unter #ifdef SGS_SISM (t, fac_sb, rho_clamp_hits), zwingend VOR tile_slot (TS_P ist der letzte Parameter)
			fac_sb = Memory<float>(device, 6ull*aktiv);
			for(ulong q=0ull;q<6ull*aktiv;q++) fac_sb[q]=0.0f;
			fac_sb.write_to_device();
			kernel_sgs_fdwand.add_parameters(t, fac_sb, rho_clamp_hits, 0u); // sbar_out = 0: Lage 1 bekommt wie bisher ein fertiges w (Geistermoden-Fix) // Position 5/6/7; rho_clamp_hits haengt schon an stream_collide -- Mehrfachbindung desselben cl_mem ist unproblematisch
			print_info("SISM gebunden: fac_sb "+to_string((float)(24ull*aktiv)/1048576.0f,1u)+" MB fuer "+to_string(aktiv)+" Facetten, EMA T = "+to_string((ulong)sism_T)+" Schritte (alpha = 1/T im Kernel), klassisch bis Schritt "+to_string(sism_ab)+"; Slot 126/127 am Laufende.");
		}
		if(sparse_on) kernel_sgs_fdwand.add_parameters(tile_slot); // gleiche B-7-Lehre wie sgs_gdiag
		{ const uint fwix=fac_param_pos+4u+(fac_ema_on?1u:0u)+(fac_pema_on?1u:0u)+(diagz_gebaut?1u:0u)+(fac_elibb_on?1u:0u)+(fac_kdiag_on?1u:0u)+(nachbar_on?1u:0u); // +nachbar_on (03.09.): fac_nb sitzt VOR fac_wfd
		  kernel_stream_collide.set_parameters(fwix, fac_wfd); band_param_pos = fwix+1u; } // ★ 08.09.: dieselbe Rechnung fuer das Band merken -- alloc_sgs_band laeuft spaeter und sieht diagz_gebaut nicht mehr // ★ Rebind NACH dem Neubau -- der Rebind stand zuerst VOR dem Move-Assignment und band den gleich darauf ZERSTOERTEN Platzhalter (CL -52 beim ersten Enqueue; exakt die DIAGZ-Use-after-free-Lektion, 02.09. erneut bezahlt)
		print_info("SGS-GEISTERMODEN-FIX (CFD_SGS_FDWAND): w an "+to_string(aktiv)+" Facettenzellen aus |S|_FD (u-Feld, geistermodenfrei) statt aus dem Pi-Tensor; FD-Kernel je Schritt nach stream_collide (ein Schritt Versatz, deterministisch), Wirkpfad Slot 76 (B70).");
	}
	if(sgs_gdiag>0u) { // ★ g-DIAGNOSE (31.08., Parameter statt Statik seit 02.09.): Akkumulator + eigener Kernel (Liste oben).
		// KEIN Eingriff in stream_collide, keine Signaturaenderung, kein JIT-Define -- der Kernel ist
		// immer kompiliert und wird nur hier gebunden und spaeter explizit gerufen. Default-Bitgleichheit
		// ist damit trivial (Schalter aus = weder Puffer noch Launch).
		gdiag_on = true;
		fac_gd = Memory<float>(device, 8ull*aktiv);
		for(ulong q8=0ull;q8<8ull*aktiv;q8++) fac_gd[q8]=0.0f;
		fac_gd.write_to_device();
		kernel_sgs_gdiag = Kernel(device, aktiv, "sgs_gdiag", fi, u, flags, gd_zellen, (uint)aktiv, fac_gd, t, fx, fy, fz, s_sgs_guo?1u:0u);
		if(sparse_on) kernel_sgs_gdiag.add_parameters(tile_slot); // Pruefbefund B-7: TS_P haengt an SPARSE_TILES -- ohne dieses Argument stuerbe der erste Launch mit CL_INVALID_KERNEL_ARGS
		print_info("g-DIAGNOSE (CFD_SGS_GDIAG): "+to_string(aktiv)+" Wandzellen, "+to_string((ulong)(40ull*aktiv/1048576ull))+" MB -- misst |S|_FD, |S|_Pi, D_WALE, D_Sigma, |Omega| je Zelle; Physik unangetastet.");
	}
	facetten_bound = true;
	print_info("Facetten gebunden: "+to_string(aktiv)+" aktiv, "+to_string(ausgeschlossen)+" markiert (BB bleibt), Indexfeld "
		+to_string((float)(fac_idx_voll_on ? 4ull*FN : 8ull*((FN+31ull)/32ull))/1048576.0f,1u)+(fac_idx_voll_on?" MB (Vollfeld)":" MB (Bitmaske+Praefixsumme)")+", Geometrie "+to_string((float)(aktiv*32ull)/1048576.0f,1u)+" MB auf "+device.info.name+".");
}

void LBM::alloc_facetten(const std::vector<Facette>& F, const std::unordered_map<ulong,std::array<uchar,18>>* qmap, const uint sgs_gdiag, const uint sgs_fdwand, const uint sgs_sism) {
	if(get_D()!=1u) { print_error("CFD_FACETTEN ist nur fuer eine Domaene gebaut (dd = zwei getrennte Instanzen)."); return; }
	lbm_domain[0]->alloc_facetten_domain(F, (uint)get_Nx(), (uint)get_Ny(), qmap, sgs_gdiag, sgs_fdwand, sgs_sism); // 02.09.: BEIDE Parameter wirklich durchreichen (der Regex-Umbau hatte diese Zeile verfehlt -- Lauf 3 ist am neuen No-Op-Waechter LAUT gescheitert, genau dafuer ist er da)
}

void LBM_Domain::finalize_sparse_tiles() {
	// Nach der Voxelisierung aufrufen: erst dann steht fest, welche Tiles voll solid sind.
	// Eine Tile ist tot, wenn sie SAMT 2-Zell-Halo vollstaendig solid ist. Der Halo muss 2 sein, nicht 1:
	// update_force_field liest via load_f die Nachbarn wand-adjazenter SOLID-Zellen, greift also bis zu
	// zwei Zellen weit -- jede Zelle, die hoechstens 2 von Fluid entfernt ist, muss in einer aktiven Tile
	// bleiben. Mit 1-Halo waeren die Kraefte an der Wand still falsch.
	if(!sparse_on) return;
	const uint Nx=(uint)get_Nx(), Ny=(uint)get_Ny(), Nz=(uint)get_Nz(), T=sparse_T;
	const ulong n_tiles = (ulong)sparse_tiles_x*sparse_tiles_y*sparse_tiles_z;
	auto IDX = [&](const uint x, const uint y, const uint z) { return (ulong)x+((ulong)y+(ulong)z*(ulong)Ny)*(ulong)Nx; };
	std::vector<uint> slot(n_tiles, 0xFFFFFFFFu);
	uint n_active = 0u;
	for(uint tz=0u; tz<sparse_tiles_z; tz++) for(uint ty=0u; ty<sparse_tiles_y; ty++) for(uint tx=0u; tx<sparse_tiles_x; tx++) {
		bool all_solid = true;
		const int x0=(int)(tx*T)-2, x1=(int)(tx*T+T+1), y0=(int)(ty*T)-2, y1=(int)(ty*T+T+1), z0=(int)(tz*T)-2, z1=(int)(tz*T+T+1);
		for(int z=z0; z<=z1&&all_solid; z++) for(int y=y0; y<=y1&&all_solid; y++) for(int x=x0; x<=x1&&all_solid; x++) {
			if(x<0||x>=(int)Nx||y<0||y>=(int)Ny||z<0||z>=(int)Nz) continue; // Domaenenrand zaehlt nicht als Fluid
			if((flags[IDX((uint)x,(uint)y,(uint)z)]&TYPE_S)==0u) all_solid = false;
		}
		// Slots werden ab 1 vergeben: Slot 0 ist ein PAPIERKORB. Grund: store_f schreibt nicht nur an den
		// eigenen Index, sondern auch an die Indizes der Nachbarn. Eine Zelle in einer AKTIVEN Tile kann
		// sehr wohl einen Nachbarn in einer toten Tile haben (sie ist dann selbst solid) -- und cell_base
		// liefert fuer tote Tiles den Notfallwert 0. Zaehlte man die echten Slots ab 0, landeten diese
		// Schreibzugriffe mitten in einer echten Tile und zerstoerten sie. Genau daran sind die ersten
		// T=8- und T=4-Laeufe divergiert (Cd 18.4 bzw. 22.4), auch noch mit is_dead_tile-Ausstieg.
		if(!all_solid) slot[(ulong)tx+(ulong)sparse_tiles_x*((ulong)ty+(ulong)sparse_tiles_y*tz)] = 1u + n_active++;
	}
	for(ulong i=0ull; i<n_tiles; i++) tile_slot[i] = slot[i];
	tile_slot.write_to_device();

	const ulong sparse_cells = (ulong)(n_active+1u)*(ulong)T*T*T; // +1 fuer den Papierkorb-Slot 0
	const double full_gb = (double)get_N()*velocity_set*sizeof(fpxx)/1e9;
	const double sparse_gb = (double)sparse_cells*velocity_set*sizeof(fpxx)/1e9;
	fi = Memory<fpxx>(device, sparse_cells, velocity_set, false); // Move-Assign gibt nur den Platzhalter frei
	kernel_initialize.set_parameters(0u, fi);
	kernel_stream_collide.set_parameters(0u, fi);
	kernel_update_fields.set_parameters(0u, fi);
	kernel_boden_eq.set_parameters(0u, fi); // XL-Audit B2 (Pruefagent R2: NICHT unter FORCE_FIELD): ohne Rebind hielte boden_eq das cl_mem des FREIGEGEBENEN Platzhalters
	kernel_einlass_eq.set_parameters(0u, fi); // EINLASS_EQ: dito (Sparse-Rebind AUSSERHALB von FORCE_FIELD)
	if(rho_rek_max>0ull) kernel_rho_rek_ebene.set_parameters(0u, fi);     // ★ 15.09. RHO_RAND C1/C2a (C2-Plan N7): heute nie gesetzt (Alloc erst nach dem Init),
	if(rho_aus_max>0ull) kernel_rho_ausgabe_ebene.set_parameters(0u, fi); // aber ohne Rebind hielte ein frueher Alloc das freigegebene fi
	if(apg_on&&fac_N>0ull) kernel_fac_apg.set_parameters(4u, fi); // ★ 16.09. (Pruefagent NIEDRIG): fi ist Position 4 in fac_apg_ab; heute nie erreicht (Alloc nach dem Init), aber ein frueherer Alloc hielte sonst das freigegebene fi
#ifdef FORCE_FIELD
	kernel_update_force_field.set_parameters(0u, fi);
#endif // FORCE_FIELD
	print_info("[SPARSE] "+to_string(n_active)+"/"+to_string(n_tiles)+" Tiles aktiv (T="+to_string(T)+") -> fi "
		+to_string((float)sparse_gb,2u)+" GB statt "+to_string((float)full_gb,2u)+" GB, also "
		+to_string((float)(full_gb-sparse_gb),2u)+" GB frei.");
}

void LBM_Domain::enqueue_initialize() { // call kernel_initialize
	kernel_initialize.enqueue_run();
	if(nachbar_on&&fac_N>0ull) kernel_fac_nachbar.enqueue_run(); // ★ 16.09. (Pruefagent NIEDRIG): fac_apg_ab NICHT im Init -- fac_nb[2..4] ist 0-initialisiert (= kein Gradient, Aequilibrium-Start), und ein Init-Launch bei t=1 zaehlte [306] am Zaehltakt 1 ohne Facettenbesuch
}
void LBM_Domain::enqueue_stream_collide() { // call kernel_stream_collide to perform one LBM time step
	// ★ Invarianten-Waechter (Pruefagent Rang-1-Remat, NIEDRIG-3): der Remat-Block im Kernel
	// verlaesst sich darauf, dass t ein monotoner Schrittzaehler < 2^62 bleibt (t>>62 == 0).
	if(t>=(1ull<<62)) print_error("enqueue_stream_collide: t >= 2^62 -- die Remat-Invariante (t>>62==0) waere verletzt.");
	// ★ TODO 2 Schritt 1: rho_voll = 1 heisst "schreibe rho ueberall" (heutiges Verhalten).
	// rho_takt = 0 -> immer 1, also bitgleich zum Stand vor der Aenderung. Sonst 1 nur an dem
	// Schritt, NACH dem der Host das ganze Feld liest (Sample-Kadenz = sample_every*ratio feine
	// Schritte), oder wenn der Host es ausdruecklich erzwingt (Abschlusspfad, unregelmaessige Lesung).
	// BITFELD: Bit 0 = rho ueberall schreiben, Bit 1 = u ueberall schreiben. Takt 0 heisst "aus",
	// dann ist das Bit immer gesetzt und das Verhalten bitgleich zum Stand vor der Aenderung.
	// ★ BERICHTIGT 12.09.2026, nach einer GESCHEITERTEN Abnahme: hier stand fuer beide Felder eine
	// Takt-Arithmetik ueber t ((t+1)%takt==0). Sie hat im FERNFELD nicht getroffen -- 23.425.595
	// Zellen standen im Feld-Dump veraltet, exakt die Zahl, die der Zaehler als uebersprungen meldete.
	// Der Host weiss genau, wann er das ganze Feld liest; er sagt es jetzt ausdruecklich an
	// (rho_voll_zwang), statt dass die Domaene es aus ihrem Schrittzaehler erraet.
	// Was BLEIBT ist die Substep-Regel fuer u: dass am letzten Substep jedes Grobschritts voll
	// geschrieben werden muss, folgt aus der N2F-Entnahme und ist eine Eigenschaft der Domaene,
	// keine Absprache mit dem Host.
	const uint rho_bit = (rho_takt==0u||rho_voll_zwang) ? 1u : 0u;
	const uint u_bit   = (u_takt==0u  ||rho_voll_zwang||((t+1ull)%(ulong)u_takt)==0ull) ? 2u : 0u;
	felder_voll_h = rho_bit|u_bit;
	kernel_stream_collide.set_parameters(4u, t, fx, fy, fz, felder_voll_h).enqueue_run();
	if(fdwand_on&&fac_N>0ull) { if(sism_on) kernel_sgs_fdwand.set_parameters(5u, t); kernel_sgs_fdwand.enqueue_run(); }
	if(band_on&&band_N>0ull&&!band_pi_on) { if(sism_on) kernel_sgs_band.set_parameters(5u, t); kernel_sgs_band.enqueue_run(); } // ★ 22.09. Plan C: im Pi-Modus kein FD-Bandkernel // ★ 08.09. SGS-BAND: zweiter Launch desselben Kernels ueber die Bandzellen, dieselbe In-Order-Queue -> derselbe Determinismus wie Lage 1 // ★ Audit-Befund 11 (07.09.): Waechter auf fac_N statt fac_wfd.length()>1 -- bei GENAU EINER aktiven Facette ist die Laenge 1 und der FD-Kernel wurde still uebersprungen (Platzhalter und Einzelfacette nicht unterscheidbar; dieselbe Falle wie fac_nb 03.09.). fac_N wird nur in alloc_facetten_domain gesetzt. // ★ 07.09. SISM: t je Schritt nachfuehren (Muster sgs_gdiag/boden_eq), Position 5 = erstes SGS_SISM-Argument; der FD-Kernel sieht dasselbe t wie der eben gerechnete Schritt (increment_time_step folgt erst danach)
	if(nachbar_on&&fac_N>0ull) { kernel_fac_nachbar.enqueue_run();
		if(apg_on) { kernel_fac_apg.set_parameters(5u, t);
			if(timer_apg>0u) { // ★ 22.09.2026 CFD_TIMER_APG: fac_apg_ab ISOLIERT. Erst alles Vorherige einholen (sonst misst die Uhr den Rueckstau von stream_collide und fac_nachbar_ab mit), dann den Vorkernel allein.
				finish_queue(); Clock c_apg; kernel_fac_apg.enqueue_run(); finish_queue();
				const double dt_ = c_apg.stop();
				apg_t_summe += dt_; apg_t_n++; if(dt_<apg_t_min) apg_t_min = dt_; if(dt_>apg_t_max) apg_t_max = dt_;
			} else kernel_fac_apg.enqueue_run(); // ★ AUS = bitgleich: kein finish_queue, kein Clock, kein veraenderter Aufrufpfad
		} } // ★ 16.09. APG: t (Position 5 in fac_apg_ab) je Schritt -- der Vorkernel liest load_f(t+1), das rho des naechsten Schritts. // ★ 03.09. Nachbarabtastung fuer den NAECHSTEN Schritt (Waechter fac_N>0: Platzhalter hat Laenge 2, Pruefagent Pass 2), in-order nach stream_collide (deterministisch); length-Guard = nie auf dem Platzhalter // ★ Geistermoden-Fix: FD-w fuer den NAECHSTEN Schritt, in-order nach stream_collide (deterministisch); length-Guard = nie auf dem Platzhalter
}
void LBM_Domain::enqueue_boden_eq() { // ★ V1-Port: post-stream Boden-Equilibrium (Staggered-Mode-Kur); No-Op bei n==0
	if(boden_eq_n==0u) return;
	if(!(boden_eq_u>=0.0f)) print_error("BODEN_EQ ist aktiv (n = "+to_string(boden_eq_n)+"), aber boden_eq_u traegt noch den Sentinel "+to_string(boden_eq_u,4u)+" -- das Setup hat LBM_Domain::s_boden_eq_u nicht auf sein u_lat gesetzt. Die mitbewegte Fahrbahn liefe mit einer anderen Geschwindigkeit als die Stroemung (Pruefbefund B7, 12.09.2026).");
	kernel_boden_eq.set_parameters(2u, t, boden_eq_u, boden_eq_n, boden_eq_down, boden_eq_split, boden_eq_abstand).enqueue_run();
}
void LBM_Domain::enqueue_einlass_eq() { // ★ V1-Port apply_inlet_velocity: post-stream Einlass-Equilibrium x=1..nx; No-Op bei n==0
	if(einlass_eq_n==0u) return;
	if(!(einlass_eq_u>=0.0f)) print_error("EINLASS_EQ ist aktiv (n = "+to_string(einlass_eq_n)+"), aber einlass_eq_u traegt noch den Sentinel "+to_string(einlass_eq_u,4u)+" -- das Setup hat LBM_Domain::s_einlass_eq_u nicht auf sein u_lat gesetzt (Pruefbefund B7, 12.09.2026).");
	kernel_einlass_eq.set_parameters(2u, t, einlass_eq_u, einlass_eq_n).enqueue_run();
}
void LBM_Domain::sgs_gdiag_gpu() { // ★ g-Diagnose: ein Mess-Launch ueber die Wandzellenliste (31.08.)
	if(!gdiag_on) return;
	kernel_sgs_gdiag.set_parameters(6u, t, fx, fy, fz).run();
	if(band_gdiag_on) kernel_band_gdiag.set_parameters(6u, t, fx, fy, fz).run(); // ★ 22.09. Band-g-Diagnose an derselben Kadenz (Besuche je Bandzelle == Besuche je Facette, Ist=Soll im Bericht) // t UND fx/fy/fz aktualisieren (Pruefbefund B-6a: der Kanal REGELT fx je Chunk -- der eingefrorene Startwert verfaelschte den Guo-Term unter SGS_GUO=1); run mit finish
}
void LBM_Domain::enqueue_update_fields() { // update fields (rho, u, T) manually
#ifndef UPDATE_FIELDS
	if(t!=t_last_update_fields) { // only run kernel_update_fields if the time step has changed since last update
		kernel_update_fields.set_parameters(4u, t, fx, fy, fz).enqueue_run();
		t_last_update_fields = t;
	}
#endif // UPDATE_FIELDS
}
#ifdef SURFACE
void LBM_Domain::enqueue_surface_0() {
	kernel_surface_0.set_parameters(7u, t, fx, fy, fz).enqueue_run();
}
void LBM_Domain::enqueue_surface_1() {
	kernel_surface_1.enqueue_run();
}
void LBM_Domain::enqueue_surface_2() {
	kernel_surface_2.set_parameters(4u, t).enqueue_run();
}
void LBM_Domain::enqueue_surface_3() {
	kernel_surface_3.enqueue_run();
}
#endif // SURFACE
#ifdef FORCE_FIELD
void LBM_Domain::enqueue_update_force_field() { // calculate forces from fluid on TYPE_S cells
	if(t!=t_last_force_field) { // only run kernel_update_force_field if the time step has changed since last update
		kernel_update_force_field.set_parameters(2u, t).enqueue_run();
		t_last_force_field = t;
	}
}
void LBM_Domain::enqueue_object_center_of_mass(const uchar flag_marker) { // calculate center of mass of all cells flagged with flag_marker
	object_sum.x[0] = 0.0f; // reset object_sum
	object_sum.y[0] = 0.0f;
	object_sum.z[0] = 0.0f;
	object_sum.enqueue_write_to_device();
	kernel_object_center_of_mass.set_parameters(1u, flag_marker).enqueue_run();
	object_sum.enqueue_read_from_device();
}
void LBM_Domain::enqueue_object_force(const uchar flag_marker) { // add up force for all cells flagged with flag_marker
	enqueue_update_force_field(); // update force field if it is not yet up-to-date
	// Kein Nullen mehr noetig: object_force_final schreibt mit "=", und jeder Teilsummen-Slot wird
	// von seiner Arbeitsgruppe jeden Lauf unbedingt geschrieben (Muster po_final_mean, 849b14f).
	kernel_object_force.set_parameters(3u, flag_marker).enqueue_run(); // ★ 03.09.: Index 2 -> 3, f_maske sitzt seit der F-Markerliste zwischen F und flags
	kernel_object_force_final.enqueue_run();
	object_sum.enqueue_read_from_device();
}
void LBM_Domain::enqueue_object_force_zband(const uchar flag_marker, const uint z_lo, const uint z_hi) { // FORK Kraft-Zerlegung: object_force auf das z-Band [z_lo,z_hi); object_sum WIEDERVERWENDET -- strikt sequenziell zu enqueue_object_force
	enqueue_update_force_field(); // update force field if it is not yet up-to-date
	// Kein Nullen mehr noetig: object_force_final schreibt mit "=", und jeder Teilsummen-Slot wird
	// von seiner Arbeitsgruppe jeden Lauf unbedingt geschrieben (Muster po_final_mean, 849b14f).
	kernel_object_force_zband.set_parameters(3u, flag_marker, z_lo, z_hi).enqueue_run(); // ★ 03.09.: Index 2 -> 3 (f_maske)
	kernel_object_force_final.enqueue_run();
	object_sum.enqueue_read_from_device();
}
void LBM_Domain::enqueue_object_torque(const float3& rotation_center, const uchar flag_marker) { // add up torque around specified rotation_center for all cells flagged with flag_marker
	enqueue_update_force_field(); // update force field if it is not yet up-to-date
	object_sum.x[0] = 0.0f; // reset object_sum
	object_sum.y[0] = 0.0f;
	object_sum.z[0] = 0.0f;
	object_sum.enqueue_write_to_device();
	kernel_object_torque.set_parameters(3u, flag_marker, rotation_center.x, rotation_center.y, rotation_center.z).enqueue_run(); // ★ 03.09.: Index 2 -> 3 (f_maske)
	object_sum.enqueue_read_from_device();
}
// ★ kraft_facetten-GPU-Reduktion (Muster init_pressure_outlet/set_pressure_outlet_faces): Liste der
// Markerzellen hochladen, Gruppenpuffer anlegen, Kernel binden. Der Schluessel (marker,z_per) steht
// in kf_marker/kf_zper -- der Aufrufer (setup.cpp kraft_facetten) bindet bei Wechsel neu.
void LBM_Domain::bind_kraft_facetten(const std::vector<ulong>& liste, const uchar marker, const bool z_per, const bool band_slot) {
	// ★ FORK Kraft-Zerlegung: band_slot=true waehlt den kfb_*-Membersatz (z-Band-Teilliste), sonst
	// laeuft alles wortgleich ueber den Hauptslot. kfb_zband setzt der Aufrufer (setup.cpp).
	Memory<uint>& liste_m = band_slot ? kfb_liste : kf_liste;
	Memory<float>& psum_m  = band_slot ? kfb_psum  : kf_psum;
	Memory<uint>&  pcnt_m  = band_slot ? kfb_pcnt  : kf_pcnt;
	Kernel& kernel_m = band_slot ? kernel_kraft_facetten_band : kernel_kraft_facetten;
	const ulong liste_n = (ulong)liste.size();
	if(liste_n>0xFFFFFFFFull) print_error("kraft_facetten-Liste ueberschreitet 2^32 Zellen -- uint-Cast im Kernel-Argument wuerde stumm abschneiden (R1-N4)."); // praktisch unerreichbar, aber billig
	if(band_slot) { kfb_N = liste_n; kfb_marker = marker; kfb_zper = z_per; kfb_bound = true; } // eigene Schluessel (Pruefagent M)
	else { kf_N = liste_n; kf_marker = marker; kf_zper = z_per; kf_bound = true; }
	if(liste_n==0ull) return; // leere Liste: kraft_facetten_gpu liefert Nullen ohne Launch
	liste_m = Memory<uint>(device, liste_n); // Ctor-Nullinit + zweiter Voll-Write = ein verschenkter 16-MB-Transfer, EINMALIG beim Bind -- bewusst toleriert (Pruefagent N2)
	if(get_N()>0xFFFFFFFFull) print_error("kf_liste/kfb_liste sind seit 08.09. uint (VRAM) -- bei N > 2^32 wuerde jeder Zellindex still abgeschnitten. Der Waechter oben prueft die LISTENLAENGE, nicht den Indexwert (Pruefagent-Befund B1).");
	for(ulong i=0ull; i<liste_n; i++) liste_m[i] = (uint)liste[i];
	liste_m.write_to_device();
	// ★ 11.09.2026 HOST-SPIEGEL FREIGEBEN (VRAM-Audit). Dieser Puffer wird EINMAL gefuellt,
	// hochgeladen und danach host-seitig nie wieder angefasst -- weder per read_from_device()
	// noch per Index. delete_host_buffer() ist seit 03.09. entschaerft (Aux-Zeiger, Double-Free,
	// Zero-Copy-Waechter), wurde aber nirgends gerufen.
	// DIE BEDINGUNG IST EINSEITIG UND DESHALB SICHER: is_zero_copy verlangt uses_ram, also
	// schliesst !uses_ram Zero-Copy aus. Auf der iGPU IST der Host-Puffer der Geraetespeicher,
	// dort wuerde die Freigabe die laufende Rechnung lautlos zerstoeren -- deshalb nur dGPU.
	if(!device.info.uses_ram) liste_m.delete_host_buffer();
	const ulong gruppen = (liste_n+(ulong)WORKGROUP_SIZE-1ull)/(ulong)WORKGROUP_SIZE; // = ceil(liste_n/64.0)
	psum_m = Memory<float>(device, 3ull*gruppen);
	pcnt_m = Memory<uint>(device, 3ull*gruppen);
	// Im AUS-Arm (!facetten_on) existieren fac_idx/fac_tau_n/fac_geo NICHT (allocate bindet sie nur
	// mit facetten_on, s.o.) -- 1-Element-Dummies anlegen, damit der Kernel gueltige Puffer bekommt.
	// NUR wenn noch nie alloziert (length()==0): ein Move-Assignment auf einen bereits als Kernel-Arg
	// gebundenen Puffer waere genau der DIAGZ-Use-after-free (Nachpruefer-Lektion in alloc_facetten_domain).
	if(!facetten_on) {
		if(fac_idx.length()==0ull)   { fac_idx   = Memory<uint>(device, 2ull);  fac_idx[0]=0u; fac_idx[1]=0u; fac_idx.write_to_device(); } // Bitmaske: Maske 0 = nie ein fid
		if(fac_tau_n.length()==0ull) { fac_tau_n = Memory<uint>(device, 1ull);  fac_tau_n[0]=0u;        fac_tau_n.write_to_device(); }
		if(fac_geo.length()==0ull)   { fac_geo   = Memory<float>(device, 8ull); for(ulong q=0ull;q<8ull;q++) fac_geo[q]=0.0f; fac_geo.write_to_device(); }
	}
	kernel_m = Kernel(device, liste_n, "kraft_facetten_gpu", F, f_maske, liste_m, (uint)liste_n,
		fac_idx, fac_tau_n, fac_geo, facetten_on?1u:0u, z_per?1u:0u, psum_m, pcnt_m);
}
void LBM_Domain::kraft_facetten_gpu(double& px, double& py, double& pz, ulong& n_voll, ulong& n_proj, ulong& n_unklar, const bool band_slot) {
	px=py=pz=0.0; n_voll=n_proj=n_unklar=0ull;
	const ulong liste_n = band_slot ? kfb_N : kf_N; // ★ FORK Kraft-Zerlegung: band_slot -> kfb_*-Satz
	if(liste_n==0ull) return; // keine Markerzellen: Nullen ohne Launch
	// ★ run() MIT finish, nicht enqueue_run(): kf_psum/kf_pcnt sind auf CPU/iGPU ZERO-COPY-Puffer
	// (CL_MEM_USE_HOST_PTR) -- dort erzwingt der "blockierende" read_from_device KEINE Ausfuehrung
	// der wartenden Kommandos, und der Kernel lief erst mit dem naechsten Queue-Flush. Gemessen als
	// Ein-Aufruf-Versatz im PRUEF-Doppellauf (GPU-Werte = Host-Werte des VORHERIGEN Aufrufs, erster
	// Aufruf Nullen). Die in-order-Queue stellt zugleich sicher, dass enqueue_update_force_field
	// davor abgearbeitet ist.
	Memory<float>& psum_m = band_slot ? kfb_psum : kf_psum;
	Memory<uint>&  pcnt_m = band_slot ? kfb_pcnt : kf_pcnt;
	(band_slot ? kernel_kraft_facetten_band : kernel_kraft_facetten).run();
	psum_m.read_from_device();
	pcnt_m.read_from_device();
	const ulong gruppen = (liste_n+(ulong)WORKGROUP_SIZE-1ull)/(ulong)WORKGROUP_SIZE;
	for(ulong g=0ull; g<gruppen; g++) { // double-Endsumme in FESTER Gruppenreihenfolge (deterministisch)
		px += (double)psum_m[3ull*g]; py += (double)psum_m[3ull*g+1ull]; pz += (double)psum_m[3ull*g+2ull];
		n_voll += (ulong)pcnt_m[3ull*g]; n_proj += (ulong)pcnt_m[3ull*g+1ull]; n_unklar += (ulong)pcnt_m[3ull*g+2ull];
	}
}
#endif // FORCE_FIELD
#ifdef MOVING_BOUNDARIES
void LBM_Domain::enqueue_update_moving_boundaries() { // mark/unmark cells next to TYPE_S cells with velocity!=0 with TYPE_MS
	kernel_update_moving_boundaries.enqueue_run();
}
#endif // MOVING_BOUNDARIES
#ifdef PARTICLES
void LBM_Domain::enqueue_integrate_particles(const uint time_step_multiplicator) { // intgegrate particles forward in time and couple particles to fluid
#ifdef FORCE_FIELD
	if(particles_rho!=1.0f) kernel_reset_force_field.enqueue_run(); // only reset force field if particles have buoyancy and apply forces on fluid
	kernel_integrate_particles.set_parameters(5u, fx, fy, fz);
#endif // FORCE_FIELD
	kernel_integrate_particles.set_parameters(3u, (float)time_step_multiplicator).enqueue_run();
}
#endif // PARTICLES

void LBM_Domain::increment_time_step(const ulong steps) {
	t += steps; // increment time step
#ifdef UPDATE_FIELDS
	t_last_update_fields = t;
#endif // UPDATE_FIELDS
}
void LBM_Domain::reset_time_step() {
	t = 0ull; // increment time step
#ifdef UPDATE_FIELDS
	t_last_update_fields = t;
#endif // UPDATE_FIELDS
}
void LBM_Domain::flush_queue() { device.flush_queue(); } // Perf-Audit: siehe run_async

void LBM_Domain::finish_queue() {
	device.finish_queue();
}

uint LBM_Domain::get_velocity_set() const {
	return velocity_set;
}

void LBM_Domain::voxelize_mesh_on_device(const Mesh* mesh, const uchar flag, const float3& rotation_center, const float3& linear_velocity, const float3& rotational_velocity) { // voxelize triangle mesh
	Memory<float3> p0(device, mesh->triangle_number, 1u, mesh->p0);
	Memory<float3> p1(device, mesh->triangle_number, 1u, mesh->p1);
	Memory<float3> p2(device, mesh->triangle_number, 1u, mesh->p2);
	Memory<float> bounding_box_and_velocity(device, 16u);
	const float x0=mesh->pmin.x-2.0f, y0=mesh->pmin.y-2.0f, z0=mesh->pmin.z-2.0f, x1=mesh->pmax.x+2.0f, y1=mesh->pmax.y+2.0f, z1=mesh->pmax.z+2.0f; // use bounding box of mesh to speed up voxelization; add tolerance of 2 cells for re-voxelization of moving objects
	bounding_box_and_velocity[ 0] = as_float(mesh->triangle_number);
	bounding_box_and_velocity[ 1] = x0;
	bounding_box_and_velocity[ 2] = y0;
	bounding_box_and_velocity[ 3] = z0;
	bounding_box_and_velocity[ 4] = x1;
	bounding_box_and_velocity[ 5] = y1;
	bounding_box_and_velocity[ 6] = z1;
	bounding_box_and_velocity[ 7] = rotation_center.x;
	bounding_box_and_velocity[ 8] = rotation_center.y;
	bounding_box_and_velocity[ 9] = rotation_center.z;
	bounding_box_and_velocity[10] = linear_velocity.x;
	bounding_box_and_velocity[11] = linear_velocity.y;
	bounding_box_and_velocity[12] = linear_velocity.z;
	bounding_box_and_velocity[13] = rotational_velocity.x;
	bounding_box_and_velocity[14] = rotational_velocity.y;
	bounding_box_and_velocity[15] = rotational_velocity.z;
	uint direction = 0u;
	if(length(rotational_velocity)==0.0f) { // choose direction of minimum bounding-box cross-section area
		float v[3] = { (y1-y0)*(z1-z0), (z1-z0)*(x1-x0), (x1-x0)*(y1-y0) };
		float vmin = v[0];
		for(uint i=1u; i<3u; i++) {
			if(v[i]<vmin) {
				vmin = v[i];
				direction = i;
			}
		}
	} else { // choose direction closest to rotation axis
		float v[3] = { fabsf(rotational_velocity.x), fabsf(rotational_velocity.y), fabsf(rotational_velocity.z) };
		float vmax = v[0];
		for(uint i=1u; i<3u; i++) {
			if(v[i]>vmax) {
				vmax = v[i];
				direction = i; // find direction of minimum bounding-box cross-section area
			}
		}
	}
	const ulong A[3] = { (ulong)Ny*(ulong)Nz, (ulong)Nz*(ulong)Nx, (ulong)Nx*(ulong)Ny };
	Kernel kernel_voxelize_mesh(device, A[direction], "voxelize_mesh", direction, fi, u, flags, t+1ull, flag, p0, p1, p2, bounding_box_and_velocity);
#ifdef SURFACE
	kernel_voxelize_mesh.add_parameters(mass, massex);
#endif // SURFACE
	if(sparse_on) kernel_voxelize_mesh.add_parameters(tile_slot); // TS_P haengt tile_slot hinten an
	p0.write_to_device();
	p1.write_to_device();
	p2.write_to_device();
	bounding_box_and_velocity.write_to_device();
	kernel_voxelize_mesh.run();
}
void LBM_Domain::enqueue_unvoxelize_mesh_on_device(const Mesh* mesh, const uchar flag) { // remove voxelized triangle mesh from LBM grid
	const float x0=mesh->pmin.x, y0=mesh->pmin.y, z0=mesh->pmin.z, x1=mesh->pmax.x, y1=mesh->pmax.y, z1=mesh->pmax.z; // remove all flags in bounding box of mesh
	Kernel kernel_unvoxelize_mesh(device, get_N(), "unvoxelize_mesh", flags, flag, x0, y0, z0, x1, y1, z1);
	kernel_unvoxelize_mesh.run();
}

// ★ Auditor-B B-1 (26.08.): werkzeuge/scratch_gate/gen_main.cpp FRIERT diese Define-Liste fuer
// das Offline-Scratch-Gate ein (Kanal-Referenzpunkt). Wer hier Defines aendert/ergaenzt, zieht
// den Zwilling nach -- sonst prueft das Gate still eine Quelle, die niemand mehr faehrt.
// Voller Drift-Anker (Gate difft gegen frischen CFD_DUMP_DEFINES-Dump): Folgepunkt im Plan.
// ★★ P-TRT-EMISSION UND IHRE WAECHTER (10.09.2026 abends, nach der Diff-Pruefung).
// DREI FALLEN, die der Pruefagent an der ersten Fassung fand:
//  (1) atof bricht am ersten unbrauchbaren Zeichen ab. CFD_PTRT=1,9 in deutscher
//      Kommaschreibweise haette still 1.0 ergeben -- und 1,0 ist ausgerechnet das schlechte
//      Ende der Skala (e-Faltung 158 gegen 463 im heutigen Stand). Der Lauf haette
//      "omega_g = 1,9" geheissen und 1,0 gerechnet. Deshalb strtod mit Endzeigerpruefung.
//  (2) Ohne Ansage im Log ist hinterher nicht nachvollziehbar, welcher Wert wirklich
//      emittiert wurde. to_string(float, 12u) klemmt intern auf 8 Nachkommastellen.
//  (3) Der Kernelblock ist D3Q19-spezifisch (sieben Korrekturwerte, def_w0/ws/we). Unter
//      D3Q27 waere er kein Projektor mehr und wuerde still Masse einspeisen, unter D2Q9
//      schriebe er ueber fhn[8] hinaus. Deshalb wird PTRT ausserhalb von D3Q19 GAR NICHT
//      emittiert, und der Schalter meldet sich als Fehler statt still zu wirken.
string ptrt_defines() {
	const char* roh = getenv("CFD_PTRT");
	if(roh==nullptr||roh[0]=='\0') return "";
	char* ende = nullptr;
	const double wert = strtod(roh, &ende);
	while(ende!=nullptr&&(*ende==' '||*ende=='\t'||*ende=='\r'||*ende=='\n')) ende++; // ★ auch Tab und CR, sonst scheitert eine CRLF-Seriendatei
	if(ende==nullptr||*ende!='\0') print_error("CFD_PTRT = \""+string(roh)+"\" ist keine reine Zahl (Rest: \""+string(ende==nullptr?"":ende)+"\"). Dezimaltrenner ist der PUNKT: CFD_PTRT=1.95, nicht 1,95. Ein stillschweigend abgeschnittener Wert waere hier besonders teuer, weil 1,0 und 1,95 auf entgegengesetzten Enden der Skala liegen.");
	// ★ AB HIER GEGEN DEN FLOAT PRUEFEN, nicht gegen den double (Host-Pruefer 10.09. nachts):
	// emittiert wird to_string((float)wert), und CFD_PTRT=1.99999995 ist als double < 2, als
	// float aber EXAKT 2.0f. Der Waechter haette genau den Fall durchgelassen, den er verhindern
	// soll. Und die Bedingung muss POSITIV formuliert sein: bei NaN sind sowohl wert<=0 als auch
	// wert>=2 falsch, NaN passierte beide Waechter und landete als "NaNf" im Kernel.
	const float wf = (float)wert;
	if(!(wf>0.0f&&wf<2.0f)) {
		if(wf!=wf) print_error("CFD_PTRT = \""+string(roh)+"\" ergibt NaN. omega_g muss eine Zahl in (0, 2) sein.");
		if(wf>=2.0f) print_error("CFD_PTRT = "+to_string(wf,6u)+" ist >= 2. Der Geistanteil waechst dann je Schritt um |1-omega_g| >= 1, der Lauf ist unbedingt instabil. Erlaubt ist 0 < omega_g < 2.");
		return ""; // wf <= 0 heisst "aus" -- ohne Meldung, das ist der dokumentierte Aus-Zustand
	}
#ifndef D3Q19
	print_error("CFD_PTRT ist gesetzt, aber dieser Build ist nicht D3Q19. Der P-TRT-Block kennt nur die drei D3Q19-Geistmoden und die Gewichte def_w0/def_ws/def_we; unter D3Q27 waere er kein Projektor mehr (er speiste Masse ein), unter D2Q9 schriebe er ueber das DDF-Feld hinaus. Velocity set in defines.hpp aendern oder CFD_PTRT weglassen.");
	return "";
#else
	print_info("P-TRT AKTIV: omega_g = "+to_string(wf,8u)+" (Geistanteil des geraden Nichtgleichgewichts relaxiert mit dieser Rate statt mit der Kollisionsrate w). Wirkpfad Slots 199/200/201, Abnahme pruefe_ptrt.");
	return (string)"\n	#define PTRT"
	      +"\n	#define def_omega_g "+to_string(wf, 12u)+"f";
#endif // D3Q19
}

string LBM_Domain::device_defines(const Device_Info& device_info) const { return
	"\n	#define def_Nx "+to_string(Nx)+"u"
	"\n	#define def_Ny "+to_string(Ny)+"u"
	"\n	#define def_Nz "+to_string(Nz)+"u"
	"\n	#define def_N "+to_string(get_N())+"ul"
	"\n	#define uxx "+(get_N()<=(ulong)max_uint ? "uint" : "ulong")+"" // switchable data type for index calculation (32-bit uint / 64-bit ulong)

	"\n	#define def_GNx "+to_string((Nx-2u*(uint)(Dx>1u))*Dx)+"u" // global LBM grid resolution of all domains together
	"\n	#define def_GNy "+to_string((Ny-2u*(uint)(Dy>1u))*Dy)+"u"
	"\n	#define def_GNz "+to_string((Nz-2u*(uint)(Dz>1u))*Dz)+"u"

	"\n	#define def_Dx "+to_string(Dx)+"u"
	"\n	#define def_Dy "+to_string(Dy)+"u"
	"\n	#define def_Dz "+to_string(Dz)+"u"

	"\n	#define def_Ox "+to_string(Ox)+"" // offsets are signed integer!
	"\n	#define def_Oy "+to_string(Oy)+""
	"\n	#define def_Oz "+to_string(Oz)+""

	"\n	#define def_Ax "+to_string(Ny*Nz)+"u"
	"\n	#define def_Ay "+to_string(Nz*Nx)+"u"
	"\n	#define def_Az "+to_string(Nx*Ny)+"u"

	"\n	#define def_domain_offset_x "+to_string(0.5f*(float)((int)Nx+2*Ox+(int)Dx*(2*(int)(Dx>1u)-(int)Nx)))+"f"
	"\n	#define def_domain_offset_y "+to_string(0.5f*(float)((int)Ny+2*Oy+(int)Dy*(2*(int)(Dy>1u)-(int)Ny)))+"f"
	"\n	#define def_domain_offset_z "+to_string(0.5f*(float)((int)Nz+2*Oz+(int)Dz*(2*(int)(Dz>1u)-(int)Nz)))+"f"

	"\n	#define D"+to_string(dimensions)+"Q"+to_string(velocity_set)+"" // D2Q9/D3Q15/D3Q19/D3Q27
	"\n	#define def_velocity_set "+to_string(velocity_set)+"u" // LBM velocity set (D2Q9/D3Q15/D3Q19/D3Q27)
	"\n	#define def_dimensions "+to_string(dimensions)+"u" // number spatial dimensions (2D or 3D)
	"\n	#define def_transfers "+to_string(transfers)+"u" // number of DDFs that are transferred between multiple domains

	+(klemm_haken_env()==2u ? "\n	#define def_c 0.05000000f" : "\n	#define def_c 0.57735027f") // lattice speed of sound c = 1/sqrt(3)*dt; ★ Klemmen-Haken 2 (nur Testarme): u-Klemme auf 0,05. Z2a: dieselbe c_s wie RHO_KLEMM_CS2 (defines.hpp) -- u_max = c_s ist die Grenze der Konsistenzhuelle, aus der RHO_CLAMP_MIN/MAX folgen
	+	"\n	#define def_w " +to_string(1.0f/get_tau())+"f" // relaxation rate w = dt/tau = dt/(nu/c^2+dt/2) = 1/(3*nu+1/2)
#if defined(D2Q9)
	"\n	#define def_w0 (1.0f/2.25f)" // center (0)
	"\n	#define def_ws (1.0f/9.0f)" // straight (1-4)
	"\n	#define def_we (1.0f/36.0f)" // edge (5-8)
#elif defined(D3Q15)
	"\n	#define def_w0 (1.0f/4.5f)" // center (0)
	"\n	#define def_ws (1.0f/9.0f)" // straight (1-6)
	"\n	#define def_wc (1.0f/72.0f)" // corner (7-14)
#elif defined(D3Q19)
	"\n	#define def_w0 (1.0f/3.0f)" // center (0)
	"\n	#define def_ws (1.0f/18.0f)" // straight (1-6)
	"\n	#define def_we (1.0f/36.0f)" // edge (7-18)
#elif defined(D3Q27)
	"\n	#define def_w0 (1.0f/3.375f)" // center (0)
	"\n	#define def_ws (1.0f/13.5f)" // straight (1-6)
	"\n	#define def_we (1.0f/54.0f)" // edge (7-18)
	"\n	#define def_wc (1.0f/216.0f)" // corner (19-26)
#endif // D3Q27

#if defined(SRT)
	"\n	#define SRT"
#elif defined(TRT)
	"\n	#define TRT"
	// ★★ LAMBDA ALS LAUFZEITPARAMETER, 2026-08-09. Bisher stand 0.1875f hart im Kernel.
	// Gerechnet (von-Neumann, D3Q19, tau+ = 0,5000071, u_lat = 0,075):
	//   Lambda = 3/16  -> tau- = 26409, max|Eigenwert| = 1,005543, e-Faltung 181 Schritte
	//   Lambda = 1/4   -> praktisch unveraendert (w- ist laengst null) -- der Umbau brachte NICHTS
	//   SRT            -> max|Eigenwert| = 1,003480, e-Faltung 288 Schritte  (GEMESSEN 2,6x besser)
	//   Lambda = 9,1e-8 (w- = 1,95) -> max|Eigenwert| = 1,000485, e-Faltung 2062 Schritte
	// ★ NACHGERECHNET 10.09.2026, unabhaengige Neuimplementierung werkzeuge/vonneumann.py:
	//   ALLE VIER ZEILEN BESTAETIGT. 3/16 und 9,1e-8 auf die letzte Stelle, SRT 1,003455/290
	//   (k-Gitter 96^3 mit Nachoptimierung) gegen 1,003480/288 hier.
	//   WER SIE NICHT REPRODUZIERT, HAT EIN ZU GROBES k-GITTER: bei 16^3 kommt fuer SRT
	//   1,000259 heraus -- falscher Betrag UND falsche Reihenfolge. Genau dieser Fehler liess
	//   am 10.09. einen Planungsagenten die Tabelle fuer falsch halten. Das SRT-Maximum sitzt
	//   in einem schmalen Gebiet bei kx ~ 0,02 pi, die TRT-Maxima nicht.
	// SRT ist der Sonderfall Lambda = (tau-1/2)^2. Ein Knopf deckt damit ALLE Operatoren ab, und
	// ungesetzt bleibt der Quelltext bit-identisch zum bisherigen Stand.
	+"\n	#define def_lambda "+to_string(getenv("CFD_LAMBDA")!=nullptr?(float)atof(getenv("CFD_LAMBDA")):0.1875f, 12u)+"f"

#endif // TRT

	// ★★ P-TRT (CFD_PTRT = omega_g), 10.09.2026. STEHT AUSSERHALB DES SRT/TRT-#elif, WEIL DIESER
	// FORK SRT RECHNET (defines.hpp:10; TRT ist in defines.hpp:19 auskommentiert). Zuerst stand er
	// drinnen und war damit wirkungslos -- die Abnahme fing es am selben Abend (Slot 199 = 0).
	// P-TRT gilt fuer beide Operatoren: relaxiert wird der Geistanteil des GERADEN Nichtgleich-
	// gewichts, den SRT wie TRT sonst mit der Kollisionsrate w behandeln. Relaxiert den Geistanteil des symmetrischen
	// Nichtgleichgewichts mit einer eigenen Rate statt mit wp. Ungesetzt oder <= 0 wird NICHTS
	// emittiert -- dann ist der OpenCL-Quelltext zeichengleich zum bisherigen Stand und der Lauf
	// bitgleich. Das ist der EINZIGE bitgleiche Kontrollarm: omega_g = wp taugt NICHT als Nullarm,
	// weil wp unter SUBGRID/SPONGE/FDWAND zellweise ist, def_omega_g aber eine JIT-Konstante --
	// (wp - omega_g) wird dann nirgends exakt null.
	// WELCHER WERT? Gerechnet mit werkzeuge/vonneumann.py auf der SRT-BASIS (dem Operator, den
	// dieser Fork wirklich rechnet), am Betriebspunkt tau = 0,50002832 (4-mm-Nahfeld) und
	// u_lat = 0,075. BEIDE Angaben gehoeren dazu, und dazu DIE k-AUFLOESUNG -- ohne sie ist die
	// Tabelle nicht reproduzierbar. Werte aus k-Gitter 72^3 mit Nachoptimierung, konvergenzgeprueft:
	//   omega_g   max|Eigenwert|   e-Faltung   Akkumulation 1/(1-|1-omega_g|)
	//     1,0        1,006362          158            1
	//     1,80       1,001000         1000            5
	//     1,84       1,000771         1298            6
	//     1,87       1,000603         1659            8
	//     1,90       1,000447         2236           10     <- OPTIMUM
	//     1,93       1,000554         1806           14
	//     1,95       1,000772         1296           20
	//     w (heute)  1,003387          296         8828
	// Bei 1,90 verbessern sich BEIDE Kriterien: e-Faltung Faktor 7,6 gegen heute, Akkumulation
	// von 8828 auf 10. Der Verlauf ist NICHT monoton, das Maximum liegt zwischen 1,87 und 1,93.
	// VOLLE PURIFIKATION (omega_g = 1,0) IST DIE SCHLECHTESTE WAHL: 158 liegt UNTER dem heutigen
	// Stand. Sie steht im Preprint als "Geist-Eigenwert auf null", ist hier aber der schlechteste
	// Punkt der Skala.
	// ★ ACHTUNG, TEUER GELERNT: eine fruehere Fassung dieser Tabelle stammte von einem 40er-Gitter
	// und nannte 1,95 als Optimum mit Faktor 7,3. Beides falsch. Ein feineres k-Gitter findet nur
	// GROESSERE Maxima, jeder zu grobe Wert ist also zu optimistisch -- und zwar je Arm
	// unterschiedlich stark, weshalb die REIHENFOLGE kippt. Unter n = 72 mit --fein ist hier keine
	// Zahl belastbar; die Begruendung steht im Kopf von werkzeuge/vonneumann.py.
	// Schranke: omega_g >= 2 ist unbedingt instabil (|1-omega_g| >= 1, der Geistanteil waechst
	// je Schritt), omega_g <= 0 hiesse "gar nicht relaxieren". Beides faengt ptrt_defines().
	+ptrt_defines()

	+"\n	#define TYPE_S 0x01" // 0b00000001 // (stationary or moving) solid boundary
	"\n	#define TYPE_E 0x02" // 0b00000010 // equilibrium boundary (inflow/outflow)
	"\n	#define TYPE_T 0x04" // 0b00000100 // temperature boundary
	"\n	#define TYPE_F 0x08" // 0b00001000 // fluid
	"\n	#define TYPE_I 0x10" // 0b00010000 // interface
	"\n	#define TYPE_G 0x20" // 0b00100000 // gas
	"\n	#define TYPE_X 0x40" // 0b01000000 // reserved type X
	"\n	#define TYPE_Y 0x80" // 0b10000000 // reserved type Y

	// ★★ DAEMPFUNGSZONE (Sponge), 2026-08-09 -- GEOMETRISCH an den Domaenenflaechen x-/x+/y-/y+/z+
	// verankert (Abstand zur Flaeche, KEIN Flag-Test; R2-Korrektur: der alte Text behauptete eine
	// TYPE_E-Bindung, die es nie gab -- in einem periodischen Setup rampte die Zone an Flaechen
	// ohne Rand, heute nur per Konvention verhindert, der Kanal setzt sie nicht). DER Weg, der nach drei
	// gescheiterten Randumbauten uebrig bleibt -- und der einzige, der durch Messung gestuetzt ist:
	// im leeren Fernfeld hat AUSSCHLIESSLICH die Viskositaet gedaempft (nu x1000: Streuung 0,040 ->
	// 0,025), waehrend jede Aenderung der Randgleichung das Klingeln verstaerkte. Die Zone hebt nu
	// nur in einem Streifen vor den Raendern an (quadratische Rampe), also dort, wo die Quelle sitzt
	// und die Reflexionen laufen -- nicht im Messvolumen. Sie liest nichts zurueck, erhaelt Masse
	// und Impuls (reine Aenderung der Relaxationsrate) und ist unter Esoteric Pull trivial sicher.
	// Der Boden z=0 ist AUSGENOMMEN: dort ist Fahrbahn, keine TYPE_E-Flaeche, und die Grenzschicht
	// darf nicht kuenstlich verdickt werden. V1s wirksame Klemmschicht war der harte Vorlaeufer
	// dieser Idee -- gleicher Ort, aber als f-Reset statt als Viskositaet.
	// Nur emittiert, wenn CFD_SPONGE_N gesetzt ist; ohne die Variable ist der Quelltext bit-identisch.
	// ★ Aus den STATIKEN, nicht aus getenv -- siehe die Begruendung bei ihrer Deklaration in lbm.hpp.
	// Das Setup setzt sie vor jedem Konstruktor; damit ist die Zone pro Domaene schaltbar, und die
	// Auswertung der Umgebungsvariablen laeuft ueber env_u/env_f im Setup, die den WERT auswerten
	// (CFD_SPONGE_N=0 heisst aus) statt nur auf das Literal "0" zu pruefen.
	+((s_sponge_n>0u) ? (string)
	"\n	#define SPONGE"
	"\n	#define def_sponge_n "+to_string(s_sponge_n)+"u"
	"\n	#define def_sponge_a "+to_string(s_sponge_a,4u)+"f" // 4 Nachkommastellen: bei kleinen a war 1 Stelle irrefuehrend
	"\n	#define def_sponge_wmin "+to_string(s_sponge_wmin,4u)+"f"
	: (string)"")
	// FORK: REG_E(i) ist der Randwert einer TYPE_E-Zelle -- reines Gleichgewicht wie bisher, oder mit
	// REGULARIZED_BOUNDARIES zusaetzlich der rekonstruierte Nichtgleichgewichtsanteil.
	//
	// ★★ ZWEI LEHREN VOM 2026-08-08 stecken in diesen paar Zeilen:
	// (1) Das Makro war einmal der VOLLE 19-Richtungs-Ausdruck und expandierte 19-fach in die ternaere
	//     Kollisionszeile. Der Intel-Uebersetzer blieb daran haengen; beim zweiten Mal fror der ganze
	//     Rechner ein, weil der Desktop auf derselben GPU laeuft. Jetzt ist es ein Aufruf der kleinen
	//     Funktion reg_fneq(), und der TYPE_E-Zweig ist in der Kollision herausgehoben.
	// (2) Der Schalter ist eine LAUFZEIT-Entscheidung (CFD_REG_BC), keine Compile-Zeit-Entscheidung.
	//     Damit liefert DASSELBE Binary mit CFD_REG_BC=0 exakt den alten OpenCL-Quelltext -- das ist
	//     der bit-genaue Kontrollarm fuer jedes A/B, und zugleich der Rettungsanker, falls der
	//     GPU-Uebersetzer am regularisierten Code doch wieder haengen sollte.
	// ★★ GEMESSEN UND DEFAULT AUS, 2026-08-09. A/B im leeren Fernfeld, gleiches Binary, Kontrollarm:
	//   bei 0,08 s Streuung 0,0719 (aus) gegen 0,1042 (an), Zellen ueber 10 % daneben 12,1 gegen 20,2 %.
	// Der regularisierte Einlass VERSTAERKT das Klingeln. Es ist der dritte Randumbau, der am selben
	// Muster scheitert: S wird aus dem u[] der Nachbarn gebildet, und die erste Fluidzelle ist genau
	// die, die die Stoerung traegt -- der Rand koppelt das Rauschen auf sich selbst zurueck, und bei
	// w -> 2 daempft die Kollision nichts, sie spiegelt. Der reine Gleichgewichts-Reset ist in diesem
	// Regime der am wenigsten schaedliche Rand, WEIL er nichts zurueckliest.
	// Der Code bleibt (mathematisch korrekt, Erhaltung symbolisch bestaetigt) fuer Regimes mit
	// ordentlichem tau; CFD_REG_BC=1 schaltet ihn ein.
#ifdef REGULARIZED_BOUNDARIES
	// Audit-Nacharbeit 15: atoi statt "alles ausser '0' ist an" -- CFD_REG_BC=false hiess vorher AN.
	+((getenv("CFD_REG_BC")!=nullptr&&atoi(getenv("CFD_REG_BC"))>0) ? (string)
	"\n	#define REGULARIZED_BOUNDARIES"
	"\n	#define REG_E(i) (feq[i]+reg_fneq(i, regf, Sxx, Syy, Szz, Sxy, Sxz, Syz, trS3))"
	: (string)
	"\n	#define REG_E(i) (feq[i])")
	+
#else
	+ "\n	#define REG_E(i) (feq[i])" // ★ 17.09.2026: hier fehlte das '+' -- der Zweig ohne REGULARIZED_BOUNDARIES war nie gebaut worden (erster D3Q27-Build)
#endif // REGULARIZED_BOUNDARIES
	#ifdef RHO_CLAMP
	"\n	#define RHO_CLAMP"
	+((klemm_haken_env()==1u||klemm_haken_env()==3u||klemm_haken_env()==4u) ? string("\n	#define RHO_CLAMP_MIN 1.0010f\n	#define RHO_CLAMP_MAX 1.0020f") // ★ Klemmen-Haken 1/3/4: nur Testarme
	  : (rho_huelle_env()>0u ? string("\n	#define RHO_CLAMP_MIN (20.0f/32768.0f)\n	#define RHO_CLAMP_MAX (1.0f+65504.0f/32768.0f)\n	#define RHO_HUELLE") // ★ Z2f: numerische Huelle als BRUECHE (to_string schnitte 20/32768 auf 0,0006)
	    +(klemm_haken_env()==5u ? string("\n	#define def_rho_kons_lo (1.0f-1.0f/32768.0f)\n	#define def_rho_kons_hi (1.0f+1.0f/32768.0f)") // ★ Pruefbefund Z2f M2: Haken 5 verengt auch die Konsistenzhuelle -> [298]+[299] MUSS feuern (nur Zaehler)
	      : "\n	#define def_rho_kons_lo "+to_string(RHO_CLAMP_MIN, 6u)+"f\n	#define def_rho_kons_hi "+to_string(RHO_CLAMP_MAX, 6u)+"f")
	  : "\n	#define RHO_CLAMP_MIN "+to_string(RHO_CLAMP_MIN,4u)+"f\n	#define RHO_CLAMP_MAX "+to_string(RHO_CLAMP_MAX,4u)+"f")) // Vorgabe: Zeichenfolge wie bisher (\"0.5000f\")
	+((klemm_bilanz_env()||tor_huelle_env()>0u) ? (klemm_haken_env()==5u ? string("\n	#define def_tor_lo (1.0f-1.0f/32768.0f)\n	#define def_tor_hi (1.0f+1.0f/32768.0f)") // ★ Pruefbefund Z2b M1: Haken 5 verengt die Bildhuelle kuenstlich -> [300] MUSS feuern, das Tor (0,5; 2,0) bleibt, Physik unveraendert
	  : "\n	#define def_tor_lo (1.0f-1.5625f*"+to_string(RHO_CLAMP_MAX-1.0f, 6u)+"f)"+"\n	#define def_tor_hi (1.0f+1.5625f*"+to_string(RHO_CLAMP_MAX-1.0f, 6u)+"f)") : string("")) // ★ Z2b/Z2e: Bildhuelle aus dem PHYSIKALISCHEN RHO_CLAMP_MAX (Host-Makro), Lambda^2 = 1,5625; ausserhalb von SRT, weil Lift-Kernel und Waechter immer gebaut werden
	+(tor_huelle_env()>0u ? string("\n	#define def_tor_gate_lo (def_tor_lo-16.0f/32768.0f)\n	#define def_tor_gate_hi (def_tor_hi+16.0f/32768.0f)\n	#define def_w210_lo (def_tor_lo-32.0f/32768.0f)\n	#define def_w210_hi (def_tor_hi+32.0f/32768.0f)") // ★ Z2e: groesste RHO_FP16-ULP unter 2^15 ist 16
	  : (rho_huelle_aktiv() ? string("\n	#define def_tor_gate_lo RHO_CLAMP_MIN\n	#define def_tor_gate_hi RHO_CLAMP_MAX\n	#define def_w210_lo 0.0f\n	#define def_w210_hi 3.0f") // ★ Z2f: numerische Huelle
	  : string("\n	#define def_tor_gate_lo 0.5f\n	#define def_tor_gate_hi 2.0f\n	#define def_w210_lo 0.4f\n	#define def_w210_hi 2.1f"))) // Vorgabe: Werte wie vor Z2e
#ifdef SRT // Pruefpass S0b NIEDRIG: die Buchung dj = w*rho*du gilt nur fuer SRT (unter TRT relaxiert der Impuls mit wm)
	+(klemm_bilanz_env() ? string("\n	#define KLEMM_BILANZ\n	#define def_klemm_s 16384.0f") : string("")) // ★ 15.09.2026 Klemmen S0b; S = 2^14 (Plan §4)
	+(klemm_bilanz_env()||u_klemme_env()>0u ? string("\n	#define def_u2max (def_c*def_c)") : string("")) // ★ Z2b/Z2d: Betragshuelle folgt def_c (Haken 2 schrumpft sie mit, gewollt); die Klemme haengt NICHT am Instrument
	+(u_klemme_env()>0u ? string("\n	#define U_BETRAG") : string("")) // ★ Z2d: CFD_U_KLEMME=1
	+(klemm_bilanz_env()&&klemm_haken_env()==3u ? string("\n	#define KLEMM_HAKEN3") : string(""))
#if defined(D3Q19)&&defined(FP16S) // ★ 15.09.2026 Klemmen Stufe 1 P1a: bei CFD_POSITIV=0 leer (Kernelquelle zeichengleich); Sperren im Konstruktor
	+(klemm_bilanz_env() ? positiv_defines(positiv_env(), positiv_haken_env(), positiv_facette_env(), true, (unsigned long long)get_N(), get_Nx(), get_Ny()) : string(""))
#elif defined(D3Q19)&&!defined(FP16C)
	+(klemm_bilanz_env() ? positiv_defines(positiv_env(), positiv_haken_env(), positiv_facette_env(), false, (unsigned long long)get_N(), get_Nx(), get_Ny()) : string(""))
#endif
#endif // SRT
#endif // RHO_CLAMP
	// ★ Audit-Nacharbeit 2: SGS_WANDFREI und WANDFUNKTION standen im #ifdef-SUBGRID-Block -- mit
	// abgeschaltetem SUBGRID (die Kugel-Validierung verlangt das) waeren beide LAUTLOSE No-Ops
	// gewesen. Jetzt ausserhalb emittiert; SGS_WANDFREI ohne SUBGRID ist sinnlos und wird im
	// Konstruktor hart abgewiesen, die WFB ist von SUBGRID unabhaengig.
	+((s_sgs_wandfrei) ? (string)"\n	#define SGS_WANDFREI" : (string)"")
	+((s_sgs_guo)      ? (string)"\n	#define SGS_GUO"      : (string)"") // ★ 2026-08-25 Guo-Korrektur von Pi^neq, Default AN
	+((s_sgs_diag)     ? (string)"\n	#define SGS_DIAG"     : (string)"")
	+((s_sgs_diag)     ? (string)"\n	#define def_sgs_diag_ab "+to_string(s_sgs_diag_ab)+"ul" : (string)"")
	+((s_wandfunktion) ? (string)"\n	#define WANDFUNKTION"
	"\n	#define def_wf_Y "+to_string(0.5f/nu,8u)+"f"
	"\n	#define def_wf_tau "+to_string(s_wf_tau,4u)+"f"
	"\n	#define def_wf_spalding_it "+to_string(max(1u,env_u("CFD_SPALDING_IT",3u)))+"u" : (string)"")
	// ★ C1b Stufe 2 (FACETTEN-STUFE2.md F4): def_fac_Y ueber WOERTLICH dieselbe Emissionskette wie
	// def_wf_Y -- der Aequivalenznachweis am Kanal (yw=0,5, fac_a=1) kollabiert dann bitgenau.
	+((s_facetten) ? (string)"\n	#define FACETTEN"
	"\n	#define def_fac_Y "+to_string(0.5f/nu,8u)+"f"
	+"\n	#define def_fac_utkorr "+to_string(s_fac_utkorr, 6u)+"f" // ★ 3/2-Abtastpunkt-Messarm (CFD_FAC_UTKORR, Default 1,0 = bitgleich)
	"\n	#define def_fac_tau "+to_string(s_fac_tau,4u)+"f"
	// ★ MLS-Blende (Baustein 1, 26.08.): chi-Nenner 1/(tau0+0,5), tau0=3nu+0,5 (SRT, cs^2=1/3).
	// INTERIM I1 (deklariert im kernel.cpp-MLS-Block): tau0 statt lokalem SUBGRID-tau_eff.
	"\n	#define def_fac_chifak "+to_string(1.0f/(3.0f*nu+1.0f),8u)+"f"
	"\n	#define def_fac_budget "+to_string(s_fac_budget,4u)+"f"
	"\n	#define def_fac_budget_sn "+to_string(s_fac_budget_sn,4u)+"f"
	"\n	#define def_fac_isogate "+to_string(s_fac_isogate,4u)+"f"
	"\n	#define def_fac_deteps "+to_string(s_fac_deteps,4u)+"f"
	"\n	#define def_wf_spalding_it "+to_string(max(1u,env_u("CFD_SPALDING_IT",3u)))+"u" : (string)"")
	+"\n	#define def_zaehl_takt "+to_string(zaehl_takt())+"ul" // ★ gemeinsamer Zaehltakt, siehe zaehl_takt()
	+((s_wandfunktion||s_facetten) ? spalding_tabelle() : (string)"") // ★ Spalding-Tabelle, nur wenn CFD_SPALDING_TAB=1
	+((s_facetten&&s_fac_imem) ? (string)"\n	#define FACETTEN_IMEM" : (string)"") // iMEM-Umbau: Arme 3/4 (Splice ausserhalb R() -- Werkzeugfalle)
	+((s_facetten&&s_fac_imem&&s_fac_ema>0.0f) ? (string)"\n	#define FACETTEN_EMA"
	"\n	#define def_fac_ema "+to_string(s_fac_ema,6u)+"f" : (string)"") // EMA nur wenn gesetzt -- ungesetzt bitgleich zum 3x3-ohne-EMA
	+((s_facetten&&s_fac_imem&&s_fac_satgate) ? (string)"\n	#define FACETTEN_SATGATE" : (string)"") // (a-strich): Klemme -> BB-Rueckfall
	+((s_facetten&&s_fac_imem&&s_fac_alpha>0u) ? (string)"\n	#define FACETTEN_ALPHA" : (string)"") // J4-alpha: Massenkorrektur, Sum q = 0 je Facette
	+((s_facetten&&s_fac_imem&&s_fac_alpha>1u&&s_fac_masse_alle==0u) ? (string)"\n	#define FACETTEN_ALPHA2" : (string)"") // das Downdate ist die Buchhaltung der WANDLINK-Verteilung -- unter MASSE_ALLE gibt es sie nicht mehr
	+((s_facetten&&s_fac_imem&&s_fac_masse_alle>0u) ? (string)"\n	#define FACETTEN_MASSE_ALLE" : (string)"") // ★ 04.09.: Massenkompensation nicht mehr nur ueber die Wandlinks
	+((s_facetten&&s_fac_imem&&s_fac_uw>0u) ? (string)"\n	#define FACETTEN_UW"
	"\n	#define def_fac_nu (0.5f/def_fac_Y)" // ★ Pruefbefund M3: to_string(nu,9u) ist Festkomma mit 8 Nachkommastellen -- bei nu ~ 5,8e-6 blieben DREI signifikante Stellen, unter 5e-9 waere es 0.0 und y+ wuerde inf. def_fac_Y = 0.5f/nu traegt volle float-Genauigkeit, also hier ableiten statt neu drucken.
	"\n	#define def_fac_uwkappa "+to_string(fmax(0.05f, fmin(1.0f, env_f("CFD_FAC_UWKAPPA", 0.41f))),4u)+"f" : (string)"") // ★ 06.09.: u_w = u_B - u_tau^2*yw/(nu*(1+kappa*y+)); nu MUSS mit, y+ wird im Kernel gebraucht
	+((s_facetten&&s_fac_imem&&s_fac_uw>0u&&s_fac_uw_sn) ? (string)"\n	#define FACETTEN_UW_SN" : (string)"") // A/B-Arm: Normalnullung trotz u_w -- dann fallen die Ein-Link-Facetten wie heute zurueck
	+((s_facetten&&s_fac_imem&&s_fac_masse_alle==2u) ? (string)"\n	#define FACETTEN_MASSE_F0" : (string)"") // Modus 2 (VERWORFEN 04.09.): alles auf f_0. ==2u, nicht >=2u -- sonst bekaeme Modus 3 diese Injektion (Bauplan V1)
	+((s_facetten&&s_fac_imem&&s_fac_masse_alle==3u) ? (string)"\n	#define FACETTEN_MASSE_X" : (string)"") // ★ ARM X: Modus-1-Injektion, aber Rueckfall-Entscheid im SCHATTEN wie ALPHA2 -- X gegen Basis = (B) rein, Modus 1 gegen X = (A) rein
	+((s_facetten&&s_fac_imem&&s_fac_messnur>0u) ? (string)"\n	#define FACETTEN_MESSNUR" : (string)"") // ★ 30.08. BB-Physik, nur messen
	+((s_facetten&&s_fac_imem&&s_fac_nachbar>0u) ? (string)"\n	#define FACETTEN_NACHBAR" : (string)"") // ★ 30.08. Eingang aus der zweiten Fluidzelle
	+(string)"\n	#define def_nb_roff "+string((s_facetten&&s_fac_imem&&s_fac_nachbar>0u&&s_fac_apg!=0.0f) ? "5ul" : "2ul") // ★ 23.09. Stufe A: Offset der Nachbar-RICHTUNG in fac_nb, HINTER den APG-Gradienten -- so wandert keine bestehende Lesestelle
	+(string)"\n	#define def_nb_stride "+string(to_string((ulong)(((s_facetten&&s_fac_imem&&s_fac_nachbar>0u&&s_fac_apg!=0.0f) ? 5ull : 2ull) + (fac_rek_on ? nb_rek_floats : 0ull)))+"ul") // ★ 16.09. fac_nb: 2 float je Facette, unter APG 5 (grad rho in [2..4]) -- UNBEDINGT emittiert: fac_nachbar_ab steht in JEDER Domaene im Quelltext (Gate-Befund 16.09.: ohne NACHBAR undeclared identifier, das haette das Fernfeld im dd-Fall gekillt); Host-Allokation MUSS mitziehen
	+((s_facetten&&s_fac_imem&&s_fac_apg!=0.0f&&s_fac_apg_haken==2u) ? (string)"\n	#define FACETTEN_APG_HAKEN" : (string)"") // ★ 16.09. Testhaken: Konstantgradient im Vorkernel
	+((s_facetten&&s_fac_imem&&s_fac_apg!=0.0f&&s_fac_apg_haken==3u) ? (string)"\n	#define FACETTEN_APG_HAKEN3" : (string)"") // ★ 16.09. Testhaken 3 (Pruefagent MITTEL-1): analytisches rho = 1 + x/1024 im Vorkernel, gz := kx -- Host prueft gx exakt
	+((s_facetten&&s_fac_imem&&s_fac_kdiag>0u) ? (string)"\n	#define FACETTEN_KDIAG" : (string)"") // ★ 30.08. Klassen-Diagnostik
	+((s_rho_takt>0u&&s_smbox[3]>0u) ? (string)"\n	#define RHO_SMBOX" : (string)"") // ★ TODO 2: im Fernfeld deckt die rho-Maske auch die Entnahmeebenen ab
	+((s_u_takt>0u) ? (string)"\n	#define U_SPARSAM" : (string)"") // ★ TODO 2 Schritt 3: gattert die u-Schreibstelle; ohne das Define ist der Geraetecode dort zeichengleich zu vorher
	+((s_rho_rand>0u) ? (string)"\n	#define RHO_RAND"+"\n	#define def_RR_N "+to_string(r1_anzahl((uint)get_Nx(), (uint)get_Ny(), (uint)get_Nz()))+"ul" : (string)"") // ★ 15.09. RHO_RAND C2c: rho nur in R1; Fernfeld bleibt ohne (Statik vor lbm_c genullt)
	+((s_rho_takt>0u) ? (string)"\n	#define RHO_SPARSAM" : (string)"") // ★ TODO 2 Schritt 1: gattert die rho-Schreibstelle in stream_collide; ohne das Define ist der Geraetecode ZEICHENGLEICH zu vorher
	+((s_facetten&&s_sgs_fdwand>0u) ? (string)"\n	#define SGS_FDWAND" : (string)"") // ★ 02.09. Geistermoden-Fix (braucht Facetten fuer fac_idx, nicht zwingend iMEM -- wirkt auch im MESSNUR/BB-Arm)
	+((s_facetten&&s_sgs_fdwand>0u&&s_sgs_vandriest>0u) ? (string)"\n	#define SGS_VANDRIEST"
	"\n	#define def_sgs_vd_aplus "+to_string(s_sgs_vd_aplus,4u)+"f"
	"\n	#define def_sgs_vd_ab "+to_string(s_sgs_vd_ab)+"ul" : (string)"") // ★ 08.09. van Driest auf Facetten; A+ als Konstante emittiert (Literatur 26.0)
	+((s_facetten&&s_sgs_fdwand>0u&&s_sgs_vandriest>1u) ? (string)"\n	#define SGS_VANDRIEST_ANWENDEN" : (string)"") // Modus 2 legt erst hier die Wirkung auf w um; Modus 1 bleibt bitgleich
	+((s_facetten&&s_sgs_fdwand>0u&&s_sgs_nut_skal!=1.0f) ? (string)"\n	#define SGS_NUT_SKAL"
	"\n	#define def_sgs_nut_skal "+to_string(s_sgs_nut_skal,6u)+"f" : (string)"") // ★ 10.09. Diskriminator: nu_t am klassischen Modell skaliert. Ohne Schalter kein Define -> Kontrollarm bitgleich. Sechs Nachkommastellen reichen (Faktor der Groessenordnung 0,1; Pruefbefund M3 betraf Groessen ~1e-6).
	+((s_facetten&&s_sgs_fdwand>0u&&s_sgs_band>0u) ? (string)"\n	#define SGS_BAND" : (string)"")
	+((s_facetten&&s_sgs_fdwand>0u&&s_sgs_band>0u&&s_sgs_sism>0u&&s_sgs_band_pi>0u) ? (string)"\n	#define SGS_BAND_PI" : (string)"") // ★ 22.09. Plan C: Pi-konsistente EMA im Band (braucht SGS_SISM fuer def_sgs_sism_T/ab) // ★ 08.09. SGS-BAND: Wandlagen 2..N ueber eine eigene Zellenliste; KEINE neue Kernelfunktion, nur zwei Argumente und ein Leserzweig
	+((s_facetten&&s_sgs_fdwand>0u&&s_sgs_sism>0u) ? (string)"\n	#define SGS_SISM"
	"\n	#define def_sgs_sism_T "+to_string((ulong)s_sgs_sism_T)+"u" // T in SCHRITTEN als uint; alpha = 1.0f/(float)def_sgs_sism_T erst im Kernel (Pruefbefund M3: to_string(float) ist Festkomma -- alpha ~1e-4 wuerde auf 0,4 % quantisiert)
	"\n	#define def_sgs_sism_ab "+to_string(s_sgs_sism_ab)+"ul" : (string)"") // ★ 07.09.2026 SHEAR-IMPROVED SMAGORINSKY im FD-Kernel (Leveque 2007); ab wie def_sgs_diag_ab. Ohne Schalter: kein Define, keine Signaturaenderung -> Kontrollarm bitgleich
	+((s_facetten&&s_fac_imem&&s_fac_elibb) ? (string)"\n	#define FACETTEN_ELIBB" : (string)"") // ★ B2 (2026-08-25): ELIBB 18-Link, q aus der Facettenebene
	+((s_facetten&&s_fac_imem&&s_fac_elibb_pur) ? (string)"\n	#define FACETTEN_ELIBB_PUR" : (string)"") // ★ Pur-Arm: NUR Geometrie-Blende (CFD_FAC_ELIBB=2)
	+((s_facetten&&s_fac_imem&&s_fac_lsq) ? (string)"\n	#define FACETTEN_LSQ" : (string)"")
	+((s_facetten&&s_fac_imem&&s_fac_quergate) ? (string)"\n	#define FACETTEN_QUERGATE" : (string)"")
	+((s_facetten&&s_fac_imem&&s_fac_kraft>0u) ? (string)"\n	#define FACETTEN_KRAFT\n	#define def_fac_kraft "+to_string(min(2u,s_fac_kraft))+"u" : (string)"") // ★ 30.08. Zellkraft statt Slip (Weg F) // ★ 2026-08-25 Querimpuls-Gate, Slot 64 // ★ 2026-08-25 kleinste Quadrate statt Skalar-Rueckfall (CFD_FAC_LSQ, Default 1)
	+((s_facetten&&s_fac_imem&&s_fac_rdiag>0u) ? (string)"\n	#define FACETTEN_RDIAG" : (string)"") // ★ 07.09. Rueckfall-Diagnose, bitneutral
	+((s_facetten&&s_fac_imem&&s_fac_apg!=0.0f) ? (string)"\n	#define FACETTEN_APG"
	"\n	#define def_fac_apg "+to_string(s_fac_apg,6u)+"f"
	+((s_fac_apg_moz>0u)
	  ? (string)"\n	#define FACETTEN_APG_MOZ"
	    "\n	#define def_fac_apg_c "+to_string(s_fac_apg_c,6u)+"f"
	    "\n	#define def_fac_apg_ap0 "+to_string(s_fac_apg_ap0,6u)+"f"
	  : (string)"")
	: (string)"") // APG-Messarm: Emission nur bei kappa != 0 (Kommentar-Verklebung R2 geloest) /* ALPHA2 setzt ALPHA voraus (S0/alph undeklariert sonst) -- die >1/>0-Paarung hier ist die einzige Garantie (Audit 1/3) */ // J4-alpha Stufe 2: Momenten-Downdate (Impuls-Projektion)
	+((s_facetten&&s_fac_imem&&s_fac_pema>0.0f) ? (string)"\n	#define FACETTEN_PEMA"
	"\n	#define def_fac_pema "+to_string(s_fac_pema,6u)+"f" : (string)"") // PEMA (Weg A): Eingangs-Filterung
	+((s_facetten&&s_fac_imem&&s_fac_diagz>=0l) ? (string)"\n	#define FACETTEN_DIAGZ" : (string)"") // Ziel-fid zur Laufzeit in fac_diag[16]
	+"\n	#define TYPE_MS 0x03" // 0b00000011 // cell next to moving solid boundary
	"\n	#define TYPE_BO 0x03" // 0b00000011 // any flag bit used for boundaries (temperature excluded)
	"\n	#define TYPE_IF 0x18" // 0b00011000 // change from interface to fluid
	"\n	#define TYPE_IG 0x30" // 0b00110000 // change from interface to gas
	"\n	#define TYPE_GI 0x38" // 0b00111000 // change from gas to interface
	"\n	#define TYPE_SU 0x38" // 0b00111000 // any flag bit used for SURFACE
	"\n	#define TYPE_XY 0xC0" // 0b11000000 // any flag bit used for X or Y markers

#if defined(FP16S)
	"\n	#define fpxx half" // switchable data type (scaled IEEE-754 16-bit floating-point format: 1-5-10, exp-30, +-1.99902344, +-1.86446416E-9, +-1.81898936E-12, 3.311 digits)
	"\n	#define fpxx_copy ushort" // switchable data type for direct copying (scaled IEEE-754 16-bit floating-point format: 1-5-10, exp-30, +-1.99902344, +-1.86446416E-9, +-1.81898936E-12, 3.311 digits)
	"\n	#define load(p,o) (vload_half(o,p)*3.0517578E-5f)" // special function for loading half
	"\n	#define store(p,o,x) vstore_half_rte((x)*32768.0f,o,p)" // special function for storing half
#elif defined(FP16C)
	"\n	#define fpxx ushort" // switchable data type (custom 16-bit floating-point format: 1-4-11, exp-15, +-1.99951168, +-6.10351562E-5, +-2.98023224E-8, 3.612 digits), 12.5% slower than IEEE-754 16-bit
	"\n	#define fpxx_copy ushort" // switchable data type for direct copying (custom 16-bit floating-point format: 1-4-11, exp-15, +-1.99951168, +-6.10351562E-5, +-2.98023224E-8, 3.612 digits), 12.5% slower than IEEE-754 16-bit
	"\n	#define load(p,o) half_to_float_custom((p)[o])" // special function for loading half
	"\n	#define store(p,o,x) (p)[o]=float_to_half_custom(x)" // special function for storing half
#else // FP32
	"\n	#define fpxx float" // switchable data type (regular 32-bit float)
	"\n	#define fpxx_copy float" // switchable data type for direct copying (regular 32-bit float)
	"\n	#define load(p,o) (p)[o]" // regular float read
	"\n	#define store(p,o,x) (p)[o]=(x)" // regular float write
#endif // FP32

// ★ TODO 2 Schritt 4 (12.09.2026) -- Speicherformat von rho auf der GERAETESEITE.
// Gespeichert wird rho-1, nicht rho: bei rho ~ 1 ist der half-ULP 9,8e-4 und damit so gross wie das
// Signal. Die Wandlung ist VERLUSTFREI SKALIERT -- 3.0517578E-5f ist BITGENAU 2^-15 und 32768.0f ist
// 2^15, beide Multiplikationen runden also nicht. Zusammen mit der Sterbenz-Exaktheit von (x)-1.0f
// auf [0,5; 2,0] macht das die Kette Laden->Speichern zu einem Fixpunkt (Beweis siehe lbm.hpp bei
// rho_pack). Es macht die Makros ausserdem unempfindlich gegen -cl-mad-enable (opencl.hpp): eine
// Kontraktion zu mad() kann nichts aendern, weil h*2^-15 EXAKT ist und auf einem exakten Produkt
// fma und mul+add dieselbe einzige Rundung liefern. (Die Addition von 1.0f rundet sehr wohl --
// berichtigt 12.09., Pruefer A; die frueher hier stehende Begruendung "nichts zu runden" war falsch.)
//
// ZWEI LADEMAKROS, und das ist kein Luxus: load_rho liefert rho, load_drho liefert rho-1 OHNE den
// Umweg ueber die Addition von 1. Wer rho-1 braucht und trotzdem load_rho nimmt, rechnet
// (h*2^-15 + 1.0f) - 1.0f und hat den Wert auf das float32-Raster bei 1,0 gerundet, also einen
// absoluten Boden von 5,96e-8 eingebaut. Genau davor warnt der Kommentar an po_reduce_mean, der fuer
// seine Abweichungsablage 1e-9 beansprucht -- mit load_rho waere das lautlos 60-fach verfehlt.
#ifdef RHO_FP16
	"\n	#define RHO_FP16" // ★ 12.09. (Pruefer A, N5): geraeteseitig heute UNBENUTZT -- kernel.cpp traegt kein einziges #ifdef RHO_FP16 mehr, seit die Quantisierungsmessung entfallen ist. Bewusst emittiert: CFD_DUMP_DEFINES und CFD_DUMP_CL machen den Arm damit am Quelltext erkennbar, und der naechste rho-Zweig braucht es wieder.
	"\n	#define rhoxx half" // rho als range-verschobenes IEEE-754-FP16, 2 statt 4 Byte je Zelle
	"\n	#define load_rho(p,o) (vload_half(o,p)*3.0517578E-5f+1.0f)"
	"\n	#define load_drho(p,o) (vload_half(o,p)*3.0517578E-5f)" // rho-1, ohne Ausloeschung
	"\n	#define store_rho(p,o,x) vstore_half_rte(((x)-1.0f)*32768.0f,o,p)"
#else // RHO_FP16
	"\n	#define rhoxx float" // unveraendert: rho als float32
	"\n	#define load_rho(p,o) ((p)[o])"
	"\n	#define load_drho(p,o) ((p)[o]-1.0f)"
	"\n	#define store_rho(p,o,x) ((p)[o]=(x))"
#endif // RHO_FP16

// ★ TODO 2 Schritt 4 (12.09.2026) -- Speicherformat von u auf der GERAETESEITE.
// Wortgleich zur fpxx-Form der Verteilungen: KEINE Verschiebung, weil u um 0 zentriert ist und
// damit keinen Sockel hat, gegen den der half-ULP anlaufen muesste (die vollstaendige Begruendung
// steht an U_FP16 in defines.hpp). 3.0517578E-5f ist bitgenau 2^-15, 32768.0f ist 2^15; beide
// Multiplikationen runden nicht, ein Unterlauf ist ausgeschlossen (kleinstes Ergebnis 1,8e-12).
// Die Kette store_u(load_u(w)) ist deshalb ein WORT-Fixpunkt und nicht nur ein Wert-Fixpunkt --
// staerker als bei rho, wo die Addition von 1 sehr wohl rundet. Dasselbe Argument macht die Makros
// unempfindlich gegen -cl-mad-enable (opencl.hpp): auf einem exakten Produkt liefern fma und
// mul+add dieselbe einzige Rundung.
// NUR EIN LADEMAKRO, und das ist der Unterschied zu rho: dort gibt es load_rho und load_drho, weil
// der Umweg "+1, dann -1" den Wert auf das float32-Raster bei 1,0 runden wuerde. Bei u gibt es
// keinen Sockel und damit auch keine Ausloeschung, gegen die man rechnen muesste.
#ifdef U_FP16
	"\n	#define U_FP16" // geraeteseitig heute unbenutzt, bewusst emittiert: CFD_DUMP_DEFINES und CFD_DUMP_CL machen den Arm damit am Quelltext erkennbar
	"\n	#define velxx half" // u als IEEE-754-FP16 mit fester Skalierung, 2 statt 4 Byte je Komponente
	"\n	#define load_u(p,o) (vload_half(o,p)*3.0517578E-5f)"
	"\n	#define store_u(p,o,x) vstore_half_rte((x)*32768.0f,o,p)"
#else // U_FP16
	"\n	#define velxx float" // unveraendert: u als drei float32
	"\n	#define load_u(p,o) ((p)[o])"
	"\n	#define store_u(p,o,x) ((p)[o]=(x))"
#endif // U_FP16

#ifdef UPDATE_FIELDS
	"\n	#define UPDATE_FIELDS"
#endif // UPDATE_FIELDS

#ifdef VOLUME_FORCE
	"\n	#define VOLUME_FORCE"
#endif // VOLUME_FORCE

#ifdef MOVING_BOUNDARIES
	"\n	#define MOVING_BOUNDARIES"
#endif // MOVING_BOUNDARIES

#ifdef EQUILIBRIUM_BOUNDARIES
	"\n	#define EQUILIBRIUM_BOUNDARIES"
#endif // EQUILIBRIUM_BOUNDARIES

#ifdef FORCE_FIELD
	"\n	#define FORCE_FIELD"
#endif // FORCE_FIELD

#ifdef SURFACE
	"\n	#define SURFACE"
	"\n	#define def_6_sigma "+to_string(6.0f*sigma)+"f" // rho_laplace = 2*o*K, rho = 1-rho_laplace/c^2 = 1-(6*o)*K
#endif // SURFACE

#ifdef TEMPERATURE
	"\n	#define TEMPERATURE"
	"\n	#define def_w_T "+to_string(1.0f/(2.0f*alpha+0.5f))+"f" // wT = dt/tauT = 1/(2*alpha+1/2), alpha = thermal diffusion coefficient
	"\n	#define def_beta "+to_string(beta)+"f" // thermal expansion coefficient
	"\n	#define def_T_avg "+to_string(T_avg)+"f" // average temperature
#endif // TEMPERATURE

#ifdef SUBGRID
	"\n	#define SUBGRID"+(string)"" // Klebefuge: der Block muss als string enden (Audit-Nacharbeit 2 hat die Ternaere hier herausgezogen)
#endif // SUBGRID

#ifdef PARTICLES
	"\n	#define PARTICLES"
	"\n	#define def_particles_N "+to_string(particles_N)+"ul"
	"\n	#define def_particles_rho "+to_string(particles_rho)+"f"
#endif // PARTICLES

	// FORK -- F-Bounding-Box: der Kernel braucht Ursprung, Ausdehnung und Stride der Box.
	// Bei voller Domaene ist def_FBN == def_N und der Index identisch -- bit-identisch zu Upstream.
#ifdef FORCE_FIELD
	+"\n	#define def_SMX0 "+to_string(smx0)+"u" // ★ TODO 2: Schreibmasken-Box (Nahfeld = F-BBox, Fernfeld = Nahfeld-Fussabdruck)
	+"\n	#define def_SMY0 "+to_string(smy0)+"u"
	+"\n	#define def_SMZ0 "+to_string(smz0)+"u"
	+"\n	#define def_SMNX "+to_string(smnx)+"u"
	+"\n	#define def_SMNY "+to_string(smny)+"u"
	+"\n	#define def_SMNZ "+to_string(smnz)+"u"
	+"\n	#define def_FBX0 "+to_string(fbx0)+"u"
	+"\n	#define def_FBY0 "+to_string(fby0)+"u"
	+"\n	#define def_FBZ0 "+to_string(fbz0)+"u"
	+"\n	#define def_FBNX "+to_string(fbnx)+"u"
	+"\n	#define def_FBNY "+to_string(fbny)+"u"
	+"\n	#define def_FBNZ "+to_string(fbnz)+"u"
	+"\n	#define def_FBN "+to_string((ulong)fbnx*(ulong)fbny*(ulong)fbnz)+"ul"
	// ★ 03.09.2026 F-MARKERLISTE (F_LISTE). Der Stride von F -- die Zahl der Slots -- steht erst nach
	// dem Maskenbau fest, also NACH dem Konstruktor und damit nach device_defines(). Er kann deshalb
	// kein Define sein. Statt dafuer einen weiteren Kernelparameter durch fuenf Signaturen zu faedeln
	// (die Fehlerklasse vom 02.09.), steht er IM MASKENPUFFER, an einem aus def_FBN berechenbaren
	// Platz: f_maske[2*ceil(def_FBN/32)]. Im Vollfeld-Arm wird er nie gelesen -- dort ist der Stride
	// def_FBN, wortgleich zum Stand davor, also bit-identisch.
	+(f_liste_on ? (string)"\n	#define F_LISTE" : (string)"")
	+(fac_idx_voll_on ? (string)"\n	#define FAC_IDX_VOLL" : (string)"")
	+(fac_pinv_on ? (string)"\n	#define FACETTEN_PINV" : (string)"")
	+(fac_r1q_on ? (string)"\n	#define FAC_R1Q" : (string)"")+((fac_r1q_on&&s_fac_r1q>=2u) ? (string)"\n	#define FAC_R1Q_AN" : (string)"")+((fac_r1q_on&&s_fac_r1q>=3u) ? (string)"\n	#define FAC_R1Q_VR" : (string)"")+((fac_r1q_on&&s_fac_r1q>=4u) ? (string)"\n	#define FAC_R1Q_OHNE_DRUCK" : (string)"") // ★ 28.09. R1Q: Messblock / Anwendung getrennt, damit der Messarm quelltextgleich zum Anwendungsarm rechnet
	+(fac_rek_on ? (string)"\n	#define FAC_REK" : (string)"")+((fac_rek_on&&s_fac_rek>=2u) ? (string)"\n	#define FAC_REK_R3" : (string)"")+((fac_rek_on&&s_fac_rek>=3u) ? (string)"\n	#define FAC_REK_S2" : (string)"")+((fac_rek_on&&s_fac_rek>=3u) ? (string)"\n	#define def_fac_rek_leiter "+to_string(s_fac_rek_leiter,4u)+"f" : (string)"") // ★ 24.09. Diagnoseleiter: NUR unter Arm 3 emittiert, damit der Kerneltext aller anderen Arme zeichengleich bleibt (die Preprozessor-Diff-Abnahme haengt daran) // ★ 23.09. R3: eigener Arm (CFD_FAC_REK=2) fuer Gate + Buchung, damit Tor-Wirkung und eps-Wirkung trennbar bleiben. ★ BERICHTIGT 23.09. spaet: hier stand, das Gate sei auch bei eps=0 eine Physikaenderung und schalte den Solve an ALLEN Rang-0-Facetten ab -- am kipp26 ist das GEMESSEN falsch (Slot 331 = 0, Hash = Anker), am 8-mm-Fahrzeug dagegen richtig (30 335). Die Trennung lohnt also, aber sie zeigt, WO das Tor ueberhaupt wirkt // ★ 22.09. S0: schaltet NUR den Block ein; die Amplitude kommt als Laufzeitwert aus fac_geo[8i+6], damit IGC die Identitaet bei eps=0 NICHT wegoptimieren kann
	+(f_liste_on ? (string)"\n	#define F_STRIDE ((ulong)f_maske[2ul*((def_FBN+31ul)/32ul)])"
	             : (string)"\n	#define F_STRIDE def_FBN")
#ifndef PARTICLES
	// F-Null-Read-Gate: Default AN. PARTICLES-Guard hart im Praeprozessor -- spread_force
	// schriebe F an Fluidzellen, das Gate waere still falsch (Wirkpfad-Absicherung Bein 2).
	+(f_nur_solid_an() ? (string)"\n	#define F_NUR_SOLID" : (string)"")
#endif // PARTICLES
#endif // FORCE_FIELD

	// FORK -- Block-Tiling. index_f() wird per Makro auf index_f_impl(..., tile_slot) umgeschrieben, damit
	// alle Aufrufstellen unveraendert bleiben; nur die Signaturen bekommen tile_slot ueber TS_P.
	// AUS = beide Makros leer = der erzeugte Device-Code ist bit-identisch zu Upstream.
	+(sparse_on ? (string)(
		"\n	#define SPARSE_TILES"
		"\n	#define def_TILE "+to_string(sparse_T)+"u"
		"\n	#define def_TILES_X "+to_string(((uint)get_Nx()+sparse_T-1u)/sparse_T)+"u"
		"\n	#define def_TILES_Y "+to_string(((uint)get_Ny()+sparse_T-1u)/sparse_T)+"u"
		"\n	#define def_TILE_DEAD 4294967295u"
		"\n	#define TS_P , const global uint* tile_slot"
		"\n	#define TS_A , tile_slot"
		"\n	#define index_f(n, i) index_f_impl((n), (i), tile_slot)"
	) : (string)(
		"\n	#define TS_P"
		"\n	#define TS_A"
	))
;}

#ifdef GRAPHICS
void LBM_Domain::Graphics::allocate(Device& device) {
	bitmap = Memory<int>(device, camera.width*camera.height);
	zbuffer = Memory<int>(device, camera.width*camera.height, 1u, lbm->get_D()>1u); // if there are multiple domains, allocate zbuffer also on host side
	camera_parameters = Memory<float>(device, 15u);
	kernel_clear = Kernel(device, bitmap.length(), "graphics_clear", bitmap, zbuffer);
	kernel_graphics_flags = Kernel(device, lbm->get_N(), "graphics_flags", camera_parameters, bitmap, zbuffer, lbm->flags);
	{
#ifndef FORCE_FIELD
		const uint cache_required = (cb(GRAPHICS_LSF+1u)* 1u+1023u)/1024u; // in KB
#else // FORCE_FIELD
		const uint cache_required = (cb(GRAPHICS_LSF+1u)*13u+1023u)/1024u; // in KB
#endif // FORCE_FIELD
		const bool enable_ls = GRAPHICS_LSF>0u&&device.info.max_workgroup_size>=cb(GRAPHICS_LSF)&&device.info.local_cache>=cache_required;
		if(GRAPHICS_LSF>0u&&!enable_ls) print_warning(device.info.name+" does not support local memory optimization with GRAPHICS_LSF = "+to_string(GRAPHICS_LSF)+" (max supported workgroup size: "+to_string(device.info.max_workgroup_size)+" (required: "+to_string(cb(GRAPHICS_LSF))+"), cache: "+to_string(device.info.local_cache)+"KB (required: "+to_string(cache_required)+"KB)). Disabling local memory optimization.");
		const ulong N = enable_ls ? (ulong)((lbm->get_Nx()+GRAPHICS_LSF-2u)/GRAPHICS_LSF)*(ulong)((lbm->get_Ny()+GRAPHICS_LSF-2u)/GRAPHICS_LSF)*(ulong)((lbm->get_Nz()+GRAPHICS_LSF-2u)/GRAPHICS_LSF)*(ulong)cb(GRAPHICS_LSF) : (ulong)(lbm->get_Nx()-1u)*(ulong)(lbm->get_Ny()-1u)*(ulong)(lbm->get_Nz()-1u);
		const uint workgroup_size = enable_ls ? cb(GRAPHICS_LSF) : WORKGROUP_SIZE;
		kernel_graphics_flags_mc = Kernel(device, N, workgroup_size, "graphics_flags_mc", camera_parameters, bitmap, zbuffer, lbm->flags);
	}
	kernel_graphics_field = Kernel(device, lbm->get_D()==1u ? camera.width*camera.height : lbm->get_N(), lbm->get_D()==1u ? "graphics_field_rt" : "graphics_field", camera_parameters, bitmap, zbuffer, 0, lbm->rho, lbm->u, lbm->flags); // raytraced field visualization only works for single-GPU
	kernel_graphics_field_slice = Kernel(device, lbm->get_N(), "graphics_field_slice", camera_parameters, bitmap, zbuffer, 0, 0, 0, 0, 0, lbm->rho, lbm->u, lbm->flags);
#ifndef D2Q9
	kernel_graphics_streamline = Kernel(device, (lbm->get_Nx()/GRAPHICS_STREAMLINE_SPARSE)*(lbm->get_Ny()/GRAPHICS_STREAMLINE_SPARSE)*(lbm->get_Nz()/GRAPHICS_STREAMLINE_SPARSE), "graphics_streamline", camera_parameters, bitmap, zbuffer, 0, 0, 0, 0, 0, lbm->rho, lbm->u, lbm->flags); // 3D
#else // D2Q9
	kernel_graphics_streamline = Kernel(device, (lbm->get_Nx()/GRAPHICS_STREAMLINE_SPARSE)*(lbm->get_Ny()/GRAPHICS_STREAMLINE_SPARSE), "graphics_streamline", camera_parameters, bitmap, zbuffer, 0, 0, 0, 0, 0, lbm->rho, lbm->u, lbm->flags); // 2D
#endif // D2Q9
	{
		const uint cache_required = (cb(GRAPHICS_LSQ+3u)*12u+1023u)/1024u; // in KB
		const bool enable_ls = GRAPHICS_LSQ>0u&&device.info.max_workgroup_size>=cb(GRAPHICS_LSQ)&&device.info.local_cache>=cache_required;
		if(GRAPHICS_LSQ>0u&&!enable_ls) print_warning(device.info.name+" does not support local memory optimization with GRAPHICS_LSQ = "+to_string(GRAPHICS_LSQ)+" (max supported workgroup size: "+to_string(device.info.max_workgroup_size)+" (required: "+to_string(cb(GRAPHICS_LSQ))+"), cache: "+to_string(device.info.local_cache)+"KB (required: "+to_string(cache_required)+"KB)). Disabling local memory optimization.");
		const ulong N = enable_ls ? (ulong)((lbm->get_Nx()+GRAPHICS_LSQ-2u)/GRAPHICS_LSQ)*(ulong)((lbm->get_Ny()+GRAPHICS_LSQ-2u)/GRAPHICS_LSQ)*(ulong)((lbm->get_Nz()+GRAPHICS_LSQ-2u)/GRAPHICS_LSQ)*(ulong)cb(GRAPHICS_LSQ) : (ulong)(lbm->get_Nx()-1u)*(ulong)(lbm->get_Ny()-1u)*(ulong)(lbm->get_Nz()-1u);
		const uint workgroup_size = enable_ls ? cb(GRAPHICS_LSQ) : WORKGROUP_SIZE;
		kernel_graphics_q = Kernel(device, N, workgroup_size, "graphics_q", camera_parameters, bitmap, zbuffer, 0, lbm->rho, lbm->u);
	}

#ifdef FORCE_FIELD
	kernel_graphics_flags.add_parameters(lbm->F);
	kernel_graphics_flags_mc.add_parameters(lbm->F);
#endif // FORCE_FIELD

#ifdef SURFACE
	skybox = Memory<int>(device, skybox_image->width()*skybox_image->height(), 1u, skybox_image->data());
	{
		const uint cache_required = (cb(GRAPHICS_LSP+1u)*4u+1023u)/1024u; // in KB
		const bool enable_ls = GRAPHICS_LSP>0u&&device.info.max_workgroup_size>=cb(GRAPHICS_LSP)&&device.info.local_cache>=cache_required;
		if(GRAPHICS_LSP>0u&&!enable_ls) print_warning(device.info.name+" does not support local memory optimization with GRAPHICS_LSP = "+to_string(GRAPHICS_LSP)+" (max supported workgroup size: "+to_string(device.info.max_workgroup_size)+" (required: "+to_string(cb(GRAPHICS_LSP))+"), cache: "+to_string(device.info.local_cache)+"KB (required: "+to_string(cache_required)+"KB)). Disabling local memory optimization.");
		const ulong N = enable_ls ? (ulong)((lbm->get_Nx()+GRAPHICS_LSP-2u)/GRAPHICS_LSP)*(ulong)((lbm->get_Ny()+GRAPHICS_LSP-2u)/GRAPHICS_LSP)*(ulong)((lbm->get_Nz()+GRAPHICS_LSP-2u)/GRAPHICS_LSP)*(ulong)cb(GRAPHICS_LSP) : (ulong)(lbm->get_Nx()-1u)*(ulong)(lbm->get_Ny()-1u)*(ulong)(lbm->get_Nz()-1u);
		const uint workgroup_size = enable_ls ? cb(GRAPHICS_LSP) : WORKGROUP_SIZE;
		kernel_graphics_rasterize_phi = Kernel(device, N, workgroup_size, "graphics_rasterize_phi", camera_parameters, bitmap, zbuffer, lbm->phi);
	}
	kernel_graphics_raytrace_phi = Kernel(device, bitmap.length(), "graphics_raytrace_phi", camera_parameters, bitmap, skybox, lbm->phi, lbm->flags);
	kernel_graphics_q.add_parameters(lbm->flags);
#endif // SURFACE

#ifdef TEMPERATURE
	kernel_graphics_field.add_parameters(lbm->T);
	kernel_graphics_field_slice.add_parameters(lbm->T);
	kernel_graphics_streamline.add_parameters(lbm->T);
	kernel_graphics_q.add_parameters(lbm->T);
#endif // TEMPERATURE

#ifdef PARTICLES
	kernel_graphics_particles = Kernel(device, lbm->particles.length(), "graphics_particles", camera_parameters, bitmap, zbuffer, lbm->particles);
#endif // PARTICLES
}

bool LBM_Domain::Graphics::update_camera() {
	camera.update_matrix();
	bool change = false;
	for(uint i=0u; i<15u; i++) {
		const float data = camera.data(i);
		change |= (camera_parameters[i]!=data);
		camera_parameters[i] = data;
	}
	return change; // return false if camera parameters remain unchanged
}
bool LBM_Domain::Graphics::enqueue_draw_frame(const int visualization_modes, const int field_mode, const int slice_mode, const int slice_x, const int slice_y, const int slice_z, const bool visualization_change) {
	const bool camera_update = update_camera();
#if defined(INTERACTIVE_GRAPHICS)||defined(INTERACTIVE_GRAPHICS_ASCII)
	if(!visualization_change&&!camera_update&&lbm->get_t()==t_last_rendered_frame) return false; // don't render a new frame if the scene hasn't changed since last frame
#endif // INTERACTIVE_GRAPHICS||INTERACTIVE_GRAPHICS_ASCII
	t_last_rendered_frame = lbm->get_t();
	if(camera_update) camera_parameters.enqueue_write_to_device(); // camera_parameters PCIe transfer and kernel_clear execution can happen simulataneously
	kernel_clear.enqueue_run();
	const int sx=slice_x-lbm->Ox, sy=slice_y-lbm->Oy, sz=slice_z-lbm->Oz; // subtract domain offsets
#ifdef SURFACE
	if((visualization_modes&VIS_PHI_RAYTRACE)&&lbm->get_D()==1u) kernel_graphics_raytrace_phi.enqueue_run(); // disable raytracing for multi-GPU (domain decomposition rendering doesn't work for raytracing)
	if(visualization_modes&VIS_PHI_RASTERIZE) kernel_graphics_rasterize_phi.enqueue_run();
#endif // SURFACE
	if(visualization_modes&VIS_FLAG_LATTICE) kernel_graphics_flags.enqueue_run();
	if(visualization_modes&VIS_FLAG_SURFACE) kernel_graphics_flags_mc.enqueue_run();
	if(visualization_modes&VIS_STREAMLINES) kernel_graphics_streamline.set_parameters(3u, field_mode, slice_mode, sx, sy, sz).enqueue_run();
	if(visualization_modes&VIS_Q_CRITERION) kernel_graphics_q.set_parameters(3u, field_mode).enqueue_run();
#ifdef PARTICLES
	if(visualization_modes&VIS_PARTICLES) kernel_graphics_particles.enqueue_run();
#endif // PARTICLES
	if(visualization_modes&VIS_FIELD) {
		switch(slice_mode) { // 0 (no slice), 1 (x), 2 (y), 3 (z), 4 (xz), 5 (xyz), 6 (yz), 7 (xy)
			case 0: // no slice
				kernel_graphics_field.set_parameters(3u, field_mode).enqueue_run();
				break;
			case 1: case 2: case 3: // x/y/z
				kernel_graphics_field_slice.set_ranges(lbm->get_area((uint)clamp(slice_mode-1, 0, 2))).set_parameters(3u, field_mode, slice_mode, sx, sy, sz).enqueue_run();
				break;
			case 4: // xz
				kernel_graphics_field_slice.set_ranges(lbm->get_area(0u)).set_parameters(3u, field_mode, 0u+1u, sx, sy, sz).enqueue_run();
				kernel_graphics_field_slice.set_ranges(lbm->get_area(2u)).set_parameters(3u, field_mode, 2u+1u, sx, sy, sz).enqueue_run();
				break;
			case 5: // xyz
				kernel_graphics_field_slice.set_ranges(lbm->get_area(0u)).set_parameters(3u, field_mode, 0u+1u, sx, sy, sz).enqueue_run();
				kernel_graphics_field_slice.set_ranges(lbm->get_area(1u)).set_parameters(3u, field_mode, 1u+1u, sx, sy, sz).enqueue_run();
				kernel_graphics_field_slice.set_ranges(lbm->get_area(2u)).set_parameters(3u, field_mode, 2u+1u, sx, sy, sz).enqueue_run();
				break;
			case 6: // yz
				kernel_graphics_field_slice.set_ranges(lbm->get_area(1u)).set_parameters(3u, field_mode, 1u+1u, sx, sy, sz).enqueue_run();
				kernel_graphics_field_slice.set_ranges(lbm->get_area(2u)).set_parameters(3u, field_mode, 2u+1u, sx, sy, sz).enqueue_run();
				break;
			case 7: // xy
				kernel_graphics_field_slice.set_ranges(lbm->get_area(0u)).set_parameters(3u, field_mode, 0u+1u, sx, sy, sz).enqueue_run();
				kernel_graphics_field_slice.set_ranges(lbm->get_area(1u)).set_parameters(3u, field_mode, 1u+1u, sx, sy, sz).enqueue_run();
				break;
		}
	}
	bitmap.enqueue_read_from_device();
	if(lbm->get_D()>1u) zbuffer.enqueue_read_from_device();
	return true; // new frame has been rendered
}
int* LBM_Domain::Graphics::get_bitmap() { // returns pointer to zbuffer
	return bitmap.data();
}
int* LBM_Domain::Graphics::get_zbuffer() { // returns pointer to zbuffer
	return zbuffer.data();
}

string LBM_Domain::Graphics::device_defines(const Device_Info& device_info) const { return
	"\n	#define GRAPHICS"
	"\n	#define def_background_color " +to_string(GRAPHICS_BACKGROUND_COLOR)+""
	"\n	#define def_screen_width "     +to_string(camera.width)+"u"
	"\n	#define def_screen_height "    +to_string(camera.height)+"u"
	"\n	#define def_scale_u "          +to_string(1.0f/(0.57735027f*(GRAPHICS_U_MAX)))+"f"
	"\n	#define def_scale_rho "        +to_string(0.5f/(GRAPHICS_RHO_DELTA))+"f"
	"\n	#define def_scale_T "          +to_string(0.5f/(GRAPHICS_T_DELTA))+"f"
	"\n	#define def_scale_F "          +to_string(0.5f/(GRAPHICS_F_MAX))+"f"
	"\n	#define def_scale_Q_min "      +to_string(GRAPHICS_Q_CRITERION)+"f"
	"\n	#define def_streamline_sparse "+to_string(GRAPHICS_STREAMLINE_SPARSE)+"u"
	"\n	#define def_streamline_length "+to_string(GRAPHICS_STREAMLINE_LENGTH)+"u"
	"\n	#define def_n "                +to_string(1.333f)+"f" // refractive index of water for raytracing graphics
	"\n	#define def_attenuation "      +to_string(ln(clamp(GRAPHICS_RAYTRACING_TRANSMITTANCE, 1E-9f, 1.0f))/(float)max(max(lbm->get_Nx(), lbm->get_Ny()), lbm->get_Nz()))+"f" // (negative) attenuation parameter for raytracing graphics
	"\n	#define def_absorption_color " +to_string(GRAPHICS_RAYTRACING_COLOR)+"" // absorption color of fluid for raytracing graphics

	"\n	#define COLOR_S (127<<16|127<<8|127)" // (stationary or moving) solid boundary
	"\n	#define COLOR_E (  0<<16|255<<8|  0)" // equilibrium boundary (inflow/outflow)
	"\n	#define COLOR_M (255<<16|  0<<8|255)" // cells next to moving solid boundary
	"\n	#define COLOR_T (255<<16|  0<<8|  0)" // temperature boundary
	"\n	#define COLOR_F (  0<<16|  0<<8|255)" // fluid
	"\n	#define COLOR_I (  0<<16|255<<8|255)" // interface
	"\n	#define COLOR_0 (127<<16|127<<8|127)" // regular cell or gas
	"\n	#define COLOR_X (255<<16|127<<8|  0)" // reserved type X
	"\n	#define COLOR_Y (255<<16|255<<8|  0)" // reserved type Y
	"\n	#define COLOR_P (255<<16|255<<8|191)" // particles

#ifdef GRAPHICS_TRANSPARENCY
	"\n	#define GRAPHICS_TRANSPARENCY "+to_string(GRAPHICS_TRANSPARENCY)+"f"
#endif // GRAPHICS_TRANSPARENCY

#ifndef SURFACE
	"\n	#define def_skybox_width 1u"
	"\n	#define def_skybox_height 1u"
#else // SURFACE
	"\n	#define def_skybox_width " +to_string(skybox_image->width() )+"u"
	"\n	#define def_skybox_height "+to_string(skybox_image->height())+"u"
#endif // SURFACE

#ifndef FORCE_FIELD
	"\n	#define LSF "+to_string((GRAPHICS_LSF>0u&&device_info.max_workgroup_size>=cb(GRAPHICS_LSF)&&device_info.local_cache>=(cb(GRAPHICS_LSF+1u)* 1u+1023u)/1024u) ? GRAPHICS_LSF : 0u)+"u" // local box size for graphics_flags_mc() kernel (default: 4)
#else // FORCE_FIELD
	"\n	#define LSF "+to_string((GRAPHICS_LSF>0u&&device_info.max_workgroup_size>=cb(GRAPHICS_LSF)&&device_info.local_cache>=(cb(GRAPHICS_LSF+1u)*13u+1023u)/1024u) ? GRAPHICS_LSF : 0u)+"u" // local box size for graphics_flags_mc() kernel (default: 4)
#endif // FORCE_FIELD
	"\n	#define LSQ "+to_string((GRAPHICS_LSQ>0u&&device_info.max_workgroup_size>=cb(GRAPHICS_LSQ)&&device_info.local_cache>=(cb(GRAPHICS_LSQ+3u)*12u+1023u)/1024u) ? GRAPHICS_LSQ : 0u)+"u" // local box size for graphics_q() kernel (default: 8)
	"\n	#define LSP "+to_string((GRAPHICS_LSP>0u&&device_info.max_workgroup_size>=cb(GRAPHICS_LSP)&&device_info.local_cache>=(cb(GRAPHICS_LSP+1u)* 4u+1023u)/1024u) ? GRAPHICS_LSP : 0u)+"u" // local box size for graphics_rasterize_phi() kernel (default: 4)
;}
#endif // GRAPHICS



vector<Device_Info> smart_device_selection(const uint D) {
	const vector<Device_Info>& devices = get_devices(); // a vector of all available OpenCL devices
	vector<Device_Info> device_infos(D);
	const int user_specified_devices = (int)main_arguments.size();
	if(user_specified_devices>0) { // user has selevted specific devices as command line arguments
		if(user_specified_devices==D) { // as much specified devices as domains
			for(uint d=0; d<D; d++) device_infos[d] = select_device_with_id(to_uint(main_arguments[d]), devices); // use list of devices IDs specified by user
		} else {
			print_warning("Incorrect number of devices specified. Using single fastest device for all domains.");
			for(uint d=0; d<D; d++) device_infos[d] = select_device_with_most_flops(devices);
		}
	} else { // device auto-selection
		vector<vector<Device_Info>> device_type_ids; // a vector of all different devices, containing vectors of their device IDs
		for(uint i=0u; i<(uint)devices.size(); i++) {
			const string name_i = devices[i].name;
			bool already_exists = false;
			for(uint j=0u; j<(uint)device_type_ids.size(); j++) {
				const string name_j = device_type_ids[j][0].name;
				if(name_i==name_j) {
					device_type_ids[j].push_back(devices[i]);
					already_exists = true;
				}
			}
			if(!already_exists) device_type_ids.push_back(vector<Device_Info>(1, devices[i]));
		}
		float best_value = -1.0f;
		int best_j = -1;
		for(uint j=0u; j<(uint)device_type_ids.size(); j++) {
			const float value = device_type_ids[j][0].tflops;
			if((uint)device_type_ids[j].size()>=D && value>best_value) {
				best_value = value;
				best_j = j;
			}
		}
		if(best_j>=0) { // select all devices of fastest device type with at least D devices of the same type
			for(uint d=0; d<D; d++) device_infos[d] = device_type_ids[best_j][d];
		} else {
			print_warning("Not enough devices of the same type available. Using single fastest device for all domains.");
			for(uint d=0; d<D; d++) device_infos[d] = select_device_with_most_flops(devices);
		}
		//for(uint j=0u; j<(uint)device_type_ids.size(); j++) print_info("Device Type "+to_string(j)+" ("+device_type_ids[j][0].name+"): "+to_string((uint)device_type_ids[j].size())+"x");
	}
	return device_infos;
}

LBM::LBM(const uint Nx, const uint Ny, const uint Nz, const float nu, const float fx, const float fy, const float fz, const float sigma, const float alpha, const float beta, const uint particles_N, const float particles_rho) // single device
	:LBM(Nx, Ny, Nz, 1u, 1u, 1u, nu, fx, fy, fz, sigma, alpha, beta, particles_N, particles_rho) { // delegating constructor
}
LBM::LBM(const uint Nx, const uint Ny, const uint Nz, const float nu, const float fx, const float fy, const float fz, const uint particles_N, const float particles_rho)
	:LBM(Nx, Ny, Nz, 1u, 1u, 1u, nu, fx, fy, fz, 0.0f, 0.0f, 0.0f, particles_N, particles_rho) { // delegating constructor
}
LBM::LBM(const uint Nx, const uint Ny, const uint Nz, const float nu, const uint particles_N, const float particles_rho)
	:LBM(Nx, Ny, Nz, 1u, 1u, 1u, nu, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, particles_N, particles_rho) { // delegating constructor
}
LBM::LBM(const uint3 N, const uint Dx, const uint Dy, const uint Dz, const float nu, const float fx, const float fy, const float fz, const float sigma, const float alpha, const float beta, const uint particles_N, const float particles_rho)
	:LBM(N.x, N.y, N.z, Dx, Dy, Dz, nu, fx, fy, fz, sigma, alpha, beta, particles_N, particles_rho) { // delegating constructor
}
LBM::LBM(const uint3 N, const float nu, const float fx, const float fy, const float fz, const float sigma, const float alpha, const float beta, const uint particles_N, const float particles_rho) // single device
	:LBM(N.x, N.y, N.z, 1u, 1u, 1u, nu, fx, fy, fz, sigma, alpha, beta, particles_N, particles_rho) { // delegating constructor
}
LBM::LBM(const uint3 N, const float nu, const uint particles_N, const float particles_rho)
	:LBM(N.x, N.y, N.z, 1u, 1u, 1u, nu, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, particles_N, particles_rho) { // delegating constructor
}
LBM::LBM(const uint3 N, const float nu, const float fx, const float fy, const float fz, const uint particles_N, const float particles_rho)
	:LBM(N.x, N.y, N.z, 1u, 1u, 1u, nu, fx, fy, fz, 0.0f, 0.0f, 0.0f, particles_N, particles_rho) { // delegating constructor
}
LBM::LBM(const uint Nx, const uint Ny, const uint Nz, const uint Dx, const uint Dy, const uint Dz, const float nu, const float fx, const float fy, const float fz, const float sigma, const float alpha, const float beta, const uint particles_N, const float particles_rho) { // multiple devices
	const uint NDx=(Nx/Dx)*Dx, NDy=(Ny/Dy)*Dy, NDz=(Nz/Dz)*Dz; // make resolution equally divisible by domains
	if(NDx!=Nx||NDy!=Ny||NDz!=Nz) print_warning("LBM grid ("+to_string(Nx)+"x"+to_string(Ny)+"x"+to_string(Nz)+") is not equally divisible in domains ("+to_string(Dx)+"x"+to_string(Dy)+"x"+to_string(Dz)+"). Changing resolution to ("+to_string(NDx)+"x"+to_string(NDy)+"x"+to_string(NDz)+").");
	this->Nx = NDx; this->Ny = NDy; this->Nz = NDz;
	this->Dx = Dx; this->Dy = Dy; this->Dz = Dz;
	const uint D = Dx*Dy*Dz;
	const uint Hx=Dx>1u, Hy=Dy>1u, Hz=Dz>1u; // halo offsets
	const vector<Device_Info>& device_infos = smart_device_selection(D);
	sanity_checks_constructor(device_infos, this->Nx, this->Ny, this->Nz, Dx, Dy, Dz, nu, fx, fy, fz, sigma, alpha, beta, particles_N, particles_rho);
	lbm_domain = new LBM_Domain*[D];
	for(uint d=0u; d<D; d++) { // parallel_for((ulong)D, D, [&](ulong d) {
		const uint x=((uint)d%(Dx*Dy))%Dx, y=((uint)d%(Dx*Dy))/Dx, z=(uint)d/(Dx*Dy); // d = x+(y+z*Dy)*Dx
		lbm_domain[d] = new LBM_Domain(device_infos[d], this->Nx/Dx+2u*Hx, this->Ny/Dy+2u*Hy, this->Nz/Dz+2u*Hz, Dx, Dy, Dz, (int)(x*this->Nx/Dx)-(int)Hx, (int)(y*this->Ny/Dy)-(int)Hy, (int)(z*this->Nz/Dz)-(int)Hz, nu, fx, fy, fz, sigma, alpha, beta, particles_N, particles_rho);
	} // });
	{
		Memory<rhoxx>** buffers_rho = new Memory<rhoxx>*[D];
		for(uint d=0u; d<D; d++) buffers_rho[d] = &(lbm_domain[d]->rho);
		rho = Memory_Container(this, buffers_rho, "rho");
		if(lbm_domain[0]->rho_rand_on) rho.binde_rand(this); // ★ 15.09. RHO_RAND C2c: Fassade im RAND-Betrieb (Plan K5)
	} {
		Memory<velxx>** buffers_u = new Memory<velxx>*[D];
		for(uint d=0u; d<D; d++) buffers_u[d] = &(lbm_domain[d]->u);
		u = Memory_Container(this, buffers_u, "u");
	} {
		Memory<uchar>** buffers_flags = new Memory<uchar>*[D];
		for(uint d=0u; d<D; d++) buffers_flags[d] = &(lbm_domain[d]->flags);
		flags = Memory_Container(this, buffers_flags, "flags");
	} {
#ifdef FORCE_FIELD
		Memory<float>** buffers_F = new Memory<float>*[D];
		for(uint d=0u; d<D; d++) buffers_F[d] = &(lbm_domain[d]->F);
		F = Memory_Container(this, buffers_F, "F");
#endif // FORCE_FIELD
	} {
#ifdef SURFACE
		Memory<float>** buffers_phi = new Memory<float>*[D];
		for(uint d=0u; d<D; d++) buffers_phi[d] = &(lbm_domain[d]->phi);
		phi = Memory_Container(this, buffers_phi, "phi");
#endif // SURFACE
	} {
#ifdef TEMPERATURE
		Memory<float>** buffers_T = new Memory<float>*[D];
		for(uint d=0u; d<D; d++) buffers_T[d] = &(lbm_domain[d]->T);
		T = Memory_Container(this, buffers_T, "T");
#endif // TEMPERATURE
	} {
#ifdef PARTICLES
		particles = &(lbm_domain[0]->particles);
#endif // PARTICLES
	}
#ifdef GRAPHICS
	graphics = Graphics(this);
#endif // GRAPHICS
}
// FORK Doppel-Domaene: Ein-Geraete-Konstruktor mit EXPLIZITEM Device_Info.
// Der Standardweg (smart_device_selection) liefert immer das schnellste Geraet; fuer die gekoppelte
// Rechnung brauchen wir zwei LBM-Instanzen auf zwei verschiedenen GPUs. Sonst identisch zum
// Ein-Geraete-Pfad oben (D=1, keine Halos, kein Offset).
LBM::LBM(const uint3 N, const float nu, const Device_Info& device_info, const float fx, const float fy, const float fz, const float sigma, const float alpha, const float beta, const uint particles_N, const float particles_rho) {
	this->Nx = N.x; this->Ny = N.y; this->Nz = N.z;
	this->Dx = 1u; this->Dy = 1u; this->Dz = 1u;
	const vector<Device_Info> device_infos(1u, device_info);
	sanity_checks_constructor(device_infos, this->Nx, this->Ny, this->Nz, 1u, 1u, 1u, nu, fx, fy, fz, sigma, alpha, beta, particles_N, particles_rho);
	lbm_domain = new LBM_Domain*[1u];
	lbm_domain[0] = new LBM_Domain(device_info, this->Nx, this->Ny, this->Nz, 1u, 1u, 1u, 0, 0, 0, nu, fx, fy, fz, sigma, alpha, beta, particles_N, particles_rho);
	{
		Memory<rhoxx>** buffers_rho = new Memory<rhoxx>*[1u];
		buffers_rho[0] = &(lbm_domain[0]->rho);
		rho = Memory_Container(this, buffers_rho, "rho");
		if(lbm_domain[0]->rho_rand_on) rho.binde_rand(this); // ★ 15.09. RHO_RAND C2c: Fassade im RAND-Betrieb (Plan K5)
	} {
		Memory<velxx>** buffers_u = new Memory<velxx>*[1u];
		buffers_u[0] = &(lbm_domain[0]->u);
		u = Memory_Container(this, buffers_u, "u");
	} {
		Memory<uchar>** buffers_flags = new Memory<uchar>*[1u];
		buffers_flags[0] = &(lbm_domain[0]->flags);
		flags = Memory_Container(this, buffers_flags, "flags");
	}
#ifdef FORCE_FIELD
	{
		Memory<float>** buffers_F = new Memory<float>*[1u];
		buffers_F[0] = &(lbm_domain[0]->F);
		F = Memory_Container(this, buffers_F, "F");
	}
#endif // FORCE_FIELD
#ifdef SURFACE
	{
		Memory<float>** buffers_phi = new Memory<float>*[1u];
		buffers_phi[0] = &(lbm_domain[0]->phi);
		phi = Memory_Container(this, buffers_phi, "phi");
	}
#endif // SURFACE
#ifdef TEMPERATURE
	{
		Memory<float>** buffers_T = new Memory<float>*[1u];
		buffers_T[0] = &(lbm_domain[0]->T);
		T = Memory_Container(this, buffers_T, "T");
	}
#endif // TEMPERATURE
#ifdef PARTICLES
	particles = &(lbm_domain[0]->particles);
#endif // PARTICLES
#ifdef GRAPHICS
	graphics = Graphics(this);
#endif // GRAPHICS
}
LBM::~LBM() {
#ifdef GRAPHICS
	camera.allow_rendering = false;
#endif // GRAPHICS
	info.print_finalize();
	for(uint d=0u; d<get_D(); d++) delete lbm_domain[d];
	delete[] lbm_domain;
}

void LBM::sanity_checks_constructor(const vector<Device_Info>& device_infos, const uint Nx, const uint Ny, const uint Nz, const uint Dx, const uint Dy, const uint Dz, const float nu, const float fx, const float fy, const float fz, const float sigma, const float alpha, const float beta, const uint particles_N, const float particles_rho) { // sanity checks on grid resolution and extension support
	if((ulong)Nx*(ulong)Ny*(ulong)Nz==0ull) print_error("Grid point number is 0: "+to_string(Nx)+"x"+to_string(Ny)+"x"+to_string(Nz)+" = 0.");
	if(Dx*Dy*Dz==0u) print_error("You specified 0 LBM grid domains ("+to_string(Dx)+"x"+to_string(Dy)+"x"+to_string(Dz)+"). There has to be at least 1 domain in every direction. Check your input in LBM constructor.");
	const uint local_Nx=Nx/Dx+2u*(Dx>1u), local_Ny=Ny/Dy+2u*(Dy>1u), local_Nz=Nz/Dz+2u*(Dz>1u);
	uint memory_available = max_uint; // in MB
	for(Device_Info device_info : device_infos) memory_available = min(memory_available, device_info.memory);
	// ★ FORK 2026-08-29 (Variante C+D1+D2 nach Planungsschritt). Die Vorpruefung war in BEIDE
	// Richtungen falsch und lehnte deshalb ein Gitter ab, das real passt:
	//   ZU PESSIMISTISCH: sie rechnet FORCE_FIELD mit 12 B/Zelle ueber das VOLLE Gitter, obwohl
	//     allocate() F laengst nur ueber die Bounding-Box anlegt (lbm.cpp:344-347). Beim
	//     4-mm-Fahrzeug sind das 4.109 MB zuviel (Lauflog: "F-BBox: F auf 1118x468x306 statt
	//     508701465 Zellen -> 4.18 GB gespart").
	//   ZU OPTIMISTISCH: sie kennt fac_idx (4 B je F-BBox-Zelle, lbm.cpp:513) gar nicht -- 611 MB.
	// F_N ist hier bereits bekannt: setup.cpp ruft set_force_bbox VOR dem Konstruktor, und der
	// Konstruktor ruft diese Pruefung VOR new LBM_Domain (das s_fbbox erst ausliest und nullt).
	// Also DERSELBE Ausdruck wie in allocate(), keine Schaetzung. Bei Dx*Dy*Dz>1 ist F voll-
	// domaenig (lbm.cpp:432 sperrt F-BBox im Mehrgeraetefall), dort bleibt die alte Rechnung.
	const bool fbb_gilt = (Dx*Dy*Dz==1u) && LBM_Domain::s_fbbox[3]>0u && LBM_Domain::s_fbbox[4]>0u && LBM_Domain::s_fbbox[5]>0u;
	const ulong N_dom = (ulong)Nx*(ulong)Ny*(ulong)Nz/((ulong)(Dx*Dy*Dz));
	const ulong F_N   = fbb_gilt ? (ulong)LBM_Domain::s_fbbox[3]*(ulong)LBM_Domain::s_fbbox[4]*(ulong)LBM_Domain::s_fbbox[5] : N_dom;
	const ulong b_zelle = (ulong)bytes_per_cell_device();
#ifdef FORCE_FIELD
	const ulong b_ohne_F = b_zelle-12ull;   // F wird unten mit F_N statt N verrechnet
#else
	const ulong b_ohne_F = b_zelle;
#endif // FORCE_FIELD
	ulong bytes_bekannt = N_dom*b_ohne_F;
#ifdef FORCE_FIELD
	// ★ 08.09. (Pruefagent-Befund B3): unter CFD_F_LISTE ist 12*F_N eine um Faktor ~40 zu hohe Obergrenze
	// (4 mm: 1832 MiB gebucht gegen 43 MiB real). Die Vorpruefung lief damit weiter gegen den ALTEN Bedarf --
	// der freigewordene Speicher waere zwar zur Laufzeit da, aber der Deckel bei "memory_required + reserve >
	// memory_available" haette weiter jedes feinere Gitter abgelehnt. Genau die Klage, die in
	// basis/fahrzeug_dd.basis schon steht. Die Slotzahl steht erst nach der Voxelisierung fest, also wird
	// hier mit dem gemessenen Wandsolid-Anteil abgeschaetzt: 3.739.681 von 160.106.544 F-BBox-Zellen am
	// 4-mm-Fahrzeug = 2,34 % (logs/zg_pinv4.log). Aufschlag auf 6 % plus Maske und Liste -- grosszuegig
	// gegen jede Geometrie, aber nicht mehr um Faktor 40 daneben. Der harte Schutz bleibt der
	// VRAM-Waechter in alloc_facetten_domain, der gegen den GEMESSENEN Frei-Wert prueft.
	if(LBM_Domain::s_f_liste>0u) bytes_bekannt += (ulong)(0.06*12.0*(double)F_N) + 8ull*(((ulong)F_N+31ull)/32ull) + (ulong)(0.06*4.0*(double)F_N);
	else bytes_bekannt += 12ull*F_N;
#endif // FORCE_FIELD
	if(LBM_Domain::s_facetten) bytes_bekannt += (LBM_Domain::s_fac_idx_voll>0u ? 4ull*(ulong)F_N : 8ull*(((ulong)F_N+31ull)/32ull)); // fac_idx als Bitmaske+Praefixsumme (03.09.) -- die Pruefung kannte den Posten frueher gar nicht
	// ★ 11.09.2026 (VRAM-Audit, Befund G2): bis hier kannte die Bilanz von der Facettenkette
	// AUSSCHLIESSLICH fac_idx -- die eigentlichen Puffer (bei 4 mm 581,3 MB) und das SGS-Band
	// (118,8 MB) standen in KEINEM Term und in KEINEM Reserveposten. Netto blieben rund
	// 596 MB ungedeckt. Dass die Groessenordnung der Bilanz trotzdem stimmte, lag allein am
	// grosszuegigen Desktop-Posten -- also am Zufall, nicht an der Rechnung.
	//
	// Die Facettenzahl steht hier noch nicht fest (sie entsteht erst bei der Voxelisierung),
	// also wird sie geschaetzt -- wie schon beim F_LISTE-Term darueber und mit derselben
	// Ehrlichkeit: gemessen sind 3.129.185 Facetten auf 160.106.544 F-BBox-Zellen am
	// 4-mm-Fahrzeug = 1,954 % (logs/p4dt_deteps.log). Angesetzt werden 3 %, also gut das
	// Anderthalbfache. KEIN Faktor 40 wie frueher beim Kraftfeld -- ein zu grosser Aufschlag
	// lehnt Gitter ab, die passen, und genau diese Klage steht in basis/fahrzeug_dd.basis.
	// Die Bytes je Facette sind AUSGEZAEHLT, nicht geschaetzt, und folgen den Schaltern:
	//   fac_geo 32 + fac_tau 24 + fac_tau_n 4 = 60 B unbedingt
	//   + fac_q 18 (ELIBB) + gd_zellen 4 und fac_wfd 4 (FDWAND) + fac_nb 8 (NACHBAR)
	//   + fac_sb 24 (SISM) + fac_kd 64 (KDIAG)
	// Der HARTE Schutz bleibt der Waechter in alloc_facetten_domain und der seit heute
	// ergaenzte in alloc_sgs_band -- beide pruefen gegen den GEMESSENEN Frei-Wert.
	if(LBM_Domain::s_facetten) {
		ulong b_fac = 60ull;
		if(LBM_Domain::s_fac_elibb>0u)  b_fac += 18ull;
		if(LBM_Domain::s_sgs_fdwand>0u) b_fac += 8ull;
		if(LBM_Domain::s_fac_nachbar>0u)b_fac += (LBM_Domain::s_fac_apg!=0.0f ? 20ull : 8ull) + (LBM_Domain::s_fac_rek>0u ? 4ull*LBM_Domain::nb_rek_floats : 0ull); // ★ 23.09. Stufe A2: +16 B (nb_rek_floats = 4: drei Richtungsfloats + der Impuls-Akkumulator), sonst ist der VRAM-Waechter darueber blind (KDIAG-Lehre) // ★ 16.09. MITTEL-2 (Pruefagent): unter APG 5 float je Facette
		if(LBM_Domain::s_sgs_sism>0u)   b_fac += 24ull;
		if(LBM_Domain::s_fac_kdiag>0u)  b_fac += 64ull;
		const ulong fac_est = (ulong)(0.03*(double)F_N);
		bytes_bekannt += fac_est*b_fac;
		if(LBM_Domain::s_sgs_band>=2u) { // Bandzellen je Lage ~ Facettenzahl (gemessen 0,84x, angesetzt 1,0x)
			const ulong band_est = fac_est*(ulong)(LBM_Domain::s_sgs_band-1u);
			bytes_bekannt += 8ull*(((ulong)F_N+31ull)/32ull) + band_est*(LBM_Domain::s_sgs_sism>0u ? 32ull : 8ull);
		}
	}
	if(LBM_Domain::s_rho_rand>0u&&Dx*Dy*Dz==1u) { // ★ 15.09. RHO_RAND C2c: rho nur R1 + Papierkorb, dazu der Ausgabepuffer (groesste Ebene)
		bytes_bekannt -= N_dom*(ulong)sizeof(rhoxx);
		bytes_bekannt += (r1_anzahl(Nx, Ny, Nz)+1ull)*(ulong)sizeof(rhoxx) + 4ull*std::max({(ulong)Nx*(ulong)Nz, (ulong)Nx*(ulong)Ny, (ulong)Ny*(ulong)Nz}); // dieselbe Formel wie der Lazy-Alloc in rho_ausgabe_ebene
	}
	uint memory_required = (uint)(bytes_bekannt/1048576ull); // in MB
	// D1: RESERVE. ★ Pruefagent A-1: die Pruefung sieht `device_info.memory`, also den
	// GESAMTspeicher -- `memory_used` wird hier nicht abgezogen (und Device_Info ist eine Kopie,
	// Belegungen der ersten Domaene erreichen die zweite Pruefung ohnehin nicht). Der freie
	// Speicher entsteht erst durch den Abzug der Reserve unten; teilen sich zwei Domaenen EIN
	// Geraet, schuetzt sie nicht. Heiko-Vorgabe 2026-08-29: 1,0-1,5 GB
	// Restluft sind legitim. Ohne benannte Reserve verschiebt die Korrektur oben den Deckel nur
	// und laesst wieder Nutzlast zu, die es nicht gibt. Drei Posten, jeder belegt:
	//   320 MB  Spaetpuffer, die erst nach dem Konstruktor entstehen (Facettengeometrie, Schale,
	//           Kopplungsebene, kf_liste) -- gemessen am 4-mm-Lauf p4_v3b
	//  1152 MB  DESKTOP: die B70 treibt den Bildschirm (card0-DP-5). memory_used sieht davon
	//           nichts. Der Abbruch vom 22.08.2026 kam genau daher: 29,3 GB Lauf + 1,1 GB Desktop.
	//  1024 MB  Mindestluft nach Heikos Untergrenze
	// Nur fuer echte Geraete -- ein Fall, der im System-RAM rechnet (iGPU-Fernfeld), bekaeme
	// sonst einen Deckel, der mit dem Hostbedarf kollidiert.
	bool nur_ram = true;
	for(Device_Info di : device_infos) nur_ram = nur_ram && di.uses_ram;
	const uint reserve = nur_ram ? 0u : (uint)env_u("CFD_VRAM_RESERVE_MB", 2496u);
	// D2: Speicherplan drucken. Kostet nichts und macht jeden kuenftigen Lauf nachrechenbar --
	// genau das fehlte, als die alte Pruefung ein passendes Gitter ablehnte.
	print_info("SPEICHERPLAN je Domaene: bekannt "+to_string(memory_required)+" MB"
		+(fbb_gilt?string(" (F ueber BBox "+to_string(LBM_Domain::s_fbbox[3])+"x"+to_string(LBM_Domain::s_fbbox[4])+"x"+to_string(LBM_Domain::s_fbbox[5])+", nicht ueber das volle Gitter)"):string(" (F voll-domaenig)"))
		+", Reserve "+to_string(reserve)+" MB (Spaetpuffer+Desktop+Mindestluft, CFD_VRAM_RESERVE_MB)"
		+", verfuegbar "+to_string(memory_available)+" MB, Schlupf "
		+((ulong)memory_required+(ulong)reserve<=(ulong)memory_available ? to_string(memory_available-memory_required-reserve)+" MB" : string("NEGATIV")));
	if((ulong)memory_required+(ulong)reserve>(ulong)memory_available) {
		// ★ Pruefagent A-4: die Reserve ist KONSTANT, sie skaliert nicht mit N^3. Sie gehoert
		// deshalb vom Verfuegbaren abgezogen, nicht zum Bedarf addiert -- sonst schlaegt die
		// Meldung eine zu kleine Aufloesung vor, und genau sie soll zum Skalieren anleiten.
		float factor = cbrt((float)(memory_available>reserve?memory_available-reserve:1u)/(float)memory_required);
		memory_required += reserve; // fuer die Textausgabe: was insgesamt gebraucht wird
		const uint maxNx=(uint)(factor*(float)Nx), maxNy=(uint)(factor*(float)Ny), maxNz=(uint)(factor*(float)Nz);
		string message = "Grid resolution ("+to_string(Nx)+", "+to_string(Ny)+", "+to_string(Nz)+") is too large: "+to_string(Dx*Dy*Dz)+"x "+to_string(memory_required)+" MB required, "+to_string(Dx*Dy*Dz)+"x "+to_string(memory_available)+" MB available. Largest possible resolution is ("+to_string(maxNx)+", "+to_string(maxNy)+", "+to_string(maxNz)+"). Restart the simulation with lower resolution or on different device(s) with more memory.";
#if !defined(FP16S)&&!defined(FP16C)
		uint memory_required_fp16 = (uint)((ulong)Nx*(ulong)Ny*(ulong)Nz/((ulong)(Dx*Dy*Dz))*(ulong)(bytes_per_cell_device()-velocity_set*2u)/1048576ull); // in MB
		float factor_fp16 = cbrt((float)memory_available/(float)memory_required_fp16);
		const uint maxNx_fp16=(uint)(factor_fp16*(float)Nx), maxNy_fp16=(uint)(factor_fp16*(float)Ny), maxNz_fp16=(uint)(factor_fp16*(float)Nz);
		message += " Consider using FP16S/FP16C memory compression to double maximum grid resolution to a maximum of ("+to_string(maxNx_fp16)+", "+to_string(maxNy_fp16)+", "+to_string(maxNz_fp16)+"); for this, uncomment \"#define FP16S\" or \"#define FP16C\" in defines.hpp.";
#endif // !FP16S&&!FP16C
		print_error(message);
	}
	if(nu==0.0f) print_error("Viscosity cannot be 0. Change it in setup.cpp."); // sanity checks for viscosity
	else if(nu<0.0f) print_error("Viscosity cannot be negative. Remove the \"-\" in setup.cpp.");
#ifdef D2Q9
	if(Nz!=1u) print_error("D2Q9 is the 2D velocity set. You have to set Nz=1u in the LBM constructor! Currently you have set Nz="+to_string(Nz)+"u.");
#endif // D2Q9
#if !defined(SRT)&&!defined(TRT)
	print_error("No LBM collision operator selected. Uncomment either \"#define SRT\" or \"#define TRT\" in defines.hpp");
#elif defined(SRT)&&defined(TRT)
	print_error("Too many LBM collision operators selected. Comment out either \"#define SRT\" or \"#define TRT\" in defines.hpp");
#endif // SRT && TRT
// ★ FORK 2026-08-08: dieselbe Absicherung fuer das Zahlenformat, und zwar aus Erfahrung. Am 2026-08-08
// waren FP16S UND FP16C gleichzeitig gesetzt; die #if defined(FP16S) / #elif defined(FP16C)-Ketten in
// device_defines() und info.cpp lassen dann still FP16S gewinnen. Die gesamte Kugelvalidierung lief
// dadurch im falschen Format, ohne eine einzige Meldung. Fuer SRT/TRT gab es diesen Waechter schon --
// fuer FP16 nicht, obwohl der Fehler dort genauso lautlos ist.
#if defined(FP16S)&&defined(FP16C)
	print_error("FP16S und FP16C sind beide gesetzt. Die #if/#elif-Ketten lassen dann still FP16S gewinnen. Genau eines von beiden in defines.hpp auskommentieren.");
#endif // FP16S && FP16C
#ifndef VOLUME_FORCE
	if(fx!=0.0f||fy!=0.0f||fz!=0.0f) print_error("Volume force is set in LBM constructor in main_setup(), but VOLUME_FORCE is not enabled. Uncomment \"#define VOLUME_FORCE\" in defines.hpp.");
#else // VOLUME_FORCE
#ifndef FORCE_FIELD
	if(fx==0.0f&&fy==0.0f&&fz==0.0f) print_warning("The VOLUME_FORCE extension is enabled but the volume force in LBM constructor is set to zero. You may disable the extension by commenting out \"#define VOLUME_FORCE\" in defines.hpp.");
#endif // FORCE_FIELD
#endif // VOLUME_FORCE
#ifndef SURFACE
	if(sigma!=0.0f) print_error("Surface tension is set in LBM constructor in main_setup(), but SURFACE is not enabled. Uncomment \"#define SURFACE\" in defines.hpp.");
#endif // SURFACE
#ifndef TEMPERATURE
	if(alpha!=0.0f||beta!=0.0f) print_error("Thermal diffusion/expansion coefficients are set in LBM constructor in main_setup(), but TEMPERATURE is not enabled. Uncomment \"#define TEMPERATURE\" in defines.hpp.");
#else // TEMPERATURE
	if(alpha==0.0f&&beta==0.0f) print_warning("The TEMPERATURE extension is enabled but the thermal diffusion/expansion coefficients alpha/beta in the LBM constructor are both set to zero. You may disable the extension by commenting out \"#define TEMPERATURE\" in defines.hpp.");
#endif // TEMPERATURE
#ifdef PARTICLES
	if(particles_N==0u) print_error("The PARTICLES extension is enabled but the number of particles is set to 0. Comment out \"#define PARTICLES\" in defines.hpp.");
#if !defined(VOLUME_FORCE)||!defined(FORCE_FIELD)
	if(particles_rho!=1.0f) print_error("Particle density is set unequal to 1, but particle-fluid 2-way-coupling is not enabled. Uncomment both \"#define VOLUME_FORCE\" and \"#define FORCE_FIELD\" in defines.hpp.");
#endif // !VOLUME_FORCE||!FORCE_FIELD
#ifdef FORCE_FIELD
	if(particles_rho==1.0f) print_warning("Particle density is set to 1, so particles behave as passive tracers without acting a force on the fluid, but particle-fluid 2-way-coupling is enabled. You may comment out \"#define FORCE_FIELD\" in defines.hpp.");
#endif // FORCE_FIELD
#else // PARTICLES
	if(particles_N>0u) print_error("The PARTICLES extension is disabled but the number of particles is set to "+to_string(particles_N)+">0. Uncomment \"#define PARTICLES\" in defines.hpp.");
#endif // PARTICLES
}

void LBM::sanity_checks_initialization() { // sanity checks during initialization on used extensions based on used flags
	uchar flags_used = 0u;
	bool moving_boundaries_used=false, equilibrium_boundaries_used=false, surface_used=false, temperature_used=false; // identify used extensions based used flags
	const uint threads = thread::hardware_concurrency();
	vector<uchar> t_flags_used(threads, 0u);
	vector<char> t_moving_boundaries_used(threads, false); // don't use vector<bool> as it uses bit-packing which is broken for multithreading
	vector<char> t_equilibrium_boundaries_used(threads, false); // don't use vector<bool> as it uses bit-packing which is broken for multithreading
	parallel_for(get_N(), threads, [&](ulong n, uint t) {
		const uchar flagsn = flags[n];
		const uchar flagsn_bo = flagsn&(TYPE_S|TYPE_E);
		t_flags_used[t] = t_flags_used[t]|flagsn;
		if(flagsn_bo&TYPE_S) t_moving_boundaries_used[t] = t_moving_boundaries_used[t] || (((flagsn_bo==TYPE_S)&&(u.x[n]!=0.0f||u.y[n]!=0.0f||u.z[n]!=0.0f))||(flagsn_bo==(TYPE_S|TYPE_E)));
		t_equilibrium_boundaries_used[t] = t_equilibrium_boundaries_used[t] || flagsn_bo==TYPE_E;
	});
	for(uint t=0u; t<threads; t++) {
		flags_used = flags_used|t_flags_used[t];
		moving_boundaries_used = moving_boundaries_used || t_moving_boundaries_used[t];
		equilibrium_boundaries_used = equilibrium_boundaries_used || t_equilibrium_boundaries_used[t];
	}
	surface_used = (bool)(flags_used&(TYPE_F|TYPE_I|TYPE_G));
	temperature_used = (bool)(flags_used&TYPE_T);
#ifndef MOVING_BOUNDARIES
	if(moving_boundaries_used) print_warning("Some boundary cells have non-zero velocity, but MOVING_BOUNDARIES is not enabled. If you intend to use moving boundaries, uncomment \"#define MOVING_BOUNDARIES\" in defines.hpp.");
#else // MOVING_BOUNDARIES
	if(!moving_boundaries_used) print_warning("The MOVING_BOUNDARIES extension is enabled but no moving boundary cells (TYPE_S flag and velocity unequal to zero) are placed in the simulation box. You may disable the extension by commenting out \"#define MOVING_BOUNDARIES\" in defines.hpp.");
#endif // MOVING_BOUNDARIES
#ifndef EQUILIBRIUM_BOUNDARIES
	if(equilibrium_boundaries_used) print_error("Some cells are set as equilibrium boundaries with the TYPE_E flag, but EQUILIBRIUM_BOUNDARIES is not enabled. Uncomment \"#define EQUILIBRIUM_BOUNDARIES\" in defines.hpp.");
#else // EQUILIBRIUM_BOUNDARIES
	if(!equilibrium_boundaries_used) print_warning("The EQUILIBRIUM_BOUNDARIES extension is enabled but no equilibrium boundary cells (TYPE_E flag) are placed in the simulation box. You may disable the extension by commenting out \"#define EQUILIBRIUM_BOUNDARIES\" in defines.hpp.");
#endif // EQUILIBRIUM_BOUNDARIES
#ifndef SURFACE
	if(surface_used) print_error("Some cells are set as fluid/interface/gas with the TYPE_F/TYPE_I/TYPE_G flags, but SURFACE is not enabled. Uncomment \"#define SURFACE\" in defines.hpp.");
#else // SURFACE
	if(!surface_used) print_error("The SURFACE extension is enabled but no fluid/interface/gas cells (TYPE_F/TYPE_I/TYPE_G flags) are placed in the simulation box. Disable the extension by commenting out \"#define SURFACE\" in defines.hpp.");
#endif // SURFACE
#ifndef TEMPERATURE
	if(temperature_used) print_error("Some cells are set as temperature boundary with the TYPE_T flag, but TEMPERATURE is not enabled. Uncomment \"#define TEMPERATURE\" in defines.hpp.");
#endif // TEMPERATURE
}

void LBM::initialize() { // write all data fields to device and call kernel_initialize
#ifndef BENCHMARK
	sanity_checks_initialization();
#endif // BENCHMARK

	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->rho.enqueue_write_to_device();
	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->u.enqueue_write_to_device();
	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->flags.enqueue_write_to_device();
#ifdef FORCE_FIELD
#ifndef PARTICLES
	// ★ F-Waechter (Wirkpfad-Absicherung des F-Null-Read-Gates, Bein 1): das Gate behauptet
	// "F ist an Nicht-Solid-Zellen 0" -- hier wird die Behauptung am Host-Puffer HART geprueft,
	// bevor F aufs Geraet geht. Faengt den einzigen realistischen kuenftigen Verletzer
	// (host-seitiges Saeen von Fluid-Volumenkraeften a la Upstream-Doku). Bewusst STRENGER
	// als die Lesemenge (prueft auch Gas-/Halozellen) -- fail-safe-Richtung (Pruefagent NIEDRIG-1).
	// ★ 03.09.2026 F-MARKERLISTE: unter CFD_F_LISTE ist die Praemisse STRUKTURELL wahr -- an
	// Nicht-Wandsolidzellen gibt es gar keinen Speicherplatz mehr, den jemand belegen koennte.
	// Der Waechter hat dort nichts mehr zu pruefen. Er faellt aber nicht ersatzlos: an seine
	// Stelle tritt der Wirkpfad-Zaehler Slot 77 (store3_F ohne Slot, Soll 0 am Laufende) und die
	// Bit-Abnahme der Maske in alloc_f_liste.
	if(f_nur_solid_an()&&lbm_domain[0]->f_liste_on) {
		print_info("F-Waechter uebersprungen: unter der F-Markerliste ist die F-NUR-SOLID-Praemisse strukturell wahr (kein Speicher an Nicht-Wandsolid). Ersatz: Wirkpfad-Zaehler Slot 77, Soll 0.");
	} else if(f_nur_solid_an()) {
		ulong geprueft = 0ull;
		for(uint d=0u; d<get_D(); d++) {
			LBM_Domain* dom = lbm_domain[d];
			const uint Nx=dom->get_Nx(), Ny=dom->get_Ny();
			for(uint zb=0u; zb<dom->fbnz; zb++) for(uint yb=0u; yb<dom->fbny; yb++) for(uint xb=0u; xb<dom->fbnx; xb++) {
				const ulong fbi = (ulong)xb+((ulong)yb+(ulong)zb*(ulong)dom->fbny)*(ulong)dom->fbnx;
				const ulong n   = (ulong)(dom->fbx0+xb)+((ulong)(dom->fby0+yb)+(ulong)(dom->fbz0+zb)*(ulong)Ny)*(ulong)Nx;
				if((dom->flags[n]&(TYPE_S|TYPE_E))!=TYPE_S&&(dom->F(fbi,0u)!=0.0f||dom->F(fbi,1u)!=0.0f||dom->F(fbi,2u)!=0.0f)) // Host-Maske: TYPE_BO existiert nur device-seitig; (S|E)!=S = exakt die update_force_field-Schreibbedingung invertiert
					print_error("F-NUR-SOLID aktiv, aber F != 0 an Nicht-Solid-Zelle n="+to_string(n)+" (Domaene "+to_string(d)+") -- dieses Setup nutzt Fluid-Volumenkraefte: CFD_F_NUR_SOLID=0 setzen.");
				geprueft++;
			}
		}
		print_info("F-Waechter: "+to_string(geprueft)+" F-BBox-Zellen geprueft, F an Nicht-Solid ueberall 0 -- F-NUR-SOLID-Praemisse haelt.");
	}
#endif // PARTICLES
	for(uint d=0u; d<get_D(); d++) if(lbm_domain[d]->f_liste_on&&lbm_domain[d]->f_slots==0ull)
		print_error("CFD_F_LISTE war gesetzt, aber alloc_f_liste ist an Domaene "+to_string(d)+" NIE GELAUFEN -- F steht noch auf dem 1-Element-Platzhalter. Dieser Fall (Setup ruft alloc_f_liste nicht) waere ein stiller Totalausfall der Kraftrechnung.");
	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->F.enqueue_write_to_device();
	communicate_F();
#endif // FORCE_FIELD
#ifdef SURFACE
	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->phi.enqueue_write_to_device();
#endif // SURFACE
#ifdef TEMPERATURE
	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->T.enqueue_write_to_device();
#endif // TEMPERATURE
#ifdef PARTICLES
	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->particles.enqueue_write_to_device();
	communicate_particles();
#endif // PARTICLES

	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->increment_time_step(); // the communicate calls at initialization need an odd time step
	communicate_rho_u_flags();
#ifdef SURFACE
	communicate_phi_massex_flags();
#endif // SURFACE
	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->enqueue_initialize(); // odd time step is baked-in the kernel
	communicate_rho_u_flags();
#ifdef SURFACE
	communicate_phi_massex_flags();
#endif // SURFACE
	communicate_fi(); // time step must be odd here
#ifdef TEMPERATURE
	communicate_T(); // T halo data is required for field_slice rendering
	communicate_gi(); // time step must be odd here
#endif // TEMPERATURE
	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->finish_queue();
	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->reset_time_step(); // set time step to 0 again
	initialized = true;
}

void LBM::do_time_step(const bool sync_single_gpu) { // call kernel_stream_collide to perform one LBM time step
#ifdef SURFACE
	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->enqueue_surface_0();
#endif // SURFACE
	// FORK: u am Druck-Auslass VOR stream_collide setzen.
	// ★ KORREKTUR 2026-08-08 (Pruefer-Befund): hier stand, der Geschwindigkeitsteil sei innerhalb eines
	// Chunks idempotent und hinke um bis zu CFD_SAMPLE_EVERY Schritte hinterher. Das galt fuer einen
	// Stand OHNE UPDATE_FIELDS. UPDATE_FIELDS ist inzwischen fest eingeschaltet (defines.hpp), also
	// schreibt stream_collide rho und u in JEDEM Schritt (der Block ist nur fuer TYPE_E ausgenommen).
	// Der Per-Schritt-Dispatch ist damit nicht Vorrat, sondern noetig: die Neumann-Bedingung sieht das
	// Innenfeld des unmittelbar vorangegangenen Schritts. Der alte Kommentar war zu pessimistisch und
	// widersprach dem, was setup.cpp an derselben Sache richtig beschreibt.
	// ★ Audit-Nacharbeit 9 (E9), Fallzuordnung in R2 korrigiert: die 1580 vi-po-Doppelzellen
	// (Auslassebenen-Kanten) entstehen im FERNFELD-Fall mit CFD_FERN_VI=1 -- der dd-Fall setzt
	// keinen velocity_inlet (enqueue ist dort ein No-Op). Reihenfolge bewusst: erst Einlass, dann
	// Auslass -- auf Doppelzellen GEWINNT der Druck-Auslass; in-order-Queue macht es deterministisch.
	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->enqueue_apply_velocity_inlet(); // FORK: rho am Einlass mitlaufen lassen
	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->enqueue_apply_pressure_outlet();
	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->enqueue_stream_collide(); // run LBM stream_collide kernel after domain communication
	// ★ REIHENFOLGE-UMSTELLUNG 2026-08-22 (Heiko-Vorgabe): die Aufpraegung nah->fern laeuft jetzt
	// VOR dem Moving-Floor-Fix, nicht mehr danach. Der Boden-Fix behaelt damit das letzte Wort an
	// der Fahrbahn.
	//
	// ★ RICHTIGSTELLUNG DERSELBEN SITZUNG (Pruefagent-Befund 2, MITTEL/HOCH): hier stand als
	// Begruendung, der Blend habe vorher "genau das Bodenband ueberschrieben, das der Boden-Fix
	// eben gesetzt hatte". DAS IST FALSCH. z_lo = max(1u, CFD_FERN_BODEN_EQ+1u) steht schon in
	// a4c3fa5 in BEIDEN Listenbauern (dort Zeilen 2772 und 2861) -- die Zellmengen waren immer
	// disjunkt, es gab kein Ueberschreiben zu reparieren. Die Umstellung bleibt (sie ist Heikos
	// Vorgabe und die Rangfolge ist die physikalisch gewollte), aber sie repariert nichts.
	//
	// WAS SIE STATTDESSEN TUT -- und das ist der eigentliche Punkt: sie ist NICHT wirkungslos,
	// obwohl die Mengen disjunkt sind. Esoteric-Pull laesst store_f(n) fuer jedes ungerade i in
	// den Speicher des NACHBARN j[i] schreiben, und load_f liest genau diese Slots. Die unterste
	// Blend-Lage (z = nz+1) und die oberste boden_eq-Zelle (z = nz) sind direkte Nachbarn. Wer
	// zuerst schreibt, bestimmt, was der andere liest. Beide Reihenfolgen koppeln also, nur in
	// die jeweils andere Richtung. FOLGE: jeder Lauf mit CFD_FERN_BODEN_EQ>0 UND aktivem N2F ist
	// gegen a4c3fa5 nicht mehr bitgleich -- und das ist der Normalfall. Die frueheren Schalen-A/Bs
	// sind mit diesem Binary nicht reproduzierbar.
	//
	// FOLGE FUER DEN LISTENBAUER: der z_lo-Ausschluss ist NICHT redundant (die gegenteilige Notiz
	// in setup.cpp war ebenfalls falsch und ist dort korrigiert). Ohne ihn schriebe der Blend in
	// Zellen, die boden_eq danach vollstaendig ueberschreibt -- ein echter, teilweise wirkungsloser
	// Blend plus zusaetzliche Nachbarkopplung. Er bleibt und ist tragend.
	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->enqueue_schale_blend(); // ★ P9c N2F-SCHALE (No-Op wenn aus ODER alpha==0)
	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->enqueue_boden_eq(); // V1-Port (No-Op wenn aus) -- liegt jetzt UEBER dem Blend
	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->enqueue_einlass_eq(); // V1-Port apply_inlet_velocity (No-Op wenn aus; Ecken-Ueberlapp mit boden_eq unkritisch, s. Kernel-Kommentar)
#if defined(SURFACE) || defined(GRAPHICS)
	communicate_rho_u_flags(); // rho/u/flags halo data is required for SURFACE extension, and u halo data is required for Q-criterion rendering
#endif // SURFACE || GRAPHICS
#ifdef SURFACE
	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->enqueue_surface_1();
	communicate_flags();
	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->enqueue_surface_2();
	communicate_flags();
	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->enqueue_surface_3();
	communicate_phi_massex_flags();
#endif // SURFACE
	communicate_fi();
#ifdef TEMPERATURE
#ifdef GRAPHICS
	communicate_T(); // T halo data is required for field_slice rendering
#endif // GRAPHICS
	communicate_gi();
#endif // TEMPERATURE
#ifdef PARTICLES
	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->enqueue_integrate_particles(); // intgegrate particles forward in time and couple particles to fluid
	communicate_particles(); // communicate_F() is not required in do_time_step()
#endif // PARTICLES
	// FORK: sync_single_gpu=false ueberspringt diese Barriere -- dann wartet run_async() nicht, und der Aufrufer
	// setzt die Barriere selbst per finish(). Im Mehr-Domaenen-Fall liefern die communicate_*-Aufrufe die Barrieren ohnehin.
	if(sync_single_gpu && get_D()==1u) for(uint d=0u; d<get_D(); d++) lbm_domain[d]->finish_queue(); // this additional domain synchronization barrier is only required in single-GPU, as communication calls already provide all necessary synchronization barriers in multi-GPU
	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->increment_time_step();
}

void LBM::run(const ulong steps, const ulong total_steps) { // initializes the LBM simulation (copies data to device and runs initialize kernel), then runs LBM
	info.append(steps, total_steps, get_t()); // total_steps parameter is just for runtime estimation
	// ★ C1b: Schalter an, aber nie gebunden = der lautlose No-Op, den dieses Projekt jagt -- hart.
	for(uint d=0u; d<get_D(); d++) if(lbm_domain[d]->facetten_on&&!lbm_domain[d]->facetten_bound)
		print_error("CFD_FACETTEN ist gesetzt, aber alloc_facetten() wurde nie gerufen -- der Kernel rechnete mit 1-Element-Platzhaltern.");
	if(!initialized) {
		initialize();
		info.print_initialize(this); // only print setup info if the setup is new (run() was not called before)
#ifdef GRAPHICS
		camera.allow_rendering = true;
#endif // GRAPHICS
	}
	Clock clock;
	for(ulong i=1ull; i<=steps; i++) {
#if defined(INTERACTIVE_GRAPHICS)||defined(INTERACTIVE_GRAPHICS_ASCII)
		while(!key_P&&running) sleep(0.016);
		if(!running) break;
#endif // INTERACTIVE_GRAPHICS_ASCII || INTERACTIVE_GRAPHICS
		clock.start();
		do_time_step();
		info.update(clock.stop());
	}
	if(get_D()>1u) for(uint d=0u; d<get_D(); d++) lbm_domain[d]->finish_queue(); // wait for everything to finish (multi-GPU only)
}

// FORK Doppel-Domaene: `steps` Zeitschritte in die Warteschlange stellen und SOFORT zurueckkehren.
// Damit rechnet die Coarse-Domaene auf der iGPU, waehrend der Host die Ebenen liftet und die Fine-Domaene
// auf der dGPU ihre r Unterschritte macht. Wer danach rho/u/flags/fi liest, MUSS vorher finish() rufen.
void LBM::run_async(const ulong steps) {
	if(!initialized) { print_error("LBM::run_async vor der Initialisierung aufgerufen. Erst run() einmal rufen, dann run_async."); return; }
	info.append(steps, max_ulong, get_t());
	for(ulong i=1ull; i<=steps; i++) do_time_step(false);
	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->flush_queue(); // Perf-Audit 2026-08-20: ohne Flush haengt die Submission am NEO-Treiberverhalten -- der iGPU-Overlap war bisher Glueck, jetzt Garantie
}

void LBM::finish() { // FORK: Barriere ueber alle Warteschlangen dieser LBM-Instanz
	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->finish_queue();
}

void LBM::update_fields() { // update fields (rho, u, T) manually
	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->enqueue_update_fields();
	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->finish_queue();
}

void LBM::finalize_sparse_tiles() { // FORK: Block-Tiling abschliessen (no-op wenn ausgeschaltet)
	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->finalize_sparse_tiles();
}


// FORK -- Druck-Auslass, allgemeine Fassung.
//
// WAS ES IST, ehrlich benannt: ein Gleichgewichtsrand mit VORGESCHRIEBENER DICHTE und aus dem Inneren
// EXTRAPOLIERTER GESCHWINDIGKEIT. Jeden Schritt vor stream_collide wird an den Auslasszellen
// rho = rho_out gesetzt und u aus der zugehoerigen Innenzelle kopiert (Nullgradient). Die TYPE_E-Logik
// in stream_collide macht daraus f = f_eq(rho_out, u_extrapoliert).
// Das ist NICHT das Zou-He-Schema: Zou-He rekonstruiert die unbekannten Verteilungen aus den bekannten
// ueber den Nichtgleichgewichts-Anteil. Hier wird f_neq am Rand verworfen. Der Rand ist damit erster
// Ordnung und leicht ueberdaempft -- aber er ist stabil, allgemein und nachpruefbar. Ein echter
// Nichtgleichgewichts-Rand muesste fi in der Esoteric-Pull-Ablage ueberschreiben und braucht eine
// eigene Validierungskampagne; siehe die Notiz am Ende dieser Funktion.
//
// Bits im face_mask: 1=x_min 2=x_max 4=y_min 8=y_max 16=z_min 32=z_max
//
// WICHTIG: die Flags werden vom HOST gelesen. Zur Setup-Zeit ist der Host aktuell; ein
// read_from_device() wuerde die gerade gesetzten Randbedingungen mit altem Stand ueberschreiben.
// FORK -- gemeinsamer Sammler fuer BEIDE vorgeschriebenen Raender (Druck-Auslass und
// Geschwindigkeits-Einlass). Bewusst EINE Fassung statt zweier Kopien: die Zuordnung Randzelle ->
// Innenzelle ist die heikle Stelle (Kanten, Ecken, Dimensionen der Dicke 1), sie wurde am
// 2026-08-08 von drei Pruefern durchgesehen und zweimal korrigiert. Eine zweite Kopie waere die
// naechste Stelle, an der beide auseinanderlaufen, ohne dass es jemand merkt. `wofuer` geht nur
// in die Meldungen ein.
bool LBM_Domain::collect_boundary_pairs(const uint face_mask, const string& wofuer, std::vector<ulong>& cells, std::vector<ulong>& interior) {
	if(face_mask==0u) { print_warning(wofuer+": face_mask=0, nichts gesetzt."); return false; }
	const ulong NxNy = (ulong)Nx*(ulong)Ny;
	auto IDX = [&](const int x, const int y, const int z) { return (ulong)x + (ulong)y*(ulong)Nx + (ulong)z*NxNy; };
	auto is_fluid = [&](const int x, const int y, const int z) { // echte Innenzelle: weder Rand noch Solid
		if(x<0||x>=(int)Nx||y<0||y>=(int)Ny||z<0||z>=(int)Nz) return false;
		// TYPE_BO existiert nur device-seitig; host-seitig ist die Maske TYPE_S|TYPE_E.
		// HINWEIS: TYPE_G (Gas) und TYPE_I (Interface) gelten hier als Fluid. Solange SURFACE aus ist,
		// koennen sie nicht auftreten (lbm.cpp bricht sonst beim Sanity-Check ab). Wird SURFACE je
		// eingeschaltet, muesste die Maske erweitert werden -- sonst extrapoliert der Auslass aus einer Gaszelle.
		return (flags[IDX(x,y,z)]&(TYPE_S|TYPE_E))==0u;
	};
	// Welche Flaechen sind angefordert? Reihenfolge = Bitreihenfolge, Vorzeichen = einwaerts.
	const bool f_xmin=(face_mask&1u)!=0u, f_xmax=(face_mask&2u)!=0u;
	const bool f_ymin=(face_mask&4u)!=0u, f_ymax=(face_mask&8u)!=0u;
	const bool f_zmin=(face_mask&16u)!=0u, f_zmax=(face_mask&32u)!=0u;

	cells.clear(); interior.clear();
	ulong n_composite=0ull, n_fallback=0ull, n_skipped=0ull, n_degenerate=0ull, n_on_face=0ull;
	// Jede Randzelle GENAU EINMAL: ueber alle Zellen laufen und pruefen, ob sie auf einer der
	// angeforderten Flaechen liegt. Der frueher benutzte Weg (pro Flaeche sammeln) trug Kanten- und
	// Eckzellen mehrfach ein -- zwei Work-Items schrieben dann dieselbe Zelle, Reihenfolge undefiniert.
	for(uint z=0u; z<Nz; z++) for(uint y=0u; y<Ny; y++) for(uint x=0u; x<Nx; x++) {
		const ulong n = IDX((int)x,(int)y,(int)z);
		if((flags[n]&TYPE_E)==0u) continue; // nur TYPE_E; Solid und Fluid bleiben unangetastet
		int dx=0, dy=0, dz=0; // zusammengesetzte EINWAERTS-Richtung
		bool on_face = false; // ob die Zelle ueberhaupt auf einer angeforderten Flaeche liegt
		if(f_xmin && x==0u)      { dx += 1; on_face = true; }
		if(f_xmax && x==Nx-1u)   { dx -= 1; on_face = true; }
		if(f_ymin && y==0u)      { dy += 1; on_face = true; }
		if(f_ymax && y==Ny-1u)   { dy -= 1; on_face = true; }
		if(f_zmin && z==0u)      { dz += 1; on_face = true; }
		if(f_zmax && z==Nz-1u)   { dz -= 1; on_face = true; }
		if(!on_face) continue; // liegt wirklich auf keiner angeforderten Flaeche
		n_on_face++;
		// ★ 2026-08-08, von einem unabhaengigen Pruefer gefunden: die Flaechenzugehoerigkeit MUSS getrennt
		// von der Richtung bestimmt werden. Bei einer Dimension der Ausdehnung 1 gilt x==0 UND x==Nx-1
		// gleichzeitig; sind beide Bits dieser Achse gesetzt, heben sich +1 und -1 zu null auf. Die
		// vorige Fassung pruefte nur (dx==0&&dy==0&&dz==0) und verwarf solche Zellen STILL -- sie wurden
		// nicht einmal gezaehlt, und die Ausgabe meldete vollstaendige Behandlung. Reproduziert an
		// 1x6x6 mit face_mask=7: 6 Zellen verloren, keine Warnung.
		if(dx==0 && dy==0 && dz==0) { n_degenerate++; continue; } // gegenueberliegende Flaechen bei Dicke 1
		// 1. Versuch: ein Schritt entlang der zusammengesetzten Richtung. Fuer eine Kante (zwei Flaechen)
		//    ist das die Diagonale und landet damit im Inneren, nicht auf der jeweils anderen Flaeche.
		if(is_fluid((int)x+dx, (int)y+dy, (int)z+dz)) {
			cells.push_back(n); interior.push_back(IDX((int)x+dx,(int)y+dy,(int)z+dz)); n_composite++;
			continue;
		}
		// 2. Versuch: naechster echter Fluidnachbar in der 26er-Nachbarschaft, kuerzeste Distanz zuerst.
		int bx=0,by=0,bz=0; int best=99;
		for(int oz=-1; oz<=1; oz++) for(int oy=-1; oy<=1; oy++) for(int ox=-1; ox<=1; ox++) {
			if(ox==0&&oy==0&&oz==0) continue;
			const int d2 = ox*ox+oy*oy+oz*oz;
			if(d2>=best) continue;
			if(is_fluid((int)x+ox,(int)y+oy,(int)z+oz)) { best=d2; bx=ox; by=oy; bz=oz; }
		}
		if(best<99) { cells.push_back(n); interior.push_back(IDX((int)x+bx,(int)y+by,(int)z+bz)); n_fallback++; }
		else n_skipped++; // voellig eingeschlossen -> gar kein Auslass, auslassen statt raten
	}
	const uint N_po = (uint)cells.size();
	if(N_po==0u) { print_warning(wofuer+": keine TYPE_E-Zellen auf den angeforderten Flaechen gefunden."); return false; }

	// --- Selbstpruefung.
	// ★ 2026-08-08: die erste Fassung dieser Pruefung war TAUTOLOGISCH -- Eindeutigkeit und
	// "Innenzelle ist Fluid" sind durch die Sammelschleife bereits konstruktiv garantiert (je Zelle
	// hoechstens ein push_back; jede Innenzelle kam durch is_fluid). Sie konnte nie ausloesen und war
	// damit reine Beruhigung. Ein unabhaengiger Pruefer hat das nachgewiesen.
	// Was WIRKLICH brechen kann, ist die VOLLSTAENDIGKEIT: dass jede Zelle auf einer angeforderten
	// Flaeche auch irgendwo landet -- zugeordnet, uebersprungen oder degeneriert. Genau daran ist der
	// oben behobene Defekt vorbeigelaufen. Diese Bilanz wird jetzt geprueft.
	{
		if((ulong)N_po + n_skipped + n_degenerate != n_on_face) {
			print_error(wofuer+": Bilanz stimmt nicht -- "+to_string((uint)n_on_face)+" Zellen auf den Flaechen, aber nur "
				+to_string(N_po)+" zugeordnet + "+to_string((uint)n_skipped)+" uebersprungen + "+to_string((uint)n_degenerate)+" degeneriert. Es gehen Zellen still verloren.");
		}
		// Die konstruktiv garantierten Eigenschaften trotzdem pruefen -- nicht fuer heute, sondern als
		// Regressionsschutz, falls jemand die Sammelschleife umbaut. Dass sie heute nie auslesen, ist
		// kein Argument gegen sie, solange man nicht glaubt, sie wuerden etwas beweisen.
		std::vector<ulong> sorted_cells = cells;
		std::sort(sorted_cells.begin(), sorted_cells.end());
		if(std::adjacent_find(sorted_cells.begin(), sorted_cells.end())!=sorted_cells.end())
			print_error(wofuer+": mindestens eine Randzelle kommt mehrfach vor -- konkurrierende Schreibzugriffe.");
		for(uint i=0u; i<N_po; i++) {
			if((flags[interior[i]]&(TYPE_S|TYPE_E))!=0u) print_error(wofuer+": Innenzelle "+to_string(interior[i])+" ist selbst Rand oder Solid.");
			if(interior[i]==cells[i]) print_error(wofuer+": Innenzelle zeigt auf sich selbst.");
		}
	}
	print_info(wofuer+": "+to_string(N_po)+" Zellen (face_mask=0x"+to_string(face_mask)+")."
		+" Innenzelle direkt: "+to_string((uint)n_composite)+", ueber Nachbarsuche: "+to_string((uint)n_fallback)
		+(n_skipped? (", ohne Fluidnachbar uebersprungen: "+to_string((uint)n_skipped)) : string(""))
		+(n_degenerate? (", degeneriert (Dimension der Dicke 1): "+to_string((uint)n_degenerate)) : string(""))+".");
	return true;
}

void LBM_Domain::set_pressure_outlet_faces(const uint face_mask, const float rho_out) {
	po_rho = rho_out;
	{ const char* v = getenv("CFD_PO_SIGMA"); po_sigma = v ? (float)fmax(0.0, fmin(1.0, atof(v))) : 1.0f; }
	{ const char* v = getenv("CFD_PO_HART"); po_hart = (v && atoi(v)>0) ? 1u : 0u; } // 1 = alter harter Rand, der Kontrollarm (Audit-Nacharbeit 15: "false" zaehlte vorher als AN)
	std::vector<ulong> cells, interior;
	if(!collect_boundary_pairs(face_mask, "Druck-Auslass", cells, interior)) return;
	const uint N_po = (uint)cells.size();
	// Bit 1 ist x_min und damit ueblicherweise der EINLASS. Wer ihn als Auslass anfordert, verliert
	// stillschweigend die Zustroembedingung: u wird dort dann extrapoliert statt auf u_inf gehalten.
	if((face_mask&1u)!=0u) print_warning("Druck-Auslass: face_mask enthaelt x_min. Das ist normalerweise der EINLASS -- dort wird u jetzt extrapoliert statt vorgegeben.");

	po_N_active = N_po;
	po_cells    = Memory<uint>(device, (ulong)N_po);
	po_interior = Memory<uint>(device, (ulong)N_po);
	if(get_N()>0xFFFFFFFFull) print_error("Druck-Auslass: Gitter ueberschreitet 2^32 Zellen -- po_cells/po_interior sind seit 08.09. uint (VRAM).");
	for(uint i=0u; i<N_po; i++) { po_cells[i] = (uint)cells[i]; po_interior[i] = (uint)interior[i]; }
	po_cells.write_to_device();
	po_interior.write_to_device();
	po_mean = Memory<float>(device, 1ull);
	// ★ 2026-08-24: Teilsummenpuffer je Arbeitsgruppe statt atomarer Addition. Die Allokation MUSS
	// vor der Kernel-Bindung stehen -- ein Move-Assignment auf einen bereits gebundenen Puffer ist
	// die im Baum zweimal dokumentierte Use-after-free-Klasse.
	po_groups = (uint)(((ulong)N_po+(ulong)WORKGROUP_SIZE-1ull)/(ulong)WORKGROUP_SIZE);
	po_part = Memory<float>(device, (ulong)po_groups);
	kernel_po_reduce_mean = Kernel(device, (ulong)N_po, "po_reduce_mean", rho, po_interior, N_po, po_part);
	kernel_po_final_mean  = Kernel(device, 1ull, "po_final_mean", po_part, po_groups, N_po, po_mean);
	kernel_apply_pressure_outlet = Kernel(device, (ulong)N_po, "apply_pressure_outlet", u, rho, po_cells, po_interior, N_po, po_rho, po_sigma, po_mean, po_hart);
	print_info("Druck-Auslass REDUKTION: "+to_string(N_po)+" Innenzellen in "+to_string(po_groups)+" Arbeitsgruppen, Endsumme in Indexordnung (atomikfrei, bitreproduzierbar seit 2026-08-24).");
	print_info(po_hart ? ("Druck-Auslass: HARTE Klemme rho = "+to_string(po_rho,4u)+" je Zelle (CFD_PO_HART=1, der alte Zustand als Kontrollarm), u aus der Innenzelle.")
		: ("Druck-Auslass: rho_out = "+to_string(po_rho,4u)+" als FLAECHENMITTEL verankert (Ankerrate sigma = "+to_string(po_sigma,4u)
		+"), rho der Einzelzelle laeuft frei mit, u aus der Innenzelle. Der Druck ist damit global festgelegt, seine Verteilung ueber die Ebene aber nicht."));
	// OFFEN, bewusst nicht hier geloest: ein echter Nichtgleichgewichts-Rand (Guo/Zheng/Shi 2002,
	// f_i = f_eq(rho_b,u_b) + [f_i(n) - f_eq(rho_n,u_n)]) waere zweiter Ordnung statt erster. Er muesste
	// fi am Rand NACH dem Streaming ueberschreiben, und in der Esoteric-Pull-Ablage gehoeren die Slots
	// einer Zelle teilweise ihren Nachbarn -- ein Fehler dort erzeugt still falsche Ergebnisse statt
	// eines Absturzes. Das braucht eine eigene Validierung und nicht denselben Commit.
}

// FORK 2026-08-08 -- GESCHWINDIGKEITS-EINLASS mit MITLAUFENDER DICHTE.
// Gemessen an einem leeren groben Kanal (kein Fahrzeug, keine Kopplung): der bisherige Einlass
// schreibt rho UND u vor. Fuer einen kompressiblen Loeser ist das ueberbestimmt -- eine von innen
// ankommende Druckwelle kann dort weder hinaus noch absorbiert werden, und die Massenbilanz geht
// jeden Schritt nicht auf. Die Differenz landet in der ERSTEN Fluidzelle dahinter: gemessen war
// die Stoerung dort 413 (willkuerliche Einheit) gegen 0 auf der Randebene selbst und 8 weit
// stromab. Die Streuung von u_x wuchs monoton von 0,012 auf 0,125, bei 38 % der Zellen ueber
// 10 % daneben -- im LEEREN Kanal, wo u_x konstant sein muesste.
// Viskositaet ist nicht der Hebel: der zehnfache Wert aendert nichts (10,94 -> 10,77 % verdorbene
// Zellen), der hundertfache halbiert nur (4,71 %), und selbst dann ist tau erst 0,5007.
// Die Abhilfe ist die Spiegelung des Druck-Auslasses: dort wird rho vorgeschrieben und u aus der
// Innenzelle genommen -- hier wird u vorgeschrieben (das erledigt TYPE_E) und rho aus der
// Innenzelle uebernommen. Damit ist genau EINE Groesse je Rand vorgegeben, und Druckwellen
// laufen hinaus statt zurueck.
void LBM_Domain::set_velocity_inlet_faces(const uint face_mask) {
	std::vector<ulong> cells, interior;
	if(!collect_boundary_pairs(face_mask, "Geschwindigkeits-Einlass", cells, interior)) return;
	const uint N_vi = (uint)cells.size();
	vi_N_active = N_vi;
	vi_cells    = Memory<ulong>(device, (ulong)N_vi);
	vi_interior = Memory<ulong>(device, (ulong)N_vi);
	for(uint i=0u; i<N_vi; i++) { vi_cells[i] = cells[i]; vi_interior[i] = interior[i]; }
	vi_cells.write_to_device();
	vi_interior.write_to_device();
	kernel_apply_velocity_inlet = Kernel(device, (ulong)N_vi, "apply_velocity_inlet", rho, vi_cells, vi_interior, N_vi);
	print_info("Geschwindigkeits-Einlass: u bleibt vorgeschrieben, rho laeuft mit der Innenzelle mit.");
}

void LBM_Domain::enqueue_apply_velocity_inlet() {
	if(vi_N_active==0u) return;
	kernel_apply_velocity_inlet.enqueue_run();
}

void LBM::set_velocity_inlet_faces(const uint face_mask) {
	if(get_D()!=1u) { print_warning("Geschwindigkeits-Einlass: nur fuer eine einzelne Domaene validiert, uebersprungen."); return; }
	lbm_domain[0]->set_velocity_inlet_faces(face_mask);
}

void LBM::set_pressure_outlet_faces(const uint face_mask, const float rho_out) {
	if(get_D()!=1u) { print_warning("Druck-Auslass: nur fuer eine einzelne Domaene validiert, uebersprungen."); return; }
	lbm_domain[0]->set_pressure_outlet_faces(face_mask, rho_out);
}

// =====================================================================================
// FORK -- Doppel-Domaene: Host-Seite der Kopplung grob -> fein.
// Die ausfuehrliche Begruendung des Verfahrens (und die Liste dessen, was bewusst fehlt)
// steht bei den Kernels in kernel.cpp unter "Doppel-Domaene: Kopplung grobes Fernfeld".
// =====================================================================================
void LBM::alloc_coupling_planes(const ulong max_plane_cells) {
	if(get_D()!=1u) { print_error("Doppel-Domaenen-Kopplung: nur fuer je eine Domaene je LBM-Instanz gebaut."); return; }
	if(!initialized) { print_error("alloc_coupling_planes vor der Initialisierung. Erst run(1) rufen."); return; }
	lbm_domain[0]->alloc_coupling_planes(max_plane_cells);
}

// ★ Pruefer-Befund 2026-08-08: Eine Ebene, die aus der Domaene ragt, wickelt STILL um. Der Kernel prueft
// nur n>=def_N; ein Ueberlauf in x oder y bleibt darunter und landet einfach in der naechsten Zellzeile
// bzw. -ebene. Also kein Absturz, sondern richtige Werte an falschen Zellen -- genau die Fehlerklasse,
// die weder eine Norm noch ein Kraftverlauf sichtbar macht. Deshalb hier, host-seitig, vollstaendig gefasst.
bool LBM::plane_fits(const PlaneSpec& plane, const char* who) const {
	uint na=0u, nb=0u, nn=0u, oa=0u, ob=0u, on=0u; // Ausdehnung/Ursprung entlang a, entlang b, entlang der Normalen
	if(plane.axis==0u)      { na=Ny; nb=Nz; nn=Nx; oa=plane.origin.y; ob=plane.origin.z; on=plane.origin.x; }
	else if(plane.axis==1u) { na=Nx; nb=Nz; nn=Ny; oa=plane.origin.x; ob=plane.origin.z; on=plane.origin.y; }
	else if(plane.axis==2u) { na=Nx; nb=Ny; nn=Nz; oa=plane.origin.x; ob=plane.origin.y; on=plane.origin.z; }
	else { print_error(string(who)+": Ebenenachse "+to_string(plane.axis)+" gibt es nicht (erlaubt sind 0, 1, 2)."); return false; }
	if(plane.extent_a==0u || plane.extent_b==0u) { print_error(string(who)+": Ebene mit Ausdehnung 0."); return false; }
	if(on>=nn || (ulong)oa+(ulong)plane.extent_a>(ulong)na || (ulong)ob+(ulong)plane.extent_b>(ulong)nb) {
		print_error(string(who)+": Ebene ragt aus der Domaene. Ursprung ("+to_string(plane.origin.x)+","+to_string(plane.origin.y)+","+to_string(plane.origin.z)
			+"), Ausdehnung "+to_string(plane.extent_a)+"x"+to_string(plane.extent_b)+", Achse "+to_string(plane.axis)
			+", Domaene "+to_string(Nx)+"x"+to_string(Ny)+"x"+to_string(Nz)+".");
		return false;
	}
	return true;
}

void LBM::extract_plane_macros(const PlaneSpec& plane, std::vector<float>& host_buf) {
	LBM_Domain* dom = lbm_domain[0];
	const ulong n_plane = (ulong)plane.extent_a*(ulong)plane.extent_b;
	if(!plane_fits(plane, "extract_plane_macros")) return;
	if(dom->coupling_max_plane_cells==0ull) { print_error("extract_plane_macros ohne alloc_coupling_planes."); return; }
	if(n_plane>dom->coupling_max_plane_cells) { print_error("extract_plane_macros: Ebene mit "+to_string(n_plane)+" Zellen passt nicht in den Puffer ("+to_string(dom->coupling_max_plane_cells)+")."); return; }
	dom->kernel_extract_plane_macros.set_ranges(n_plane);
	dom->kernel_extract_plane_macros.set_parameters(3u, plane.axis,
		plane.origin.x, plane.origin.y, plane.origin.z, plane.extent_a, plane.extent_b);
	dom->kernel_extract_plane_macros.enqueue_run();
	dom->finish_queue();
	host_buf.resize(n_plane*4ull);
	dom->coupling_plane.read_from_device(0ull, n_plane*4ull); // nur den belegten Anfang zurueckholen
	for(ulong i=0ull; i<n_plane*4ull; i++) host_buf[i] = dom->coupling_plane[i];
}

// ★ 15.09.2026 RHO_RAND C2a (RHO_RAND-C2-PLAN.md §2.5/§3.1): Ausgabekernel, Hausmuster wie alloc_rho_rek.
void LBM_Domain::alloc_rho_ausgabe(const ulong max_plane_cells) {
	if(max_plane_cells==0ull) { print_error("alloc_rho_ausgabe mit 0 Zellen."); return; }
	if(rho_aus_max>=max_plane_cells) return;
	if(rho_aus_max>0ull) { print_error("alloc_rho_ausgabe: Vergroesserung eines gebundenen Puffers ist die Use-after-free-Klasse -- einmal gross genug anlegen."); return; }
	rho_aus_max = max_plane_cells;
	rho_aus = Memory<float>(device, max_plane_cells, 1u);
	kernel_rho_ausgabe_ebene = Kernel(device, max_plane_cells, "rho_ausgabe_ebene", fi, rho, flags, t, rho_aus, 0u, 0u, 0u, 0u, 1u, 1u, 0u, rho_clamp_hits); // fi 0, rho 1, flags 2, t 3, out 4, Ebene 5..10, zaehlen 11, hits 12, tile_slot 13
	if(sparse_on) kernel_rho_ausgabe_ebene.add_parameters(tile_slot); // Guard wie im Kernel (Falle 8)
	print_info("rho-Ausgabe (RHO_RAND C2a, Nachkollisionssumme): Puffer fuer "+to_string(max_plane_cells)+" Ebenenzellen = "+to_string((float)(4ull*max_plane_cells)/1.0e6f,2u)+" MB auf "+device.info.name+".");
}

// ★ 15.09.2026 RHO_RAND C2a: Ausgabe-rho einer Ebene. t_aus = get_t()-1 (Nachkollisions-Populationen des letzten Schritts).
void LBM::rho_ausgabe_ebene(const PlaneSpec& plane, const ulong t_aus, const bool zaehlen, std::vector<float>& out) {
	LBM_Domain* dom = lbm_domain[0];
	const ulong n_plane = (ulong)plane.extent_a*(ulong)plane.extent_b;
	if(get_D()>1u) { print_error("rho_ausgabe_ebene: nur fuer eine Domaene gebaut (D=1)."); return; }
	if(!initialized) { print_error("rho_ausgabe_ebene vor der Initialisierung."); return; }
	if(!plane_fits(plane, "rho_ausgabe_ebene")) return;
	if(dom->rho_rand_on&&get_t()==0ull) { print_error("rho_ausgabe_ebene unter RHO_RAND bei t = 0: es gibt noch keine Nachkollisions-Populationen."); return; }
	if(dom->rho_aus_max==0ull) dom->alloc_rho_ausgabe(std::max({(ulong)Nx*(ulong)Nz, (ulong)Nx*(ulong)Ny, (ulong)Ny*(ulong)Nz})); // ★ C2c: einmal in der groessten Ebenengroesse -- ein gebundener Puffer darf nicht wachsen
	if(dom->rho_aus_max<n_plane) { print_error("rho_ausgabe_ebene: Ebene mit "+to_string(n_plane)+" Zellen, Puffer "+to_string(dom->rho_aus_max)+" -- alloc_rho_ausgabe vorher gross genug rufen."); return; }
	if(t_aus+1ull!=get_t()&&dom->rho_rand_on) print_error("rho_ausgabe_ebene unter RHO_RAND: t_aus muss get_t()-1 sein (Nachkollisions-Populationen des letzten Schritts, C2-Plan Falle 7).");
	uint h219_0 = 0u, h220_0 = 0u;
	if(zaehlen) { dom->finish_queue(); dom->rho_clamp_hits.read_from_device(); h219_0 = dom->rho_clamp_hits[219]; h220_0 = dom->rho_clamp_hits[220]; }
	dom->kernel_rho_ausgabe_ebene.set_ranges(n_plane);
	dom->kernel_rho_ausgabe_ebene.set_parameters(3u, t_aus);
	dom->kernel_rho_ausgabe_ebene.set_parameters(5u, plane.axis, plane.origin.x, plane.origin.y, plane.origin.z, plane.extent_a, plane.extent_b, zaehlen ? 1u : 0u);
	dom->kernel_rho_ausgabe_ebene.enqueue_run();
	dom->finish_queue(); // Zero-Copy-Falle
	dom->rho_aus.read_from_device(0ull, n_plane);
	out.resize(n_plane);
	for(ulong i=0ull; i<n_plane; i++) out[i] = dom->rho_aus[i];
	if(zaehlen&&dom->rho_rand_on) { // ★ C2c: Ist aus den Slots, Soll aus den Geraete-Flags derselben Ebene (TYPE_E ohne TYPE_S-Bit).
		// NUR unter RHO_RAND (Pruefpass C2c, H1): der Pruefarm ohne RHO_RAND zaehlt selbst und ruft auch z-Ebenen (Nx*Ny), die
		// slice_flags (coupling_max_plane_cells) nicht fasst -- sonst schriebe extract_plane_flags ueber das Pufferende.
		if(dom->coupling_max_plane_cells<n_plane) { print_error("rho_ausgabe_ebene: gezaehlter Aufruf fuer "+to_string(n_plane)+" Zellen, slice_flags fasst nur "+to_string(dom->coupling_max_plane_cells)+"."); return; }
		dom->rho_clamp_hits.read_from_device();
		rho_aus_ist_219 += (ulong)(dom->rho_clamp_hits[219]-h219_0); rho_aus_ist_220 += (ulong)(dom->rho_clamp_hits[220]-h220_0);
		dom->kernel_extract_plane_flags.set_ranges(n_plane);
		dom->kernel_extract_plane_flags.set_parameters(2u, plane.axis, plane.origin.x, plane.origin.y, plane.origin.z, plane.extent_a, plane.extent_b);
		dom->kernel_extract_plane_flags.enqueue_run();
		dom->finish_queue();
		dom->slice_flags.read_from_device(0ull, n_plane);
		ulong ne = 0ull; for(ulong i=0ull; i<n_plane; i++) if((dom->slice_flags[i]&(TYPE_S|TYPE_E))==TYPE_E) ne++; // reines TYPE_E (TYPE_MS = TYPE_S|TYPE_E faellt heraus)
		rho_aus_gezaehlt_zellen += n_plane; rho_aus_gezaehlt_e += ne;
		// Pruefpass C2c NIEDRIG 5: Geraete-rr_idx gegen Host-rr_idx_host EXAKT -- die Ausgabe liest an TYPE_E load_rho(rho, rr_idx(n)),
		// der Host dekodiert das R1-Wort an rr_idx_host(n). Gleiches Wort -> gleicher float (Skalierung 2^-15 exakt).
		dom->rho.read_from_device();
		for(ulong i=0ull; i<n_plane; i++) {
			if((dom->slice_flags[i]&(TYPE_S|TYPE_E))!=TYPE_E) continue;
			const ulong a = i%(ulong)plane.extent_a, b = i/(ulong)plane.extent_a;
			// Pruefpass C2c Pass 2 N1: volle Zuordnung wie plane_cell_index (kernel.cpp), einschliesslich der In-Ebenen-Urspruenge.
			// Abdeckung von rr_idx: nur die Zweige, die TYPE_E-Zellen dieser Ebene treffen (y-Slice: x<2, x>=Nx-2, z>=Nz-2);
			// y-Ring und z<2 decken die Host-Bijektion (pruefe_rho_rand_c0) und der Kopplungs-Verify ab.
			const ulong x = plane.axis==0u ? (ulong)plane.origin.x : (ulong)plane.origin.x+a;
			const ulong y = plane.axis==0u ? (ulong)plane.origin.y+a : (plane.axis==1u ? (ulong)plane.origin.y : (ulong)plane.origin.y+b);
			const ulong z = plane.axis==2u ? (ulong)plane.origin.z : (ulong)plane.origin.z+b;
			const ulong n = x+(y+z*(ulong)Ny)*(ulong)Nx;
			const ulong rr = rr_idx_host(n, Nx, Ny, Nz);
			rho_aus_rr_verglichen++;
			if(rr>=dom->rr_N||as_uint(out[i])!=as_uint(rho_unpack(dom->rho[rr]))) rho_aus_rr_abw++;
		}
	}
}

// ★ 15.09.2026 RHO_RAND C1: rho einer Ebene aus den DDFs. t_rek ist normalerweise das aktuelle Domaenen-t (dann
// liefert der Kernel das rho, das stream_collide(t) gleich speichert); t-1 ist der Negativtest der Pruefung.
void LBM::rho_rek_ebene(const PlaneSpec& plane, const ulong t_rek, const uint modus, std::vector<float>& out4, std::vector<rhoxx>& worte) {
	LBM_Domain* dom = lbm_domain[0];
	const ulong n_plane = (ulong)plane.extent_a*(ulong)plane.extent_b;
	if(get_D()>1u) { print_error("rho_rek_ebene: nur fuer eine Domaene gebaut (Plan §6, D=1)."); return; }
	if(!initialized) { print_error("rho_rek_ebene vor der Initialisierung."); return; } // C2-Plan N7
	if(modus>1u) { print_error("rho_rek_ebene: modus kennt nur 0 (Identitaet) und 1 (Nachkollision)."); return; }
	if(!plane_fits(plane, "rho_rek_ebene")) return;
	if(dom->rho_rek_max<n_plane) { print_error("rho_rek_ebene: Ebene mit "+to_string(n_plane)+" Zellen, Puffer "+to_string(dom->rho_rek_max)+" -- alloc_rho_rek vorher gross genug rufen."); return; }
	dom->kernel_rho_rek_ebene.set_ranges(n_plane);
	dom->kernel_rho_rek_ebene.set_parameters(4u, t_rek);
	dom->kernel_rho_rek_ebene.set_parameters(7u, plane.axis, plane.origin.x, plane.origin.y, plane.origin.z, plane.extent_a, plane.extent_b, modus);
	dom->kernel_rho_rek_ebene.enqueue_run();
	dom->finish_queue(); // Zero-Copy-Falle: vor dem Lesen die Ausfuehrung erzwingen
	dom->rho_rek_out.read_from_device(0ull, n_plane*4ull);
	dom->rho_rek_wort.read_from_device(0ull, n_plane);
	out4.resize(n_plane*4ull); worte.resize(n_plane);
	for(ulong i=0ull; i<n_plane*4ull; i++) out4[i] = dom->rho_rek_out[i];
	for(ulong i=0ull; i<n_plane; i++) worte[i] = dom->rho_rek_wort[i];
}

// ★ Slice-Ebenen-Read (Perf-Hebel 2026-08-26, Plan "Variante b"): (rho,u,flags) EINER y-Ebene
// per Device-Gather holen und in die Host-Arrays streuen. Renderer, Diff-Schnitt und Sonden
// lesen unveraendert dieselben Host-Indizes -- es aendert sich NUR der Transportweg
// (4-mm-Nahdomaene: ~14 MB statt ~8,65 GB je Slice-Ereignis). Wertgleichheit per Konstruktion:
// identische floats, Indexkonvention exakt plane_cell_index (x + y*Nx + z*Nx*Ny).
// ★ 15.09.2026 RHO_RAND C2c: Fassade im RAND-Betrieb (lbm.hpp, Rho_Feld).
void LBM::Rho_Feld::binde_rand(LBM* l) {
	lbm_ = l; rand = true;
	if(l->get_D()!=1u) print_error("RHO_RAND: Rho_Feld nur fuer eine Domaene gebaut.");
}
void LBM::Rho_Feld::read_from_device() {
	c.read_from_device();
	if(rand) r1_t = lbm_->get_t();
}
float LBM::Rho_Feld::get_rand(const ulong n) {
	const uint Nx = lbm_->get_Nx(), Ny = lbm_->get_Ny(), Nz = lbm_->get_Nz();
	const ulong NxNy = (ulong)Nx*(ulong)Ny;
	const uint x = (uint)(n%(ulong)Nx), y = (uint)((n/(ulong)Nx)%(ulong)Ny), z = (uint)(n/NxNy);
	const bool auf_ebene = (ebene_achse==1u&&y==ebene_pos)||(ebene_achse==2u&&z==ebene_pos);
	if(auf_ebene&&ebene_t==lbm_->get_t()) { // Cache nur fuer den Zeitschritt, zu dem er entstand (C2-Plan Falle 5)
		n_cache++;
		return ebene_achse==1u ? ebene[(size_t)((ulong)x+(ulong)z*(ulong)Nx)] : ebene[(size_t)((ulong)x+(ulong)y*(ulong)Nx)];
	}
	LBM_Domain* d = lbm_->lbm_domain[0];
	const ulong rr = rr_idx_host(n, Nx, Ny, Nz);
	const bool gepflegt = rr<d->rr_N&&(((lbm_->flags[n]&(TYPE_S|TYPE_E))==TYPE_E)||x+2u>=Nx);
	if(!gepflegt) {
		if(auf_ebene) print_error("RHO_RAND: Hostzugriff auf rho an ("+to_string(x)+","+to_string(y)+","+to_string(z)+") -- die Ausgabe-Ebene ist VERALTET (t="+to_string(ebene_t)+", jetzt "+to_string(lbm_->get_t())+"). Host-Zugriffssperre.");
		print_error("RHO_RAND: Hostzugriff auf rho an ("+to_string(x)+","+to_string(y)+","+to_string(z)+") -- weder in der aktuellen Ausgabe-Ebene noch in der gepflegten Randschale (TYPE_E oder x >= Nx-2). Host-Zugriffssperre.");
	}
	if(r1_t!=lbm_->get_t()) print_error("RHO_RAND: Hostzugriff auf R1 an ("+to_string(x)+","+to_string(y)+","+to_string(z)+") -- der R1-Hostspiegel ist VERALTET (gelesen bei t="+to_string(r1_t)+", jetzt "+to_string(lbm_->get_t())+"). Vorher rho.read_from_device().");
	n_r1++;
	return rho_unpack(d->rho[rr]); // Speicherwort (an TYPE_E der gelesene Randwert; an x >= Nx-2 das von stream_collide geschriebene rhon, NICHT die Nachkollisionssumme)
}
void LBM::rho_schicht_in_host(const uint z, const bool zaehlen) {
	PlaneSpec plane; plane.origin = uint3(0u, 0u, z); plane.extent_a = Nx; plane.extent_b = Ny; plane.axis = 2u;
	std::vector<float> w;
	rho_ausgabe_ebene(plane, get_t()-1ull, zaehlen, w);
	rho.setze_ebene(2u, z, get_t(), w);
}

void LBM::lese_yslice_in_host(const uint y) {
	LBM_Domain* dom = lbm_domain[0];
#ifndef UPDATE_FIELDS
	if(initialized) dom->enqueue_update_fields(); // wie Memory_Container::read_from_device(): u/rho erst aktualisieren
#endif // UPDATE_FIELDS
	PlaneSpec plane; plane.origin = uint3(0u, y, 0u); plane.extent_a = Nx; plane.extent_b = Nz; plane.axis = 1u;
	static std::vector<float> ebene; // sequenzielle Nutzung im Hauptthread; vor jedem Aufruf geleert
	ebene.clear();
	extract_plane_macros(plane, ebene);
	const ulong n_plane = (ulong)Nx*(ulong)Nz;
	if(ebene.size()<n_plane*4ull) return; // Verteidigung: heute unerreichbar (jeder Wrapper-Abbruch endet in print_error/exit), bleibt fuer den Fall, dass print_error je nicht-fatal wird (Auditor-A NIEDRIG-1)
	dom->kernel_extract_plane_flags.set_ranges(n_plane);
	dom->kernel_extract_plane_flags.set_parameters(2u, plane.axis,
		plane.origin.x, plane.origin.y, plane.origin.z, plane.extent_a, plane.extent_b);
	dom->kernel_extract_plane_flags.enqueue_run();
	dom->finish_queue();
	dom->slice_flags.read_from_device(0ull, n_plane);
	for(uint z=0u; z<Nz; z++) for(uint x=0u; x<Nx; x++) {
		const ulong g = (ulong)x + (ulong)z*(ulong)Nx;                                // Ebenen-Index (a=x, b=z)
		const ulong n = (ulong)x + ((ulong)y + (ulong)z*(ulong)Ny)*(ulong)Nx;         // Domaenen-Index
		const ulong o = g*4ull;
		if(!dom->rho_rand_on) rho.set(n, ebene[o]); // ★ C2c: unter RHO_RAND traegt ebene[o] den NaN-Marker; rho kommt unten aus der Ausgabe
		u.x[n] = ebene[o+1ull]; u.y[n] = ebene[o+2ull]; u.z[n] = ebene[o+3ull];
		flags[n] = dom->slice_flags[g];
	}
	if(dom->rho_rand_on) { // ★ 15.09. RHO_RAND C2c: rho der Ebene als Nachkollisionssumme (Entscheidung (b)); der erste Aufruf wird gezaehlt
		std::vector<float> w;
		const bool zaehlen = rho_aus_gezaehlt_zellen==0ull;
		rho_ausgabe_ebene(plane, get_t()-1ull, zaehlen, w);
		rho.setze_ebene(1u, y, get_t(), w);
	}
}

void LBM::drive_boundary_from_coarse(const PlaneSpec& fine_plane, const std::vector<float>& coarse_face, const uint coarse_a, const uint coarse_b, const uint ratio) {
	LBM_Domain* dom = lbm_domain[0];
	const ulong n_coarse = (ulong)coarse_a*(ulong)coarse_b;
	// ratio=0 waere im Kernel eine Ganzzahldivision durch null (undefiniert), coarse_a=0 liesse
	// (coarse_a-1u)*ratio als uint unterlaufen und die Konventionspruefung unten durchgehen.
	if(ratio==0u || coarse_a==0u || coarse_b==0u) { print_error("drive_boundary_from_coarse: ratio="+to_string(ratio)+", grobe Ausdehnung "+to_string(coarse_a)+"x"+to_string(coarse_b)+" -- keines davon darf 0 sein."); return; }
	if(!plane_fits(fine_plane, "drive_boundary_from_coarse")) return;
	if(dom->coupling_max_plane_cells==0ull) { print_error("drive_boundary_from_coarse ohne alloc_coupling_planes."); return; }
	if(n_coarse>dom->coupling_max_plane_cells) { print_error("drive_boundary_from_coarse: grobe Ebene passt nicht in den Puffer."); return; }
	if((ulong)coarse_face.size()<n_coarse*4ull) { print_error("drive_boundary_from_coarse: grobe Ebene zu klein ("+to_string((ulong)coarse_face.size())+" < "+to_string((ulong)(n_coarse*4ull))+")."); return; }
	// Deckungspunkt-Konvention pruefen. Stimmt sie nicht, laege die interpolierte Ebene raeumlich
	// verschoben auf dem Rand -- ein Fehler, den kein Kraftverlauf als solchen zeigen wuerde.
	if(fine_plane.extent_a!=(coarse_a-1u)*ratio+1u || fine_plane.extent_b!=(coarse_b-1u)*ratio+1u) {
		print_error("drive_boundary_from_coarse: Ausdehnungen passen nicht zusammen. Grob "+to_string(coarse_a)+"x"+to_string(coarse_b)
			+" bei ratio="+to_string(ratio)+" verlangt fein "+to_string((coarse_a-1u)*ratio+1u)+"x"+to_string((coarse_b-1u)*ratio+1u)
			+", bekommen "+to_string(fine_plane.extent_a)+"x"+to_string(fine_plane.extent_b)+".");
		return;
	}
	for(ulong i=0ull; i<n_coarse*4ull; i++) dom->coupling_plane[i] = coarse_face[i];
	dom->coupling_plane.write_to_device(0ull, n_coarse*4ull);
	dom->kernel_drive_boundary_cubic_lift.set_ranges((ulong)fine_plane.extent_a*(ulong)fine_plane.extent_b);
	dom->kernel_drive_boundary_cubic_lift.set_parameters(4u, fine_plane.axis,
		fine_plane.origin.x, fine_plane.origin.y, fine_plane.origin.z, fine_plane.extent_a, fine_plane.extent_b,
		coarse_a, coarse_b, ratio);
	dom->kernel_drive_boundary_cubic_lift.enqueue_run();
	dom->finish_queue();
}

// ★ P9c N2F-SCHALE: LBM-Ebenen-Wrapper (Muster alloc_coupling_planes/extract_plane_macros).
void LBM::alloc_schale(const std::vector<ulong>& liste, const std::vector<float>& gewichte, const uint ratio, const uint modus, const bool blendet) {
	if(get_D()!=1u) { print_error("N2F-Schale: nur fuer je eine Domaene je LBM-Instanz gebaut."); return; }
	if(!initialized) { print_error("alloc_schale vor der Initialisierung. Erst run(0) rufen."); return; }
	lbm_domain[0]->alloc_schale(liste, gewichte, ratio, modus, blendet);
}

void LBM::schale_extract_u(std::vector<float>& out, const uint mittel) {
	LBM_Domain* dom = lbm_domain[0];
	if(dom->schale_n==0u) { print_error("schale_extract_u ohne alloc_schale."); return; }
	// Blockierend (Muster extract_plane_macros): der Aufrufer steht im Kopplungsfenster, die
	// Warteschlange dieser Instanz ist dort ohnehin leer (lbm_f nach run(ratio), lbm_c nach finish()).
	dom->kernel_schale_extract.set_parameters(5u, mittel).enqueue_run();
	dom->finish_queue();
	dom->schale_uout.read_from_device();
	const ulong m = 3ull*(ulong)dom->schale_n;
	out.resize(m);
	for(ulong i=0ull; i<m; i++) out[i] = dom->schale_uout[i];
}

void LBM::schale_upload_unear(const std::vector<float>& unear) {
	LBM_Domain* dom = lbm_domain[0];
	if(dom->schale_n==0u) { print_error("schale_upload_unear ohne alloc_schale."); return; }
	const ulong m = 3ull*(ulong)dom->schale_n;
	if((ulong)unear.size()<m) { print_error("schale_upload_unear: Puffer zu klein ("+to_string((ulong)unear.size())+" < "+to_string(m)+")."); return; }
	if(dom->schale_unear.length()<3ull*(ulong)dom->schale_n) { print_error("schale_upload_unear auf eine Domaene mit blendet=false -- der Blend-Eingang ist dort ein Dummy (VRAM-Sparmassnahme 4, 08.09.)."); return; }
	for(ulong i=0ull; i<m; i++) dom->schale_unear[i] = unear[i];
	dom->schale_unear.write_to_device();
}

void LBM::reset() { // reset simulation (takes effect in following run() call)
	initialized = false;
}

#ifdef FORCE_FIELD
void LBM::update_force_field() { // calculate forces from fluid on TYPE_S cells
	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->enqueue_update_force_field();
	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->finish_queue();
}
float3 LBM::object_center_of_mass(const uchar flag_marker) { // calculate center of mass of all cells flagged with flag_marker
	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->enqueue_object_center_of_mass(flag_marker);
	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->finish_queue();
	float3 object_com = float3(0.0f, 0.0f, 0.0f);
	ulong object_cells = 0ull;
	for(uint d=0u; d<get_D(); d++) {
		object_com += float3(lbm_domain[d]->object_sum.x[0], lbm_domain[d]->object_sum.y[0], lbm_domain[d]->object_sum.z[0]);
		object_cells += (ulong)as_uint(lbm_domain[d]->object_sum.w[0]);
	}
	return object_com/(float)object_cells;
}
float3 LBM::object_force(const uchar flag_marker) { // add up force for all cells flagged with flag_marker
	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->enqueue_object_force(flag_marker);
	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->finish_queue();
	float3 object_force = float3(0.0f, 0.0f, 0.0f);
	for(uint d=0u; d<get_D(); d++) object_force += float3(lbm_domain[d]->object_sum.x[0], lbm_domain[d]->object_sum.y[0], lbm_domain[d]->object_sum.z[0]);
	return object_force;
}
float3 LBM::object_force_zband(const uchar flag_marker, const uint z_lo, const uint z_hi) { // FORK Kraft-Zerlegung (CFD_KRAFT_ZBAND): object_force auf das z-Band [z_lo,z_hi)
	if(get_D()>1u) print_error("object_force_zband: coordinates() ist domaenenlokal -- nur D=1.");
	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->enqueue_object_force_zband(flag_marker, z_lo, z_hi);
	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->finish_queue();
	float3 object_force = float3(0.0f, 0.0f, 0.0f);
	for(uint d=0u; d<get_D(); d++) object_force += float3(lbm_domain[d]->object_sum.x[0], lbm_domain[d]->object_sum.y[0], lbm_domain[d]->object_sum.z[0]);
	return object_force;
}
float3 LBM::object_torque(const float3& rotation_center, const uchar flag_marker) { // add up torque around specified rotation center for all cells flagged with flag_marker
	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->enqueue_object_torque(rotation_center, flag_marker);
	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->finish_queue();
	float3 object_torque = float3(0.0f, 0.0f, 0.0f);
	for(uint d=0u; d<get_D(); d++) object_torque += float3(lbm_domain[d]->object_sum.x[0], lbm_domain[d]->object_sum.y[0], lbm_domain[d]->object_sum.z[0]);
	return object_torque;
}
#endif // FORCE_FIELD

#ifdef MOVING_BOUNDARIES
void LBM::update_moving_boundaries() { // mark/unmark cells next to TYPE_S cells with velocity!=0 with TYPE_MS
	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->enqueue_update_moving_boundaries();
	communicate_flags();
	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->finish_queue();
#ifdef GRAPHICS
	camera.key_update = true; // to prevent flickering of flags in interactive graphics when camera is not moved
#endif // GRAPHICS
}
#endif // MOVING_BOUNDARIES

#if defined(PARTICLES)&&!defined(FORCE_FIELD)
void LBM::integrate_particles(const ulong steps, const ulong total_steps, const uint time_step_multiplicator) { // intgegrate passive tracer particles forward in time in stationary flow field
	info.append(steps, total_steps, get_t());
	Clock clock;
	for(ulong i=1ull; i<=steps; i+=(ulong)time_step_multiplicator) {
#if defined(INTERACTIVE_GRAPHICS)||defined(INTERACTIVE_GRAPHICS_ASCII)
		while(!key_P&&running) sleep(0.016);
		if(!running) break;
#endif // INTERACTIVE_GRAPHICS_ASCII || INTERACTIVE_GRAPHICS
		clock.start();
		for(uint d=0u; d<get_D(); d++) lbm_domain[d]->enqueue_integrate_particles(time_step_multiplicator);
		for(uint d=0u; d<get_D(); d++) lbm_domain[d]->finish_queue();
		for(uint d=0u; d<get_D(); d++) lbm_domain[d]->increment_time_step(time_step_multiplicator);
		info.update(clock.stop());
	}
}
#endif // PARTICLES&&!FORCE_FIELD

void LBM::write_status(const string& path) { // write LBM status report to a .txt file
	string status = "";
	status += "Grid Resolution = "+to_string(Nx)+" x "+to_string(Ny)+" x "+to_string(Nz)+" = "+to_string(get_N())+"\n";
	status += "Grid Domains = "+to_string(Dx)+" x "+to_string(Dy)+" x "+to_string(Dz)+" = "+to_string(get_D())+"\n";
	status += "LBM Type = D"+string(get_velocity_set()==9 ? "2" : "3")+"Q"+to_string(get_velocity_set())+" "+info.collision+"\n";
	status += "Memory Usage = CPU "+to_string(info.cpu_mem_required)+" MB, GPU "+to_string(get_D())+"x "+to_string(info.gpu_mem_required)+" MB\n";
	status += "Maximum Allocation Size = "+to_string((uint)(get_N()/(ulong)get_D()*(ulong)(get_velocity_set()*sizeof(fpxx))/1048576ull))+" MB\n";
	status += "Time Steps = "+to_string(get_t())+" / "+(info.steps==max_ulong ? "infinite" : to_string(info.steps))+"\n";
	status += "Runtime = "+print_time(info.runtime_total)+" (total) = "+print_time(info.runtime_lbm)+" (LBM) + "+print_time(info.runtime_total-info.runtime_lbm)+" (rendering and data evaluation)\n";
	status += "Average MLUPs/s = "+to_string(to_uint(1E-6*(double)get_N()*(double)get_t()/info.runtime_lbm))+"\n";
	status += "Kinematic Viscosity = "+to_string(get_nu())+"\n";
	status += "Relaxation Time = "+to_string(get_tau())+"\n";
	status += "Maximum Reynolds Number = "+to_string(get_Re_max())+"\n";
#ifdef VOLUME_FORCE
	status += "Volume Force = ("+to_string(get_fx())+", "+to_string(get_fy())+", "+to_string(get_fz())+")\n";
#endif // VOLUME_FORCE
#ifdef SURFACE
	status += "Surface Tension Coefficient = "+to_string(get_sigma())+"\n";
#endif // SURFACE
#ifdef TEMPERATURE
	status += "Thermal Diffusion Coefficient = "+to_string(get_alpha())+"\n";
	status += "Thermal Expansion Coefficient = "+to_string(get_beta())+"\n";
#endif // TEMPERATURE
	const string filename = default_filename(path, "status", ".txt", get_t());
	write_file(filename, status);
}

void LBM::voxelize_mesh_on_device(const Mesh* mesh, const uchar flag, const float3& rotation_center, const float3& linear_velocity, const float3& rotational_velocity) { // voxelize triangle mesh
	if(get_D()==1u) {
		lbm_domain[0]->voxelize_mesh_on_device(mesh, flag, rotation_center, linear_velocity, rotational_velocity); // if this crashes on Windows, create a TdrDelay 32-bit DWORD with decimal value 300 in Computer\HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\GraphicsDrivers
	} else {
		parallel_for(get_D(), get_D(), [&](uint d) {
			lbm_domain[d]->voxelize_mesh_on_device(mesh, flag, rotation_center, linear_velocity, rotational_velocity);
		});
	}
#ifdef MOVING_BOUNDARIES
	if((flag&(TYPE_S|TYPE_E))==TYPE_S&&(length(linear_velocity)>0.0f||length(rotational_velocity)>0.0f)) update_moving_boundaries();
#endif // MOVING_BOUNDARIES
	if(!initialized) {
		flags.read_from_device();
		u.read_from_device();
	}
}
void LBM::unvoxelize_mesh_on_device(const Mesh* mesh, const uchar flag) { // remove voxelized triangle mesh from LBM grid by removing all flags in mesh bounding box (only required when bounding box size changes during re-voxelization)
	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->enqueue_unvoxelize_mesh_on_device(mesh, flag);
	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->finish_queue();
}
void LBM::write_mesh_to_vtk(const Mesh* mesh, const string& path, const bool convert_to_si_units) const { // write mesh to binary .vtk file
	const string filename = default_filename(path, "mesh", ".vtk", get_t());
	const string header_1 = "# vtk DataFile Version 3.0\nFluidX3D "+filename.substr(filename.rfind('/')+1)+"\nBINARY\nDATASET POLYDATA\nPOINTS "+to_string(3u*mesh->triangle_number)+" float\n";
	const string header_2 = "POLYGONS "+to_string(mesh->triangle_number)+" "+to_string(4u*mesh->triangle_number)+"\n";
	float* points = new float[9u*mesh->triangle_number];
	int* triangles = new int[4u*mesh->triangle_number];
	const float spacing = convert_to_si_units ? units.si_x(1.0f) : 1.0f;
	const float3 offset = center();
	parallel_for(mesh->triangle_number, [&](uint i) {
		points[9u*i   ] = reverse_bytes(spacing*(mesh->p0[i].x-offset.x));
		points[9u*i+1u] = reverse_bytes(spacing*(mesh->p0[i].y-offset.y));
		points[9u*i+2u] = reverse_bytes(spacing*(mesh->p0[i].z-offset.z));
		points[9u*i+3u] = reverse_bytes(spacing*(mesh->p1[i].x-offset.x));
		points[9u*i+4u] = reverse_bytes(spacing*(mesh->p1[i].y-offset.y));
		points[9u*i+5u] = reverse_bytes(spacing*(mesh->p1[i].z-offset.z));
		points[9u*i+6u] = reverse_bytes(spacing*(mesh->p2[i].x-offset.x));
		points[9u*i+7u] = reverse_bytes(spacing*(mesh->p2[i].y-offset.y));
		points[9u*i+8u] = reverse_bytes(spacing*(mesh->p2[i].z-offset.z));
		triangles[4u*i   ] = reverse_bytes(3); // 3 vertices per triangle
		triangles[4u*i+1u] = reverse_bytes(3*(int)i  ); // vertex 0
		triangles[4u*i+2u] = reverse_bytes(3*(int)i+1); // vertex 1
		triangles[4u*i+3u] = reverse_bytes(3*(int)i+2); // vertex 2
	});
	create_folder(filename);
	std::ofstream file(filename, std::ios::out|std::ios::binary);
	file.write(header_1.c_str(), header_1.length()); // write non-binary file header
	file.write((char*)points, 4u*9u*mesh->triangle_number); // write binary data
	file.write(header_2.c_str(), header_2.length()); // write non-binary file header
	file.write((char*)triangles, 4u*4u*mesh->triangle_number); // write binary data
	file.close();
	delete[] points;
	delete[] triangles;
	info.allow_printing.lock();
	print_info("File \""+filename+"\" saved.");
	info.allow_printing.unlock();
}
void LBM::voxelize_stl(const string& path, const float3& center, const float3x3& rotation, const float size, const uchar flag) { // voxelize triangle mesh
	const Mesh* mesh = read_stl(path, this->size(), center, rotation, size);
	flags.write_to_device();
	voxelize_mesh_on_device(mesh, flag);
	delete mesh;
	flags.read_from_device();
}
void LBM::voxelize_stl(const string& path, const float3x3& rotation, const float size, const uchar flag) { // read and voxelize binary .stl file (place in box center)
	voxelize_stl(path, center(), rotation, size, flag);
}
void LBM::voxelize_stl(const string& path, const float3& center, const float size, const uchar flag) { // read and voxelize binary .stl file (no rotation)
	voxelize_stl(path, center, float3x3(1.0f), size, flag);
}
void LBM::voxelize_stl(const string& path, const float size, const uchar flag) { // read and voxelize binary .stl file (place in box center, no rotation)
	voxelize_stl(path, center(), float3x3(1.0f), size, flag);
}

#ifdef GRAPHICS
int* LBM::Graphics::draw_frame() {
#ifndef UPDATE_FIELDS
	if(visualization_modes&(VIS_FIELD|VIS_STREAMLINES|VIS_Q_CRITERION)) {
		for(uint d=0u; d<lbm->get_D(); d++) lbm->lbm_domain[d]->enqueue_update_fields(); // only call update_fields() if the time step has changed since the last rendered frame
	}
#endif // UPDATE_FIELDS
	if(key_1) { visualization_modes = (visualization_modes&~0b11)|(((visualization_modes&0b11)+1)%4); key_1 = false; }
	if(key_2) { visualization_modes ^= VIS_FIELD        ; key_2 = false; }
	if(key_3) { visualization_modes ^= VIS_STREAMLINES  ; key_3 = false; }
	if(key_4) { visualization_modes ^= VIS_Q_CRITERION  ; key_4 = false; }
	if(key_5) { visualization_modes ^= VIS_PHI_RASTERIZE; key_5 = false; }
	if(key_6) { visualization_modes ^= VIS_PHI_RAYTRACE ; key_6 = false; }
	if(key_7) { visualization_modes ^= VIS_PARTICLES    ; key_7 = false; }
	if(key_T) {
		slice_mode = (slice_mode+1)%8; key_T = false;
	}
	if(key_Z) {
#ifndef TEMPERATURE
		field_mode = (field_mode+1)%2; key_Z = false; // field_mode = { 0 (u), 1 (rho) }
#else // TEMPERATURE
		field_mode = (field_mode+1)%3; key_Z = false; // field_mode = { 0 (u), 1 (rho), 2 (T) }
#endif // TEMPERATURE
	}
	if(slice_mode==1u) {
		if(key_Q) { slice_x = clamp(slice_x-1, 0, (int)lbm->get_Nx()-1); key_Q = false; }
		if(key_E) { slice_x = clamp(slice_x+1, 0, (int)lbm->get_Nx()-1); key_E = false; }
	}
	if(slice_mode==2u) {
		if(key_Q) { slice_y = clamp(slice_y-1, 0, (int)lbm->get_Ny()-1); key_Q = false; }
		if(key_E) { slice_y = clamp(slice_y+1, 0, (int)lbm->get_Ny()-1); key_E = false; }
	}
	if(slice_mode==3u) {
		if(key_Q) { slice_z = clamp(slice_z-1, 0, (int)lbm->get_Nz()-1); key_Q = false; }
		if(key_E) { slice_z = clamp(slice_z+1, 0, (int)lbm->get_Nz()-1); key_E = false; }
	}
	const bool visualization_change = camera.key_update||last_visualization_modes!=visualization_modes||last_field_mode!=field_mode||last_slice_mode!=slice_mode||last_slice_x!=slice_x||last_slice_y!=slice_y||last_slice_z!=slice_z;
	camera.key_update = false;
	last_visualization_modes = visualization_modes;
	last_field_mode = field_mode;
	last_slice_mode = slice_mode;
	last_slice_x = slice_x;
	last_slice_y = slice_y;
	last_slice_z = slice_z;
	bool new_frame = true;
	for(uint d=0u; d<lbm->get_D(); d++) new_frame = new_frame && lbm->lbm_domain[d]->graphics.enqueue_draw_frame(visualization_modes, field_mode, slice_mode, slice_x, slice_y, slice_z, visualization_change);
	for(uint d=0u; d<lbm->get_D(); d++) lbm->lbm_domain[d]->finish_queue();
	int* bitmap = lbm->lbm_domain[0]->graphics.get_bitmap();
	int* zbuffer = lbm->lbm_domain[0]->graphics.get_zbuffer();
	for(uint d=1u; d<lbm->get_D()&&new_frame; d++) {
		const int* const bitmap_d = lbm->lbm_domain[d]->graphics.get_bitmap(); // each domain renders its own frame
		const int* const zbuffer_d = lbm->lbm_domain[d]->graphics.get_zbuffer();
		for(uint i=0u; i<camera.width*camera.height; i++) {
#ifndef GRAPHICS_TRANSPARENCY
			const int zdi = zbuffer_d[i];
			if(zdi>zbuffer[i]) {
				bitmap[i] = bitmap_d[i]; // overlay frames using their z-buffers
				zbuffer[i] = zdi;
			}
#else // GRAPHICS_TRANSPARENCY
			bitmap[i] = color_add(bitmap[i], bitmap_d[i]);
#endif // GRAPHICS_TRANSPARENCY
		}
	}
	camera.allow_labeling = new_frame; // only print new label on frame if a new frame has been rendered
	return bitmap;
}

void LBM::Graphics::set_camera_centered(const float rx, const float ry, const float fov, const float zoom) {
	camera.free = false;
	camera.rx = 0.5*pi+((double)rx*pi/180.0);
	camera.ry = pi-((double)ry*pi/180.0);
	camera.fov = clamp((float)fov, 1E-6f, 179.0f);
	camera.set_zoom(0.5f*(float)fmax(fmax(lbm->get_Nx(), lbm->get_Ny()), lbm->get_Nz())/zoom);
}
void LBM::Graphics::set_camera_free(const float3& p, const float rx, const float ry, const float fov) {
	camera.free = true;
	camera.rx = 0.5*pi+((double)rx*pi/180.0);
	camera.ry = pi-((double)ry*pi/180.0);
	camera.fov = clamp((float)fov, 1E-6f, 179.0f);
	camera.zoom = 1E16f;
	camera.pos = p;
}
bool LBM::Graphics::next_frame(const ulong total_time_steps, const float video_length_seconds) { // returns true once simulation time has progressed enough to render the next video frame for a 60fps video of specified length
	const uint new_frame = to_uint((float)lbm->get_t()/(float)total_time_steps*video_length_seconds*60.0f);
	if(new_frame!=last_exported_frame) {
		last_exported_frame = new_frame;
		return true;
	} else {
		return false;
	}
}
void LBM::Graphics::print_frame() { // preview current frame in console
#ifndef INTERACTIVE_GRAPHICS_ASCII
	camera.rendring_frame.lock(); // block rendering for other threads until finished
	camera.key_update = true; // force rendering new frame
	int* image_data = draw_frame(); // make sure the frame is fully rendered
	Image* image = new Image(camera.width, camera.height, image_data);
	info.allow_printing.lock();
	println();
	print_image(image);
	info.allow_printing.unlock();
	delete image;
	camera.rendring_frame.unlock();
#endif // INTERACTIVE_GRAPHICS_ASCII
}
void encode_image(Image* image, const string& filename, const string& extension, std::atomic_int* running_encoders) {
	if(extension==".png") write_png(filename, image);
	if(extension==".qoi") write_qoi(filename, image);
	if(extension==".bmp") write_bmp(filename, image);
	delete image; // delete image when done
	(*running_encoders)--;
}
void LBM::Graphics::write_frame(const string& path, const string& name, const string& extension, bool print_preview) { // save current frame as .png file (smallest file size, but slow)
	write_frame(0u, 0u, camera.width, camera.height, path, name, extension, print_preview);
}
void LBM::Graphics::write_frame(const uint x1, const uint y1, const uint x2, const uint y2, const string& path, const string& name, const string& extension, bool print_preview) { // save a cropped current frame with two corner points (x1,y1) and (x2,y2)
	camera.rendring_frame.lock(); // block rendering for other threads until finished
	camera.key_update = true; // force rendering new frame
	int* image_data = draw_frame(); // make sure the frame is fully rendered
	const string filename = default_filename(path, name, extension, lbm->get_t());
	const uint xa=max(min(x1, x2), 0u), xb=min(max(x1, x2), camera.width ); // sort coordinates if necessary
	const uint ya=max(min(y1, y2), 0u), yb=min(max(y1, y2), camera.height);
	Image* image = new Image(xb-xa, yb-ya); // create local copy of frame buffer
	for(uint y=0u; y<image->height(); y++) for(uint x=0u; x<image->width(); x++) image->set_color(x, y, image_data[camera.width*(ya+y)+(xa+x)]);
#ifndef INTERACTIVE_GRAPHICS_ASCII
	if(print_preview) {
		info.allow_printing.lock();
		println();
		print_image(image);
		print_info("Image \""+filename+"\" saved.");
		info.allow_printing.unlock();
	}
#endif // INTERACTIVE_GRAPHICS_ASCII
	running_encoders++;
	thread encoder(encode_image, image, filename, extension, &running_encoders); // the main bottleneck in rendering images to the hard disk is .png encoding, so encode image in new thread
	encoder.detach(); // detatch thread so it can run concurrently
	camera.rendring_frame.unlock();
}
void LBM::Graphics::write_frame_png(const string& path, bool print_preview) { // save current frame as .png file (smallest file size, but slow)
	write_frame(path, "image", ".png", print_preview);
}
void LBM::Graphics::write_frame_qoi(const string& path, bool print_preview) { // save current frame as .qoi file (small file size, fast)
	write_frame(path, "image", ".qoi", print_preview);
}
void LBM::Graphics::write_frame_bmp(const string& path, bool print_preview) { // save current frame as .bmp file (large file size, fast)
	write_frame(path, "image", ".bmp", print_preview);
}
void LBM::Graphics::write_frame_png(const uint x1, const uint y1, const uint x2, const uint y2, const string& path, bool print_preview) { // save current frame as .png file (smallest file size, but slow)
	write_frame(x1, y1, x2, y2, path, "image", ".png", print_preview);
}
void LBM::Graphics::write_frame_qoi(const uint x1, const uint y1, const uint x2, const uint y2, const string& path, bool print_preview) { // save current frame as .qoi file (small file size, fast)
	write_frame(x1, y1, x2, y2, path, "image", ".qoi", print_preview);
}
void LBM::Graphics::write_frame_bmp(const uint x1, const uint y1, const uint x2, const uint y2, const string& path, bool print_preview) { // save current frame as .bmp file (large file size, fast)
	write_frame(x1, y1, x2, y2, path, "image", ".bmp", print_preview);
}
#endif // GRAPHICS



void LBM_Domain::allocate_transfer(Device& device) { // allocate all memory for multi-device trqansfer
	ulong Amax = 0ull; // maximum domain side area of communicated directions
	if(Dx>1u) Amax = max(Amax, (ulong)Ny*(ulong)Nz); // Ax
	if(Dy>1u) Amax = max(Amax, (ulong)Nz*(ulong)Nx); // Ay
	if(Dz>1u) Amax = max(Amax, (ulong)Nx*(ulong)Ny); // Az

	transfer_buffer_p = Memory<char>(device, Amax, max(transfers*(uint)sizeof(fpxx), 17u), true, true, 0, false); // only allocate one set of transfer buffers in plus/minus directions, for all x/y/z transfers
	transfer_buffer_m = Memory<char>(device, Amax, max(transfers*(uint)sizeof(fpxx), 17u), true, true, 0, false); // these transfer buffers must not be zero-copy!

	kernel_transfer[enum_transfer_field::fi              ][0] = Kernel(device, 0ull, "transfer_extract_fi"              , 0u, t, transfer_buffer_p, transfer_buffer_m, fi);
	kernel_transfer[enum_transfer_field::fi              ][1] = Kernel(device, 0ull, "transfer__insert_fi"              , 0u, t, transfer_buffer_p, transfer_buffer_m, fi);
	kernel_transfer[enum_transfer_field::rho_u_flags     ][0] = Kernel(device, 0ull, "transfer_extract_rho_u_flags"     , 0u, t, transfer_buffer_p, transfer_buffer_m, rho, u, flags);
	kernel_transfer[enum_transfer_field::rho_u_flags     ][1] = Kernel(device, 0ull, "transfer__insert_rho_u_flags"     , 0u, t, transfer_buffer_p, transfer_buffer_m, rho, u, flags);
	kernel_transfer[enum_transfer_field::flags           ][0] = Kernel(device, 0ull, "transfer_extract_flags"           , 0u, t, transfer_buffer_p, transfer_buffer_m, flags);
	kernel_transfer[enum_transfer_field::flags           ][1] = Kernel(device, 0ull, "transfer__insert_flags"           , 0u, t, transfer_buffer_p, transfer_buffer_m, flags);
#ifdef FORCE_FIELD
	kernel_transfer[enum_transfer_field::F               ][0] = Kernel(device, 0ull, "transfer_extract_F"               , 0u, t, transfer_buffer_p, transfer_buffer_m, F);
	kernel_transfer[enum_transfer_field::F               ][1] = Kernel(device, 0ull, "transfer__insert_F"               , 0u, t, transfer_buffer_p, transfer_buffer_m, F);
#endif // FORCE_FIELD
#ifdef SURFACE
	kernel_transfer[enum_transfer_field::phi_massex_flags][0] = Kernel(device, 0ull, "transfer_extract_phi_massex_flags", 0u, t, transfer_buffer_p, transfer_buffer_m, phi, massex, flags);
	kernel_transfer[enum_transfer_field::phi_massex_flags][1] = Kernel(device, 0ull, "transfer__insert_phi_massex_flags", 0u, t, transfer_buffer_p, transfer_buffer_m, phi, massex, flags);
#endif // SURFACE
#ifdef TEMPERATURE
	kernel_transfer[enum_transfer_field::gi              ][0] = Kernel(device, 0ull, "transfer_extract_gi"              , 0u, t, transfer_buffer_p, transfer_buffer_m, gi);
	kernel_transfer[enum_transfer_field::gi              ][1] = Kernel(device, 0ull, "transfer__insert_gi"              , 0u, t, transfer_buffer_p, transfer_buffer_m, gi);
	kernel_transfer[enum_transfer_field::T               ][0] = Kernel(device, 0ull, "transfer_extract_T"               , 0u, t, transfer_buffer_p, transfer_buffer_m, T);
	kernel_transfer[enum_transfer_field::T               ][1] = Kernel(device, 0ull, "transfer__insert_T"               , 0u, t, transfer_buffer_p, transfer_buffer_m, T);
#endif // TEMPERATURE
}

ulong LBM_Domain::get_area(const uint direction) {
	const ulong A[3] = { (ulong)Ny*(ulong)Nz, (ulong)Nz*(ulong)Nx, (ulong)Nx*(ulong)Ny };
	return A[direction];
}
void LBM_Domain::enqueue_transfer_extract_field(Kernel& kernel_transfer_extract_field, const uint direction, const uint bytes_per_cell) {
	kernel_transfer_extract_field.set_ranges(get_area(direction)); // direction: x=0, y=1, z=2
	kernel_transfer_extract_field.set_parameters(0u, direction, get_t()).enqueue_run(); // selective in-VRAM copy
	transfer_buffer_p.enqueue_read_from_device(0ull, kernel_transfer_extract_field.range()*(ulong)bytes_per_cell); // PCIe copy (+)
	transfer_buffer_m.enqueue_read_from_device(0ull, kernel_transfer_extract_field.range()*(ulong)bytes_per_cell); // PCIe copy (-)
}
void LBM_Domain::enqueue_transfer_insert_field(Kernel& kernel_transfer_insert_field, const uint direction, const uint bytes_per_cell) {
	kernel_transfer_insert_field.set_ranges(get_area(direction)); // direction: x=0, y=1, z=2
	transfer_buffer_p.enqueue_write_to_device(0ull, kernel_transfer_insert_field.range()*(ulong)bytes_per_cell); // PCIe copy (+)
	transfer_buffer_m.enqueue_write_to_device(0ull, kernel_transfer_insert_field.range()*(ulong)bytes_per_cell); // PCIe copy (-)
	kernel_transfer_insert_field.set_parameters(0u, direction, get_t()).enqueue_run(); // selective in-VRAM copy
}
void LBM::communicate_field(const enum_transfer_field field, const uint bytes_per_cell) {
	if(Dx>1u) { // communicate in x-direction
		for(uint d=0u; d<get_D(); d++) lbm_domain[d]->enqueue_transfer_extract_field(lbm_domain[d]->kernel_transfer[field][0], 0u, bytes_per_cell); // selective in-VRAM copy (x) + PCIe copy
		for(uint d=0u; d<get_D(); d++) lbm_domain[d]->finish_queue(); // domain synchronization barrier
		for(uint d=0u; d<get_D(); d++) {
			const uint x=(d%(Dx*Dy))%Dx, y=(d%(Dx*Dy))/Dx, z=d/(Dx*Dy), dxp=((x+1u)%Dx)+(y+z*Dy)*Dx; // d = x+(y+z*Dy)*Dx
			lbm_domain[d]->transfer_buffer_p.exchange_host_buffer(lbm_domain[dxp]->transfer_buffer_m.exchange_host_buffer(lbm_domain[d]->transfer_buffer_p.data())); // CPU pointer swaps
		}
		for(uint d=0u; d<get_D(); d++) lbm_domain[d]-> enqueue_transfer_insert_field(lbm_domain[d]->kernel_transfer[field][1], 0u, bytes_per_cell); // PCIe copy + selective in-VRAM copy (x)
	}
	if(Dy>1u) { // communicate in y-direction
		for(uint d=0u; d<get_D(); d++) lbm_domain[d]->enqueue_transfer_extract_field(lbm_domain[d]->kernel_transfer[field][0], 1u, bytes_per_cell); // selective in-VRAM copy (y) + PCIe copy
		for(uint d=0u; d<get_D(); d++) lbm_domain[d]->finish_queue(); // domain synchronization barrier
		for(uint d=0u; d<get_D(); d++) {
			const uint x=(d%(Dx*Dy))%Dx, y=(d%(Dx*Dy))/Dx, z=d/(Dx*Dy), dyp=x+(((y+1u)%Dy)+z*Dy)*Dx; // d = x+(y+z*Dy)*Dx
			lbm_domain[d]->transfer_buffer_p.exchange_host_buffer(lbm_domain[dyp]->transfer_buffer_m.exchange_host_buffer(lbm_domain[d]->transfer_buffer_p.data())); // CPU pointer swaps
		}
		for(uint d=0u; d<get_D(); d++) lbm_domain[d]-> enqueue_transfer_insert_field(lbm_domain[d]->kernel_transfer[field][1], 1u, bytes_per_cell); // PCIe copy + selective in-VRAM copy (y)
	}
	if(Dz>1u) { // communicate in z-direction
		for(uint d=0u; d<get_D(); d++) lbm_domain[d]->enqueue_transfer_extract_field(lbm_domain[d]->kernel_transfer[field][0], 2u, bytes_per_cell); // selective in-VRAM copy (z) + PCIe copy
		for(uint d=0u; d<get_D(); d++) lbm_domain[d]->finish_queue(); // domain synchronization barrier
		for(uint d=0u; d<get_D(); d++) {
			const uint x=(d%(Dx*Dy))%Dx, y=(d%(Dx*Dy))/Dx, z=d/(Dx*Dy), dzp=x+(y+((z+1u)%Dz)*Dy)*Dx; // d = x+(y+z*Dy)*Dx
			lbm_domain[d]->transfer_buffer_p.exchange_host_buffer(lbm_domain[dzp]->transfer_buffer_m.exchange_host_buffer(lbm_domain[d]->transfer_buffer_p.data())); // CPU pointer swaps
		}
		for(uint d=0u; d<get_D(); d++) lbm_domain[d]-> enqueue_transfer_insert_field(lbm_domain[d]->kernel_transfer[field][1], 2u, bytes_per_cell); // PCIe copy + selective in-VRAM copy (z)
	}
}

void LBM::communicate_fi() {
	communicate_field(enum_transfer_field::fi, transfers*sizeof(fpxx));
}
void LBM::communicate_rho_u_flags() {
	communicate_field(enum_transfer_field::rho_u_flags, 17u);
}
void LBM::communicate_flags() {
	communicate_field(enum_transfer_field::flags, 1u);
}
#ifdef FORCE_FIELD
void LBM::communicate_F() {
	communicate_field(enum_transfer_field::F, 12u);
}
#endif // FORCE_FIELD
#ifdef SURFACE
void LBM::communicate_phi_massex_flags() {
	communicate_field(enum_transfer_field::phi_massex_flags, 9u);
}
#endif // SURFACE
#ifdef TEMPERATURE
void LBM::communicate_gi() {
	communicate_field(enum_transfer_field::gi, sizeof(fpxx));
}
void LBM::communicate_T() {
	communicate_field(enum_transfer_field::T, 4u);
}
#endif // TEMPERATURE
#ifdef PARTICLES
void LBM::communicate_particles() {
	if(get_D()>1u) {
		if(initialized) {
			for(uint d=0u; d<get_D(); d++) lbm_domain[d]->particles.enqueue_read_from_device();
			for(uint d=0u; d<get_D(); d++) lbm_domain[d]->finish_queue(); // domain synchronization barrier
			for(ulong n=0ull; n<lbm_domain[0]->particles.length(); n++) { // parallel_for(lbm_domain[0]->particles.length(), [&](ulong n) {
				for(uint d=1u; d<get_D(); d++) { // gather modified particle positions
					const float lbm_domain_d___particles_x_n_ = lbm_domain[d]->particles.x[n];
					if(as_uint(lbm_domain_d___particles_x_n_)!=0xFFFFFFFFu) { // particle was in domain d and has been modified
						lbm_domain[0]->particles.x[n] = lbm_domain_d___particles_x_n_;
						lbm_domain[0]->particles.y[n] = lbm_domain[d]->particles.y[n];
						lbm_domain[0]->particles.z[n] = lbm_domain[d]->particles.z[n];
						break; // particle can only be in one domain at a time, no need to check other domains once it has been found
					}
				}
			} // });
		}
		for(uint d=0u; d<get_D(); d++) { // broadcast unified particle positions, using pointer of lbm_domain[0] instead of memory copy
			float* lbm_domain_d_particles_data = lbm_domain[d]->particles.exchange_host_buffer(lbm_domain[0]->particles.data());
			lbm_domain[d]->particles.enqueue_write_to_device();
			lbm_domain[d]->particles.exchange_host_buffer(lbm_domain_d_particles_data);
		}
	}
}
#endif // PARTICLES