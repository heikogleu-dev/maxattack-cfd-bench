#!/usr/bin/env bash
# ★ 05.10.2026 BAU-FINGERABDRUCK (Pruefbefund FLAGS4 M1, PLAN-VRAM-URAND-FLAGS "Pruefung FLAGS4"):
# code/LAUF.txt trug den HEAD zum Zeitpunkt der SICHERUNG, nicht den Stand des BINARYS. Wurde zwischen Bau und Lauf
# committet/gemergt, stand dort ein falscher Commit samt "Arbeitsbaum SAUBER" (f4c_z1, f4c2_z1, f4i2_* am 05.10.).
# Dieses Skript schreibt beim Bau den Commit, ein dirty-Flag und den sha1 des Quell-Diffs in eine kleine C++-Datei,
# die ins Binary gelinkt wird (makefile: temp/bau_id.o; bau_q27.sh: temp/q27/bau_id.o). setup.cpp (sichere_lauf)
# gibt sie im Lauflog (println BAU-FINGERABDRUCK) und in code/LAUF.txt aus und warnt, wenn HEAD != Binary-Stand.
#
# Aufruf: werkzeuge/bau_id.sh <ziel.cpp>
# Die Datei wird NUR neu geschrieben, wenn sich der Inhalt aendert -- sonst bleibt ihr mtime stehen und make linkt nicht neu.
# Keine Bauzeit im Inhalt (sie erzwaenge bei jedem make einen Neulink); die Binary-Zeit liest setup.cpp zur Laufzeit.
# Massgeblich fuer "schmutzig" sind die Pfade, die ins Binary gehen: src/, makefile, make.sh (auch unversionierte Dateien in src/).
set -u
set -o pipefail # ★ 05.10.2026 Pruefbefund 547165d M2: ein git-Fehler darf nicht als sha1 der leeren Eingabe (= "sauber") durchgehen
ziel="$1"
cd "$(dirname "$0")/.." || exit 1
pfade="src makefile make.sh"
git_ok=1
commit=$(git rev-parse --verify -q HEAD 2>/dev/null) || { commit=unbekannt; git_ok=0; }
if aend=$(git status --porcelain -- $pfade 2>/dev/null); then :; else aend="GIT-FEHLER (git status)"; git_ok=0; fi
if [ -n "$aend" ] || [ "$git_ok" = 0 ]; then schmutz=1; else schmutz=0; fi
# sha1 des Diffs der VERSIONIERTEN Quellen gegen den Commit (leer -> sha1 der leeren Eingabe da39a3ee...).
# Zur Laufzeit rechnet setup.cpp denselben Ausdruck gegen den Binary-Commit: gleich <=> src/ entspricht dem Baustand.
# Scheitert git (sudo make: dubious ownership, kein .git, ...), wird dsha "git-fehler" -- nie der Leer-sha1, und schmutzig=1.
# Direkt gepipt (keine Zwischenvariable: $(...) wuerde Schluss-Zeilenumbrueche kappen und den sha gegen die Laufzeit-Rechnung verfaelschen);
# pipefail macht einen git-Fehler im ersten Glied sichtbar.
if [ "$git_ok" = 1 ] && dsha=$(git diff "$commit" -- $pfade 2>/dev/null | sha1sum | cut -c1-40); then :; else dsha=git-fehler; git_ok=0; schmutz=1; fi
liste=$(printf '%s' "$aend" | tr '\n' ';' | sed 's/\\/\\\\/g; s/"/\\"/g')
tmp="$ziel.tmp.$$"
{
	echo "// automatisch erzeugt von werkzeuge/bau_id.sh beim Bau -- NICHT von Hand aendern, NICHT einchecken (temp/)"
	echo "const char* bau_commit() { return \"$commit\"; }"
	echo "int bau_schmutzig() { return $schmutz; }"
	echo "const char* bau_diff_sha() { return \"$dsha\"; }"
	echo "const char* bau_aenderungen() { return \"$liste\"; }"
} > "$tmp"
if [ -f "$ziel" ] && cmp -s "$tmp" "$ziel"; then rm -f "$tmp"; else mv -f "$tmp" "$ziel"; echo "Bau-Fingerabdruck: Commit ${commit:0:12}, schmutzig=$schmutz, diff-sha1 ${dsha:0:12} -> $ziel"; fi
