#!/usr/bin/env bash
# Download the six benchmark suites (~1.3 GB unpacked) into suites/ and verify every file against run/suites.sha256.
# They are not in this repository; this fetches the original artifact repository at a pinned commit as one tarball
# and unpacks only its suites/.  Idempotent: an existing suites/ that verifies is left alone.  Needs curl, tar, sha256sum.
#
#   bash run/get_suites.sh
set -eu
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC_REPO="${SUITES_REPO:-ptraceratc-boop/ptracer-artifact}"
SRC_COMMIT="${SUITES_COMMIT:-3faa9e5c5c9067db07786b040dfb03dcb2496812}"
URL="https://codeload.github.com/$SRC_REPO/tar.gz/$SRC_COMMIT"
DST="$ROOT/suites"
TGZ="$ROOT/.suites.tar.gz"

SUMS="$ROOT/run/suites.sha256"   # sha256 of every file of suites/ at $SRC_COMMIT
verify() { ( cd "$DST" && sha256sum -c --quiet "$SUMS" ); }

if [ -d "$DST" ] && verify; then
    echo "suites/ already present and verified"; exit 0
fi
if [ -e "$DST" ]; then
    echo "FATAL: $DST exists but does not verify; move it away and run this script again"; exit 2
fi

echo "Downloading $URL (~1 GB)"
ok=0
for try in 1 2 3 4 5; do
    if curl -fL --retry 3 -o "$TGZ" "$URL"; then ok=1; break; fi
    echo "download failed (try $try of 5); retrying in 10 s"; sleep 10
done
[ $ok = 1 ] || { rm -f "$TGZ"; echo "FATAL: download failed; run this script again"; exit 2; }

echo "Unpacking suites/"
mkdir "$DST"
tar -xzf "$TGZ" -C "$DST" --strip-components=2 "${SRC_REPO#*/}-$SRC_COMMIT/suites"
rm -f "$TGZ"

echo "Verifying suites/ against run/suites.sha256"
verify
echo "OK: suites/ ready ($(du -sh "$DST" | cut -f1))"
