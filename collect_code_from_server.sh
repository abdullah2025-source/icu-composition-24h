#!/bin/bash
# Run ON THE PROJECT SERVER from inside the unzipped repo folder:
#     cd ~/icu-composition-24h && bash collect_code_from_server.sh
# Copies ONLY code (.py .sh .sql .md) and small JSON configs from ~/fyp_project into src/ and configs/,
# then scans everything for patient-level data and secrets. Nothing is pushed by this script.
set -u
PROJ=${PROJ:-$HOME/fyp_project}
REPO=$(pwd)
MAXKB=2048   # skip any file larger than 2 MB

echo "== 1. Collecting code from $PROJ"
cd "$PROJ"
find . \( -path ./mamba_env -o -path ./mimic-code -o -path ./data -o -path ./external_validation_data \
          -o -path ./mimic-iii-clinical-database-1.4 -o -path '*/__pycache__' -o -path '*/.ipynb_checkpoints' \
          -o -path ./outputs/documentation_master \) -prune -o \
       -type f \( -name '*.py' -o -name '*.sh' -o -name '*.sql' \) -size -${MAXKB}k -print \
  | grep -v -E '_backup|bad_gender|/tmp/|\.bak' > "$REPO/.code_list.txt"
while read -r f; do mkdir -p "$REPO/src/$(dirname "$f")"; cp "$f" "$REPO/src/$f"; done < "$REPO/.code_list.txt"
echo "   code files copied: $(wc -l < "$REPO/.code_list.txt")"

echo "== 2. Collecting model configurations and locked recipes (JSON only, < 2 MB)"
find ./outputs -type f -size -${MAXKB}k \( -iname 'best_params*.json' -o -iname '*recipe*.json' -o -iname '*config*.json' \
       -o -iname 'FINAL_*DECISION*.json' -o -iname '*source_weights*.json' \) > "$REPO/.cfg_list.txt"
while read -r f; do mkdir -p "$REPO/configs/$(dirname "${f#./outputs/}")"; cp "$f" "$REPO/configs/${f#./outputs/}"; done < "$REPO/.cfg_list.txt"
echo "   config files copied: $(wc -l < "$REPO/.cfg_list.txt")"
pip freeze > "$REPO/requirements_server_freeze.txt" 2>/dev/null
cd "$REPO"

echo "== 3. SAFETY SCAN"
FAIL=0
# 3a. forbidden file types anywhere in the repo
BAD=$(find src configs results -type f \( -name '*.csv.gz' -o -name '*.parquet' -o -name '*.npy' -o -name '*.pt' -o -name '*.pkl' \
      -o -name '*.db' -o -name '*.ckpt' -o -name '*.ipynb' \) 2>/dev/null)
[ -n "$BAD" ] && { echo "   [FAIL] data/model files found:"; echo "$BAD"; FAIL=1; }
# 3b. configs must not contain lists of identifiers (8-digit MIMIC ids)
for f in $(find configs -name '*.json'); do
  n=$(grep -o -E '\b[0-9]{8}\b' "$f" | grep -v -E '^2026[0-9]{4}$' | wc -l)
  [ "$n" -gt 20 ] && { echo "   [FAIL] $f contains $n 8-digit numbers (possible stay/subject ids) -> delete it"; FAIL=1; }
done
# 3c. secrets
SECRETS=$(grep -r -n -i -E "password *=|passwd|api[_-]?key|secret|token *= *['\"][A-Za-z0-9]|ghp_[A-Za-z0-9]{20}|physionet.*(user|pass)" src configs 2>/dev/null | grep -v -i "tokeniz\|max_token\|token_len\|tokens" )
[ -n "$SECRETS" ] && { echo "   [CHECK] possible secrets — open these lines and remove credentials:"; echo "$SECRETS"; FAIL=1; }
# 3d. large files
LARGE=$(find . -path ./.git -prune -o -type f -size +${MAXKB}k -print)
[ -n "$LARGE" ] && { echo "   [FAIL] files larger than 2 MB:"; echo "$LARGE"; FAIL=1; }
# 3e. personal paths (informational)
P=$(grep -r -l "/home/abdullah2025" src 2>/dev/null | wc -l)
echo "   [INFO] $P scripts contain the absolute path /home/abdullah2025 (harmless, but you may replace it with ~ or a config variable)"

if [ $FAIL -eq 0 ]; then echo "== SAFETY SCAN PASSED. Review 'du -sh src configs' and 'git status', then push."; else echo "== FIX THE ITEMS ABOVE BEFORE PUSHING."; fi
du -sh src configs results docs 2>/dev/null
