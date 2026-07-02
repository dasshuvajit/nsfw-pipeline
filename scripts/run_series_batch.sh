#!/usr/bin/env bash
# Unattended back-to-back series generator with a ComfyUI self-heal.
#
# Runs scripts/art_series.py for a rotating list of niche/tier pairs, one after
# another, until a time budget expires — designed to run detached (overnight).
# Each series renders FOREGROUND-within-this-loop (one process total). Before
# every series it health-checks ComfyUI and restarts it if it has died, so a
# ComfyUI crash mid-run auto-recovers instead of fast-failing the rest of the set
# (learned 2026-07-02: the heavy Z-Image LoRA+TE stack can OOM ComfyUI after a
# couple of series). Progress + the resume index persist in RUNDIR, so a killed
# loop resumes where it left off.
#
# Usage:
#   nohup bash scripts/run_series_batch.sh [HOURS] >/tmp/series_batch.out 2>&1 &
#     HOURS       time budget in hours (default 8; a series in flight at the
#                 deadline still finishes).
#   Env overrides:
#     RUNDIR      state/log dir (default output/.batch_state) — delete its
#                 state.env to start the sequence from the top.
#     COUNT       images per series (default 6)
#     FIGURE      --figure-focus value (default busty)
#     COMFY_DIR   ComfyUI install (default ~/AI/apps/ComfyUI)
#   Edit SEQ below to change which niche/tier sets run (all must be zimage-safe
#   and in-band for the niche's tier_band).
set -u

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

HOURS="${1:-8}"
RUNDIR="${RUNDIR:-output/.batch_state}"
COUNT="${COUNT:-6}"
FIGURE="${FIGURE:-busty}"
COMFY_DIR="${COMFY_DIR:-$HOME/AI/apps/ComfyUI}"
COMFY_URL="http://127.0.0.1:8188"

mkdir -p "$RUNDIR"
STATE="$RUNDIR/state.env"
MASTER="$RUNDIR/master.log"
LOCK="$RUNDIR/loop.lock"

# --- singleton guard: refuse to start a second concurrent loop (state races) ---
if [ -f "$LOCK" ] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; then
  echo "another batch loop is already running (pid $(cat "$LOCK")); exiting." >&2
  exit 1
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

DEADLINE=$(( $(date +%s) + HOURS * 3600 ))

# niche/tier rotation (edit freely; each must be in the niche's tier_band).
SEQ=(
  "mythology_goddess T4_explicit"
  "dark_fantasy_vampire T4_explicit"
  "old_hollywood_glamour T2_implied"
  "bohemian_naturallight T3_artnude"
  "fantasy_glamour T3_artnude"
  "south_asian_editorial T1_suggestive"
  "athletic_studio T3_artnude"
  "renaissance_baroque T4_explicit"
  "art_deco_boudoir T1_suggestive"
  "wild_nature T3_artnude"
  "goth_romantic T3_artnude"
  "arabian_nights T4_explicit"
  "angelic_divine T3_artnude"
  "iberian_flamenco T1_suggestive"
  "cyberpunk_pinup T3_artnude"
  "thermal_bathhouse T3_artnude"
  "pinup_1950s T1_suggestive"
  "burlesque_cabaret T2_implied"
  "film_noir_boudoir T3_artnude"
  "cottagecore_pastoral T2_implied"
)
N=${#SEQ[@]}

comfy_up() { curl -s -m3 "$COMFY_URL/system_stats" >/dev/null 2>&1; }
ensure_comfy() {   # 0 if up (restarting if needed), 1 if it won't come up
  comfy_up && return 0
  echo "[$(date '+%H:%M')] ComfyUI DOWN — restarting (python3)" | tee -a "$MASTER"
  pkill -9 -f "ComfyUI/main.py" 2>/dev/null; sleep 2
  # ComfyUI runs under Homebrew python3 (its venv lacks some deps); MPS watermark
  # relaxed to reduce OOM aborts.
  ( cd "$COMFY_DIR" && PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 nohup python3 main.py > /tmp/comfy_restart.log 2>&1 & )
  local n=0
  until comfy_up || [ $n -ge 90 ]; do n=$((n+1)); sleep 2; done
  comfy_up && { echo "[$(date '+%H:%M')] ComfyUI back up (~$((n*2))s)" | tee -a "$MASTER"; return 0; }
  echo "[$(date '+%H:%M')] ComfyUI restart FAILED — see /tmp/comfy_restart.log" | tee -a "$MASTER"; return 1
}

[ -f "$STATE" ] || echo "IDX=0" > "$STATE"
source "$STATE"

echo "[$(date '+%H:%M')] BATCH START — budget ${HOURS}h, resume IDX=$IDX, ${N} slots" | tee -a "$MASTER"
fails=0
while :; do
  [ "$(date +%s)" -ge "$DEADLINE" ] && { echo "[$(date '+%H:%M')] deadline reached — ran $IDX series, stopping." | tee -a "$MASTER"; break; }
  [ "$IDX" -ge "$N" ] && { echo "[$(date '+%H:%M')] SEQ exhausted ($N) — stopping." | tee -a "$MASTER"; break; }
  if ! ensure_comfy; then echo "[$(date '+%H:%M')] ComfyUI unavailable — pause 120s, retry same slot" | tee -a "$MASTER"; sleep 120; continue; fi
  entry="${SEQ[$IDX]}"; set -- $entry; niche=$1; tier=$2
  no=$(( IDX + 1 ))
  ts=$(date '+%Y%m%d_%H%M%S'); out="output/art_series/${ts}_batch"
  rlog="$RUNDIR/$(printf '%02d' $no)_${niche}_${tier}.log"
  echo "[$(date '+%H:%M')] START #$no $niche $tier -> $out" | tee -a "$MASTER"
  caffeinate -i python3 scripts/art_series.py --niche "$niche" --tier "$tier" \
      --figure-focus "$FIGURE" --count "$COUNT" --engine zimage --out-dir "$out" > "$rlog" 2>&1
  rc=$?
  imgs=$(find "$out" -path '*/package/*' -name '*.png' 2>/dev/null | wc -l | tr -d ' ')
  echo "[$(date '+%H:%M')] DONE  #$no $niche $tier rc=$rc packaged_imgs=$imgs -> $out" | tee -a "$MASTER"
  IDX=$(( IDX + 1 )); echo "IDX=$IDX" > "$STATE"
  if [ "$rc" -ne 0 ]; then fails=$(( fails+1 )); [ "$fails" -ge 3 ] && { echo "[$(date '+%H:%M')] 3 consecutive fails — pause 120s" | tee -a "$MASTER"; sleep 120; }; else fails=0; fi
done
echo "[$(date '+%H:%M')] BATCH LOOP EXIT (IDX=$IDX)" | tee -a "$MASTER"
