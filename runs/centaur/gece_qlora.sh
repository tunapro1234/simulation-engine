#!/usr/bin/env bash
# Consumer-101 Centaur-style QLoRA night on the GPU box (Qwen3.8-27B, 96 GB card). Written 23 Oct 2026; NOT run yet.
# Order = priority; every step has a wall-clock guard so the box is quiet by 07:45 (safe stop 08:00).
#   bash runs/centaur/gece_qlora.sh            # from the simulation-engine repo root
#   KESINLIK=bf16 bash runs/centaur/gece_qlora.sh   # plain LoRA on bf16 weights (fits 96 GB; likely faster than nf4)
# Outputs: /data/talp/is/centaur/qlora/{olc,taban,kd,gd}/ ; log gece.log. Morning scoring (CPU box is fine):
#   PYTHONPATH=$PWD nice -n 19 /data/talp/venv/bin/python runs/centaur/heldout.py skor --qlora /data/talp/is/centaur/qlora/{taban,kd,gd}
set -u
cd "$(dirname "$0")/../.."
export PYTHONPATH=$PWD HF_HOME=${HF_HOME:-/data/talp/hf} TOKENIZERS_PARALLELISM=false
PY="nice -n 19 /data/talp/venv/bin/python"
Q=runs/centaur/qlora.py
OUT=/data/talp/is/centaur/qlora
KESINLIK=${KESINLIK:-nf4}
N=${N:-200}            # people per held-out set: the same 200 (fixed seed) as the Claude / Qwen API arms
mkdir -p $OUT
LOG=$OUT/gece.log
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
saniye_kadar() { local t; t=$(date -d "$1" +%s); [ "$t" -le "$(date +%s)" ] && t=$(date -d "tomorrow $1" +%s); echo $(( t - $(date +%s) )); }

log "start KESINLIK=$KESINLIK N=$N"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv | tee -a $LOG
$PY -c "import peft, bitsandbytes, transformers; print('peft', peft.__version__, 'bnb', bitsandbytes.__version__, 'transformers', transformers.__version__)" 2>&1 | tee -a $LOG
$PY -c "import fla, causal_conv1d" 2>/dev/null && log "fla + causal_conv1d: OK" \
  || log "WARNING: flash-linear-attention / causal-conv1d missing -> DeltaNet layers on the slow reference path (pip install flash-linear-attention causal-conv1d)"

# 0. speed check: 12 optimizer steps of the first training file; olc.json has tok_s and the full-epoch estimate
log "0 olc"
timeout $(saniye_kadar 21:30) $PY $Q olc --kesinlik $KESINLIK --veri talp73_s0_train_kategori_disi --cikti $OUT/olc \
  --olc-adim 12 >> $LOG 2>&1 || log "olc failed or timed out (see log); continuing"

# 1. base model zero-shot, raw text, logprob readout: the untrained row (held-out task for the base model)
log "1 taban (base, zero-shot) gorev + kategori"
timeout $(saniye_kadar 22:30) $PY $Q degerlendir --kesinlik $KESINLIK --adapter yok --etiket taban \
  --ayrim gorev kategori --n $N --cikti $OUT/taban >> $LOG 2>&1 || log "taban failed or timed out"

# 2. main run: talp73 seed-0 train people with 5 categories removed (8.4 M tokens) -> held-out people (15 seen
#    categories), cover story, held-out categories, all from one adapter
log "2 egit kd (talp73_s0_train_kategori_disi)"
timeout $(saniye_kadar 04:15) $PY $Q egit --kesinlik $KESINLIK --veri talp73_s0_train_kategori_disi --cikti $OUT/kd \
  --bitis 03:00 --etiket kd --ayrim kategori kisi kapak --n $N >> $LOG 2>&1 || log "kd failed or timed out"
if [ -d $OUT/kd/adapter ] && ! ls $OUT/kd/kapak__qlora-kd__s0.parquet >/dev/null 2>&1; then
  log "2b kd evaluation (rerun from the saved adapter)"
  timeout $(saniye_kadar 05:00) $PY $Q degerlendir --kesinlik $KESINLIK --adapter $OUT/kd/adapter --etiket kd \
    --ayrim kategori kisi kapak --n $N --cikti $OUT/kd >> $LOG 2>&1 || log "kd evaluation failed or timed out"
fi

# 3. held-out task: Twin + Psych-101 only, no talp73 at all (13.2 M tokens; expect a partial epoch, stopped by --bitis)
log "3 egit gd (karisim_gorev_disi)"
timeout $(saniye_kadar 07:40) $PY $Q egit --kesinlik $KESINLIK --veri karisim_gorev_disi --cikti $OUT/gd \
  --bitis 06:30 --etiket gd --ayrim gorev twin --n $N >> $LOG 2>&1 || log "gd failed or timed out"
log "done; score in the morning: heldout.py skor --qlora $OUT/kd (and $OUT/taban, $OUT/gd)"
