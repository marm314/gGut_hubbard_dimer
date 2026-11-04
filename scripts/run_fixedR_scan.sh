#!/bin/bash
# sequential scan: continuation in s from the unmodified solution (adaptive step), one log per (tag, U)
for tag in dV0 dV2n; do for U in 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0; do
  python3 psi_fixedR.py $tag 2 $U norm 0.8 1.0 0.02 > ../psi_scan/Rfix_ng2_${tag}_U${U}_norm.log 2>&1
  echo "$tag U=$U finished: $(tail -1 ../psi_scan/Rfix_ng2_${tag}_U${U}_norm.log)" >> ../psi_scan/Rfix_scan_progress.log
done; done
echo ALLDONE >> ../psi_scan/Rfix_scan_progress.log
