#!/bin/bash
for U in 2.0 4.0 5.0 6.0 7.0; do
  python3 psi_fixedR.py dV0 0 $U norm 0.5 1.1 0.05 > ../psi_scan/Rfix_ng0_dV0_U${U}_norm.log 2>&1
  echo "Ng=0 dV0 U=$U finished: $(tail -1 ../psi_scan/Rfix_ng0_dV0_U${U}_norm.log)" >> ../psi_scan/Rfix_ng0_progress.log
done
echo ALLDONE >> ../psi_scan/Rfix_ng0_progress.log
