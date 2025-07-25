
/*
Auto-generated by CVXPYgen on July 19, 2025 at 17:32:05.
Content: Example program for updating parameters, solving, and inspecting the result.
*/

#include <stdio.h>
#include "cpg_workspace.h"
#include "cpg_solve.h"

static int i;

int main(int argc, char *argv[]){

  // Update first entry of every user-defined parameter
  cpg_update_w_tr(0.00000000000000000000);
  cpg_update_f_dt_last(1.29990010464792793421);
  cpg_update_gf_dt_last(0, -0.60544692199794125642);
  cpg_update_f_ct_last(-0.25249646740600034667);
  cpg_update_A_ct_last(0, 0.81326760757103788713);
  cpg_update_B_ct_last(0, 1.36307367398911938317);
  cpg_update_C_ct_last(0, 1.37303812719025297717);
  cpg_update_S_ct_last(0, 0.51618756488335204580);
  cpg_update_X_last(0, -2.50207931350946299887);
  cpg_update_f_bar(0, 0.22385467380247964231);
  cpg_update_A_bar(0, -1.28299915588546209477);
  cpg_update_B_bar(0, -1.85999330178220301235);
  cpg_update_C_bar(0, -1.08553419276562079787);
  cpg_update_S_bar(0, 0.84985335498746561456);
  cpg_update_U_last(0, 0.74811348147299761013);
  cpg_update_S_last(0, 0.00000000000000000000);

  // Solve the problem instance
  cpg_solve();

  // Print objective function value
  printf("obj = %f\n", CPG_Result.info->obj_val);

  // Print primal solution
  for(i=0; i<156; i++) {
    printf("dx[%d] = %f\n", i, CPG_Result.prim->dx[i]);
  }
  for(i=0; i<48; i++) {
    printf("du[%d] = %f\n", i, CPG_Result.prim->du[i]);
  }
  for(i=0; i<11; i++) {
    printf("ds[%d] = %f\n", i, CPG_Result.prim->ds[i]);
  }
  for(i=0; i<143; i++) {
    printf("nu[%d] = %f\n", i, CPG_Result.prim->nu[i]);
  }
  for(i=0; i<11; i++) {
    printf("S[%d] = %f\n", i, CPG_Result.prim->S[i]);
  }
  for(i=0; i<156; i++) {
    printf("X[%d] = %f\n", i, CPG_Result.prim->X[i]);
  }
  for(i=0; i<48; i++) {
    printf("U[%d] = %f\n", i, CPG_Result.prim->U[i]);
  }

  // Print dual solution
  for(i=0; i<13; i++) {
    printf("d0[%d] = %f\n", i, CPG_Result.dual->d0[i]);
  }
  for(i=0; i<13; i++) {
    printf("d1[%d] = %f\n", i, CPG_Result.dual->d1[i]);
  }
  for(i=0; i<13; i++) {
    printf("d2[%d] = %f\n", i, CPG_Result.dual->d2[i]);
  }
  for(i=0; i<13; i++) {
    printf("d3[%d] = %f\n", i, CPG_Result.dual->d3[i]);
  }
  for(i=0; i<13; i++) {
    printf("d4[%d] = %f\n", i, CPG_Result.dual->d4[i]);
  }
  for(i=0; i<13; i++) {
    printf("d5[%d] = %f\n", i, CPG_Result.dual->d5[i]);
  }
  for(i=0; i<13; i++) {
    printf("d6[%d] = %f\n", i, CPG_Result.dual->d6[i]);
  }
  for(i=0; i<13; i++) {
    printf("d7[%d] = %f\n", i, CPG_Result.dual->d7[i]);
  }
  for(i=0; i<13; i++) {
    printf("d8[%d] = %f\n", i, CPG_Result.dual->d8[i]);
  }
  for(i=0; i<13; i++) {
    printf("d9[%d] = %f\n", i, CPG_Result.dual->d9[i]);
  }
  for(i=0; i<13; i++) {
    printf("d10[%d] = %f\n", i, CPG_Result.dual->d10[i]);
  }
  for(i=0; i<156; i++) {
    printf("d11[%d] = %f\n", i, CPG_Result.dual->d11[i]);
  }
  for(i=0; i<48; i++) {
    printf("d12[%d] = %f\n", i, CPG_Result.dual->d12[i]);
  }
  for(i=0; i<11; i++) {
    printf("d13[%d] = %f\n", i, CPG_Result.dual->d13[i]);
  }
  for(i=0; i<11; i++) {
    printf("d14[%d] = %f\n", i, CPG_Result.dual->d14[i]);
  }
  printf("d15 = %f\n", CPG_Result.dual->d15);
  for(i=0; i<11; i++) {
    printf("d16[%d] = %f\n", i, CPG_Result.dual->d16[i]);
  }
  for(i=0; i<8; i++) {
    printf("d17[%d] = %f\n", i, CPG_Result.dual->d17[i]);
  }
  for(i=0; i<8; i++) {
    printf("d18[%d] = %f\n", i, CPG_Result.dual->d18[i]);
  }
  for(i=0; i<3; i++) {
    printf("d19[%d] = %f\n", i, CPG_Result.dual->d19[i]);
  }
  for(i=0; i<3; i++) {
    printf("d20[%d] = %f\n", i, CPG_Result.dual->d20[i]);
  }
  for(i=0; i<4; i++) {
    printf("d21[%d] = %f\n", i, CPG_Result.dual->d21[i]);
  }
  for(i=0; i<4; i++) {
    printf("d22[%d] = %f\n", i, CPG_Result.dual->d22[i]);
  }
  for(i=0; i<12; i++) {
    printf("d23[%d] = %f\n", i, CPG_Result.dual->d23[i]);
  }
  for(i=0; i<12; i++) {
    printf("d24[%d] = %f\n", i, CPG_Result.dual->d24[i]);
  }
  for(i=0; i<36; i++) {
    printf("d25[%d] = %f\n", i, CPG_Result.dual->d25[i]);
  }

  return 0;

}
