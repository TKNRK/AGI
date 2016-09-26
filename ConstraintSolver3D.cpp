#include <stdio.h>
#include <math.h>
#include <lbfgs.h>
#include <iostream>
#include "ConstraintSolver3D.h"
using namespace std;

const float ConstSolve3D::defaultInit[] = { 0.9 , -0.1 , 0.1 , -0.1 , 0.9 , 0.1 , 0.1 , 0.1 , 0.9 , 0.1 , -0.1 , 0.1 , -0.1 , 0.1 , 0.1 };

struct coefficients {
  lbfgsfloatval_t * _new_pos;
  lbfgsfloatval_t * _pre_pos;
  lbfgsfloatval_t norm_f0;
};

//Set Objective-function & Gradient Vector

static lbfgsfloatval_t evaluate(
        void *instance,
        const lbfgsfloatval_t *x,
        lbfgsfloatval_t *g,
        const int n,
        const lbfgsfloatval_t step
        ) {
  lbfgsfloatval_t fx = 0.0;
  //load coefficients
  lbfgsfloatval_t * _new = ((struct coefficients *) instance)->_new_pos;
  lbfgsfloatval_t * _pre = ((struct coefficients *) instance)->_pre_pos;
  lbfgsfloatval_t l = ((struct coefficients *) instance)->norm_f0;

  lbfgsfloatval_t x15 = (_new[0] - x[0] * _pre[0] - x[1] * _pre[1] - x[2] * _pre[2]) / l; //a10
  lbfgsfloatval_t x16 = (_new[1] - x[3] * _pre[0] - x[4] * _pre[1] - x[5] * _pre[2]) / l; //a20
  lbfgsfloatval_t x17 = (_new[2] - x[6] * _pre[0] - x[7] * _pre[1] - x[8] * _pre[2]) / l; //a30

  lbfgsfloatval_t norm_e1 = sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x15 * x15);
  lbfgsfloatval_t norm_e2 = sqrt(x[3] * x[3] + x[4] * x[4] + x[5] * x[5] + x16 * x16);
  lbfgsfloatval_t norm_e3 = sqrt(x[6] * x[6] + x[7] * x[7] + x[8] * x[8] + x17 * x17);

  lbfgsfloatval_t e1_dot_e2 = x[0] * x[3] + x[1] * x[4] + x[2] * x[5] + x15*x16;
  lbfgsfloatval_t e1_dot_e3 = x[0] * x[6] + x[1] * x[7] + x[2] * x[8] + x15*x17;
  lbfgsfloatval_t e2_dot_e3 = x[3] * x[6] + x[4] * x[7] + x[5] * x[8] + x16*x17;

  lbfgsfloatval_t norm_r1 = sqrt(x[9] * x[9] + x[10] * x[10] + x[11] * x[11]);
  lbfgsfloatval_t norm_r2 = sqrt(x[12] * x[12] + x[13] * x[13] + x[14] * x[14]);
  lbfgsfloatval_t r1_dot_r2 = x[9] * x[12] + x[10] * x[13] + x[11] * x[14];

  //Set objective function value
  lbfgsfloatval_t *f = lbfgs_malloc(n);

  f[0] = norm_e1 - 1;
  f[1] = norm_e2 - 1;
  f[2] = norm_e3 - 1;
  f[3] = (x[0] * x[3] + x[1] * x[4] + x[2] * x[5] + x15 * x16) / (norm_e1 * norm_e2);
  f[4] = (x[0] * x[6] + x[1] * x[7] + x[2] * x[8] + x15 * x17) / (norm_e1 * norm_e3);
  f[5] = (x[3] * x[6] + x[4] * x[7] + x[5] * x[8] + x16 * x17) / (norm_e2 * norm_e3);
  f[6] = (x[0] - 1) * x[9] + x[1] * x[10] + x[2] * x[11];
  f[7] = x[3] * x[9] + (x[4] - 1) * x[10] + x[5] * x[11];
  f[8] = x[6] * x[9] + x[7] * x[10] + (x[8] - 1) * x[11];
  f[9] = (x[0] - 1) * x[12] + x[1] * x[13] + x[2] * x[14];
  f[10] = x[3] * x[12] + (x[4] - 1) * x[13] + x[5] * x[14];
  f[11] = x[6] * x[12] + x[7] * x[13] + (x[8] - 1) * x[14];
  f[12] = norm_r1 - 1;
  f[13] = norm_r2 - 1;
  f[14] = (x[9] * x[12] + x[10] * x[13] + x[11] * x[14]) / (norm_r1 * norm_r2);
  //set fx
  for (int i = 0; i < n; i++) {
    fx += f[i] * f[i];
  }

  //Set GradientVector
  lbfgsfloatval_t d_x15_x0 = -1 * _pre[0] / l, d_x15_x1 = -1 * _pre[1] / l, d_x15_x2 = -1 * _pre[2] / l;
  lbfgsfloatval_t d_x16_x3 = -1 * _pre[0] / l, d_x16_x4 = -1 * _pre[1] / l, d_x16_x5 = -1 * _pre[2] / l;
  lbfgsfloatval_t d_x17_x6 = -1 * _pre[0] / l, d_x17_x7 = -1 * _pre[1] / l, d_x17_x8 = -1 * _pre[2] / l;

  lbfgsfloatval_t d_norm_e1_x0 = (x[0] + x15 * d_x15_x0) / norm_e1, d_norm_e1_x1 = (x[1] + x15 * d_x15_x1) / norm_e1, d_norm_e1_x2 = (x[2] + x15 * d_x15_x2) / norm_e1;
  lbfgsfloatval_t d_norm_e2_x3 = (x[3] + x16 * d_x16_x3) / norm_e2, d_norm_e2_x4 = (x[4] + x16 * d_x16_x4) / norm_e2, d_norm_e2_x5 = (x[5] + x16 * d_x16_x5) / norm_e2;
  lbfgsfloatval_t d_norm_e3_x6 = (x[6] + x17 * d_x17_x6) / norm_e3, d_norm_e3_x7 = (x[7] + x17 * d_x17_x7) / norm_e3, d_norm_e3_x8 = (x[8] + x17 * d_x17_x8) / norm_e3;

  lbfgsfloatval_t d_e1_dot_e2_x0 = x16 * d_x15_x0 + x[3], d_e1_dot_e2_x1 = x16 * d_x15_x1 + x[4], d_e1_dot_e2_x2 = x16 * d_x15_x2 + x[5];
  lbfgsfloatval_t d_e1_dot_e2_x3 = x15 * d_x16_x3 + x[0], d_e1_dot_e2_x4 = x15 * d_x16_x4 + x[1], d_e1_dot_e2_x5 = x15 * d_x16_x5 + x[2];
  lbfgsfloatval_t d_e1_dot_e3_x0 = x17 * d_x15_x0 + x[6], d_e1_dot_e3_x1 = x17 * d_x15_x1 + x[7], d_e1_dot_e3_x2 = x17 * d_x15_x2 + x[8];
  lbfgsfloatval_t d_e1_dot_e3_x6 = x15 * d_x17_x6 + x[0], d_e1_dot_e3_x7 = x15 * d_x17_x7 + x[1], d_e1_dot_e3_x8 = x15 * d_x17_x8 + x[2];
  lbfgsfloatval_t d_e2_dot_e3_x3 = x17 * d_x16_x3 + x[6], d_e2_dot_e3_x4 = x17 * d_x16_x4 + x[7], d_e2_dot_e3_x5 = x17 * d_x16_x5 + x[8];
  lbfgsfloatval_t d_e2_dot_e3_x6 = x16 * d_x17_x6 + x[3], d_e2_dot_e3_x7 = x16 * d_x17_x7 + x[4], d_e2_dot_e3_x8 = x16 * d_x17_x8 + x[5];

  lbfgsfloatval_t d_norm_r1_x9 = x[9] / norm_r1, d_norm_r1_x10 = x[10] / norm_r1, d_norm_r1_x11 = x[11] / norm_r1;
  lbfgsfloatval_t d_norm_r2_x12 = x[12] / norm_r2, d_norm_r2_x13 = x[13] / norm_r2, d_norm_r2_x14 = x[14] / norm_r2;

  lbfgsfloatval_t d_r1_dot_r2_x9 = x[12], d_r1_dot_r2_x10 = x[13], d_r1_dot_r2_x11 = x[14];
  lbfgsfloatval_t d_r1_dot_r2_x12 = x[9], d_r1_dot_r2_x13 = x[10], d_r1_dot_r2_x14 = x[11];

  //set elements
  g[0] = 2 * (f[0] * d_norm_e1_x0
          + f[3]*(d_e1_dot_e2_x0 / norm_e1 - e1_dot_e2 * d_norm_e1_x0 / (norm_e1 * norm_e1)) / norm_e2
          + f[4]*(d_e1_dot_e3_x0 / norm_e1 - e1_dot_e3 * d_norm_e1_x0 / (norm_e1 * norm_e1)) / norm_e3
          + f[6] * x[9] + f[9] * x[12]);

  g[1] = 2 * (f[0] * d_norm_e1_x1
          + f[3]*(d_e1_dot_e2_x1 / norm_e1 - e1_dot_e2 * d_norm_e1_x1 / (norm_e1 * norm_e1)) / norm_e2
          + f[4]*(d_e1_dot_e3_x1 / norm_e1 - e1_dot_e3 * d_norm_e1_x1 / (norm_e1 * norm_e1)) / norm_e3
          + f[6] * x[10] + f[9] * x[13]);

  g[2] = 2 * (f[0] * d_norm_e1_x2
          + f[3]*(d_e1_dot_e2_x2 / norm_e1 - e1_dot_e2 * d_norm_e1_x2 / (norm_e1 * norm_e1)) / norm_e2
          + f[4]*(d_e1_dot_e3_x2 / norm_e1 - e1_dot_e3 * d_norm_e1_x2 / (norm_e1 * norm_e1)) / norm_e3
          + f[6] * x[11] + f[9] * x[14]);

  g[3] = 2 * (f[1] * d_norm_e2_x3
          + f[3]*(d_e1_dot_e2_x3 / norm_e2 - e1_dot_e2 * d_norm_e2_x3 / (norm_e2 * norm_e2)) / norm_e1
          + f[5]*(d_e2_dot_e3_x3 / norm_e2 - e2_dot_e3 * d_norm_e2_x3 / (norm_e2 * norm_e2)) / norm_e3
          + f[7] * x[9] + f[10] * x[12]);

  g[4] = 2 * (f[1] * d_norm_e2_x4
          + f[3]*(d_e1_dot_e2_x4 / norm_e2 - e1_dot_e2 * d_norm_e2_x4 / (norm_e2 * norm_e2)) / norm_e1
          + f[5]*(d_e2_dot_e3_x4 / norm_e2 - e2_dot_e3 * d_norm_e2_x4 / (norm_e2 * norm_e2)) / norm_e3
          + f[7] * x[10] + f[10] * x[13]);

  g[5] = 2 * (f[1] * d_norm_e2_x5
          + f[3]*(d_e1_dot_e2_x5 / norm_e2 - e1_dot_e2 * d_norm_e2_x5 / (norm_e2 * norm_e2)) / norm_e1
          + f[5]*(d_e2_dot_e3_x5 / norm_e2 - e2_dot_e3 * d_norm_e2_x5 / (norm_e2 * norm_e2)) / norm_e3
          + f[7] * x[11] + f[10] * x[14]);

  g[6] = 2 * (f[2] * d_norm_e3_x6
          + f[4]*(d_e1_dot_e3_x6 / norm_e3 - e1_dot_e3 * d_norm_e3_x6 / (norm_e3 * norm_e3)) / norm_e1
          + f[5]*(d_e2_dot_e3_x6 / norm_e3 - e2_dot_e3 * d_norm_e3_x6 / (norm_e3 * norm_e3)) / norm_e2
          + f[8] * x[9] + f[11] * x[12]);

  g[7] = 2 * (f[2] * d_norm_e3_x7
          + f[4]*(d_e1_dot_e3_x7 / norm_e3 - e1_dot_e3 * d_norm_e3_x7 / (norm_e3 * norm_e3)) / norm_e1
          + f[5]*(d_e2_dot_e3_x7 / norm_e3 - e2_dot_e3 * d_norm_e3_x7 / (norm_e3 * norm_e3)) / norm_e2
          + f[8] * x[10] + f[11] * x[13]);

  g[8] = 2 * (f[2] * d_norm_e3_x8
          + f[4]*(d_e1_dot_e3_x8 / norm_e3 - e1_dot_e3 * d_norm_e3_x8 / (norm_e3 * norm_e3)) / norm_e1
          + f[5]*(d_e2_dot_e3_x8 / norm_e3 - e2_dot_e3 * d_norm_e3_x8 / (norm_e3 * norm_e3)) / norm_e2
          + f[8] * x[11] + f[11] * x[14]);

  g[9] = 2 * (f[6]*(x[0] - 1) + f[7] * x[3] + f[8] * x[6] + f[12] * d_norm_r1_x9
          + f[14]*(d_r1_dot_r2_x9 / norm_r1 - r1_dot_r2 * d_norm_r1_x9 / norm_r1 * norm_r1) / norm_r2);

  g[10] = 2 * (f[6] * x[1] + f[7]*(x[4] - 1) + f[8] * x[7] + f[12] * d_norm_r1_x10
          + f[14]*(d_r1_dot_r2_x10 / norm_r1 - r1_dot_r2 * d_norm_r1_x10 / norm_r1 * norm_r1) / norm_r2);

  g[11] = 2 * (f[6] * x[2] + f[7] * x[5] + f[8]*(x[8] - 1) + f[12] * d_norm_r1_x11
          + f[14]*(d_r1_dot_r2_x11 / norm_r1 - r1_dot_r2 * d_norm_r1_x11 / norm_r1 * norm_r1) / norm_r2);

  g[12] = 2 * (f[9]*(x[0] - 1) + f[10] * x[3] + f[11] * x[6] + f[13] * d_norm_r2_x12
          + f[14]*(d_r1_dot_r2_x12 / norm_r2 - r1_dot_r2 * d_norm_r2_x12 / norm_r2 * norm_r2) / norm_r1);

  g[13] = 2 * (f[9] * x[1] + f[10]*(x[4] - 1) + f[11] * x[7] + f[13] * d_norm_r2_x13
          + f[14]*(d_r1_dot_r2_x13 / norm_r2 - r1_dot_r2 * d_norm_r2_x13 / norm_r2 * norm_r2) / norm_r1);

  g[14] = 2 * (f[9] * x[2] + f[10] * x[5] + f[11]*(x[8] - 1) + f[13] * d_norm_r2_x14
          + f[14]*(d_r1_dot_r2_x14 / norm_r2 - r1_dot_r2 * d_norm_r2_x14 / norm_r2 * norm_r2) / norm_r1);

  lbfgs_free(f);
  return fx;
}

static int progress(
        void *instance,
        const lbfgsfloatval_t *x,
        const lbfgsfloatval_t *g,
        const lbfgsfloatval_t fx,
        const lbfgsfloatval_t xnorm,
        const lbfgsfloatval_t gnorm,
        const lbfgsfloatval_t step,
        int n,
        int k,
        int ls
        ) {
  return 0;
}

#define _N 15

float * solver3D(float * _pre, float * _new, float _norm_f0, float * init) {
  int ret = 0;
  lbfgsfloatval_t fx;
  lbfgsfloatval_t *x = lbfgs_malloc(_N);
  lbfgs_parameter_t param;

  for (int i = 0; i < _N; i++) {
    x[i] = init[i];
  }

  /* Initialize the parameters for the L-BFGS optimization. */
  lbfgs_parameter_init(&param);
  param.linesearch = LBFGS_LINESEARCH_BACKTRACKING;

  //Set pos_pre & pos__new & norm_f0
  lbfgsfloatval_t * pos_new = lbfgs_malloc(3);
  lbfgsfloatval_t * pos_pre = lbfgs_malloc(3);
  for (int i = 0; i < 3; i++) {
    pos_new[i] = _new[i];
    pos_pre[i] = _pre[i];
  }
  lbfgsfloatval_t norm_f0 = _norm_f0;
  struct coefficients c = {pos_new, pos_pre, norm_f0};

  /*
     Start the L-BFGS optimization; this will invoke the callback functions
     evaluate() and progress() when necessary.
   */
  ret = lbfgs(_N, x, &fx, evaluate, progress, &c, &param);

  //printf("L-BFGS optimization terminated with status code = %d\n", ret);
  //printf("fx = %f\n",fx);

  /* Report the result. */
  if (ret != 0) {
    lbfgs_free(x);
    return NULL;
  } else {
    //printf("L-BFGS optimization terminated with status code = %d\n", ret);
    //printf("  fx = %f\n",fx);

    float * res = new float[_N];
    for (int i = 0; i < _N; i++) {
      //printf("%f\n",x[i]);
      res[i] = x[i];
    }
    lbfgs_free(x);
    return res;
  }
}

/*
   int main(){
   float _new[3]; float _pre[3]; float init[N];
   _new[0] = 0.924851;   _new[1] = 0.0981696;   _new[2] = 0.0801997;
   _pre[0] = 0.910553;   _pre[1] = 0.0981696;   _pre[2] = 0.0801997;

   init[0] = 0.8; init[4] = 1.1; init[8] = 0.9;
   init[1] = 0.1; init[2] = -0.1; init[3] = -0.1; init[5] = -0.2; init[6] = 0.2; init[7] = 0.1;
   init[9] = 0.5; init[10] = -0.5; init[11] = 0.5; init[12] = 0.5; init[13] = 0.5; init[14] = 0.5;

   float norm_f0 = 2.045151;

   float * res = solver(_pre,_new,norm_f0, init);
   for(int i = 0; i < _N; i++){
   cout << res[i] << endl;
   }
   return 0;
   }
 */

