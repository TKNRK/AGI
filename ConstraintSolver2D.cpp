#include <stdio.h>
#include <math.h>
#include <lbfgs.h>
#include <iostream>
#include "ConstraintSolver2D.h"

using namespace std;

const float ConstSolve2D::defaultInit[] = { 0.9, -0.1, 0.1, 0.9, 0.4, 0.4 };

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

  lbfgsfloatval_t x10 = (_new[0] - x[0] * _pre[0] - x[1] * _pre[1]) / l; //a10
  lbfgsfloatval_t x20 = (_new[1] - x[2] * _pre[0] - x[3] * _pre[1]) / l; //a20

  lbfgsfloatval_t norm_e1 = sqrt(x[0] * x[0] + x[1] * x[1] + x10 * x10);
  lbfgsfloatval_t norm_e2 = sqrt(x[2] * x[2] + x[3] * x[3] + x20 * x20);
  lbfgsfloatval_t e1_dot_e2 = x[0] * x[2] + x[1] * x[3] + x10*x20;
  lbfgsfloatval_t norm_r = sqrt(x[4] * x[4] + x[5] * x[5]);

  //Set objective function value
  lbfgsfloatval_t *f = lbfgs_malloc(n);

  f[0] = norm_e1 - 1;
  f[1] = norm_e2 - 1;
  f[2] = e1_dot_e2 / (norm_e1 * norm_e2);
  f[3] = (x[0] - 1) * x[4] + x[1] * x[5];
  f[4] = x[2] * x[4] + (x[3] - 1) * x[5];
  f[5] = norm_r - 1;
  //set fx
  for (int i = 0; i < n; i++) {
    fx += f[i] * f[i];
  }

  //Set GradientVector
  lbfgsfloatval_t d_x10_x0 = -1 * _pre[0] / l, d_x10_x1 = -1 * _pre[1] / l;
  lbfgsfloatval_t d_x20_x2 = -1 * _pre[0] / l, d_x20_x3 = -1 * _pre[1] / l;

  lbfgsfloatval_t d_norm_e1_x0 = (x[0] + x10 * d_x10_x0) / norm_e1, d_norm_e1_x1 = (x[1] + x10 * d_x10_x1) / norm_e1;
  lbfgsfloatval_t d_norm_e2_x2 = (x[2] + x20 * d_x20_x2) / norm_e2, d_norm_e2_x3 = (x[3] + x20 * d_x20_x3) / norm_e2;

  lbfgsfloatval_t d_e1_dot_e2_x0 = x20 * d_x10_x0 + x[2], d_e1_dot_e2_x1 = x20 * d_x10_x1 + x[3];
  lbfgsfloatval_t d_e1_dot_e2_x2 = x10 * d_x20_x2 + x[0], d_e1_dot_e2_x3 = x10 * d_x20_x3 + x[1];

  lbfgsfloatval_t d_norm_r_x4 = x[4] / norm_r, d_norm_r_x5 = x[5] / norm_r;

  //set elements
  g[0] = 2 * (f[0] * d_norm_e1_x0
          + f[2]*(d_e1_dot_e2_x0 * norm_e1 - e1_dot_e2 * d_norm_e1_x0) / (norm_e2 * norm_e1 * norm_e1)
          + f[3] * x[4]);

  g[1] = 2 * (f[0] * d_norm_e1_x1
          + f[2]*(d_e1_dot_e2_x1 * norm_e1 - e1_dot_e2 * d_norm_e1_x1) / (norm_e2 * norm_e1 * norm_e1)
          + f[3] * x[5]);

  g[2] = 2 * (f[1] * d_norm_e2_x2
          + f[2]*(d_e1_dot_e2_x2 * norm_e2 - e1_dot_e2 * d_norm_e2_x2) / (norm_e1 * norm_e2 * norm_e2)
          + f[4] * x[4]);

  g[3] = 2 * (f[1] * d_norm_e2_x3
          + f[2]*(d_e1_dot_e2_x3 * norm_e2 - e1_dot_e2 * d_norm_e2_x3) / (norm_e1 * norm_e2 * norm_e2)
          + f[4] * x[5]);

  g[4] = 2 * (f[3]*(x[0] - 1) + f[4] * x[2] + f[5] * d_norm_r_x4);

  g[5] = 2 * (f[3] * x[1] + f[4]*(x[3] - 1) + f[5] * d_norm_r_x5);

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

#define _N 6

float * solver2D(float * _pre, float * _new, float _norm_f0, float * init) {
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
  lbfgsfloatval_t * pos_new = lbfgs_malloc(2);
  lbfgsfloatval_t * pos_pre = lbfgs_malloc(2);
  for (int i = 0; i < 2; i++) {
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
