// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Structs and helper functions to evaluate data-driven subgrid-stress modeling
/// See 'Invariant data-driven subgrid stress modeling in the strain-rate eigenframe for large eddy simulation' 2022 and 'S-frame discrepancy
/// correction models for data-informed Reynolds stress closure' 2022

#ifndef sgs_dd_model_h
#define sgs_dd_model_h

#include <ceed.h>

#include "newtonian_state.h"
#include "newtonian_types.h"
#include "sgs_dd_utils.h"
#include "utils.h"
#include "utils_eigensolver_jacobi.h"

typedef struct SGS_DD_ModelContext_ *SGS_DDModelContext;
struct SGS_DD_ModelContext_ {
  CeedInt    num_inputs, num_outputs;
  CeedInt    num_layers;
  CeedInt    num_neurons;
  CeedScalar alpha;

  struct NewtonianIdealGasContext_ gas;
  struct {
    size_t bias1, bias2;
    size_t weight1, weight2;
    size_t out_scaling;
  } offsets;
  size_t     total_bytes;
  CeedScalar data[1];
};

// @brief Denormalize outputs using min-max (de-)normalization
CEED_QFUNCTION_HELPER void DenormalizeDDOutputs(CeedScalar output[6], const CeedScalar new_bounds[6][2], const CeedScalar old_bounds[6][2]) {
  CeedScalar bounds_ratio;
  for (int i = 0; i < 6; i++) {
    bounds_ratio = (new_bounds[i][1] - new_bounds[i][0]) / (old_bounds[i][1] - old_bounds[i][0]);
    output[i]    = bounds_ratio * (output[i] - old_bounds[i][1]) + new_bounds[i][1];
  }
}

CEED_QFUNCTION_HELPER void LeakyReLU(CeedScalar *x, const CeedScalar alpha, const CeedInt N) {
  for (CeedInt i = 0; i < N; i++) x[i] *= (x[i] < 0 ? alpha : 1.);
}

CEED_QFUNCTION_HELPER void DataDrivenInference(const CeedScalar *inputs, CeedScalar *outputs, SGS_DDModelContext sgsdd_ctx) {
  const CeedInt     num_neurons = sgsdd_ctx->num_neurons;
  const CeedInt     num_inputs  = sgsdd_ctx->num_inputs;
  const CeedInt     num_outputs = sgsdd_ctx->num_outputs;
  const CeedScalar  alpha       = sgsdd_ctx->alpha;
  const CeedScalar *bias1       = &sgsdd_ctx->data[sgsdd_ctx->offsets.bias1];
  const CeedScalar *bias2       = &sgsdd_ctx->data[sgsdd_ctx->offsets.bias2];
  const CeedScalar *weight1     = &sgsdd_ctx->data[sgsdd_ctx->offsets.weight1];
  const CeedScalar *weight2     = &sgsdd_ctx->data[sgsdd_ctx->offsets.weight2];
  CeedScalar        V[20]       = {0.};

  CopyN(bias1, V, num_neurons);
  MatVecNM(weight1, inputs, num_neurons, num_inputs, CEED_NOTRANSPOSE, V);
  LeakyReLU(V, alpha, num_neurons);
  CopyN(bias2, outputs, num_outputs);
  MatVecNM(weight2, V, num_outputs, num_neurons, CEED_NOTRANSPOSE, outputs);
}

//CEED_QFUNCTION_HELPER void ComputeSGS_DDAnisotropic(const CeedScalar grad_velo_aniso[3][3], const CeedScalar km_A_ij[6], const CeedScalar delta,
//                                                    const CeedScalar viscosity, CeedScalar kmsgs_stress[6], SGS_DDModelContext sgsdd_ctx) {
//CEED_QFUNCTION_HELPER void ComputeSGS_DDAnisotropic(CeedScalar kmsgs_stress[6], SGS_DDModelContext sgsdd_ctx) {
CEED_QFUNCTION_HELPER void ComputeSGS_DDAnisotropic(CeedScalar qi[6], CeedScalar km_sgs[6], SGS_DDModelContext sgsdd_ctx) {
  //CeedScalar inputs[6], grad_velo_magnitude, eigenvectors[3][3], sgs_sframe_sym[6] = {0.};
  //CeedScalar inputs[6], sgs_sframe_sym[6] = {1.};
  //CeedScalar sgs_sframe_sym[6] = {0.};

  //ComputeSGS_DDAnisotropicInputs(grad_velo_aniso, km_A_ij, delta, viscosity, eigenvectors, inputs, &grad_velo_magnitude);

  //DataDrivenInference(inputs, sgs_sframe_sym, sgsdd_ctx);
  //DataDrivenInference(kmsgs_stress, kmsgs_stress, sgsdd_ctx);
  DataDrivenInference(qi, km_sgs, sgsdd_ctx);

  //CeedScalar old_bounds[6][2] = {{0}};
  //for (int j = 0; j < 6; j++) old_bounds[j][1] = 1;
  //const CeedScalar(*new_bounds)[2] = (const CeedScalar(*)[2]) & sgsdd_ctx->data[sgsdd_ctx->offsets.out_scaling];
  //DenormalizeDDOutputs(sgs_sframe_sym, new_bounds, old_bounds);

  // Re-dimensionalize sgs_stress
  //ScaleN(sgs_sframe_sym, Square(delta) * Square(grad_velo_magnitude), 6);

  //CeedScalar sgs_stress[3][3] = {{0.}};
  /*{  // Rotate SGS Stress back to physical frame, SGS_physical = E^T SGS_sframe E
    CeedScalar       Evec_sgs[3][3]   = {{0.}};
    const CeedScalar sgs_sframe[3][3] = {
        {sgs_sframe_sym[0], sgs_sframe_sym[3], sgs_sframe_sym[4]},
        {sgs_sframe_sym[3], sgs_sframe_sym[1], sgs_sframe_sym[5]},
        {sgs_sframe_sym[4], sgs_sframe_sym[5], sgs_sframe_sym[2]},
    };
    MatMat3(eigenvectors, sgs_sframe, CEED_TRANSPOSE, CEED_NOTRANSPOSE, Evec_sgs);
    MatMat3(Evec_sgs, eigenvectors, CEED_NOTRANSPOSE, CEED_NOTRANSPOSE, sgs_stress);
  }*/

  //KMPack(sgs_stress, kmsgs_stress);
}

// @brief Calculate subgrid stress at nodes using anisotropic data-driven model
CEED_QFUNCTION_HELPER int ComputeSGS_DDAnisotropicNodal(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out,
                                                        StateFromQi_t StateFromQi) {
  const CeedScalar(*q)[CEED_Q_VLA]            = (const CeedScalar(*)[CEED_Q_VLA])in[0];
  //const CeedScalar(*x)[CEED_Q_VLA]            = (const CeedScalar(*)[CEED_Q_VLA])in[1];
  //const CeedScalar(*grad_velo)[3][CEED_Q_VLA] = (const CeedScalar(*)[3][CEED_Q_VLA])in[2];
  //const CeedScalar(*A_ij_delta)[CEED_Q_VLA]   = (const CeedScalar(*)[CEED_Q_VLA])in[3];
  //const CeedScalar(*inv_multiplicity)         = (const CeedScalar(*))in[4];
  //CeedScalar(*v)[CEED_Q_VLA]                  = (CeedScalar(*)[CEED_Q_VLA])out[0];

  const SGS_DDModelContext       sgsdd_ctx = (SGS_DDModelContext)ctx;
  //const NewtonianIdealGasContext gas       = &sgsdd_ctx->gas;

  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    //const CeedScalar qi[5]                 = {q[0][i], q[1][i], q[2][i], q[3][i], q[4][i]};
    CeedScalar qi[6]                 = {q[0][i], q[1][i], q[2][i], q[3][i], q[4][i], q[4][i]};
  //  const CeedScalar x_i[3]                = {x[0][i], x[1][i], x[2][i]};
  //  const CeedScalar grad_velo_aniso[3][3] = {
  //      {grad_velo[0][0][i], grad_velo[0][1][i], grad_velo[0][2][i]},
  //      {grad_velo[1][0][i], grad_velo[1][1][i], grad_velo[1][2][i]},
  //      {grad_velo[2][0][i], grad_velo[2][1][i], grad_velo[2][2][i]}
  //  };
    //const CeedScalar km_A_ij[6] = {A_ij_delta[0][i], A_ij_delta[1][i], A_ij_delta[2][i], A_ij_delta[3][i], A_ij_delta[4][i], A_ij_delta[5][i]};
    //const CeedScalar delta      = A_ij_delta[6][i];
    //const State      s          = StateFromQi(gas, qi, x_i);
    CeedScalar       km_sgs[6];

    //ComputeSGS_DDAnisotropic(grad_velo_aniso, km_A_ij, delta, gas->mu / s.U.density, km_sgs, sgsdd_ctx);
    //ComputeSGS_DDAnisotropic(km_sgs, sgsdd_ctx);
    ComputeSGS_DDAnisotropic(qi, km_sgs, sgsdd_ctx);

    //for (int j = 0; j < 6; j++) v[j][i] = inv_multiplicity[i] * km_sgs[j];
  }
  return 0;
}

CEED_QFUNCTION(ComputeSGS_DDAnisotropicNodal_Prim)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return ComputeSGS_DDAnisotropicNodal(ctx, Q, in, out, StateFromY);
}

CEED_QFUNCTION(ComputeSGS_DDAnisotropicNodal_Conserv)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return ComputeSGS_DDAnisotropicNodal(ctx, Q, in, out, StateFromU);
}

// @brief Adds subgrid stress to residual (during IFunction evaluation)
CEED_QFUNCTION_HELPER int FluxSubgridStress(const StatePrimitive Y, const CeedScalar km_sgs[6], CeedScalar Flux[5][3]) {
  CeedScalar sgs[3][3];

  KMUnpack(km_sgs, sgs);
  for (CeedInt j = 0; j < 3; j++) {
    Flux[0][j] = 0.;
    for (CeedInt k = 0; k < 3; k++) Flux[k + 1][j] = sgs[k][j];
    Flux[4][j] = Y.velocity[0] * sgs[0][j] + Y.velocity[1] * sgs[1][j] + Y.velocity[2] * sgs[2][j];
  }
  return 0;
}

CEED_QFUNCTION_HELPER int IFunction_NodalSubgridStress(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out,
                                                       StateFromQi_t StateFromQi, StateFromQi_fwd_t StateFromQi_fwd) {
  const CeedScalar(*q)[CEED_Q_VLA]      = (const CeedScalar(*)[CEED_Q_VLA])in[0];
  const CeedScalar(*q_data)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[1];
  const CeedScalar(*x)[CEED_Q_VLA]      = (const CeedScalar(*)[CEED_Q_VLA])in[2];
  const CeedScalar(*km_sgs)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[3];
  CeedScalar(*Grad_v)[5][CEED_Q_VLA]    = (CeedScalar(*)[5][CEED_Q_VLA])out[0];

  SGS_DDModelContext       sgsdd_ctx = (SGS_DDModelContext)ctx;
  NewtonianIdealGasContext gas       = &sgsdd_ctx->gas;

  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    const CeedScalar qi[5]  = {q[0][i], q[1][i], q[2][i], q[3][i], q[4][i]};
    const CeedScalar x_i[3] = {x[0][i], x[1][i], x[2][i]};
    const State      s      = StateFromQi(gas, qi, x_i);

    const CeedScalar wdetJ      = q_data[0][i];
    const CeedScalar dXdx[3][3] = {
        {q_data[1][i], q_data[2][i], q_data[3][i]},
        {q_data[4][i], q_data[5][i], q_data[6][i]},
        {q_data[7][i], q_data[8][i], q_data[9][i]}
    };

    CeedScalar       Flux[5][3];
    const CeedScalar km_sgs_i[6] = {km_sgs[0][i], km_sgs[1][i], km_sgs[2][i], km_sgs[3][i], km_sgs[4][i], km_sgs[5][i]};
    FluxSubgridStress(s.Y, km_sgs_i, Flux);

    for (CeedInt j = 0; j < 3; j++) {
      for (CeedInt k = 0; k < 5; k++) {
        Grad_v[j][k][i] = -wdetJ * (dXdx[j][0] * Flux[k][0] + dXdx[j][1] * Flux[k][1] + dXdx[j][2] * Flux[k][2]);
      }
    }
  }
  return 0;
}

CEED_QFUNCTION(IFunction_NodalSubgridStress_Conserv)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return IFunction_NodalSubgridStress(ctx, Q, in, out, StateFromU, StateFromU_fwd);
}

CEED_QFUNCTION(IFunction_NodalSubgridStress_Prim)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return IFunction_NodalSubgridStress(ctx, Q, in, out, StateFromY, StateFromY_fwd);
}

#endif  // sgs_dd_model_h
