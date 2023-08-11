// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Internal header for MAGMA non-tensor basis interpolation
#ifndef CEED_MAGMA_BASIS_INTERP_DERIV_NONTENSOR_H
#define CEED_MAGMA_BASIS_INTERP_DERIV_NONTENSOR_H

#include "magma-common-nontensor.h"

//////////////////////////////////////////////////////////////////////////////////////////
template <typename T, int Q_COMP, int P, int Q, int NB>
static __device__ __inline__ void magma_basis_nontensor_device_n(const int n, CeedScalar const *dA, int ldda, CeedScalar const *dB, int lddb,
                                                                 CeedScalar *dC, int lddc, CeedScalar *shared_data) {
  const int tx      = threadIdx.x;
  const int ty      = threadIdx.y;
  const int id      = blockIdx.x * blockDim.y + ty;
  const int nblocks = MAGMA_CEILDIV(n, NB);
  const int myn     = min(NB, n - id * NB);

  dB += id * NB * lddb;
  dC += id * NB * lddc;

  // A is P x Q
  const int   slda = P;
  const int   sldb = P;
  CeedScalar *sA   = (CeedScalar *)shared_data;
  CeedScalar *sB   = sA + Q * P;
  sB += ty * sldb * NB;

  // read B once for all C's
  if (id < nblocks) {
    read_B_g2s_1D_nosync<CeedScalar, Q, P, NB>(tx, myn, dB, lddb, sB, sldb);
  }

  // init rA, rC
  CeedScalar rA[P], rC[NB];

  // unrolling this loop yields dramatic performance drop using hipcc, so let the compiler decide (no pragma unroll)
  for (int d = 0; d < Q_COMP; d++, dA += Q * P, dC += Q * n) {
    // read A (P x Q) using all threads
    read_A_trans_g2r_1D_nosync<CeedScalar, Q, P, NB>(tx, ty, dA, ldda, sA, slda, rA);

    __syncthreads();
    if (id < nblocks) {
      mul_rAsBrC_1D_nosync<CeedScalar, Q, P, NB>(tx, rA, sB, sldb, rC);

      // write C
      write_C_r2g_1D_nosync<CeedScalar, Q, P, NB>(tx, myn, rC, dC, lddc);
    }
    __syncthreads();
  }
}

//////////////////////////////////////////////////////////////////////////////////////////
template <typename T, int Q_COMP, int P, int Q, int NB>
static __device__ __inline__ void magma_basis_nontensor_device_t(const int n, CeedScalar const *dA, int ldda, CeedScalar const *dB, int lddb,
                                                                 CeedScalar *dC, int lddc, CeedScalar *shared_data) {
  const int tx      = threadIdx.x;
  const int ty      = threadIdx.y;
  const int id      = blockIdx.x * blockDim.y + ty;
  const int nblocks = MAGMA_CEILDIV(n, NB);
  const int myn     = min(NB, n - id * NB);

  // terminate threads with no work
  if (id >= nblocks) return;

  dB += id * NB * lddb;
  dC += id * NB * lddc;

  // A is P x Q
  const int   sldb = Q;
  CeedScalar *sB   = (CeedScalar *)shared_data;
  sB += ty * sldb * NB;

  // init rA, rC
  CeedScalar rA[Q], rC[NB];

  // read A (unroll first iteration)
  read_A_notrans_g2r_1D_nosync<CeedScalar, P, Q, NB>(tx, dA, ldda, NULL, 0, rA);

  // read B
  read_B_g2s_1D_nosync<CeedScalar, P, Q, NB>(tx, myn, dB, lddb, sB, sldb);

  __syncthreads();
  mul_rAsBrC_1D_nosync<CeedScalar, P, Q, NB>(tx, rA, sB, sldb, rC);
  __syncthreads();

  // unrolling this loop yields dramatic performance drop using hipcc, so let the compiler decide (no pragma unroll)
  for (int d = 1; d < Q_COMP; d++) {
    dA += P * Q;
    dB += Q * n;

    // read A
    read_A_notrans_g2r_1D_nosync<CeedScalar, P, Q, NB>(tx, dA, ldda, NULL, 0, rA);

    // read B
    read_B_g2s_1D_nosync<CeedScalar, P, Q, NB>(tx, myn, dB, lddb, sB, sldb);

    __syncthreads();
    addmul_rAsBrC_1D_nosync<CeedScalar, P, Q, NB>(tx, rA, sB, sldb, rC);
    __syncthreads();
  }

  // write C
  write_C_r2g_1D_nosync<CeedScalar, P, Q, NB>(tx, myn, rC, dC, lddc);
}

////////////////////////////////////////////////////////////////////////////////
extern "C" __launch_bounds__(MAGMA_BASIS_BOUNDS(BASIS_Q, MAGMA_MAXTHREADS_1D)) __global__
    void magma_interp_nontensor_n(const int n, CeedScalar const *dA, int ldda, CeedScalar const *dB, int lddb, CeedScalar *dC, int lddc) {
  MAGMA_DEVICE_SHARED(CeedScalar, shared_data);

  magma_basis_nontensor_device_n<CeedScalar, BASIS_Q_COMP_INTERP, BASIS_P, BASIS_Q, BASIS_NB_INTERP_N>(n, dA, ldda, dB, lddb, dC, lddc,
                                                                                                       (CeedScalar *)shared_data);
}

////////////////////////////////////////////////////////////////////////////////
extern "C" __launch_bounds__(MAGMA_BASIS_BOUNDS(BASIS_P, MAGMA_MAXTHREADS_1D)) __global__
    void magma_interp_nontensor_t(const int n, CeedScalar const *dA, int ldda, CeedScalar const *dB, int lddb, CeedScalar *dC, int lddc) {
  MAGMA_DEVICE_SHARED(CeedScalar, shared_data);

  magma_basis_nontensor_device_t<CeedScalar, BASIS_Q_COMP_INTERP, BASIS_P, BASIS_Q, BASIS_NB_INTERP_T>(n, dA, ldda, dB, lddb, dC, lddc,
                                                                                                       (CeedScalar *)shared_data);
}

////////////////////////////////////////////////////////////////////////////////
extern "C" __launch_bounds__(MAGMA_BASIS_BOUNDS(BASIS_Q, MAGMA_MAXTHREADS_1D)) __global__
    void magma_deriv_nontensor_n(const int n, CeedScalar const *dA, int ldda, CeedScalar const *dB, int lddb, CeedScalar *dC, int lddc) {
  MAGMA_DEVICE_SHARED(CeedScalar, shared_data);

  magma_basis_nontensor_device_n<CeedScalar, BASIS_Q_COMP_DERIV, BASIS_P, BASIS_Q, BASIS_NB_DERIV_N>(n, dA, ldda, dB, lddb, dC, lddc,
                                                                                                     (CeedScalar *)shared_data);
}

////////////////////////////////////////////////////////////////////////////////
extern "C" __launch_bounds__(MAGMA_BASIS_BOUNDS(BASIS_P, MAGMA_MAXTHREADS_1D)) __global__
    void magma_deriv_nontensor_t(const int n, CeedScalar const *dA, int ldda, CeedScalar const *dB, int lddb, CeedScalar *dC, int lddc) {
  MAGMA_DEVICE_SHARED(CeedScalar, shared_data);

  magma_basis_nontensor_device_t<CeedScalar, BASIS_Q_COMP_DERIV, BASIS_P, BASIS_Q, BASIS_NB_DERIV_T>(n, dA, ldda, dB, lddb, dC, lddc,
                                                                                                     (CeedScalar *)shared_data);
}

#endif  // CEED_MAGMA_BASIS_INTERP_DERIV_NONTENSOR_H
