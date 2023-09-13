// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

// magma functions specific to ceed
#ifndef CEED_MAGMA_H
#define CEED_MAGMA_H

#include <ceed.h>
#include <ceed/backend.h>
#include <magma_v2.h>

#define MAGMA_MAXTHREADS_1D 128
#define MAGMA_MAXTHREADS_2D 128
#define MAGMA_MAXTHREADS_3D 64
#define MAGMA_NONTENSOR_MAXTHREADS (128)

// Define macro for determining number of threads in y-direction
// for basis kernels
#define MAGMA_BASIS_NTCOL(x, maxt) (((maxt) < (x)) ? 1 : ((maxt) / (x)))
#define MAGMA_NONTENSOR_BASIS_NTCOL(N) (CeedIntMax(1, (MAGMA_NONTENSOR_MAXTHREADS / (N))))
#define MAGMA_CEILDIV(A, B) (((A) + (B)-1) / (B))

#define MAGMA_NONTENSOR_CUSTOM_KERNEL_MAX_P (40)
#define MAGMA_NONTENSOR_CUSTOM_KERNEL_MAX_Q (40)

// Define macro for computing the total threads in a block
// for use with __launch_bounds__()
#define MAGMA_BASIS_BOUNDS(x, maxt) (x * MAGMA_BASIS_NTCOL(x, maxt))

// Define macro for non-tensor kernel instances
#define MAGMA_NONTENSOR_KERNEL_INSTANCES (5)
#define MAGMA_NONTENSOR_N_VALUES 10240, 51200, 102400, 512000, 1024000

#ifdef CEED_MAGMA_USE_HIP
typedef hipModule_t   CeedMagmaModule;
typedef hipFunction_t CeedMagmaFunction;
#define CeedCompileMagma CeedCompile_Hip
#define CeedGetKernelMagma CeedGetKernel_Hip
#define CeedRunKernelMagma CeedRunKernel_Hip
#define CeedRunKernelDimMagma CeedRunKernelDim_Hip
#define CeedRunKernelDimSharedMagma CeedRunKernelDimShared_Hip
#else
typedef CUmodule   CeedMagmaModule;
typedef CUfunction CeedMagmaFunction;
#define CeedCompileMagma CeedCompile_Cuda
#define CeedGetKernelMagma CeedGetKernel_Cuda
#define CeedRunKernelMagma CeedRunKernel_Cuda
#define CeedRunKernelDimMagma CeedRunKernelDim_Cuda
#define CeedRunKernelDimSharedMagma CeedRunKernelDimShared_Cuda
#endif

typedef struct {
  CeedMagmaModule   module;
  CeedMagmaFunction magma_interp;
  CeedMagmaFunction magma_interp_tr;
  CeedMagmaFunction magma_grad;
  CeedMagmaFunction magma_grad_tr;
  CeedMagmaFunction magma_weight;
  CeedScalar       *d_q_ref_1d;
  CeedScalar       *d_interp_1d;
  CeedScalar       *d_grad_1d;
  CeedScalar       *d_q_weight_1d;
} CeedBasis_Magma;

typedef struct {
  CeedMagmaModule   module[MAGMA_NONTENSOR_KERNEL_INSTANCES];
  CeedMagmaFunction magma_interp_nontensor[MAGMA_NONTENSOR_KERNEL_INSTANCES];
  CeedMagmaFunction magma_interp_tr_nontensor[MAGMA_NONTENSOR_KERNEL_INSTANCES];
  CeedMagmaFunction magma_grad_nontensor[MAGMA_NONTENSOR_KERNEL_INSTANCES];
  CeedMagmaFunction magma_grad_tr_nontensor[MAGMA_NONTENSOR_KERNEL_INSTANCES];
  CeedScalar       *d_q_ref;
  CeedScalar       *d_interp;
  CeedScalar       *d_grad;
  CeedScalar       *d_q_weight;
} CeedBasisNonTensor_Magma;

CEED_INTERN void magma_weight_nontensor(magma_int_t grid, magma_int_t threads, magma_int_t num_elem, magma_int_t Q, CeedScalar *d_q_weight,
                                        CeedScalar *d_v, magma_queue_t queue);

CEED_INTERN int magma_gemm_nontensor(magma_trans_t trans_A, magma_trans_t trans_B, magma_int_t m, magma_int_t n, magma_int_t k, CeedScalar alpha,
                                     const CeedScalar *d_A, magma_int_t ldda, const CeedScalar *d_B, magma_int_t lddb, CeedScalar beta,
                                     CeedScalar *d_C, magma_int_t lddc, magma_queue_t queue);

CEED_INTERN void gemm_selector(int gpu_arch, char precision, char trans_A, int m, int n, int k, int *n_batch, int *use_magma);

CEED_INTERN CeedInt nontensor_rtc_get_nb(int gpu_arch, char precision, CeedEvalMode e_mode, CeedTransposeMode t_mode, int P_, int N, int Q_);

CEED_INTERN magma_int_t magma_isdevptr(const void *A);

CEED_INTERN int CeedBasisCreateTensorH1_Magma(CeedInt dim, CeedInt P_1d, CeedInt Q_1d, const CeedScalar *interp_1d, const CeedScalar *grad_1d,
                                              const CeedScalar *q_ref_1d, const CeedScalar *q_weight_1d, CeedBasis basis);

CEED_INTERN int CeedBasisCreateH1_Magma(CeedElemTopology topo, CeedInt dim, CeedInt num_dof, CeedInt num_qpts, const CeedScalar *interp,
                                        const CeedScalar *grad, const CeedScalar *q_ref, const CeedScalar *q_weight, CeedBasis basis);

// Comment the line below to use the default magma_is_devptr function
#define magma_is_devptr magma_isdevptr

// If magma and cuda/ref are using the null stream, then ceed_magma_queue_sync should do nothing
#define ceed_magma_queue_sync(...)

#endif  // CEED_MAGMA_H
