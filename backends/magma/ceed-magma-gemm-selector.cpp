// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include "ceed-magma-gemm-selector.h"

#include <array>
#include <limits>
#include <vector>

#include "tuning/indices.h"
#ifdef CEED_MAGMA_USE_HIP
#include "tuning/mi100.h"
#include "tuning/mi250x.h"
#include "tuning/mi250x_grad_rtc.h"
#include "tuning/mi250x_interp_rtc.h"
#else
#include "tuning/a100.h"
#include "tuning/a100_grad_rtc.h"
#include "tuning/a100_interp_rtc.h"
#include "tuning/v100.h"
#endif

////////////////////////////////////////////////////////////////////////////////
#ifdef CEED_MAGMA_USE_HIP
static inline auto gemm_selector_get_data(int gpu_arch, char precision, char transA) -> decltype(dgemm_nn_mi250x) {
  if (gpu_arch >= 910) {
    // gfx90a or newer
    return (precision == 's') ? ((transA == 'n') ? sgemm_nn_mi250x : sgemm_tn_mi250x) : ((transA == 'n') ? dgemm_nn_mi250x : dgemm_tn_mi250x);
  } else {
    // gfx908 or older
    return (precision == 's') ? ((transA == 'n') ? sgemm_nn_mi100 : sgemm_tn_mi100) : ((transA == 'n') ? dgemm_nn_mi100 : dgemm_tn_mi100);
  }
}
#else
static inline auto gemm_selector_get_data(int gpu_arch, char precision, char transA) -> decltype(dgemm_nn_a100) {
  if (gpu_arch >= 800) {
    // sm80 or newer
    return (precision == 's') ? ((transA == 'n') ? sgemm_nn_a100 : sgemm_tn_a100) : ((transA == 'n') ? dgemm_nn_a100 : dgemm_tn_a100);
  } else {
    // sm70 or older
    return (precision == 's') ? ((transA == 'n') ? sgemm_nn_v100 : sgemm_tn_v100) : ((transA == 'n') ? dgemm_nn_v100 : dgemm_tn_v100);
  }
}
#endif

////////////////////////////////////////////////////////////////////////////////
void gemm_selector(int gpu_arch, char precision, char transA, int m, int n, int k, int *nbatch, int *use_magma) {
  *nbatch    = n;
  *use_magma = 0;

  const auto &data = gemm_selector_get_data(gpu_arch, precision, transA);
  int         ir   = -1;
  double      norm = std::numeric_limits<double>::max();

  for (size_t i = 0; i < data.size(); i++) {
    int im = data[i][M_INDEX];
    int in = data[i][N_INDEX];
    int ik = data[i][K_INDEX];

    double mdiff = (double)(im - m);
    double ndiff = (double)(in - n);
    double kdiff = (double)(ik - k);

    double nrm = sqrt(mdiff * mdiff + ndiff * ndiff + kdiff * kdiff);

    if (nrm < norm) {
      norm = nrm;
      ir   = i;
    }

    if (nrm == 0) {
      // the input (m, n, k) exactly matches a record in `data`
      // no need to search further
      break;
    }
  }

  if (ir >= 0) {
    // if the closest match indicates that n = nbatch,
    // that means calling the regular non-batch gemm.
    // So nbatch is set to n instead of the 'nbatch'
    // entry of the matching record
    int n_      = data[ir][N_INDEX];
    int nbatch_ = data[ir][N_BATCH_INDEX];
    *nbatch     = (n_ == nbatch_) ? n : nbatch_;
    *use_magma  = data[ir][USE_MAGMA_INDEX];
  }
}

////////////////////////////////////////////////////////////////////////////////
#ifdef CEED_MAGMA_USE_HIP
static inline auto nontensor_rtc_get_data(int gpu_arch, char transA, int qcomp) -> decltype(dinterp_n_mi250x) {
  if (qcomp == 1) {
    return (transA == 'n') ? dinterp_n_mi250x : dinterp_t_mi250x;
  } else {
    return (transA == 'n') ? dgrad_n_mi250x : dgrad_t_mi250x;
  }
}
#else
static inline auto nontensor_rtc_get_data(int gpu_arch, char transA, int qcomp) -> decltype(dinterp_n_a100) {
  if (qcomp == 1) {
    return (transA == 'n') ? dinterp_n_a100 : dinterp_t_a100;
  } else {
    return (transA == 'n') ? dgrad_n_a100 : dgrad_t_a100;
  }
}
#endif

////////////////////////////////////////////////////////////////////////////////
CeedInt nontensor_rtc_get_nb(int gpu_arch, char transA, int qcomp, int P_, int N, int Q_) {
  CeedInt P  = (transA == 'n') ? P_ : Q_;
  CeedInt Q  = (transA == 'n') ? Q_ : P_;
  CeedInt NB = 1;

  const auto &data = nontensor_rtc_get_data(gpu_arch, transA, qcomp);
  int         ir   = -1;
  double      norm = std::numeric_limits<double>::max();

  for (size_t i = 0; i < data.size(); i++) {
    int ip = data[i][M_INDEX_RTC];
    int in = data[i][N_INDEX_RTC];
    int iq = data[i][K_INDEX_RTC];

    double pdiff = (double)(ip - P);
    double ndiff = (double)(in - N);
    double qdiff = (double)(iq - Q);
    double nrm   = sqrt(pdiff * pdiff + ndiff * ndiff + qdiff * qdiff);

    if (nrm < norm) {
      norm = nrm;
      ir   = i;
    }

    if (nrm == 0) {
      // the input (m, n, k) exactly matches a record in `data`
      // no need to search further
      break;
    }
  }

  if (ir >= 0) {
    NB = data[ir][NB_INDEX_RTC];
  }

  return NB;
}
