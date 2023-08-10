// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#ifndef CEED_MAGMA_GEMM_NONTENSOR_H
#define CEED_MAGMA_GEMM_NONTENSOR_H

#include "ceed-magma.h"

////////////////////////////////////////////////////////////////////////////////
CEED_INTERN int magma_gemm_nontensor(magma_trans_t transA, magma_trans_t transB, magma_int_t m, magma_int_t n, magma_int_t k, CeedScalar alpha,
                                     const CeedScalar *dA, magma_int_t ldda, const CeedScalar *dB, magma_int_t lddb, CeedScalar beta, CeedScalar *dC,
                                     magma_int_t lddc, magma_queue_t queue);

#endif  // CEED_MAGMA_GEMM_NONTENSOR_H
