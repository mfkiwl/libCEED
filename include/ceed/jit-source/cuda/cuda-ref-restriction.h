// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Internal header for CUDA element restriction kernels
#ifndef CEED_CUDA_REF_RESTRICTION_H
#define CEED_CUDA_REF_RESTRICTION_H

#include <ceed.h>

//------------------------------------------------------------------------------
// L-vector -> E-vector, strided
//------------------------------------------------------------------------------
extern "C" __global__ void StridedNoTranspose(const CeedInt num_elem, const CeedScalar *__restrict__ u, CeedScalar *__restrict__ v) {
  for (CeedInt node = blockIdx.x * blockDim.x + threadIdx.x; node < num_elem * RSTR_ELEM_SIZE; node += blockDim.x * gridDim.x) {
    const CeedInt loc_node = node % RSTR_ELEM_SIZE;
    const CeedInt elem     = node / RSTR_ELEM_SIZE;

    for (CeedInt comp = 0; comp < RSTR_NUM_COMP; comp++) {
      v[loc_node + comp * RSTR_ELEM_SIZE * RSTR_NUM_ELEM + elem * RSTR_ELEM_SIZE] =
          u[loc_node * RSTR_STRIDE_NODES + comp * RSTR_STRIDE_COMP + elem * RSTR_STRIDE_ELEM];
    }
  }
}

//------------------------------------------------------------------------------
// L-vector -> E-vector, standard (with offsets)
//------------------------------------------------------------------------------
extern "C" __global__ void OffsetNoTranspose(const CeedInt num_elem, const CeedInt *__restrict__ indices, const CeedScalar *__restrict__ u,
                                             CeedScalar *__restrict__ v) {
  for (CeedInt node = blockIdx.x * blockDim.x + threadIdx.x; node < num_elem * RSTR_ELEM_SIZE; node += blockDim.x * gridDim.x) {
    const CeedInt ind      = indices[node];
    const CeedInt loc_node = node % RSTR_ELEM_SIZE;
    const CeedInt elem     = node / RSTR_ELEM_SIZE;

    for (CeedInt comp = 0; comp < RSTR_NUM_COMP; comp++) {
      v[loc_node + comp * RSTR_ELEM_SIZE * RSTR_NUM_ELEM + elem * RSTR_ELEM_SIZE] = u[ind + comp * RSTR_COMP_STRIDE];
    }
  }
}

//------------------------------------------------------------------------------
// L-vector -> E-vector, oriented
//------------------------------------------------------------------------------
extern "C" __global__ void OrientedNoTranspose(const CeedInt num_elem, const CeedInt *__restrict__ indices, const bool *__restrict__ orients,
                                               const CeedScalar *__restrict__ u, CeedScalar *__restrict__ v) {
  for (CeedInt node = blockIdx.x * blockDim.x + threadIdx.x; node < num_elem * RSTR_ELEM_SIZE; node += blockDim.x * gridDim.x) {
    const CeedInt ind      = indices[node];
    const bool    orient   = orients[node];
    const CeedInt loc_node = node % RSTR_ELEM_SIZE;
    const CeedInt elem     = node / RSTR_ELEM_SIZE;

    for (CeedInt comp = 0; comp < RSTR_NUM_COMP; comp++) {
      v[loc_node + comp * RSTR_ELEM_SIZE * RSTR_NUM_ELEM + elem * RSTR_ELEM_SIZE] = u[ind + comp * RSTR_COMP_STRIDE] * (orient ? -1.0 : 1.0);
    }
  }
}

//------------------------------------------------------------------------------
// L-vector -> E-vector, curl-oriented
//------------------------------------------------------------------------------
extern "C" __global__ void CurlOrientedNoTranspose(const CeedInt num_elem, const CeedInt *__restrict__ indices,
                                                   const CeedInt8 *__restrict__ curl_orients, const CeedScalar *__restrict__ u,
                                                   CeedScalar *__restrict__ v) {
  for (CeedInt node = blockIdx.x * blockDim.x + threadIdx.x; node < num_elem * RSTR_ELEM_SIZE; node += blockDim.x * gridDim.x) {
    const CeedInt  loc_node       = node % RSTR_ELEM_SIZE;
    const CeedInt  elem           = node / RSTR_ELEM_SIZE;
    const CeedInt  ind_dl         = loc_node > 0 ? indices[node - 1] : 0;
    const CeedInt  ind_d          = indices[node];
    const CeedInt  ind_du         = loc_node < (RSTR_ELEM_SIZE - 1) ? indices[node + 1] : 0;
    const CeedInt8 curl_orient_dl = curl_orients[3 * node + 0];
    const CeedInt8 curl_orient_d  = curl_orients[3 * node + 1];
    const CeedInt8 curl_orient_du = curl_orients[3 * node + 2];

    for (CeedInt comp = 0; comp < RSTR_NUM_COMP; comp++) {
      CeedScalar value = 0.0;
      value += loc_node > 0 ? u[ind_dl + comp * RSTR_COMP_STRIDE] * curl_orient_dl : 0.0;
      value += u[ind_d + comp * RSTR_COMP_STRIDE] * curl_orient_d;
      value += loc_node < (RSTR_ELEM_SIZE - 1) ? u[ind_du + comp * RSTR_COMP_STRIDE] * curl_orient_du : 0.0;
      v[loc_node + comp * RSTR_ELEM_SIZE * RSTR_NUM_ELEM + elem * RSTR_ELEM_SIZE] = value;
    }
  }
}

//------------------------------------------------------------------------------
// L-vector -> E-vector, unsigned curl-oriented
//------------------------------------------------------------------------------
extern "C" __global__ void CurlOrientedUnsignedNoTranspose(const CeedInt num_elem, const CeedInt *__restrict__ indices,
                                                           const CeedInt8 *__restrict__ curl_orients, const CeedScalar *__restrict__ u,
                                                           CeedScalar *__restrict__ v) {
  for (CeedInt node = blockIdx.x * blockDim.x + threadIdx.x; node < num_elem * RSTR_ELEM_SIZE; node += blockDim.x * gridDim.x) {
    const CeedInt  loc_node       = node % RSTR_ELEM_SIZE;
    const CeedInt  elem           = node / RSTR_ELEM_SIZE;
    const CeedInt  ind_dl         = loc_node > 0 ? indices[node - 1] : 0;
    const CeedInt  ind_d          = indices[node];
    const CeedInt  ind_du         = loc_node < (RSTR_ELEM_SIZE - 1) ? indices[node + 1] : 0;
    const CeedInt8 curl_orient_dl = abs(curl_orients[3 * node + 0]);
    const CeedInt8 curl_orient_d  = abs(curl_orients[3 * node + 1]);
    const CeedInt8 curl_orient_du = abs(curl_orients[3 * node + 2]);

    for (CeedInt comp = 0; comp < RSTR_NUM_COMP; comp++) {
      CeedScalar value = 0.0;
      value += loc_node > 0 ? u[ind_dl + comp * RSTR_COMP_STRIDE] * curl_orient_dl : 0.0;
      value += u[ind_d + comp * RSTR_COMP_STRIDE] * curl_orient_d;
      value += loc_node < (RSTR_ELEM_SIZE - 1) ? u[ind_du + comp * RSTR_COMP_STRIDE] * curl_orient_du : 0.0;
      v[loc_node + comp * RSTR_ELEM_SIZE * RSTR_NUM_ELEM + elem * RSTR_ELEM_SIZE] = value;
    }
  }
}

//------------------------------------------------------------------------------
// E-vector -> L-vector, strided
//------------------------------------------------------------------------------
extern "C" __global__ void StridedTranspose(const CeedInt num_elem, const CeedScalar *__restrict__ u, CeedScalar *__restrict__ v) {
  for (CeedInt node = blockIdx.x * blockDim.x + threadIdx.x; node < num_elem * RSTR_ELEM_SIZE; node += blockDim.x * gridDim.x) {
    const CeedInt loc_node = node % RSTR_ELEM_SIZE;
    const CeedInt elem     = node / RSTR_ELEM_SIZE;

    for (CeedInt comp = 0; comp < RSTR_NUM_COMP; comp++) {
      v[loc_node * RSTR_STRIDE_NODES + comp * RSTR_STRIDE_COMP + elem * RSTR_STRIDE_ELEM] +=
          u[loc_node + comp * RSTR_ELEM_SIZE * RSTR_NUM_ELEM + elem * RSTR_ELEM_SIZE];
    }
  }
}

//------------------------------------------------------------------------------
// E-vector -> L-vector, standard (with offsets)
//------------------------------------------------------------------------------
extern "C" __global__ void OffsetTranspose(const CeedInt num_elem, const CeedInt *__restrict__ indices, const CeedScalar *__restrict__ u,
                                           CeedScalar *__restrict__ v) {
  for (CeedInt node = blockIdx.x * blockDim.x + threadIdx.x; node < num_elem * RSTR_ELEM_SIZE; node += blockDim.x * gridDim.x) {
    const CeedInt ind      = indices[node];
    const CeedInt loc_node = node % RSTR_ELEM_SIZE;
    const CeedInt elem     = node / RSTR_ELEM_SIZE;

    for (CeedInt comp = 0; comp < RSTR_NUM_COMP; comp++) {
      atomicAdd(v + ind + comp * RSTR_COMP_STRIDE, u[loc_node + comp * RSTR_ELEM_SIZE * RSTR_NUM_ELEM + elem * RSTR_ELEM_SIZE]);
    }
  }
}

extern "C" __global__ void OffsetTransposeDet(const CeedInt *__restrict__ l_vec_indices, const CeedInt *__restrict__ t_indices,
                                              const CeedInt *__restrict__ t_offsets, const CeedScalar *__restrict__ u, CeedScalar *__restrict__ v) {
  CeedScalar value[RSTR_NUM_COMP];

  for (CeedInt i = blockIdx.x * blockDim.x + threadIdx.x; i < RSTR_NUM_NODES; i += blockDim.x * gridDim.x) {
    const CeedInt ind     = l_vec_indices[i];
    const CeedInt range_1 = t_offsets[i];
    const CeedInt range_N = t_offsets[i + 1];

    for (CeedInt comp = 0; comp < RSTR_NUM_COMP; comp++) value[comp] = 0.0;

    for (CeedInt j = range_1; j < range_N; j++) {
      const CeedInt t_ind    = t_indices[j];
      const CeedInt loc_node = t_ind % RSTR_ELEM_SIZE;
      const CeedInt elem     = t_ind / RSTR_ELEM_SIZE;

      for (CeedInt comp = 0; comp < RSTR_NUM_COMP; comp++) {
        value[comp] += u[loc_node + comp * RSTR_ELEM_SIZE * RSTR_NUM_ELEM + elem * RSTR_ELEM_SIZE];
      }
    }

    for (CeedInt comp = 0; comp < RSTR_NUM_COMP; comp++) v[ind + comp * RSTR_COMP_STRIDE] += value[comp];
  }
}

//------------------------------------------------------------------------------
// E-vector -> L-vector, oriented
//------------------------------------------------------------------------------
extern "C" __global__ void OrientedTranspose(const CeedInt num_elem, const CeedInt *__restrict__ indices, const bool *__restrict__ orients,
                                             const CeedScalar *__restrict__ u, CeedScalar *__restrict__ v) {
  for (CeedInt node = blockIdx.x * blockDim.x + threadIdx.x; node < num_elem * RSTR_ELEM_SIZE; node += blockDim.x * gridDim.x) {
    const CeedInt ind      = indices[node];
    const bool    orient   = orients[node];
    const CeedInt loc_node = node % RSTR_ELEM_SIZE;
    const CeedInt elem     = node / RSTR_ELEM_SIZE;

    for (CeedInt comp = 0; comp < RSTR_NUM_COMP; comp++) {
      atomicAdd(v + ind + comp * RSTR_COMP_STRIDE,
                u[loc_node + comp * RSTR_ELEM_SIZE * RSTR_NUM_ELEM + elem * RSTR_ELEM_SIZE] * (orient ? -1.0 : 1.0));
    }
  }
}

//------------------------------------------------------------------------------
// E-vector -> L-vector, curl-oriented
//------------------------------------------------------------------------------
extern "C" __global__ void CurlOrientedTranspose(const CeedInt num_elem, const CeedInt *__restrict__ indices,
                                                 const CeedInt8 *__restrict__ curl_orients, const CeedScalar *__restrict__ u,
                                                 CeedScalar *__restrict__ v) {
  for (CeedInt node = blockIdx.x * blockDim.x + threadIdx.x; node < num_elem * RSTR_ELEM_SIZE; node += blockDim.x * gridDim.x) {
    const CeedInt  loc_node       = node % RSTR_ELEM_SIZE;
    const CeedInt  elem           = node / RSTR_ELEM_SIZE;
    const CeedInt  ind            = indices[node];
    const CeedInt8 curl_orient_du = loc_node > 0 ? curl_orients[3 * node - 1] : 0.0;
    const CeedInt8 curl_orient_d  = curl_orients[3 * node + 1];
    const CeedInt8 curl_orient_dl = loc_node < (RSTR_ELEM_SIZE - 1) ? curl_orients[3 * node + 3] : 0.0;

    for (CeedInt comp = 0; comp < RSTR_NUM_COMP; comp++) {
      CeedScalar value = 0.0;
      value += loc_node > 0 ? u[loc_node - 1 + comp * RSTR_ELEM_SIZE * RSTR_NUM_ELEM + elem * RSTR_ELEM_SIZE] * curl_orient_du : 0.0;
      value += u[loc_node + comp * RSTR_ELEM_SIZE * RSTR_NUM_ELEM + elem * RSTR_ELEM_SIZE] * curl_orient_d;
      value +=
          loc_node < (RSTR_ELEM_SIZE - 1) ? u[loc_node + 1 + comp * RSTR_ELEM_SIZE * RSTR_NUM_ELEM + elem * RSTR_ELEM_SIZE] * curl_orient_dl : 0.0;
      atomicAdd(v + ind + comp * RSTR_COMP_STRIDE, value);
    }
  }
}

//------------------------------------------------------------------------------
// E-vector -> L-vector, unsigned curl-oriented
//------------------------------------------------------------------------------
extern "C" __global__ void CurlOrientedUnsignedTranspose(const CeedInt num_elem, const CeedInt *__restrict__ indices,
                                                         const CeedInt8 *__restrict__ curl_orients, const CeedScalar *__restrict__ u,
                                                         CeedScalar *__restrict__ v) {
  for (CeedInt node = blockIdx.x * blockDim.x + threadIdx.x; node < num_elem * RSTR_ELEM_SIZE; node += blockDim.x * gridDim.x) {
    const CeedInt  loc_node       = node % RSTR_ELEM_SIZE;
    const CeedInt  elem           = node / RSTR_ELEM_SIZE;
    const CeedInt  ind            = indices[node];
    const CeedInt8 curl_orient_du = loc_node > 0 ? abs(curl_orients[3 * node - 1]) : 0.0;
    const CeedInt8 curl_orient_d  = abs(curl_orients[3 * node + 1]);
    const CeedInt8 curl_orient_dl = loc_node < (RSTR_ELEM_SIZE - 1) ? abs(curl_orients[3 * node + 3]) : 0.0;

    for (CeedInt comp = 0; comp < RSTR_NUM_COMP; comp++) {
      CeedScalar value = 0.0;
      value += loc_node > 0 ? u[loc_node - 1 + comp * RSTR_ELEM_SIZE * RSTR_NUM_ELEM + elem * RSTR_ELEM_SIZE] * curl_orient_du : 0.0;
      value += u[loc_node + comp * RSTR_ELEM_SIZE * RSTR_NUM_ELEM + elem * RSTR_ELEM_SIZE] * curl_orient_d;
      value +=
          loc_node < (RSTR_ELEM_SIZE - 1) ? u[loc_node + 1 + comp * RSTR_ELEM_SIZE * RSTR_NUM_ELEM + elem * RSTR_ELEM_SIZE] * curl_orient_dl : 0.0;
      atomicAdd(v + ind + comp * RSTR_COMP_STRIDE, value);
    }
  }
}

//------------------------------------------------------------------------------

#endif  // CEED_CUDA_REF_RESTRICTION_H
