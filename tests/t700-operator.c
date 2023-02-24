/// @file
/// Test assembly of mass matrix operator diagonal for H(div) space
/// \test Test assembly of mass matrix operator diagonal for H(div) space
#include "t700-operator.h"

#include <ceed.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "ceed/ceed.h"
#include "t330-basis.h"

int main(int argc, char **argv) {
  Ceed                ceed;
  CeedElemRestriction elem_restr_x, elem_restr_u;
  CeedBasis           basis_x, basis_u;
  CeedVector          X, A, U, V;
  CeedInt             P = 8, Q = 3, dim = 2, Px = 2;
  // works only for one element; TODO extend to multiple element
  CeedInt           nx = 1, ny = 1, num_elem = nx * ny, num_faces = (nx + 1) * ny + (ny + 1) * nx;
  CeedInt           num_dofs_u = num_faces * 2, num_dofs_x = (nx + 1) * (ny + 1), num_qpts = Q * Q;
  CeedInt           ind_x[num_elem * Px * Px], ind_u[num_elem * P];
  CeedScalar        x[dim * num_dofs_x], assembled_true[num_dofs_u];
  bool              orient_u[num_elem * P];
  CeedScalar        q_ref[dim * num_qpts], q_weights[num_qpts];
  CeedScalar        interp_u[dim * P * num_qpts], div[P * num_qpts];
  CeedQFunction     qf_mass;
  CeedOperator      op_mass;
  CeedScalar       *u;
  const CeedScalar *a, *v;

  CeedInit(argv[1], &ceed);

  // DoF Coordinates
  for (CeedInt i = 0; i < nx + 1; i++)
    for (CeedInt j = 0; j < ny + 1; j++) {
      x[i + j * (nx + 1) + 0 * num_dofs_x] = (CeedScalar)i / (nx);
      x[i + j * (nx + 1) + 1 * num_dofs_x] = (CeedScalar)j / (ny);
    }
  CeedVectorCreate(ceed, dim * num_dofs_x, &X);
  CeedVectorSetArray(X, CEED_MEM_HOST, CEED_USE_POINTER, x);

  // Qdata Vector

  // Element Setup
  for (CeedInt i = 0; i < num_elem; i++) {
    CeedInt col, row, offset;
    col    = i % nx;
    row    = i / nx;
    offset = col * (Px - 1) + row * (nx + 1) * (Px - 1);
    for (CeedInt j = 0; j < Px; j++)
      for (CeedInt k = 0; k < Px; k++) ind_x[Px * (Px * i + k) + j] = offset + k * (nx + 1) + j;
  }

  bool    orient_u_local[8] = {false, false, false, false, true, true, true, true};
  CeedInt ind_u_local[8]    = {0, 1, 6, 7, 2, 3, 4, 5};
  for (CeedInt j = 0; j < ny; j++) {
    for (CeedInt i = 0; i < nx; i++) {
      for (CeedInt k = 0; k < P; k++) {
        // CeedInt e = j*nx + i;
        ind_u[k]    = ind_u_local[k];
        orient_u[k] = orient_u_local[k];
      }
    }
  }

  // Restrictions
  CeedElemRestrictionCreate(ceed, num_elem, Px * Px, dim, num_dofs_x, dim * num_dofs_x, CEED_MEM_HOST, CEED_USE_POINTER, ind_x, &elem_restr_x);
  CeedElemRestrictionCreateOriented(ceed, num_elem, P, 1, 1, num_dofs_u, CEED_MEM_HOST, CEED_COPY_VALUES, ind_u, orient_u, &elem_restr_u);
  // Bases
  CeedBasisCreateTensorH1Lagrange(ceed, dim, dim, Px, Q, CEED_GAUSS, &basis_x);
  BuildHdivQuadrilateral(Q, q_ref, q_weights, interp_u, div, CEED_GAUSS);
  CeedBasisCreateHdiv(ceed, CEED_TOPOLOGY_QUAD, 1, P, num_qpts, interp_u, div, q_ref, q_weights, &basis_u);

  // QFunctions
  CeedQFunctionCreateInterior(ceed, 1, mass, mass_loc, &qf_mass);
  CeedQFunctionAddInput(qf_mass, "weight", 1, CEED_EVAL_WEIGHT);
  CeedQFunctionAddInput(qf_mass, "dx", dim * dim, CEED_EVAL_GRAD);
  CeedQFunctionAddInput(qf_mass, "u", dim, CEED_EVAL_INTERP);
  CeedQFunctionAddOutput(qf_mass, "v", dim, CEED_EVAL_INTERP);

  // Operators
  CeedOperatorCreate(ceed, qf_mass, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_mass);
  CeedOperatorSetField(op_mass, "weight", CEED_ELEMRESTRICTION_NONE, basis_x, CEED_VECTOR_NONE);
  CeedOperatorSetField(op_mass, "dx", elem_restr_x, basis_x, X);
  CeedOperatorSetField(op_mass, "u", elem_restr_u, basis_u, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_mass, "v", elem_restr_u, basis_u, CEED_VECTOR_ACTIVE);

  // Assemble diagonal
  CeedVectorCreate(ceed, num_dofs_u, &A);
  CeedOperatorLinearAssembleDiagonal(op_mass, A, CEED_REQUEST_IMMEDIATE);

  // Manually assemble diagonal
  CeedVectorCreate(ceed, num_dofs_u, &U);
  CeedVectorSetValue(U, 0.0);
  CeedVectorCreate(ceed, num_dofs_u, &V);
  for (int i = 0; i < num_dofs_u; i++) {
    // Set input
    CeedVectorGetArray(U, CEED_MEM_HOST, &u);
    u[i] = 1.0;
    if (i) u[i - 1] = 0.0;
    CeedVectorRestoreArray(U, &u);

    // Compute diag entry for DoF i
    CeedOperatorApply(op_mass, U, V, CEED_REQUEST_IMMEDIATE);

    // Retrieve entry
    CeedVectorGetArrayRead(V, CEED_MEM_HOST, &v);
    assembled_true[i] = v[i];
    CeedVectorRestoreArrayRead(V, &v);
  }

  // Check output
  CeedVectorGetArrayRead(A, CEED_MEM_HOST, &a);
  for (int i = 0; i < num_dofs_u; i++)
    if (fabs(a[i] - assembled_true[i]) > 100. * CEED_EPSILON)
      // LCOV_EXCL_START
      printf("[%" CeedInt_FMT "] Error in assembly: %f != %f\n", i, a[i], assembled_true[i]);
  // LCOV_EXCL_STOP
  CeedVectorRestoreArrayRead(A, &a);

  CeedElemRestrictionDestroy(&elem_restr_x);
  CeedElemRestrictionDestroy(&elem_restr_u);
  CeedBasisDestroy(&basis_x);
  CeedBasisDestroy(&basis_u);
  CeedVectorDestroy(&X);

  CeedDestroy(&ceed);
  return 0;
}
