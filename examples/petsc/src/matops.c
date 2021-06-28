#include "../include/matops.h"

#include "../include/petscutils.h"

// -----------------------------------------------------------------------------
// Setup apply operator context data
// -----------------------------------------------------------------------------
PetscErrorCode SetupApplyOperatorCtx(MPI_Comm comm, DM dm, Ceed ceed, CeedData ceed_data, Vec X_loc, OperatorApplyContext op_apply_ctx) {
  PetscFunctionBeginUser;

  op_apply_ctx->comm  = comm;
  op_apply_ctx->dm    = dm;
  op_apply_ctx->X_loc = X_loc;
  PetscCall(VecDuplicate(X_loc, &op_apply_ctx->Y_loc));
  op_apply_ctx->x_ceed = ceed_data->x_ceed;
  op_apply_ctx->y_ceed = ceed_data->y_ceed;
  op_apply_ctx->op     = ceed_data->op_apply;
  op_apply_ctx->ceed   = ceed;

  PetscFunctionReturn(PETSC_SUCCESS);
}

// -----------------------------------------------------------------------------
// Setup error operator context data
// -----------------------------------------------------------------------------
PetscErrorCode SetupErrorOperatorCtx(MPI_Comm comm, DM dm, Ceed ceed, CeedData ceed_data, Vec X_loc, CeedOperator op_error,
                                     OperatorApplyContext op_error_ctx) {
  PetscFunctionBeginUser;

  op_error_ctx->comm  = comm;
  op_error_ctx->dm    = dm;
  op_error_ctx->X_loc = X_loc;
  PetscCall(VecDuplicate(X_loc, &op_error_ctx->Y_loc));
  op_error_ctx->x_ceed = ceed_data->x_ceed;
  op_error_ctx->y_ceed = ceed_data->y_ceed;
  op_error_ctx->op     = op_error;
  op_error_ctx->ceed   = ceed;

  PetscFunctionReturn(PETSC_SUCCESS);
}

// -----------------------------------------------------------------------------
// This function returns the computed diagonal of the operator
// -----------------------------------------------------------------------------
PetscErrorCode MatGetDiag(Mat A, Vec D) {
  OperatorApplyContext op_apply_ctx;

  PetscFunctionBeginUser;

  PetscCall(MatShellGetContext(A, &op_apply_ctx));

  // Compute Diagonal via libCEED
  PetscScalar *y;
  PetscMemType mem_type;

  // -- Place PETSc vector in libCEED vector
  PetscCall(VecGetArrayAndMemType(op_apply_ctx->Y_loc, &y, &mem_type));
  CeedVectorSetArray(op_apply_ctx->y_ceed, MemTypeP2C(mem_type), CEED_USE_POINTER, y);

  // -- Compute Diagonal
  CeedOperatorLinearAssembleDiagonal(op_apply_ctx->op, op_apply_ctx->y_ceed, CEED_REQUEST_IMMEDIATE);

  // -- Local-to-Global
  CeedVectorTakeArray(op_apply_ctx->y_ceed, MemTypeP2C(mem_type), NULL);
  PetscCall(VecRestoreArrayAndMemType(op_apply_ctx->Y_loc, &y));
  PetscCall(VecZeroEntries(D));
  PetscCall(DMLocalToGlobal(op_apply_ctx->dm, op_apply_ctx->Y_loc, ADD_VALUES, D));

  PetscFunctionReturn(PETSC_SUCCESS);
}

// -----------------------------------------------------------------------------
// This function uses libCEED to compute the action of the Laplacian with Dirichlet boundary conditions
// -----------------------------------------------------------------------------
PetscErrorCode ApplyLocal_Ceed(Vec X, Vec Y, OperatorApplyContext op_apply_ctx) {
  PetscScalar *x, *y;
  PetscMemType x_mem_type, y_mem_type;

  PetscFunctionBeginUser;

  // Global-to-local
  PetscCall(DMGlobalToLocal(op_apply_ctx->dm, X, INSERT_VALUES, op_apply_ctx->X_loc));

  // Setup libCEED vectors
  PetscCall(VecGetArrayReadAndMemType(op_apply_ctx->X_loc, (const PetscScalar **)&x, &x_mem_type));
  PetscCall(VecGetArrayAndMemType(op_apply_ctx->Y_loc, &y, &y_mem_type));
  CeedVectorSetArray(op_apply_ctx->x_ceed, MemTypeP2C(x_mem_type), CEED_USE_POINTER, x);
  CeedVectorSetArray(op_apply_ctx->y_ceed, MemTypeP2C(y_mem_type), CEED_USE_POINTER, y);

  // Apply libCEED operator
  CeedOperatorApply(op_apply_ctx->op, op_apply_ctx->x_ceed, op_apply_ctx->y_ceed, CEED_REQUEST_IMMEDIATE);

  // Restore PETSc vectors
  CeedVectorTakeArray(op_apply_ctx->x_ceed, MemTypeP2C(x_mem_type), NULL);
  CeedVectorTakeArray(op_apply_ctx->y_ceed, MemTypeP2C(y_mem_type), NULL);
  PetscCall(VecRestoreArrayReadAndMemType(op_apply_ctx->X_loc, (const PetscScalar **)&x));
  PetscCall(VecRestoreArrayAndMemType(op_apply_ctx->Y_loc, &y));

  // Local-to-global
  PetscCall(VecZeroEntries(Y));
  PetscCall(DMLocalToGlobal(op_apply_ctx->dm, op_apply_ctx->Y_loc, ADD_VALUES, Y));

  PetscFunctionReturn(PETSC_SUCCESS);
}

// -----------------------------------------------------------------------------
// This function wraps the libCEED operator for a MatShell
// -----------------------------------------------------------------------------
PetscErrorCode MatMult_Ceed(Mat A, Vec X, Vec Y) {
  OperatorApplyContext op_apply_ctx;

  PetscFunctionBeginUser;

  PetscCall(MatShellGetContext(A, &op_apply_ctx));

  // libCEED for local action of residual evaluator
  PetscCall(ApplyLocal_Ceed(X, Y, op_apply_ctx));

  PetscFunctionReturn(PETSC_SUCCESS);
}

// -----------------------------------------------------------------------------
// This function uses libCEED to compute the action of the prolongation operator
// -----------------------------------------------------------------------------
PetscErrorCode MatMult_Prolong(Mat A, Vec X, Vec Y) {
  ProlongRestrContext pr_restr_ctx;
  PetscScalar        *c, *f;
  PetscMemType        c_mem_type, f_mem_type;

  PetscFunctionBeginUser;

  PetscCall(MatShellGetContext(A, &pr_restr_ctx));

  // Global-to-local
  PetscCall(VecZeroEntries(pr_restr_ctx->loc_vec_c));
  PetscCall(DMGlobalToLocal(pr_restr_ctx->dmc, X, INSERT_VALUES, pr_restr_ctx->loc_vec_c));

  // Setup libCEED vectors
  PetscCall(VecGetArrayReadAndMemType(pr_restr_ctx->loc_vec_c, (const PetscScalar **)&c, &c_mem_type));
  PetscCall(VecGetArrayAndMemType(pr_restr_ctx->loc_vec_f, &f, &f_mem_type));
  CeedVectorSetArray(pr_restr_ctx->ceed_vec_c, MemTypeP2C(c_mem_type), CEED_USE_POINTER, c);
  CeedVectorSetArray(pr_restr_ctx->ceed_vec_f, MemTypeP2C(f_mem_type), CEED_USE_POINTER, f);

  // Apply libCEED operator
  CeedOperatorApply(pr_restr_ctx->op_prolong, pr_restr_ctx->ceed_vec_c, pr_restr_ctx->ceed_vec_f, CEED_REQUEST_IMMEDIATE);

  // Restore PETSc vectors
  CeedVectorTakeArray(pr_restr_ctx->ceed_vec_c, MemTypeP2C(c_mem_type), NULL);
  CeedVectorTakeArray(pr_restr_ctx->ceed_vec_f, MemTypeP2C(f_mem_type), NULL);
  PetscCall(VecRestoreArrayReadAndMemType(pr_restr_ctx->loc_vec_c, (const PetscScalar **)&c));
  PetscCall(VecRestoreArrayAndMemType(pr_restr_ctx->loc_vec_f, &f));

  // Multiplicity
  PetscCall(VecPointwiseMult(pr_restr_ctx->loc_vec_f, pr_restr_ctx->loc_vec_f, pr_restr_ctx->mult_vec));

  // Local-to-global
  PetscCall(VecZeroEntries(Y));
  PetscCall(DMLocalToGlobal(pr_restr_ctx->dmf, pr_restr_ctx->loc_vec_f, ADD_VALUES, Y));

  PetscFunctionReturn(PETSC_SUCCESS);
}

// -----------------------------------------------------------------------------
// This function uses libCEED to compute the action of the restriction operator
// -----------------------------------------------------------------------------
PetscErrorCode MatMult_Restrict(Mat A, Vec X, Vec Y) {
  ProlongRestrContext pr_restr_ctx;
  PetscScalar        *c, *f;
  PetscMemType        c_mem_type, f_mem_type;

  PetscFunctionBeginUser;

  PetscCall(MatShellGetContext(A, &pr_restr_ctx));

  // Global-to-local
  PetscCall(VecZeroEntries(pr_restr_ctx->loc_vec_f));
  PetscCall(DMGlobalToLocal(pr_restr_ctx->dmf, X, INSERT_VALUES, pr_restr_ctx->loc_vec_f));

  // Multiplicity
  PetscCall(VecPointwiseMult(pr_restr_ctx->loc_vec_f, pr_restr_ctx->loc_vec_f, pr_restr_ctx->mult_vec));

  // Setup libCEED vectors
  PetscCall(VecGetArrayReadAndMemType(pr_restr_ctx->loc_vec_f, (const PetscScalar **)&f, &f_mem_type));
  PetscCall(VecGetArrayAndMemType(pr_restr_ctx->loc_vec_c, &c, &c_mem_type));
  CeedVectorSetArray(pr_restr_ctx->ceed_vec_f, MemTypeP2C(f_mem_type), CEED_USE_POINTER, f);
  CeedVectorSetArray(pr_restr_ctx->ceed_vec_c, MemTypeP2C(c_mem_type), CEED_USE_POINTER, c);

  // Apply CEED operator
  CeedOperatorApply(pr_restr_ctx->op_restrict, pr_restr_ctx->ceed_vec_f, pr_restr_ctx->ceed_vec_c, CEED_REQUEST_IMMEDIATE);

  // Restore PETSc vectors
  CeedVectorTakeArray(pr_restr_ctx->ceed_vec_c, MemTypeP2C(c_mem_type), NULL);
  CeedVectorTakeArray(pr_restr_ctx->ceed_vec_f, MemTypeP2C(f_mem_type), NULL);
  PetscCall(VecRestoreArrayReadAndMemType(pr_restr_ctx->loc_vec_f, (const PetscScalar **)&f));
  PetscCall(VecRestoreArrayAndMemType(pr_restr_ctx->loc_vec_c, &c));

  // Local-to-global
  PetscCall(VecZeroEntries(Y));
  PetscCall(DMLocalToGlobal(pr_restr_ctx->dmc, pr_restr_ctx->loc_vec_c, ADD_VALUES, Y));

  PetscFunctionReturn(PETSC_SUCCESS);
}

// -----------------------------------------------------------------------------
// This function sets up the BDDC preconditioner
// -----------------------------------------------------------------------------
PetscErrorCode PCShellSetup_BDDC(PC pc) {
  BDDCApplyContext bddc_ctx;

  PetscFunctionBeginUser;

  PetscCall(PCShellGetContext(pc, (void *)&bddc_ctx));

  // Assemble mat for element Schur AMG
  PetscCall(VecZeroEntries(bddc_ctx->X_Pi_r_loc));
  PetscCall(SNESComputeJacobianDefaultColor(bddc_ctx->snes_Pi_r, bddc_ctx->X_Pi_r_loc, bddc_ctx->mat_S_Pi_r, bddc_ctx->mat_S_Pi_r, NULL));
  PetscCall(MatAssemblyBegin(bddc_ctx->mat_S_Pi_r, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(bddc_ctx->mat_S_Pi_r, MAT_FINAL_ASSEMBLY));

  // Assemble mat for Schur AMG
  PetscCall(VecZeroEntries(bddc_ctx->X_Pi));
  PetscCall(SNESComputeJacobianDefaultColor(bddc_ctx->snes_Pi, bddc_ctx->X_Pi, bddc_ctx->mat_S_Pi, bddc_ctx->mat_S_Pi, NULL));
  PetscCall(MatAssemblyBegin(bddc_ctx->mat_S_Pi, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(bddc_ctx->mat_S_Pi, MAT_FINAL_ASSEMBLY));

  PetscFunctionReturn(PETSC_SUCCESS);
};

// -----------------------------------------------------------------------------
// This function provides the action of the element Schur compliment
// -----------------------------------------------------------------------------
PetscErrorCode MatMult_BDDCElementSchur(BDDCApplyContext bddc_ctx, Vec X_Pi_r_loc, Vec Y_Pi_r_loc) {
  CeedDataBDDC data = bddc_ctx->ceed_data_bddc;
  PetscScalar *x, *y;
  PetscMemType x_mem_type, y_mem_type;

  PetscFunctionBeginUser;

  // Set arrays in libCEED
  PetscCall(VecGetArrayReadAndMemType(X_Pi_r_loc, (const PetscScalar **)&x, &x_mem_type));
  PetscCall(VecGetArrayAndMemType(Y_Pi_r_loc, &y, &y_mem_type));
  CeedVectorSetArray(data->x_Pi_r_ceed, MemTypeP2C(x_mem_type), CEED_USE_POINTER, x);
  CeedVectorSetArray(data->y_Pi_r_ceed, MemTypeP2C(y_mem_type), CEED_USE_POINTER, y);

  // Apply action on Schur compliment
  // Y_Pi_r = -B A_r,r^-1 B^T X_Pi_r
  // -- X_r = B^T X_Pi_r
  CeedOperatorApply(data->op_inject_Pi_r, data->x_Pi_r_ceed, data->x_r_ceed, CEED_REQUEST_IMMEDIATE);
  // -- Y_r = A_r,r^-1 X_r
  CeedOperatorApply(data->op_r_r_inv, data->x_r_ceed, data->y_r_ceed, CEED_REQUEST_IMMEDIATE);
  // -- Y_Pi_r = -B Y_r
  CeedOperatorApply(data->op_restrict_Pi_r, data->y_r_ceed, data->y_Pi_r_ceed, CEED_REQUEST_IMMEDIATE);
  CeedVectorScale(data->y_Pi_r_ceed, -1.0);

  // Restore arrays
  CeedVectorTakeArray(data->x_Pi_r_ceed, MemTypeP2C(x_mem_type), &x);
  CeedVectorTakeArray(data->y_Pi_r_ceed, MemTypeP2C(y_mem_type), &y);
  PetscCall(VecRestoreArrayReadAndMemType(X_Pi_r_loc, (const PetscScalar **)&x));
  PetscCall(VecRestoreArrayAndMemType(Y_Pi_r_loc, &y));

  PetscFunctionReturn(PETSC_SUCCESS);
};

// -----------------------------------------------------------------------------
// This function assembles the element Schur compliment for the dummy SNES
// -----------------------------------------------------------------------------
PetscErrorCode FormResidual_BDDCElementSchur(SNES snes, Vec X_Pi_r_loc, Vec Y_Pi_r_loc, void *ctx) {
  BDDCApplyContext bddc_ctx = (BDDCApplyContext)ctx;

  PetscFunctionBeginUser;

  PetscCall(MatMult_BDDCElementSchur(bddc_ctx, X_Pi_r_loc, Y_Pi_r_loc));

  PetscFunctionReturn(PETSC_SUCCESS);
};

// -----------------------------------------------------------------------------
// This function provides the action of the element inverse
// -----------------------------------------------------------------------------
PetscErrorCode BDDCArrInv(BDDCApplyContext bddc_ctx, CeedVector x_r_ceed, CeedVector y_r_ceed) {
  CeedDataBDDC data = bddc_ctx->ceed_data_bddc;
  PetscScalar *x, *y;
  PetscMemType x_mem_type, y_mem_type;

  PetscFunctionBeginUser;

  // Y_r = A_r,r^-1 (I + B S^-1 B^T A_r,r^-1) X_r
  // -- X_r = (I + B S^-1 B^T A_r,r^-1) X_r
  // ---- Y_r = A_r,r^-1 X_r
  CeedVectorPointwiseMult(x_r_ceed, x_r_ceed, data->mask_r_ceed);
  CeedOperatorApply(data->op_r_r_inv, x_r_ceed, y_r_ceed, CEED_REQUEST_IMMEDIATE);
  // ---- Y_Pi_r = B^T Y_r
  PetscCall(VecGetArrayAndMemType(bddc_ctx->Y_Pi_r_loc, &y, &y_mem_type));
  CeedVectorSetArray(data->y_Pi_r_ceed, MemTypeP2C(y_mem_type), CEED_USE_POINTER, y);
  CeedOperatorApply(data->op_restrict_Pi_r, y_r_ceed, data->y_Pi_r_ceed, CEED_REQUEST_IMMEDIATE);
  CeedVectorTakeArray(data->y_Pi_r_ceed, MemTypeP2C(y_mem_type), &y);
  PetscCall(VecRestoreArrayAndMemType(bddc_ctx->Y_Pi_r_loc, &y));
  // ---- X_Pi_r = S^-1 Y_Pi_r
  PetscCall(KSPSolve(bddc_ctx->ksp_S_Pi_r, bddc_ctx->Y_Pi_r_loc, bddc_ctx->X_Pi_r_loc));
  // ---- X_r += B X_Pi_r
  PetscCall(VecGetArrayReadAndMemType(bddc_ctx->X_Pi_r_loc, (const PetscScalar **)&x, &x_mem_type));
  CeedVectorSetArray(data->x_Pi_r_ceed, MemTypeP2C(x_mem_type), CEED_USE_POINTER, x);
  CeedOperatorApplyAdd(data->op_inject_Pi_r, data->x_Pi_r_ceed, x_r_ceed, CEED_REQUEST_IMMEDIATE);
  CeedVectorTakeArray(data->x_Pi_r_ceed, MemTypeP2C(x_mem_type), &x);
  PetscCall(VecRestoreArrayReadAndMemType(bddc_ctx->X_Pi_r_loc, (const PetscScalar **)&x));
  // -- Y_r = A_r,r^-1 X_r
  CeedOperatorApply(data->op_r_r_inv, x_r_ceed, y_r_ceed, CEED_REQUEST_IMMEDIATE);

  PetscFunctionReturn(PETSC_SUCCESS);
};

// -----------------------------------------------------------------------------
// This function provides the action of the Schur compliment
// -----------------------------------------------------------------------------
PetscErrorCode MatMult_BDDCSchur(BDDCApplyContext bddc_ctx, Vec X_Pi, Vec Y_Pi) {
  CeedDataBDDC data = bddc_ctx->ceed_data_bddc;
  PetscScalar *x, *y;
  PetscMemType x_mem_type, y_mem_type;

  PetscFunctionBeginUser;

  // Global-to-Local
  PetscCall(VecZeroEntries(bddc_ctx->X_Pi_loc));
  PetscCall(DMGlobalToLocal(bddc_ctx->dm_Pi, X_Pi, INSERT_VALUES, bddc_ctx->X_Pi_loc));
  // Set arrays in libCEED
  PetscCall(VecGetArrayReadAndMemType(bddc_ctx->X_Pi_loc, (const PetscScalar **)&x, &x_mem_type));
  PetscCall(VecGetArrayAndMemType(bddc_ctx->Y_Pi_loc, &y, &y_mem_type));
  CeedVectorSetArray(data->x_Pi_ceed, MemTypeP2C(x_mem_type), CEED_USE_POINTER, x);
  CeedVectorSetArray(data->y_Pi_ceed, MemTypeP2C(y_mem_type), CEED_USE_POINTER, y);

  // Apply action on Schur compliment
  // Y_Pi  = (A_Pi,Pi - A_Pi,r A_r,r^-1 A_r,Pi) X_Pi
  // -- X_r = A_r,Pi X_Pi
  CeedOperatorApply(data->op_r_Pi, data->x_Pi_ceed, data->x_r_ceed, CEED_REQUEST_IMMEDIATE);
  // -- Y_r = A_r,r^-1 X_r
  PetscCall(BDDCArrInv(bddc_ctx, data->x_r_ceed, data->y_r_ceed));
  // -- Y_Pi = -A_Pi,r Y_r
  CeedVectorPointwiseMult(data->y_r_ceed, data->y_r_ceed, data->mask_r_ceed);
  CeedOperatorApply(data->op_Pi_r, data->y_r_ceed, data->y_Pi_ceed, CEED_REQUEST_IMMEDIATE);
  CeedVectorScale(data->y_Pi_ceed, -1.0);
  // -- Y_Pi += A_Pi,Pi X_Pi
  CeedOperatorApplyAdd(data->op_Pi_Pi, data->x_Pi_ceed, data->y_Pi_ceed, CEED_REQUEST_IMMEDIATE);

  // Restore arrays
  CeedVectorTakeArray(data->x_Pi_ceed, MemTypeP2C(x_mem_type), &x);
  CeedVectorTakeArray(data->y_Pi_ceed, MemTypeP2C(y_mem_type), &y);
  PetscCall(VecRestoreArrayReadAndMemType(bddc_ctx->X_Pi_loc, (const PetscScalar **)&x));
  PetscCall(VecRestoreArrayAndMemType(bddc_ctx->Y_Pi_loc, &y));
  // Local-to-Global
  PetscCall(VecZeroEntries(Y_Pi));
  PetscCall(DMLocalToGlobal(bddc_ctx->dm_Pi, bddc_ctx->Y_Pi_loc, ADD_VALUES, Y_Pi));

  PetscFunctionReturn(PETSC_SUCCESS);
};

// -----------------------------------------------------------------------------
// This function assembles the Schur compliment for the dummy SNES
// -----------------------------------------------------------------------------
PetscErrorCode FormResidual_BDDCSchur(SNES snes, Vec X_Pi, Vec Y_Pi, void *ctx) {
  BDDCApplyContext bddc_ctx = (BDDCApplyContext)ctx;

  PetscFunctionBeginUser;

  PetscCall(MatMult_BDDCSchur(bddc_ctx, X_Pi, Y_Pi));

  PetscFunctionReturn(PETSC_SUCCESS);
};

// -----------------------------------------------------------------------------
// This function uses libCEED to compute the action of the BDDC preconditioner
// -----------------------------------------------------------------------------
PetscErrorCode PCShellApply_BDDC(PC pc, Vec X, Vec Y) {
  BDDCApplyContext bddc_ctx;
  CeedDataBDDC     data;
  PetscScalar     *x, *y;
  PetscMemType     x_mem_type, y_mem_type;

  PetscFunctionBeginUser;

  PetscCall(PCShellGetContext(pc, (void *)&bddc_ctx));
  data = bddc_ctx->ceed_data_bddc;

  // Inject to broken space
  // -- Scaled injection, point multiply by 1/multiplicity
  // ---- Global-to-Local
  PetscCall(VecZeroEntries(bddc_ctx->X_loc));
  PetscCall(DMGlobalToLocal(bddc_ctx->dm, X, INSERT_VALUES, bddc_ctx->X_loc));
  // ---- Inject to Y_r
  PetscCall(VecGetArrayReadAndMemType(bddc_ctx->X_loc, (const PetscScalar **)&x, &x_mem_type));
  CeedVectorSetArray(data->x_ceed, MemTypeP2C(x_mem_type), CEED_USE_POINTER, x);
  CeedOperatorApply(data->op_inject_r, data->x_ceed, data->y_r_ceed, CEED_REQUEST_IMMEDIATE);
  CeedVectorTakeArray(data->x_ceed, MemTypeP2C(x_mem_type), &x);
  PetscCall(VecRestoreArrayReadAndMemType(bddc_ctx->X_loc, (const PetscScalar **)&x));
  // -- Harmonic injection, scaled with jump map
  if (bddc_ctx->is_harmonic) {
    CeedVectorPointwiseMult(data->x_r_ceed, data->y_r_ceed, data->mask_I_ceed);
    // ---- Z_r = A_I,I^-1 X_r
    PetscCall(BDDCArrInv(bddc_ctx, data->x_r_ceed, data->z_r_ceed));
    // ---- X_r = - A_Gamma,I Z_r
    CeedVectorPointwiseMult(data->z_r_ceed, data->z_r_ceed, data->mask_I_ceed);
    CeedOperatorApply(data->op_r_r, data->z_r_ceed, data->x_r_ceed, CEED_REQUEST_IMMEDIATE);
    CeedVectorPointwiseMult(data->x_r_ceed, data->x_r_ceed, data->mask_Gamma_ceed);
    // ---- J^T (jump map)
    CeedVectorPointwiseMult(data->z_r_ceed, data->x_r_ceed, data->mult_ceed);
    // ------ Local-to-Global
    PetscCall(VecGetArrayAndMemType(bddc_ctx->Y_loc, &y, &y_mem_type));
    CeedVectorSetArray(data->y_ceed, MemTypeP2C(y_mem_type), CEED_USE_POINTER, y);
    CeedOperatorApply(data->op_restrict_r, data->z_r_ceed, data->y_ceed, CEED_REQUEST_IMMEDIATE);
    CeedVectorTakeArray(data->y_ceed, MemTypeP2C(y_mem_type), &y);
    PetscCall(VecRestoreArrayAndMemType(bddc_ctx->Y_loc, &y));
    PetscCall(VecZeroEntries(Y));
    PetscCall(DMLocalToGlobal(bddc_ctx->dm, bddc_ctx->Y_loc, ADD_VALUES, Y));
    // ------ Global-to-Local
    PetscCall(VecZeroEntries(bddc_ctx->Y_loc));
    PetscCall(DMGlobalToLocal(bddc_ctx->dm, Y, INSERT_VALUES, bddc_ctx->Y_loc));
    PetscCall(VecGetArrayReadAndMemType(bddc_ctx->Y_loc, (const PetscScalar **)&y, &y_mem_type));
    CeedVectorSetArray(data->y_ceed, MemTypeP2C(y_mem_type), CEED_USE_POINTER, y);
    CeedOperatorApply(data->op_inject_r, data->y_ceed, data->z_r_ceed, CEED_REQUEST_IMMEDIATE);
    CeedVectorAXPY(data->z_r_ceed, -1.0, data->x_r_ceed);
    // ---- Y_r -=  J^T (- A_Gamma,I A_I,I^-1) Y_r
    CeedVectorPointwiseMult(data->y_r_ceed, data->y_r_ceed, data->mult_ceed);
    CeedVectorAXPY(data->y_r_ceed, -1.0, data->z_r_ceed);
  } else {
    CeedVectorPointwiseMult(data->y_r_ceed, data->y_r_ceed, data->mult_ceed);
  }
  // ---- Inject to Y_Pi
  PetscCall(VecGetArrayAndMemType(bddc_ctx->Y_Pi_loc, &y, &y_mem_type));
  CeedVectorSetArray(data->y_Pi_ceed, MemTypeP2C(y_mem_type), CEED_USE_POINTER, y);
  CeedOperatorApply(data->op_inject_Pi, data->y_r_ceed, data->y_Pi_ceed, CEED_REQUEST_IMMEDIATE);
  CeedVectorTakeArray(data->y_Pi_ceed, MemTypeP2C(y_mem_type), &y);
  PetscCall(VecRestoreArrayAndMemType(bddc_ctx->Y_Pi_loc, &y));
  // ---- Global-To-Local
  PetscCall(VecZeroEntries(bddc_ctx->Y_Pi));
  PetscCall(DMLocalToGlobal(bddc_ctx->dm_Pi, bddc_ctx->Y_Pi_loc, ADD_VALUES, bddc_ctx->Y_Pi));
  // Note: current values in Y_Pi, Y_r

  // K_u^-T - update nodal values from subdomain
  // -- X_r = A_r,r^-1 Y_r
  PetscCall(BDDCArrInv(bddc_ctx, data->y_r_ceed, data->x_r_ceed));
  // -- X_Pi = A_Pi,r X_r
  PetscCall(VecGetArrayAndMemType(bddc_ctx->X_Pi_loc, &x, &x_mem_type));
  CeedVectorSetArray(data->x_Pi_ceed, MemTypeP2C(x_mem_type), CEED_USE_POINTER, x);
  CeedVectorPointwiseMult(data->x_r_ceed, data->x_r_ceed, data->mask_r_ceed);
  CeedOperatorApply(data->op_Pi_r, data->x_r_ceed, data->x_Pi_ceed, CEED_REQUEST_IMMEDIATE);
  CeedVectorTakeArray(data->x_Pi_ceed, MemTypeP2C(x_mem_type), &x);
  PetscCall(VecRestoreArrayAndMemType(bddc_ctx->X_Pi_loc, &x));
  PetscCall(VecZeroEntries(bddc_ctx->X_Pi));
  PetscCall(DMLocalToGlobal(bddc_ctx->dm_Pi, bddc_ctx->X_Pi_loc, ADD_VALUES, bddc_ctx->X_Pi));
  // -- Y_Pi -= A_Pi_r A_r,r^-1 Y_r == X_Pi
  PetscCall(VecAXPY(bddc_ctx->Y_Pi, -1.0, bddc_ctx->X_Pi));
  // Note: current values in Y_Pi, Y_r

  // P^-1 - subdomain and Schur compliment solve
  // -- X_r = A_r,r^-1 Y_r
  PetscCall(BDDCArrInv(bddc_ctx, data->y_r_ceed, data->x_r_ceed));
  // -- X_Pi = S_Pi^-1 Y_Pi
  PetscCall(KSPSolve(bddc_ctx->ksp_S_Pi, bddc_ctx->Y_Pi, bddc_ctx->X_Pi));
  // Note: current values in X_Pi, X_r

  // K_u^-1 - update subdomain values from nodes
  // -- Y_r = A_r,Pi X_Pi
  PetscCall(VecZeroEntries(bddc_ctx->X_Pi_loc));
  PetscCall(DMGlobalToLocal(bddc_ctx->dm_Pi, bddc_ctx->X_Pi, INSERT_VALUES, bddc_ctx->X_Pi_loc));
  PetscCall(VecGetArrayReadAndMemType(bddc_ctx->X_Pi_loc, (const PetscScalar **)&x, &x_mem_type));
  CeedVectorSetArray(data->x_Pi_ceed, MemTypeP2C(x_mem_type), CEED_USE_POINTER, x);
  CeedOperatorApply(data->op_r_Pi, data->x_Pi_ceed, data->y_r_ceed, CEED_REQUEST_IMMEDIATE);
  CeedVectorTakeArray(data->x_Pi_ceed, MemTypeP2C(x_mem_type), &x);
  PetscCall(VecRestoreArrayReadAndMemType(bddc_ctx->X_Pi_loc, (const PetscScalar **)&x));
  // -- Z_r = A_r,r^-1 Y_r
  PetscCall(BDDCArrInv(bddc_ctx, data->y_r_ceed, data->z_r_ceed));
  // -- X_r -= A_r,r^-1 A_r,Pi X_Pi == Z_r
  CeedVectorAXPY(data->x_r_ceed, -1.0, data->z_r_ceed);
  // Note: current values in X_Pi, X_r

  // Restrict to fine space
  // -- Scaled restriction, point multiply by 1/multiplicity
  // ---- Copy X_Pi to X_r
  PetscCall(VecGetArrayReadAndMemType(bddc_ctx->X_Pi_loc, (const PetscScalar **)&x, &x_mem_type));
  CeedVectorSetArray(data->x_Pi_ceed, MemTypeP2C(x_mem_type), CEED_USE_POINTER, x);
  CeedVectorPointwiseMult(data->x_r_ceed, data->x_r_ceed, data->mask_r_ceed);
  CeedOperatorApplyAdd(data->op_restrict_Pi, data->x_Pi_ceed, data->x_r_ceed, CEED_REQUEST_IMMEDIATE);
  CeedVectorTakeArray(data->x_Pi_ceed, MemTypeP2C(x_mem_type), &x);
  PetscCall(VecRestoreArrayReadAndMemType(bddc_ctx->X_Pi_loc, (const PetscScalar **)&x));
  // -- Harmonic injection, scaled with jump map
  if (bddc_ctx->is_harmonic) {
    // ---- J^T (jump map)
    CeedVectorPointwiseMult(data->z_r_ceed, data->x_r_ceed, data->mult_ceed);
    // ------ Local-to-Global
    PetscCall(VecGetArrayAndMemType(bddc_ctx->Y_loc, &y, &y_mem_type));
    CeedVectorSetArray(data->y_ceed, MemTypeP2C(y_mem_type), CEED_USE_POINTER, y);
    CeedOperatorApply(data->op_restrict_r, data->z_r_ceed, data->y_ceed, CEED_REQUEST_IMMEDIATE);
    CeedVectorTakeArray(data->y_ceed, MemTypeP2C(y_mem_type), &y);
    PetscCall(VecRestoreArrayAndMemType(bddc_ctx->Y_loc, &y));
    PetscCall(VecZeroEntries(Y));
    PetscCall(DMLocalToGlobal(bddc_ctx->dm, bddc_ctx->Y_loc, ADD_VALUES, Y));
    // ------ Global-to-Local
    PetscCall(VecZeroEntries(bddc_ctx->Y_loc));
    PetscCall(DMGlobalToLocal(bddc_ctx->dm, Y, INSERT_VALUES, bddc_ctx->Y_loc));
    PetscCall(VecGetArrayReadAndMemType(bddc_ctx->Y_loc, (const PetscScalar **)&y, &y_mem_type));
    CeedVectorSetArray(data->y_ceed, MemTypeP2C(y_mem_type), CEED_USE_POINTER, y);
    CeedOperatorApply(data->op_inject_r, data->y_ceed, data->z_r_ceed, CEED_REQUEST_IMMEDIATE);
    CeedVectorAXPY(data->z_r_ceed, -1.0, data->x_r_ceed);
    CeedVectorTakeArray(data->y_ceed, MemTypeP2C(y_mem_type), &y);
    PetscCall(VecRestoreArrayAndMemType(bddc_ctx->Y_loc, &y));
    // ---- Y_r = A_I,Gamma Z_r
    CeedVectorPointwiseMult(data->z_r_ceed, data->z_r_ceed, data->mask_Gamma_ceed);
    CeedOperatorApply(data->op_r_r, data->z_r_ceed, data->y_r_ceed, CEED_REQUEST_IMMEDIATE);
    // ---- Z_r = A_I,I^-1 Y_r
    PetscCall(BDDCArrInv(bddc_ctx, data->y_r_ceed, data->z_r_ceed));
    // ---- X_r += A_I,I^-1 A_I,Gamma J X_r
    CeedVectorPointwiseMult(data->z_r_ceed, data->z_r_ceed, data->mask_I_ceed);
    CeedVectorPointwiseMult(data->x_r_ceed, data->x_r_ceed, data->mult_ceed);
    CeedVectorAXPY(data->x_r_ceed, -1.0, data->z_r_ceed);
  } else {
    CeedVectorPointwiseMult(data->x_r_ceed, data->x_r_ceed, data->mult_ceed);
  }
  // ---- Restrict to Y
  PetscCall(VecGetArrayAndMemType(bddc_ctx->Y_loc, &y, &y_mem_type));
  CeedVectorSetArray(data->y_ceed, MemTypeP2C(y_mem_type), CEED_USE_POINTER, y);
  CeedOperatorApply(data->op_restrict_r, data->x_r_ceed, data->y_ceed, CEED_REQUEST_IMMEDIATE);
  CeedVectorTakeArray(data->y_ceed, MemTypeP2C(y_mem_type), &y);
  PetscCall(VecRestoreArrayAndMemType(bddc_ctx->Y_loc, &y));
  // ---- Local-to-Global
  PetscCall(VecZeroEntries(Y));
  PetscCall(DMLocalToGlobal(bddc_ctx->dm, bddc_ctx->Y_loc, ADD_VALUES, Y));
  // Note: current values in Y

  PetscFunctionReturn(PETSC_SUCCESS);
};

// -----------------------------------------------------------------------------
// This function calculates the error in the final solution
// -----------------------------------------------------------------------------
PetscErrorCode ComputeL2Error(Vec X, PetscScalar *l2_error, OperatorApplyContext op_error_ctx) {
  Vec E;

  PetscFunctionBeginUser;

  PetscCall(VecDuplicate(X, &E));
  PetscCall(ApplyLocal_Ceed(X, E, op_error_ctx));
  PetscScalar error_sq = 1.0;
  PetscCall(VecSum(E, &error_sq));
  *l2_error = sqrt(error_sq);

  PetscFunctionReturn(PETSC_SUCCESS);
}

// -----------------------------------------------------------------------------
