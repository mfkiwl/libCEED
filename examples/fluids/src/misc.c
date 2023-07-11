// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Miscellaneous utility functions

#include <ceed.h>
#include <petscdm.h>
#include <petscsf.h>
#include <petscts.h>

#include "../navierstokes.h"
#include "../qfunctions/mass.h"

PetscErrorCode ICs_FixMultiplicity(DM dm, CeedData ceed_data, User user, Vec Q_loc, Vec Q, CeedScalar time) {
  CeedVector   mult_vec;
  PetscMemType m_mem_type;
  Vec          multiplicity_loc, multiplicity;
  PetscFunctionBeginUser;

  // Update time for evaluation
  if (user->phys->ics_time_label) PetscCall(UpdateContextLabel(user->comm, time, ceed_data->op_ics_ctx->op, user->phys->ics_time_label));

  // Place PETSc vector in CEED vector
  PetscCall(ApplyCeedOperatorLocalToGlobal(NULL, Q, ceed_data->op_ics_ctx));

  // CEED Restriction
  CeedElemRestrictionCreateVector(ceed_data->elem_restr_q, &mult_vec, NULL);

  // Place PETSc vector in CEED vector
  PetscCall(DMGetLocalVector(dm, &multiplicity_loc));
  PetscCall(VecP2C(multiplicity_loc, &m_mem_type, mult_vec));

  // Get multiplicity
  CeedElemRestrictionGetMultiplicity(ceed_data->elem_restr_q, mult_vec);

  // Restore vectors
  PetscCall(VecC2P(mult_vec, m_mem_type, multiplicity_loc));

  // Local-to-Global
  PetscCall(DMGetGlobalVector(dm, &multiplicity));
  PetscCall(VecZeroEntries(multiplicity));
  PetscCall(DMLocalToGlobal(dm, multiplicity_loc, ADD_VALUES, multiplicity));

  // Fix multiplicity
  PetscCall(VecPointwiseDivide(Q, Q, multiplicity));
  PetscCall(VecPointwiseDivide(Q_loc, Q_loc, multiplicity_loc));

  // Restore vectors
  PetscCall(DMRestoreLocalVector(dm, &multiplicity_loc));
  PetscCall(DMRestoreGlobalVector(dm, &multiplicity));

  // Cleanup
  CeedVectorDestroy(&mult_vec);

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMPlexInsertBoundaryValues_NS(DM dm, PetscBool insert_essential, Vec Q_loc, PetscReal time, Vec face_geom_FVM, Vec cell_geom_FVM,
                                             Vec grad_FVM) {
  Vec Qbc, boundary_mask;
  PetscFunctionBeginUser;

  // Mask (zero) Strong BC entries
  PetscCall(DMGetNamedLocalVector(dm, "boundary mask", &boundary_mask));
  PetscCall(VecPointwiseMult(Q_loc, Q_loc, boundary_mask));
  PetscCall(DMRestoreNamedLocalVector(dm, "boundary mask", &boundary_mask));

  PetscCall(DMGetNamedLocalVector(dm, "Qbc", &Qbc));
  PetscCall(VecAXPY(Q_loc, 1., Qbc));
  PetscCall(DMRestoreNamedLocalVector(dm, "Qbc", &Qbc));

  PetscFunctionReturn(PETSC_SUCCESS);
}

// @brief Load vector from binary file, possibly with embedded solution time and step number
PetscErrorCode LoadFluidsBinaryVec(MPI_Comm comm, PetscViewer viewer, Vec Q, PetscReal *time, PetscInt *step_number) {
  PetscInt   file_step_number;
  PetscInt32 token;
  PetscReal  file_time;
  PetscFunctionBeginUser;

  // Attempt
  PetscCall(PetscViewerBinaryRead(viewer, &token, 1, NULL, PETSC_INT32));
  if (token == FLUIDS_FILE_TOKEN_32 || token == FLUIDS_FILE_TOKEN_64 ||
      token == FLUIDS_FILE_TOKEN) {  // New style format; we're reading a file with step number and time in the header
    PetscCall(PetscViewerBinaryRead(viewer, &file_step_number, 1, NULL, PETSC_INT));
    PetscCall(PetscViewerBinaryRead(viewer, &file_time, 1, NULL, PETSC_REAL));
    if (time) *time = file_time;
    if (step_number) *step_number = file_step_number;
  } else if (token == VEC_FILE_CLASSID) {  // Legacy format of just the vector, encoded as [VEC_FILE_CLASSID, length, ]
    PetscInt length, N;
    PetscCall(PetscViewerBinaryRead(viewer, &length, 1, NULL, PETSC_INT));
    PetscCall(VecGetSize(Q, &N));
    PetscCheck(length == N, comm, PETSC_ERR_ARG_INCOMP, "File Vec has length %" PetscInt_FMT " but DM has global Vec size %" PetscInt_FMT, length, N);
    PetscCall(PetscViewerBinarySetSkipHeader(viewer, PETSC_TRUE));
  } else SETERRQ(comm, PETSC_ERR_FILE_UNEXPECTED, "Not a fluids header token or a PETSc Vec in file");

  // Load Q from existent solution
  PetscCall(VecLoad(Q, viewer));

  PetscFunctionReturn(PETSC_SUCCESS);
}

// Compare reference solution values with current test run for CI
PetscErrorCode RegressionTests_NS(AppCtx app_ctx, Vec Q) {
  Vec         Qref;
  PetscViewer viewer;
  PetscReal   error, Qrefnorm;
  MPI_Comm    comm = PetscObjectComm((PetscObject)Q);
  PetscFunctionBeginUser;

  // Read reference file
  PetscCall(VecDuplicate(Q, &Qref));
  PetscCall(PetscViewerBinaryOpen(comm, app_ctx->test_file_path, FILE_MODE_READ, &viewer));
  PetscCall(LoadFluidsBinaryVec(comm, viewer, Qref, NULL, NULL));

  // Compute error with respect to reference solution
  PetscCall(VecAXPY(Q, -1.0, Qref));
  PetscCall(VecNorm(Qref, NORM_MAX, &Qrefnorm));
  PetscCall(VecScale(Q, 1. / Qrefnorm));
  PetscCall(VecNorm(Q, NORM_MAX, &error));

  // Check error
  if (error > app_ctx->test_tol) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Test failed with error norm %g\n", (double)error));
  }

  // Cleanup
  PetscCall(PetscViewerDestroy(&viewer));
  PetscCall(VecDestroy(&Qref));

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode ComputeL2Error(MPI_Comm comm, Vec Q_loc, PetscReal l2_error[5], OperatorApplyContext op_error_ctx, CeedContextFieldLabel time_label,
                              CeedScalar time) {
  Vec       E;
  PetscReal l2_norm[5];
  PetscFunctionBeginUser;

  if (time_label) PetscCall(UpdateContextLabel(comm, time, op_error_ctx->op, time_label));
  PetscCall(VecDuplicate(Q_loc, &E));
  PetscCall(ApplyCeedOperatorLocalToGlobal(Q_loc, E, op_error_ctx));
  PetscCall(VecStrideNormAll(E, NORM_1, l2_norm));
  for (int i = 0; i < 5; i++) l2_error[i] = sqrt(l2_norm[i]);

  PetscCall(VecDestroy(&E));

  PetscFunctionReturn(PETSC_SUCCESS);
};

// Get error for problems with true solutions
PetscErrorCode GetError_NS(CeedData ceed_data, DM dm, User user, ProblemData *problem, Vec Q, PetscScalar final_time) {
  Vec         Q_loc;
  PetscReal   l2_error[5];
  const char *state_var_source = "Conservative";
  PetscFunctionBeginUser;

  // Get the local values of the final solution
  PetscCall(DMGetLocalVector(dm, &Q_loc));
  PetscCall(DMGlobalToLocal(dm, Q, INSERT_VALUES, Q_loc));
  PetscCall(UpdateBoundaryValues(user, Q_loc, final_time));

  // Compute the L2 error in the source state variables
  if (user->phys->ics_time_label) PetscCall(UpdateContextLabel(user->comm, final_time, ceed_data->op_ics_ctx->op, user->phys->ics_time_label));
  CeedOperatorApply(ceed_data->op_ics_ctx->op, ceed_data->x_coord, ceed_data->q_true, CEED_REQUEST_IMMEDIATE);
  PetscCall(ComputeL2Error(user->comm, Q_loc, l2_error, ceed_data->op_error_ctx, user->phys->ics_time_label, final_time));

  // Print the error
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\nL2 Error:\n"));
  if (user->phys->state_var == STATEVAR_PRIMITIVE) state_var_source = "Primitive";
  for (int i = 0; i < 5; i++) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  %s variables-Component %d: %g\n", state_var_source, i, (double)l2_error[i]));

  if (problem->convert_error.qfunction) {
    PetscReal   l2_error_converted[5];
    const char *state_var_target = "Primitive";

    // Convert the L2 error to the target state variable
    PetscCall(ComputeL2Error(user->comm, Q_loc, l2_error_converted, ceed_data->op_convert_error_ctx, user->phys->ics_time_label, final_time));

    // Print the error
    if (user->phys->state_var == STATEVAR_PRIMITIVE) state_var_target = "Conservative";
    for (int i = 0; i < 5; i++) {
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  %s variables-Component %d: %g\n", state_var_target, i, (double)l2_error_converted[i]));
    }
    // Cleanup
    CeedQFunctionDestroy(&ceed_data->qf_convert_error);
    PetscCall(OperatorApplyContextDestroy(ceed_data->op_convert_error_ctx));
  }

  // Cleanup
  PetscCall(DMRestoreLocalVector(dm, &Q_loc));
  CeedVectorDestroy(&ceed_data->q_true);
  PetscCall(OperatorApplyContextDestroy(ceed_data->op_error_ctx));

  PetscFunctionReturn(PETSC_SUCCESS);
}

// Post-processing
PetscErrorCode PostProcess_NS(TS ts, CeedData ceed_data, DM dm, ProblemData *problem, User user, Vec Q, PetscScalar final_time) {
  PetscInt          steps;
  TSConvergedReason reason;
  PetscFunctionBeginUser;

  // Print relative error
  if (problem->has_true_soln && user->app_ctx->test_type == TESTTYPE_NONE) {
    PetscCall(GetError_NS(ceed_data, dm, user, problem, Q, final_time));
  }

  // Print final time and number of steps
  PetscCall(TSGetStepNumber(ts, &steps));
  PetscCall(TSGetConvergedReason(ts, &reason));
  if (user->app_ctx->test_type == TESTTYPE_NONE) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Time integrator %s on time step %" PetscInt_FMT " with final time %g\n", TSConvergedReasons[reason],
                          steps, (double)final_time));
  }

  // Output numerical values from command line
  PetscCall(VecViewFromOptions(Q, NULL, "-vec_view"));

  // Compare reference solution values with current test run for CI
  if (user->app_ctx->test_type == TESTTYPE_SOLVER) {
    PetscCall(RegressionTests_NS(user->app_ctx, Q));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

const PetscInt32 FLUIDS_FILE_TOKEN    = 0xceedf00;  // for backwards compatibility
const PetscInt32 FLUIDS_FILE_TOKEN_32 = 0xceedf32;
const PetscInt32 FLUIDS_FILE_TOKEN_64 = 0xceedf64;

// Gather initial Q values in case of continuation of simulation
PetscErrorCode SetupICsFromBinary(MPI_Comm comm, AppCtx app_ctx, Vec Q) {
  PetscViewer viewer;

  PetscFunctionBeginUser;

  PetscCall(PetscViewerBinaryOpen(comm, app_ctx->cont_file, FILE_MODE_READ, &viewer));
  PetscCall(LoadFluidsBinaryVec(comm, viewer, Q, &app_ctx->cont_time, &app_ctx->cont_steps));
  PetscCall(PetscViewerDestroy(&viewer));

  PetscFunctionReturn(PETSC_SUCCESS);
}

// Record boundary values from initial condition
PetscErrorCode SetBCsFromICs_NS(DM dm, Vec Q, Vec Q_loc) {
  Vec Qbc, boundary_mask;
  PetscFunctionBeginUser;

  PetscCall(DMGetNamedLocalVector(dm, "Qbc", &Qbc));
  PetscCall(VecCopy(Q_loc, Qbc));
  PetscCall(VecZeroEntries(Q_loc));
  PetscCall(DMGlobalToLocal(dm, Q, INSERT_VALUES, Q_loc));
  PetscCall(VecAXPY(Qbc, -1., Q_loc));
  PetscCall(DMRestoreNamedLocalVector(dm, "Qbc", &Qbc));
  PetscCall(PetscObjectComposeFunction((PetscObject)dm, "DMPlexInsertBoundaryValues_C", DMPlexInsertBoundaryValues_NS));

  PetscCall(DMGetNamedLocalVector(dm, "boundary mask", &boundary_mask));
  PetscCall(DMGetGlobalVector(dm, &Q));
  PetscCall(VecZeroEntries(boundary_mask));
  PetscCall(VecSet(Q, 1.0));
  PetscCall(DMGlobalToLocal(dm, Q, INSERT_VALUES, boundary_mask));
  PetscCall(DMRestoreNamedLocalVector(dm, "boundary mask", &boundary_mask));

  PetscFunctionReturn(PETSC_SUCCESS);
}

// Free a plain data context that was allocated using PETSc; returning libCEED error codes
int FreeContextPetsc(void *data) {
  if (PetscFree(data)) return CeedError(NULL, CEED_ERROR_ACCESS, "PetscFree failed");
  return CEED_ERROR_SUCCESS;
}

// Return mass qfunction specification for number of components N
PetscErrorCode CreateMassQFunction(Ceed ceed, CeedInt N, CeedInt q_data_size, CeedQFunction *qf) {
  PetscFunctionBeginUser;

  switch (N) {
    case 1:
      CeedQFunctionCreateInterior(ceed, 1, Mass_1, Mass_1_loc, qf);
      break;
    case 5:
      CeedQFunctionCreateInterior(ceed, 1, Mass_5, Mass_5_loc, qf);
      break;
    case 7:
      CeedQFunctionCreateInterior(ceed, 1, Mass_7, Mass_7_loc, qf);
      break;
    case 9:
      CeedQFunctionCreateInterior(ceed, 1, Mass_9, Mass_9_loc, qf);
      break;
    case 22:
      CeedQFunctionCreateInterior(ceed, 1, Mass_22, Mass_22_loc, qf);
      break;
    default:
      SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_SUP, "Could not find mass qfunction of size %d", N);
  }

  CeedQFunctionAddInput(*qf, "u", N, CEED_EVAL_INTERP);
  CeedQFunctionAddInput(*qf, "qdata", q_data_size, CEED_EVAL_NONE);
  CeedQFunctionAddOutput(*qf, "v", N, CEED_EVAL_INTERP);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* @brief L^2 Projection of a source FEM function to a target FEM space
 *
 * To solve system using a lumped mass matrix, pass a KSP object with ksp_type=preonly, pc_type=jacobi, pc_jacobi_type=rowsum.
 *
 * @param[in]  source_vec    Global Vec of the source FEM function. NULL indicates using rhs_matop_ctx->X_loc
 * @param[out] target_vec    Global Vec of the target (result) FEM function. NULL indicates using rhs_matop_ctx->Y_loc
 * @param[in]  rhs_matop_ctx MatopApplyContext for performing the RHS evaluation
 * @param[in]  ksp           KSP for solving the consistent projection problem
 */
PetscErrorCode ComputeL2Projection(Vec source_vec, Vec target_vec, OperatorApplyContext rhs_matop_ctx, KSP ksp) {
  PetscFunctionBeginUser;

  PetscCall(ApplyCeedOperatorGlobalToGlobal(source_vec, target_vec, rhs_matop_ctx));
  PetscCall(KSPSolve(ksp, target_vec, target_vec));

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NodalProjectionDataDestroy(NodalProjectionData context) {
  PetscFunctionBeginUser;
  if (context == NULL) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(DMDestroy(&context->dm));
  PetscCall(KSPDestroy(&context->ksp));

  PetscCall(OperatorApplyContextDestroy(context->l2_rhs_ctx));

  PetscCall(PetscFree(context));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
 * @brief Open a PHASTA *.dat file, grabbing dimensions and file pointer
 *
 * This function opens the file specified by `path` using `PetscFOpen` and passes the file pointer in `fp`.
 * It is not closed in this function, thus `fp` must be closed sometime after this function has been called (using `PetscFClose` for example).
 *
 * Assumes that the first line of the file has the number of rows and columns as the only two entries, separated by a single space.
 *
 * @param[in]  comm           MPI_Comm for the program
 * @param[in]  path           Path to the file
 * @param[in]  char_array_len Length of the character array that should contain each line
 * @param[out] dims           Dimensions of the file, taken from the first line of the file
 * @param[out] fp File        pointer to the opened file
 */
PetscErrorCode PHASTADatFileOpen(const MPI_Comm comm, const char path[PETSC_MAX_PATH_LEN], const PetscInt char_array_len, PetscInt dims[2],
                                 FILE **fp) {
  int    ndims;
  char   line[char_array_len];
  char **array;

  PetscFunctionBeginUser;
  PetscCall(PetscFOpen(comm, path, "r", fp));
  PetscCall(PetscSynchronizedFGets(comm, *fp, char_array_len, line));
  PetscCall(PetscStrToArray(line, ' ', &ndims, &array));
  PetscCheck(ndims == 2, comm, PETSC_ERR_FILE_UNEXPECTED, "Found %d dimensions instead of 2 on the first line of %s", ndims, path);

  for (PetscInt i = 0; i < ndims; i++) dims[i] = atoi(array[i]);
  PetscCall(PetscStrToArrayDestroy(ndims, array));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
 * @brief Get the number of rows for the PHASTA file at path.
 *
 * Assumes that the first line of the file has the number of rows and columns as the only two entries, separated by a single space.
 *
 * @param[in]  comm  MPI_Comm for the program
 * @param[in]  path  Path to the file
 * @param[out] nrows Number of rows
 */
PetscErrorCode PHASTADatFileGetNRows(const MPI_Comm comm, const char path[PETSC_MAX_PATH_LEN], PetscInt *nrows) {
  const PetscInt char_array_len = 512;
  PetscInt       dims[2];
  FILE          *fp;

  PetscFunctionBeginUser;
  PetscCall(PHASTADatFileOpen(comm, path, char_array_len, dims, &fp));
  *nrows = dims[0];
  PetscCall(PetscFClose(comm, fp));

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PHASTADatFileReadToArrayReal(MPI_Comm comm, const char path[PETSC_MAX_PATH_LEN], PetscReal array[]) {
  PetscInt       dims[2];
  int            ndims;
  FILE          *fp;
  const PetscInt char_array_len = 512;
  char           line[char_array_len];
  char         **row_array;
  PetscFunctionBeginUser;

  PetscCall(PHASTADatFileOpen(comm, path, char_array_len, dims, &fp));

  for (PetscInt i = 0; i < dims[0]; i++) {
    PetscCall(PetscSynchronizedFGets(comm, fp, char_array_len, line));
    PetscCall(PetscStrToArray(line, ' ', &ndims, &row_array));
    PetscCheck(ndims == dims[1], comm, PETSC_ERR_FILE_UNEXPECTED,
               "Line %" PetscInt_FMT " of %s does not contain enough columns (%d instead of %" PetscInt_FMT ")", i, path, ndims, dims[1]);

    for (PetscInt j = 0; j < dims[1]; j++) {
      array[i * dims[1] + j] = (PetscReal)atof(row_array[j]);
    }
  }

  PetscCall(PetscFClose(comm, fp));

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscLogEvent       FLUIDS_CeedOperatorApply;
PetscLogEvent       FLUIDS_CeedOperatorAssemble;
PetscLogEvent       FLUIDS_CeedOperatorAssembleDiagonal;
PetscLogEvent       FLUIDS_CeedOperatorAssemblePointBlockDiagonal;
static PetscClassId libCEED_classid;

PetscErrorCode RegisterLogEvents() {
  PetscFunctionBeginUser;
  PetscCall(PetscClassIdRegister("libCEED", &libCEED_classid));
  PetscCall(PetscLogEventRegister("CeedOpApply", libCEED_classid, &FLUIDS_CeedOperatorApply));
  PetscCall(PetscLogEventRegister("CeedOpAsm", libCEED_classid, &FLUIDS_CeedOperatorAssemble));
  PetscCall(PetscLogEventRegister("CeedOpAsmD", libCEED_classid, &FLUIDS_CeedOperatorAssembleDiagonal));
  PetscCall(PetscLogEventRegister("CeedOpAsmPBD", libCEED_classid, &FLUIDS_CeedOperatorAssemblePointBlockDiagonal));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
  @brief Translate array of CeedInt to PetscInt.
    If the types differ, `array_ceed` is freed with `free()` and `array_petsc` is allocated with `malloc()`.
    Caller is responsible for freeing `array_petsc` with `free()`.

  @param[in]      num_entries  Number of array entries
  @param[in,out]  array_ceed   Array of CeedInts
  @param[out]     array_petsc  Array of PetscInts
**/
PetscErrorCode IntArrayC2P(PetscInt num_entries, CeedInt **array_ceed, PetscInt **array_petsc) {
  CeedInt  int_c = 0;
  PetscInt int_p = 0;

  PetscFunctionBeginUser;
  if (sizeof(int_c) == sizeof(int_p)) {
    *array_petsc = (PetscInt *)*array_ceed;
  } else {
    *array_petsc = malloc(num_entries * sizeof(PetscInt));
    for (PetscInt i = 0; i < num_entries; i++) (*array_petsc)[i] = (*array_ceed)[i];
    free(*array_ceed);
  }
  *array_ceed = NULL;

  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
  @brief Translate array of PetscInt to CeedInt.
    If the types differ, `array_petsc` is freed with `PetscFree()` and `array_ceed` is allocated with `PetscMalloc1()`.
    Caller is responsible for freeing `array_ceed` with `PetscFree()`.

  @param[in]      num_entries  Number of array entries
  @param[in,out]  array_petsc  Array of PetscInts
  @param[out]     array_ceed   Array of CeedInts
**/
PetscErrorCode IntArrayP2C(PetscInt num_entries, PetscInt **array_petsc, CeedInt **array_ceed) {
  CeedInt  int_c = 0;
  PetscInt int_p = 0;

  PetscFunctionBeginUser;
  if (sizeof(int_c) == sizeof(int_p)) {
    *array_ceed = (CeedInt *)*array_petsc;
  } else {
    PetscCall(PetscMalloc1(num_entries, array_ceed));
    for (PetscInt i = 0; i < num_entries; i++) (*array_ceed)[i] = (*array_petsc)[i];
    PetscCall(PetscFree(*array_petsc));
  }
  *array_petsc = NULL;

  PetscFunctionReturn(PETSC_SUCCESS);
}

// Print information about the given simulation run
PetscErrorCode PrintRunInfo(User user, Physics phys_ctx, ProblemData *problem, MPI_Comm comm) {
  PetscFunctionBeginUser;
  // Header and rank
  char        host_name[PETSC_MAX_PATH_LEN];
  PetscMPIInt rank, comm_size;
  PetscCall(PetscGetHostName(host_name, sizeof host_name));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCallMPI(MPI_Comm_size(comm, &comm_size));
  PetscCall(PetscPrintf(comm,
                        "\n-- Navier-Stokes solver - libCEED + PETSc --\n"
                        "  MPI:\n"
                        "    Host Name                          : %s\n"
                        "    Total ranks                        : %d\n",
                        host_name, comm_size));

  // Problem specific info
  PetscCall(problem->print_info(problem, user->app_ctx));

  // libCEED
  const char *used_resource;
  CeedMemType mem_type_backend;
  CeedGetResource(user->ceed, &used_resource);
  CeedGetPreferredMemType(user->ceed, &mem_type_backend);
  PetscCall(PetscPrintf(comm,
                        "  libCEED:\n"
                        "    libCEED Backend                    : %s\n"
                        "    libCEED Backend MemType            : %s\n",
                        used_resource, CeedMemTypes[mem_type_backend]));
  // PETSc
  char box_faces_str[PETSC_MAX_PATH_LEN] = "3,3,3";
  if (problem->dim == 2) box_faces_str[3] = '\0';
  PetscCall(PetscOptionsGetString(NULL, NULL, "-dm_plex_box_faces", box_faces_str, sizeof(box_faces_str), NULL));
  MatType mat_type;
  VecType vec_type;
  PetscCall(DMGetMatType(user->dm, &mat_type));
  PetscCall(DMGetVecType(user->dm, &vec_type));
  PetscCall(PetscPrintf(comm,
                        "  PETSc:\n"
                        "    Box Faces                          : %s\n"
                        "    DM MatType                         : %s\n"
                        "    DM VecType                         : %s\n"
                        "    Time Stepping Scheme               : %s\n",
                        box_faces_str, mat_type, vec_type, phys_ctx->implicit ? "implicit" : "explicit"));
  if (user->app_ctx->cont_steps) {
    PetscCall(PetscPrintf(comm,
                          "  Continue:\n"
                          "    Filename:                          : %s\n"
                          "    Step:                              : %" PetscInt_FMT "\n"
                          "    Time:                              : %g\n",
                          user->app_ctx->cont_file, user->app_ctx->cont_steps, user->app_ctx->cont_time));
  }
  // Mesh
  const PetscInt num_comp_q = 5;
  PetscInt       glob_dofs, owned_dofs, local_dofs;
  const CeedInt  num_P = user->app_ctx->degree + 1, num_Q = num_P + user->app_ctx->q_extra;
  // -- Get global size
  PetscCall(DMGetGlobalVectorInfo(user->dm, &owned_dofs, &glob_dofs, NULL));
  // -- Get local size
  PetscCall(DMGetLocalVectorInfo(user->dm, &local_dofs, NULL, NULL));
  PetscCall(PetscPrintf(comm,
                        "  Mesh:\n"
                        "    Number of 1D Basis Nodes (P)       : %" CeedInt_FMT "\n"
                        "    Number of 1D Quadrature Points (Q) : %" CeedInt_FMT "\n"
                        "    Global DoFs                        : %" PetscInt_FMT "\n"
                        "    DoFs per node                      : %" PetscInt_FMT "\n"
                        "    Global %" PetscInt_FMT "-DoF nodes                 : %" PetscInt_FMT "\n",
                        num_P, num_Q, glob_dofs, num_comp_q, num_comp_q, glob_dofs / num_comp_q));
  // -- Get Partition Statistics
  PetscCall(PetscPrintf(comm, "  Partition:                             (min,max,median,max/median)\n"));
  {
    PetscInt *gather_buffer = NULL;
    PetscInt  part_owned_dofs[3], part_local_dofs[3], part_boundary_dofs[3], part_neighbors[3];
    PetscInt  median_index = comm_size % 2 ? comm_size / 2 : comm_size / 2 - 1;
    if (!rank) PetscCall(PetscMalloc1(comm_size, &gather_buffer));

    PetscCallMPI(MPI_Gather(&owned_dofs, 1, MPIU_INT, gather_buffer, 1, MPIU_INT, 0, comm));
    if (!rank) {
      PetscCall(PetscSortInt(comm_size, gather_buffer));
      part_owned_dofs[0]             = gather_buffer[0];              // min
      part_owned_dofs[1]             = gather_buffer[comm_size - 1];  // max
      part_owned_dofs[2]             = gather_buffer[median_index];   // median
      PetscReal part_owned_dof_ratio = (PetscReal)part_owned_dofs[1] / (PetscReal)part_owned_dofs[2];
      PetscCall(PetscPrintf(
          comm, "    Global Vector %" PetscInt_FMT "-DoF nodes          : %" PetscInt_FMT ", %" PetscInt_FMT ", %" PetscInt_FMT ", %f\n", num_comp_q,
          part_owned_dofs[0] / num_comp_q, part_owned_dofs[1] / num_comp_q, part_owned_dofs[2] / num_comp_q, part_owned_dof_ratio));
    }

    PetscCallMPI(MPI_Gather(&local_dofs, 1, MPIU_INT, gather_buffer, 1, MPIU_INT, 0, comm));
    if (!rank) {
      PetscCall(PetscSortInt(comm_size, gather_buffer));
      part_local_dofs[0]             = gather_buffer[0];              // min
      part_local_dofs[1]             = gather_buffer[comm_size - 1];  // max
      part_local_dofs[2]             = gather_buffer[median_index];   // median
      PetscReal part_local_dof_ratio = (PetscReal)part_local_dofs[1] / (PetscReal)part_local_dofs[2];
      PetscCall(PetscPrintf(
          comm, "    Local Vector %" PetscInt_FMT "-DoF nodes           : %" PetscInt_FMT ", %" PetscInt_FMT ", %" PetscInt_FMT ", %f\n", num_comp_q,
          part_local_dofs[0] / num_comp_q, part_local_dofs[1] / num_comp_q, part_local_dofs[2] / num_comp_q, part_local_dof_ratio));
    }

    PetscInt num_remote_roots_total = 0, num_remote_leaves_total = 0, num_ghost_interface_ranks = 0, num_owned_interface_ranks = 0;
    {
      PetscSF            sf;
      PetscInt           nrranks, niranks;
      const PetscInt    *roffset, *rmine, *rremote, *ioffset, *irootloc;
      const PetscMPIInt *rranks, *iranks;
      PetscCall(DMGetSectionSF(user->dm, &sf));
      PetscCall(PetscSFGetRootRanks(sf, &nrranks, &rranks, &roffset, &rmine, &rremote));
      PetscCall(PetscSFGetLeafRanks(sf, &niranks, &iranks, &ioffset, &irootloc));
      for (PetscInt i = 0; i < nrranks; i++) {
        if (rranks[i] == rank) continue;  // Ignore same-part global->local transfers
        num_remote_roots_total += roffset[i + 1] - roffset[i];
        num_ghost_interface_ranks++;
      }
      for (PetscInt i = 0; i < niranks; i++) {
        if (iranks[i] == rank) continue;
        num_remote_leaves_total += ioffset[i + 1] - ioffset[i];
        num_owned_interface_ranks++;
      }
    }
    PetscCallMPI(MPI_Gather(&num_remote_roots_total, 1, MPIU_INT, gather_buffer, 1, MPIU_INT, 0, comm));
    if (!rank) {
      PetscCall(PetscSortInt(comm_size, gather_buffer));
      part_boundary_dofs[0]           = gather_buffer[0];              // min
      part_boundary_dofs[1]           = gather_buffer[comm_size - 1];  // max
      part_boundary_dofs[2]           = gather_buffer[median_index];   // median
      PetscReal part_shared_dof_ratio = (PetscReal)part_boundary_dofs[1] / (PetscReal)part_boundary_dofs[2];
      PetscCall(PetscPrintf(
          comm, "    Ghost Interface %" PetscInt_FMT "-DoF nodes        : %" PetscInt_FMT ", %" PetscInt_FMT ", %" PetscInt_FMT ", %f\n", num_comp_q,
          part_boundary_dofs[0] / num_comp_q, part_boundary_dofs[1] / num_comp_q, part_boundary_dofs[2] / num_comp_q, part_shared_dof_ratio));
    }

    PetscCallMPI(MPI_Gather(&num_ghost_interface_ranks, 1, MPIU_INT, gather_buffer, 1, MPIU_INT, 0, comm));
    if (!rank) {
      PetscCall(PetscSortInt(comm_size, gather_buffer));
      part_neighbors[0]              = gather_buffer[0];              // min
      part_neighbors[1]              = gather_buffer[comm_size - 1];  // max
      part_neighbors[2]              = gather_buffer[median_index];   // median
      PetscReal part_neighbors_ratio = (PetscReal)part_neighbors[1] / (PetscReal)part_neighbors[2];
      PetscCall(PetscPrintf(comm, "    Ghost Interface Ranks              : %" PetscInt_FMT ", %" PetscInt_FMT ", %" PetscInt_FMT ", %f\n",
                            part_neighbors[0], part_neighbors[1], part_neighbors[2], part_neighbors_ratio));
    }

    PetscCallMPI(MPI_Gather(&num_remote_leaves_total, 1, MPIU_INT, gather_buffer, 1, MPIU_INT, 0, comm));
    if (!rank) {
      PetscCall(PetscSortInt(comm_size, gather_buffer));
      part_boundary_dofs[0]           = gather_buffer[0];              // min
      part_boundary_dofs[1]           = gather_buffer[comm_size - 1];  // max
      part_boundary_dofs[2]           = gather_buffer[median_index];   // median
      PetscReal part_shared_dof_ratio = (PetscReal)part_boundary_dofs[1] / (PetscReal)part_boundary_dofs[2];
      PetscCall(PetscPrintf(
          comm, "    Owned Interface %" PetscInt_FMT "-DoF nodes        : %" PetscInt_FMT ", %" PetscInt_FMT ", %" PetscInt_FMT ", %f\n", num_comp_q,
          part_boundary_dofs[0] / num_comp_q, part_boundary_dofs[1] / num_comp_q, part_boundary_dofs[2] / num_comp_q, part_shared_dof_ratio));
    }

    PetscCallMPI(MPI_Gather(&num_owned_interface_ranks, 1, MPIU_INT, gather_buffer, 1, MPIU_INT, 0, comm));
    if (!rank) {
      PetscCall(PetscSortInt(comm_size, gather_buffer));
      part_neighbors[0]              = gather_buffer[0];              // min
      part_neighbors[1]              = gather_buffer[comm_size - 1];  // max
      part_neighbors[2]              = gather_buffer[median_index];   // median
      PetscReal part_neighbors_ratio = (PetscReal)part_neighbors[1] / (PetscReal)part_neighbors[2];
      PetscCall(PetscPrintf(comm, "    Owned Interface Ranks              : %" PetscInt_FMT ", %" PetscInt_FMT ", %" PetscInt_FMT ", %f\n",
                            part_neighbors[0], part_neighbors[1], part_neighbors[2], part_neighbors_ratio));
    }

    if (!rank) PetscCall(PetscFree(gather_buffer));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}
