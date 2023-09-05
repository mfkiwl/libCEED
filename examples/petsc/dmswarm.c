// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

//                        libCEED + PETSc DMSwarm Example
//
// This example demonstrates a simple usage of libCEED with DMSwarm.
// This example combines elements of PETSc src/impls/dm/swam/tutorials/ex1.c and src/impls/dm/swarm/tests/ex6.c
//
// Build with:
//
//     make dmswarm [PETSC_DIR=</path/to/petsc>] [CEED_DIR=</path/to/libceed>]
//
// Sample runs:
//
//     ./dmswarm
///
//TESTARGS -ceed {ceed_resource} -test -tolerance 1e-3 -dm_plex_dim 3 -dm_plex_box_faces 3,3,3 -dm_plex_box_lower -1.0,-1.0,-1.0 -dm_plex_simplex 0 -dm_plex_hash_location true

/// @file
/// libCEED example using PETSc with DMSwarm
const char help[] = "libCEED example using PETSc with DMSwarm\n";

#include <ceed.h>
#include <petscdmplex.h>
#include <petscdmswarm.h>
#include <petscds.h>
#include <petscfe.h>
#include <petscksp.h>
#include <petsc/private/petscfeimpl.h> /* For interpolation */

#include "include/petscutils.h"

typedef PetscErrorCode (*DMFunc)(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx);

const char DMSwarmField_u[] = "u";

static PetscScalar EvalU(PetscInt dim, const PetscScalar x[]) {
  PetscScalar result = 1, center = 0.1;
  for (PetscInt d = 0; d < dim; d++) {
    result *= tanh(x[d] - center);
    center += 0.1;
  }
  return result;
}

static PetscErrorCode EvalU_proj(PetscInt dim, PetscReal t, const PetscReal x[], PetscInt num_comp, PetscScalar *u, void *ctx) {
  PetscFunctionBeginUser;
  for (PetscInt c = 0; c < num_comp; c++) u[c] = EvalU(dim, x);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscScalar EvaldU(PetscInt dim, const PetscScalar x[], PetscInt direction) {
  PetscScalar result = 1, center = 0.1;
  for (PetscInt d = 0; d < dim; d++) {
    if (d == direction) result *= 1.0 / cosh(x[d] - center) / cosh(x[d] - center);
    else result *= tanh(x[d] - center);
    center += 0.1;
  }
  return result;
}

PetscErrorCode DMSwarmCreateReferenceCoordinates(DM dm_swarm, IS *is_points, Vec *ref_coords) {
  DM                 dm_mesh;
  PetscInt           cell_start, cell_end, dim, num_points_local, *point_cell_numbers;
  PetscScalar       *coords_points_ref;
  const PetscScalar *coords_points_true;

  PetscFunctionBeginUser;
  PetscCall(DMSwarmGetCellDM(dm_swarm, &dm_mesh));

  // Create vector to hold reference coordinates
  {
    Vec true_coords;

    PetscCall(DMSwarmCreateLocalVectorFromField(dm_swarm, DMSwarmPICField_coor, &true_coords));
    PetscCall(VecDuplicate(true_coords, ref_coords));
    PetscCall(DMSwarmDestroyLocalVectorFromField(dm_swarm, DMSwarmPICField_coor, &true_coords));
  }

  // Allocate index set array
  PetscCall(VecGetLocalSize(*ref_coords, &num_points_local));
  PetscCall(DMGetDimension(dm_mesh, &dim));
  num_points_local /= dim;
  PetscCall(PetscMalloc1(num_points_local, &point_cell_numbers));

  // Get reference coordinates for each swarm point wrt the elements in the background mesh
  PetscCall(DMSwarmSortGetAccess(dm_swarm));
  PetscCall(DMPlexGetHeightStratum(dm_mesh, 0, &cell_start, &cell_end));
  PetscCall(DMSwarmGetField(dm_swarm, DMSwarmPICField_coor, NULL, NULL, (void **)&coords_points_true));
  PetscCall(VecGetArray(*ref_coords, &coords_points_ref));
  for (PetscInt cell = cell_start; cell < cell_end; cell++) {
    PetscInt *cell_points;
    PetscInt  num_points_in_cell;
    PetscReal v[3], J[9], invJ[9], detJ, v0ref[3] = {-1.0, -1.0, -1.0};

    PetscCall(DMSwarmSortGetPointsPerCell(dm_swarm, cell, &num_points_in_cell, &cell_points));
    // -- Reference coordinates for swarm points in background mesh element
    PetscCall(DMPlexComputeCellGeometryFEM(dm_mesh, cell, NULL, v, J, invJ, &detJ));
    for (PetscInt p = 0; p < num_points_in_cell; p++) {
      point_cell_numbers[cell_points[p]] = cell;
      CoordinatesRealToRef(dim, dim, v0ref, v, invJ, &coords_points_true[cell_points[p] * dim], &coords_points_ref[cell_points[p] * dim]);
    }

    // -- Cleanup
    PetscCall(PetscFree(cell_points));
  }

  // Cleanup
  PetscCall(DMSwarmRestoreField(dm_swarm, DMSwarmPICField_coor, NULL, NULL, (void **)&coords_points_true));
  PetscCall(VecRestoreArray(*ref_coords, &coords_points_ref));
  PetscCall(DMSwarmSortRestoreAccess(dm_swarm));

  // Create index set
  PetscCall(ISCreateGeneral(PETSC_COMM_SELF, num_points_local, point_cell_numbers, PETSC_OWN_POINTER, is_points));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv) {
  MPI_Comm            comm;
  char                ceed_resource[PETSC_MAX_PATH_LEN] = "/cpu/self";
  PetscBool           test_mode                         = PETSC_FALSE;
  DM                  dm_mesh, dm_swarm;
  Vec                 U_mesh;
  PetscInt            dim = 3, num_comp = 1, num_points = 40, geometry_order = 1, mesh_order = 3, q_extra = 3;
  Ceed                ceed;
  CeedBasis           basis_mesh_u;
  CeedElemRestriction restriction_mesh_u;
  PetscScalar         tolerance = 1E-3;

  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  comm = PETSC_COMM_WORLD;

  // Read command line options
  PetscOptionsBegin(comm, NULL, "libCEED example using PETSc with DMSwarm", NULL);

  PetscCall(PetscOptionsBool("-test", "Testing mode (do not print unless error is large)", NULL, test_mode, &test_mode, NULL));
  PetscCall(PetscOptionsInt("-order", "Order of mesh solution space", NULL, mesh_order, &mesh_order, NULL));
  PetscCall(PetscOptionsInt("-q_extra", "Number of extra quadrature points", NULL, q_extra, &q_extra, NULL));
  PetscCall(PetscOptionsInt("-points", "Number of swarm points", NULL, num_points, &num_points, NULL));
  PetscCall(PetscOptionsScalar("-tolerance", "Absolute tolerance for swarm point values", NULL, tolerance, &tolerance, NULL));
  PetscCall(PetscOptionsString("-ceed", "CEED resource specifier", NULL, ceed_resource, ceed_resource, sizeof(ceed_resource), NULL));

  PetscOptionsEnd();

  // Initialize libCEED
  CeedInit(ceed_resource, &ceed);

  // Create background mesh
  PetscCall(DMCreate(comm, &dm_mesh));
  PetscCall(DMSetType(dm_mesh, DMPLEX));
  PetscCall(DMSetFromOptions(dm_mesh));

  // -- Check for tensor product mesh
  {
    PetscBool is_simplex;

    PetscCall(DMPlexIsSimplex(dm_mesh, &is_simplex));
    PetscCheck(!is_simplex, comm, PETSC_ERR_USER, "Only tensor-product background meshes supported");
  }

  // -- Mesh FE space
  PetscCall(DMGetDimension(dm_mesh, &dim));
  {
    PetscFE fe;

    PetscCall(DMGetDimension(dm_mesh, &dim));
    PetscCall(PetscFECreateLagrange(comm, dim, num_comp, PETSC_FALSE, mesh_order, mesh_order + q_extra, &fe));
    PetscCall(DMAddField(dm_mesh, NULL, (PetscObject)fe));
    PetscCall(PetscFEDestroy(&fe));
  }
  PetscCall(DMCreateDS(dm_mesh));

  // -- Coordinate FE space
  {
    PetscFE fe_coord;

    PetscCall(PetscFECreateLagrange(comm, dim, dim, PETSC_FALSE, geometry_order, mesh_order + q_extra, &fe_coord));
    PetscCall(DMProjectCoordinates(dm_mesh, fe_coord));
    PetscCall(PetscFEDestroy(&fe_coord));
  }

  // -- Set tensor permutation
  {
    DM dm_coord;

    PetscCall(DMGetCoordinateDM(dm_mesh, &dm_coord));
    PetscCall(DMPlexSetClosurePermutationTensor(dm_mesh, PETSC_DETERMINE, NULL));
    PetscCall(DMPlexSetClosurePermutationTensor(dm_coord, PETSC_DETERMINE, NULL));
  }

  // -- Final background mesh
  PetscCall(PetscObjectSetName((PetscObject)dm_mesh, "Background Mesh"));
  PetscCall(DMViewFromOptions(dm_mesh, NULL, "-dm_mesh_view"));

  // -- libCEED objects from background mesh
  {
    BPData bp_data = {.q_mode = CEED_GAUSS};

    PetscCall(CreateBasisFromPlex(ceed, dm_mesh, NULL, 0, 0, 0, bp_data, &basis_mesh_u));
    PetscCall(CreateRestrictionFromPlex(ceed, dm_mesh, 0, NULL, 0, &restriction_mesh_u));
  }

  // Create particle swarm
  PetscCall(DMCreate(comm, &dm_swarm));
  PetscCall(DMSetType(dm_swarm, DMSWARM));
  PetscCall(DMSetDimension(dm_swarm, dim));
  PetscCall(DMSwarmSetType(dm_swarm, DMSWARM_PIC));
  PetscCall(DMSwarmSetCellDM(dm_swarm, dm_mesh));

  // -- Swarm field
  PetscCall(DMSwarmRegisterPetscDatatypeField(dm_swarm, DMSwarmField_u, num_comp, PETSC_SCALAR));
  PetscCall(DMSwarmFinalizeFieldRegister(dm_swarm));
  PetscCall(DMSwarmSetLocalSizes(dm_swarm, num_points, 0));
  PetscCall(DMSetFromOptions(dm_swarm));

  // -- Set swarm point locations
  {
    PetscScalar *point_coords;

    PetscCall(DMSwarmGetField(dm_swarm, DMSwarmPICField_coor, NULL, NULL, (void **)&point_coords));
    for (PetscInt p = 0; p < num_points; p++) {
      point_coords[p * dim + 0] = -PetscCosReal((PetscReal)(p + 1) / (PetscReal)(num_points + 1) * PETSC_PI);
      if (dim > 1) point_coords[p * dim + 1] = -PetscSinReal((PetscReal)(p + 1) / (PetscReal)(num_points + 1) * PETSC_PI);
      if (dim > 2) point_coords[p * dim + 2] = PetscSinReal((PetscReal)(p + 1) / (PetscReal)(num_points + 1) * PETSC_PI);
    }
    PetscCall(DMSwarmRestoreField(dm_swarm, DMSwarmPICField_coor, NULL, NULL, (void **)&point_coords));
  }
  // -- Set uniform point locations in each cell
  // {
  //   PetscInt    npoints[]       = {5, 5, 5};
  //   PetscScalar points[125 * 3] = {};
  //   for (PetscInt ix = 0; ix < npoints[0]; ix++) {
  //     for (PetscInt iy = 0; iy < npoints[1]; iy++) {
  //       for (PetscInt iz = 0; iz < npoints[2]; iz++) {
  //         PetscInt p = (ix * npoints[1] + iy) * npoints[2] + iz;
  //         points[p * dim + 0] = 2.0 * (PetscReal)(ix + 1) / (PetscReal)(npoints[0] + 1) - 1;
  //         points[p * dim + 1] = 2.0 * (PetscReal)(iy + 1) / (PetscReal)(npoints[1] + 1) - 1;
  //         points[p * dim + 2] = 2.0 * (PetscReal)(iz + 1) / (PetscReal)(npoints[2] + 1) - 1;
  //       }
  //     }
  //   }
  //   PetscCall(DMSwarmSetPointCoordinatesCellwise(dm_swarm, npoints[0] * npoints[1] * npoints[2], points));
  // }
  PetscCall(DMSwarmMigrate(dm_swarm, PETSC_TRUE));

  // -- Final particle swarm
  PetscCall(PetscObjectSetName((PetscObject)dm_swarm, "Particle Swarm"));
  PetscCall(DMViewFromOptions(dm_swarm, NULL, "-dm_swarm_view"));

  // Set field values on background mesh
  PetscCall(DMCreateGlobalVector(dm_mesh, &U_mesh));
  {
    DMFunc fn[1] = {EvalU_proj};
    PetscCall(DMProjectFunction(dm_mesh, 0.0, fn, NULL, INSERT_VALUES, U_mesh));
  }

  // Visualize background mesh
  PetscCall(VecViewFromOptions(U_mesh, NULL, "-u_mesh_view"));

  // Interpolate from mesh to points via PETSc
  {
    PetscDS            ds;
    PetscFE            fe;
    PetscFEGeom        fe_geometry;
    PetscQuadrature    quadrature;
    Vec                U_loc;
    PetscInt           cell_start, cell_end;
    PetscScalar       *u_all_points_array;
    const PetscScalar *coords_points;
    const PetscReal    v0ref[3] = {-1.0, -1.0, -1.0};
    PetscBool          success  = PETSC_TRUE;
    PetscSection       section_u_mesh_loc;

    {
      PetscSection section_u_mesh_loc_clperm;

      PetscCall(DMGetLocalSection(dm_mesh, &section_u_mesh_loc_clperm));
      PetscCall(PetscSectionClone(section_u_mesh_loc_clperm, &section_u_mesh_loc));
      PetscCall(PetscSectionResetClosurePermutation(section_u_mesh_loc));
    }

    PetscCall(DMGetLocalVector(dm_mesh, &U_loc));
    PetscCall(VecZeroEntries(U_loc));
    PetscCall(DMGlobalToLocal(dm_mesh, U_mesh, INSERT_VALUES, U_loc));

    PetscCall(DMGetDS(dm_mesh, &ds));
    PetscCall(PetscDSGetDiscretization(ds, 0, (PetscObject *)&fe));
    PetscCall(DMSwarmSortGetAccess(dm_swarm));
    PetscCall(DMPlexGetHeightStratum(dm_mesh, 0, &cell_start, &cell_end));
    PetscCall(DMSwarmGetField(dm_swarm, DMSwarmPICField_coor, NULL, NULL, (void **)&coords_points));
    PetscCall(DMSwarmGetField(dm_swarm, DMSwarmField_u, NULL, NULL, (void **)&u_all_points_array));

    // -- Interpolate values to each swarm point, one element in the background mesh at a time
    for (PetscInt cell = cell_start; cell < cell_end; cell++) {
      PetscTabulation tabulation;
      PetscScalar    *u_cell = NULL, *coords_points_true, *coords_points_ref;
      PetscReal       v[dim], J[dim * dim], invJ[dim * dim], detJ;
      PetscInt       *points;
      PetscInt        num_points_in_cell;

      PetscCall(DMSwarmSortGetPointsPerCell(dm_swarm, cell, &num_points_in_cell, &points));
      PetscCall(DMGetWorkArray(dm_mesh, num_points_in_cell * dim, MPIU_REAL, &coords_points_true));
      PetscCall(DMGetWorkArray(dm_mesh, num_points_in_cell * dim, MPIU_REAL, &coords_points_ref));
      // ---- Reference coordinates for swarm points in background mesh element
      for (PetscInt p = 0; p < num_points_in_cell; p++) {
        for (PetscInt d = 0; d < dim; d++) coords_points_true[p * dim + d] = coords_points[points[p] * dim + d];
      }
      PetscCall(DMPlexComputeCellGeometryFEM(dm_mesh, cell, NULL, v, J, invJ, &detJ));
      for (PetscInt p = 0; p < num_points_in_cell; p++) {
        CoordinatesRealToRef(dim, dim, v0ref, v, invJ, &coords_points_true[p * dim], &coords_points_ref[p * dim]);
      }
      // ---- Interpolate values from current element in background mesh to swarm points
      PetscCall(PetscFECreateTabulation(fe, 1, num_points_in_cell, coords_points_ref, 1, &tabulation));
      PetscCall(DMPlexVecGetClosure(dm_mesh, section_u_mesh_loc, U_loc, cell, NULL, &u_cell));
      PetscCall(PetscFEGetQuadrature(fe, &quadrature));
      PetscCall(PetscFECreateCellGeometry(fe, quadrature, &fe_geometry));
      for (PetscInt p = 0; p < num_points_in_cell; p++) {
        PetscScalar x[dim], u_true = 0.0;

        PetscCall(PetscFEInterpolateAtPoints_Static(fe, tabulation, u_cell, &fe_geometry, p, &u_all_points_array[points[p]]));
        for (PetscInt d = 0; d < dim; d++) x[d] = coords_points_true[p * dim + d];
        u_true = EvalU(dim, x);
        if (PetscAbs(u_all_points_array[points[p]] - u_true) > tolerance) {
          success = PETSC_FALSE;
          PetscPrintf(comm, "Incorrect interpolated value from PETSc, cell %" PetscInt_FMT " point %" PetscInt_FMT ", found %f expected %f\n", cell,
                      p, u_all_points_array[points[p]], u_true);
        }
      }

      // ---- Cleanup
      PetscCall(PetscFEDestroyCellGeometry(fe, &fe_geometry));
      PetscCall(DMPlexVecRestoreClosure(dm_mesh, section_u_mesh_loc, U_loc, cell, NULL, &u_cell));
      PetscCall(DMRestoreWorkArray(dm_mesh, num_points_in_cell * dim, MPIU_REAL, &coords_points_true));
      PetscCall(DMRestoreWorkArray(dm_mesh, num_points_in_cell * dim, MPIU_REAL, &coords_points_ref));
      PetscCall(PetscTabulationDestroy(&tabulation));
      PetscCall(PetscFree(points));
    }

    // -- Cleanup
    PetscCall(DMSwarmRestoreField(dm_swarm, DMSwarmPICField_coor, NULL, NULL, (void **)&coords_points));
    PetscCall(DMSwarmRestoreField(dm_swarm, "u", NULL, NULL, (void **)&u_all_points_array));
    PetscCall(DMSwarmSortRestoreAccess(dm_swarm));
    PetscCall(DMRestoreLocalVector(dm_mesh, &U_loc));
    PetscCall(PetscSectionDestroy(&section_u_mesh_loc));
    PetscCheck(!test_mode || success, comm, PETSC_ERR_USER, "Petsc interpolation to swarm points not within tolerance");
  }

  if (!test_mode) PetscCall(DMSwarmViewXDMF(dm_swarm, "swarm_petsc.xmf"));

  // Interpolate from mesh to points via libCEED
  {
    IS                 is_points;
    Vec                U_loc, ref_coords;
    PetscInt           num_points_local;
    const PetscInt    *all_points;
    PetscInt           cell_start, cell_end;
    PetscScalar       *u_all_points_array;
    const PetscScalar *coords_points_ref;
    CeedVector         u_evec, u_cell;
    PetscBool          success_ceed = PETSC_TRUE, success_cmp = PETSC_TRUE;

    // -- Get mesh values
    PetscCall(DMGetLocalVector(dm_mesh, &U_loc));
    PetscCall(VecZeroEntries(U_loc));
    PetscCall(DMGlobalToLocal(dm_mesh, U_mesh, INSERT_VALUES, U_loc));
    {
      const PetscScalar *u_array;
      CeedVector         u_lvec;

      PetscCall(VecGetArrayRead(U_loc, &u_array));
      CeedElemRestrictionCreateVector(restriction_mesh_u, &u_lvec, &u_evec);
      CeedVectorSetArray(u_lvec, CEED_MEM_HOST, CEED_USE_POINTER, (CeedScalar *)u_array);
      CeedElemRestrictionApply(restriction_mesh_u, CEED_NOTRANSPOSE, u_lvec, u_evec, CEED_REQUEST_IMMEDIATE);
      CeedVectorTakeArray(u_lvec, CEED_MEM_HOST, (CeedScalar **)&u_array);
      PetscCall(VecRestoreArrayRead(U_loc, &u_array));
      CeedVectorDestroy(&u_lvec);
    }
    {
      CeedInt elem_size;

      CeedElemRestrictionGetElementSize(restriction_mesh_u, &elem_size);
      CeedVectorCreate(ceed, elem_size, &u_cell);
    }

    // -- Get swarm values
    PetscCall(DMPlexGetHeightStratum(dm_mesh, 0, &cell_start, &cell_end));
    PetscCall(DMSwarmGetField(dm_swarm, DMSwarmField_u, NULL, NULL, (void **)&u_all_points_array));
    PetscCall(DMSwarmCreateReferenceCoordinates(dm_swarm, &is_points, &ref_coords));
    PetscCall(ISGetSize(is_points, &num_points_local));
    PetscCall(ISGetIndices(is_points, &all_points));
    PetscCall(VecGetArrayRead(ref_coords, &coords_points_ref));

    // -- Interpolate values to each swarm point, one element in the background mesh at a time
    for (PetscInt cell = cell_start; cell < cell_end; cell++) {
      CeedVector u_points, x_points;
      PetscInt   num_points_in_cell = 0;

      for (PetscInt i = 0; i < num_points_local; i++) {
        if (all_points[i] == cell) num_points_in_cell++;
      }
      if (!num_points_in_cell) continue;
      CeedVectorCreate(ceed, num_points_in_cell, &u_points);
      CeedVectorCreate(ceed, num_points_in_cell * dim, &x_points);

      // ---- Reference coordinates for swarm points in background mesh element
      {
        PetscInt    p = 0;
        CeedScalar *x;

        CeedVectorGetArrayWrite(x_points, CEED_MEM_HOST, &x);
        for (PetscInt i = 0; i < num_points_local; i++) {
          if (all_points[i] == cell) {
            for (PetscInt d = 0; d < dim; d++) x[p * dim + d] = coords_points_ref[i * dim + d];
            p++;
          }
        }
        CeedVectorRestoreArray(x_points, &x);
      }

      // ---- Interpolate values from current element in background mesh to swarm points
      {
        const CeedScalar *u_evec_array, *u_cell_array;
        CeedInt           P;

        CeedBasisGetNumNodes(basis_mesh_u, &P);
        CeedVectorGetArrayRead(u_evec, CEED_MEM_HOST, &u_evec_array);
        u_cell_array = &u_evec_array[cell * P];
        CeedVectorSetArray(u_cell, CEED_MEM_HOST, CEED_USE_POINTER, (CeedScalar *)u_cell_array);
        CeedBasisApplyAtPoints(basis_mesh_u, num_points_in_cell, CEED_NOTRANSPOSE, CEED_EVAL_INTERP, x_points, u_cell, u_points);

        CeedVectorTakeArray(u_cell, CEED_MEM_HOST, (CeedScalar **)&u_cell_array);
        CeedVectorRestoreArrayRead(u_evec, &u_evec_array);
      }

      // ---- Verify
      {
        const CeedScalar *u_points_array, *coords_points;
        CeedInt           p = 0;

        CeedVectorGetArrayRead(u_points, CEED_MEM_HOST, &u_points_array);
        PetscCall(DMSwarmGetField(dm_swarm, DMSwarmPICField_coor, NULL, NULL, (void **)&coords_points));
        for (PetscInt i = 0; i < num_points_local; i++) {
          if (all_points[i] == cell) {
            PetscScalar x[dim], u_true = 0.0;

            for (PetscInt d = 0; d < dim; d++) x[d] = coords_points[i * dim + d];
            u_true = EvalU(dim, x);
            if (PetscAbs(u_points_array[p] - u_true) > tolerance) {
              success_ceed = PETSC_FALSE;
              PetscPrintf(comm, "Incorrect interpolated value from libCEED, cell %" PetscInt_FMT " point %" PetscInt_FMT ", found %f expected %f\n",
                          cell, i, u_points_array[p], u_true);
            }
            if (PetscAbs(u_points_array[p] - u_all_points_array[i]) > tolerance) {
              success_cmp = PETSC_FALSE;
              PetscPrintf(comm,
                          "Significant difference between libCEED and PETSc, cell %" PetscInt_FMT " point %" PetscInt_FMT ", found %f expected %f\n",
                          cell, i, u_points_array[p], u_all_points_array[i]);
            }
            p++;
          }
        }
        PetscCall(DMSwarmRestoreField(dm_swarm, DMSwarmPICField_coor, NULL, NULL, (void **)&coords_points));
        CeedVectorRestoreArrayRead(u_points, &u_points_array);
      }

      {
        const CeedScalar *u_points_array;

        CeedVectorGetArrayRead(u_points, CEED_MEM_HOST, &u_points_array);
        for (PetscInt i = 0, p = 0; i < num_points_local; i++) {
          if (all_points[i] == cell) u_all_points_array[i] = u_points_array[p++];
        }
        CeedVectorRestoreArrayRead(u_points, &u_points_array);
      }

      // ---- Cleanup
      CeedVectorDestroy(&u_points);
      CeedVectorDestroy(&x_points);
    }

    PetscCheck(!test_mode || success_ceed, comm, PETSC_ERR_USER, "libCEED interpolation to swarm points not within tolerance");
    PetscCheck(!test_mode || success_cmp, comm, PETSC_ERR_USER, "Significant difference between libCEED and PETSc interpolation to swarm points");

    // -- Cleanup
    PetscCall(DMSwarmRestoreField(dm_swarm, DMSwarmField_u, NULL, NULL, (void **)&u_all_points_array));
    PetscCall(DMRestoreLocalVector(dm_mesh, &U_loc));
    PetscCall(ISRestoreIndices(is_points, &all_points));
    PetscCall(ISDestroy(&is_points));
    PetscCall(VecRestoreArrayRead(ref_coords, &coords_points_ref));
    PetscCall(VecDestroy(&ref_coords));
    CeedVectorDestroy(&u_evec);
    CeedVectorDestroy(&u_cell);
  }

  // View result
  if (!test_mode) PetscCall(DMSwarmViewXDMF(dm_swarm, "swarm_ceed.xmf"));

  // Cleanup
  CeedDestroy(&ceed);
  CeedBasisDestroy(&basis_mesh_u);
  CeedElemRestrictionDestroy(&restriction_mesh_u);
  PetscCall(DMDestroy(&dm_swarm));
  PetscCall(DMDestroy(&dm_mesh));
  PetscCall(VecDestroy(&U_mesh));
  return PetscFinalize();
}
