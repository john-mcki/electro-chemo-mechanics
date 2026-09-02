// SPDX-FileCopyrightText: © 2025 PRISMS Center at the University of Michigan
// SPDX-License-Identifier: GNU Lesser General Public Version 2.1

#include "custom_pde.h"

#include <prismspf/core/parse_cmd_options.h>
#include <prismspf/core/problem.h>

using namespace prismspf;

int
main(int argc, char *argv[])
{
  // Initialize MPI
  prismspf::MPIInitFinalize mpi_init(argc, argv);

  // Parse the command line options (if there are any) to get the name of the
  // input file
  ParseCMDOptions cli_options(argc, argv);
  std::string     parameters_filename = cli_options.get_parameters_filename();

  constexpr unsigned int dim    = 2; // TODO change to 3 (original app)
  constexpr unsigned int degree = 1; // TODO change to 1 (original app)

  std::vector<FieldAttributes> fields = {
    FieldAttributes("c"),
    FieldAttributes("u", Vector),
    FieldAttributes("mu"),
    FieldAttributes("psi"),
    FieldAttributes("particle_concentration"),
    FieldAttributes("overpotential"),
    FieldAttributes("elastic_potential"),
    FieldAttributes("sigma_x"),
    FieldAttributes("sigma_y"),
    FieldAttributes("sigma_xy"),
    FieldAttributes("rxn"),
    FieldAttributes("del_phi")
    // FieldAttributes("Cel1", Vector),
    // FieldAttributes("Cel2", Vector),
    // FieldAttributes("Cel3", Vector),
    // FieldAttributes("eig1", Vector),
    // FieldAttributes("eig2"),
    // FieldAttributes("D1", Vector),
    // FieldAttributes("D2")
  };

  SolveBlock constant_block;
  constant_block.id           = -1;
  constant_block.solve_type   = Constant;
  constant_block.solve_timing = Initialized;
  // constant_block.field_indices = {3, 16, 17, 18, 19, 20, 21, 22};
  constant_block.field_indices = {3};

  SolveBlock c_block;
  c_block.id            = 0;
  c_block.solve_type    = Explicit;
  c_block.solve_timing  = Initialized;
  c_block.field_indices = {0, 10};
  c_block.dependencies_rhs =
    make_dependency_set(fields,
                        {"old_1(mu)", "grad(old_1(mu))", "old_1(c)", "psi", "grad(psi)"});

  SolveBlock u_block;
  u_block.id               = 1;
  u_block.solve_type       = Linear;
  u_block.solve_timing     = Uninitialized;
  u_block.field_indices    = {1};
  u_block.dependencies_rhs = make_dependency_set(fields, {"c", "psi"});
  u_block.dependencies_lhs = make_dependency_set(fields, {"grad(lhs(u))", "psi"});

  SolveBlock mu_block;
  mu_block.id               = 2;
  mu_block.solve_type       = Explicit;
  mu_block.solve_timing     = Initialized;
  mu_block.field_indices    = {2};
  mu_block.dependencies_rhs = make_dependency_set(fields, {"grad(u)", "c", "psi"});

  SolveBlock pp_block;
  pp_block.id               = 3;
  pp_block.solve_type       = Explicit;
  pp_block.solve_timing     = PostProcess;
  pp_block.field_indices    = {4, 5, 6, 7, 8, 9, 11};
  pp_block.dependencies_rhs = make_dependency_set(fields,
                                                  {
                                                    "c",
                                                    "old_1(c)",
                                                    "grad(u)",
                                                    "mu",
                                                    "grad(mu)",
                                                    "psi",
                                                    "grad(psi)"
                                                    //"Cel1",
                                                    //"Cel2",
                                                    //"Cel3",
                                                    //"eig1",
                                                    //"eig2",
                                                    //"D1",
                                                    //"D2"
                                                  });

  std::vector<SolveBlock> solve_blocks(
    {constant_block, c_block, u_block, mu_block, pp_block});

  UserInputParameters<dim>       user_inputs(parameters_filename);
  PhaseFieldTools<dim>           pf_tools;
  CustomPDE<dim, degree, double> pde_operator(user_inputs, pf_tools);
  Problem<dim, degree, double>   problem(fields,
                                         solve_blocks,
                                         user_inputs,
                                         pf_tools,
                                         pde_operator);
  problem.solve();

  return 0;
}
