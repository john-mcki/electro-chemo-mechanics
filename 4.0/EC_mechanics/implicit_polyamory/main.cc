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

  // Parse the command line options (if there are any) to get the name of the input
  // file
  ParseCMDOptions cli_options(argc, argv);
  std::string     parameters_filename = cli_options.get_parameters_filename();

  constexpr unsigned int dim    = 2; // TODO change to 3 (original app)
  constexpr unsigned int degree = 2; // TODO change to 1 (original app)

  std::vector<FieldAttributes> fields = {FieldAttributes("u", Vector),
                                         FieldAttributes("mu"),
                                         FieldAttributes("c"),
                                         FieldAttributes("psi")};
  // FieldAttributes("particle_concentration"),
  // FieldAttributes("conf_energy"),
  // FieldAttributes("mech_energy")};

  SolveBlock constant_block;
  constant_block.id            = -1;
  constant_block.solve_type    = Constant;
  constant_block.solve_timing  = Initialized;
  constant_block.field_indices = {3};

  SolveBlock c_block;
  c_block.id               = 0;
  c_block.solve_type       = Newton;
  c_block.solve_timing     = Initialized;
  c_block.field_indices    = {0, 1, 2};
  c_block.dependencies_rhs = make_dependency_set(
    fields,
    {"grad(u)", "mu", "grad(mu)", "old_1(c)", "c", "psi", "grad(psi)"});
  c_block.dependencies_lhs = make_dependency_set(fields,
                                                 {"grad(change(u))",
                                                  "mu",
                                                  "change(mu)",
                                                  "grad(change(mu))",
                                                  "c",
                                                  "change(c)",
                                                  "psi",
                                                  "grad(psi)"});

  /*
  SolveBlock pp_block;
  pp_block.id               = 1;
  pp_block.solve_type       = Explicit;
  pp_block.solve_timing     = PostProcess;
  pp_block.field_indices    = {4, 5, 6};
  pp_block.dependencies_rhs = make_dependency_set(fields, {"grad(u)", "c", "psi"});
  */

  // std::vector<SolveBlock> solve_blocks({constant_block, c_block, pp_block});
  std::vector<SolveBlock> solve_blocks({constant_block, c_block});

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
