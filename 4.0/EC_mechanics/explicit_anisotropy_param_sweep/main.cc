// SPDX-FileCopyrightText: © 2025 PRISMS Center at the University of Michigan
// SPDX-License-Identifier: GNU Lesser General Public Version 2.1

#include "custom_pde.h"
#include "field_registry.h"

#include <prismspf/core/parse_cmd_options.h>
#include <prismspf/core/problem.h>

using namespace prismspf;

int main(int argc, char *argv[]) {
  // Initialize MPI
  prismspf::MPIInitFinalize mpi_init(argc, argv);

  // Parse the command line options (if there are any) to get the name of the
  // input file
  ParseCMDOptions cli_options(argc, argv);
  std::string parameters_filename = cli_options.get_parameters_filename();

  constexpr unsigned int dim = 2;    // TODO change to 3 (original app)
  constexpr unsigned int degree = 2; // TODO change to 1 (original app)

  using Fields = FieldStruct<dim, false>;

  // Build field attributes vector automatically from field_registry
  std::vector<FieldAttributes> fields = Fields::get_field_attributes();

  using Subsets = FieldSubsets<dim, false>;

  SolveBlock constant_block;
  constant_block.id = -1;
  constant_block.solve_type = Constant;
  constant_block.solve_timing = Initialized;
  constant_block.field_indices = collect_indices(
      Fields::psi,
      Subsets::stiffness(),
      Subsets::eigenstrain(),
      Subsets::diffusion()
    );
  SolveBlock c_block;
  c_block.id = 0;
  c_block.solve_type = Explicit;
  c_block.solve_timing = Initialized;
  c_block.field_indices = collect_indices(
      Fields::c,
      Fields::rxn
    );
  c_block.dependencies_rhs = make_dependency_set(
    fields, collect_deps(
      Fields::mu.old_1(),
      Fields::mu.old_1().grad(),
      Fields::c.old_1(),
      Fields::psi,
      Fields::psi.grad(),
      Subsets::diffusion()
    ));

  SolveBlock u_block;
  u_block.id = 1;
  u_block.solve_type = Linear;
  u_block.solve_timing = Uninitialized;
  u_block.field_indices = collect_indices(Fields::u);
  u_block.dependencies_rhs =
    make_dependency_set(fields, collect_deps(
      Fields::c.old_1(),
      Fields::psi,
      Subsets::stiffness(),
      Subsets::eigenstrain()
    ));
  u_block.dependencies_lhs =
    make_dependency_set(fields, collect_deps(
      Fields::u.lhs().grad(),
      Fields::psi,
      Subsets::stiffness()
    ));

  SolveBlock mu_block;
  mu_block.id = 2;
  mu_block.solve_type = Explicit;
  mu_block.solve_timing = Initialized;
  mu_block.field_indices = collect_indices(Fields::mu);
  mu_block.dependencies_rhs =
    make_dependency_set(fields, collect_deps(
      Fields::c,
      Fields::u.grad(),
      Fields::psi,
      Subsets::stiffness(),
      Subsets::eigenstrain()
    ));

  SolveBlock pp_block;
  pp_block.id = 3;
  pp_block.solve_type = Explicit;
  pp_block.solve_timing = PostProcess;
  pp_block.field_indices = collect_indices(Subsets::postprocess());
  pp_block.dependencies_rhs =
  make_dependency_set(fields, collect_deps(
      Fields::c,
      Fields::u.grad(),
      Fields::mu,
      Fields::psi,
      Subsets::stiffness(),
      Subsets::eigenstrain()
    ));

  std::vector<SolveBlock> solve_blocks(
      {constant_block, c_block, u_block, mu_block, pp_block});

  UserInputParameters<dim> user_inputs(parameters_filename, 15);
  PhaseFieldTools<dim> pf_tools;
  CustomPDE<dim, degree, double> pde_operator(user_inputs, pf_tools);
  Problem<dim, degree, double> problem(fields, solve_blocks, user_inputs,
                                       pf_tools, pde_operator);
  problem.solve();

  return 0;
}
