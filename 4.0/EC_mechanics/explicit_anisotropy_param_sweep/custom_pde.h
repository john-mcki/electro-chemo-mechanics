// SPDX-FileCopyrightText: © 2025 PRISMS Center at the University of Michigan
// SPDX-License-Identifier: GNU Lesser General Public Version 2.1

#include <prismspf/core/pde_operator_base.h>

#include <prismspf/utilities/logger.h>
#include <prismspf/utilities/mechanics.h>

#include "field_registry.h"
#include "ocv.h"
#include "tensor_helper.h"

PRISMS_PF_BEGIN_NAMESPACE

template <unsigned int dim, unsigned int degree, typename number>
class CustomPDE : public PDEOperatorBase<dim, degree, number>
{
public:
  using ScalarValue = dealii::VectorizedArray<number>;
  using ScalarGrad  = dealii::Tensor<1, dim, ScalarValue>;
  using ScalarHess  = dealii::Tensor<2, dim, ScalarValue>;
  using VectorValue = dealii::Tensor<1, dim, ScalarValue>;
  using VectorGrad  = dealii::Tensor<2, dim, ScalarValue>;
  using VectorHess  = dealii::Tensor<3, dim, ScalarValue>;
  using PDEOperatorBase<dim, degree, number>::get_user_inputs;
  using PDEOperatorBase<dim, degree, number>::get_pf_tools;
  using Fields       = FieldStruct<dim, false>;
  using Subsets      = FieldSubsets<dim, false>;
  using TensorHelper = TensorHelper<dim, ScalarValue>;

  /**
   * @brief Constructor.
   */
  CustomPDE(const UserInputParameters<dim> &_user_inputs, PhaseFieldTools<dim> &_pf_tools)
    : PDEOperatorBase<dim, degree, number>(_user_inputs, _pf_tools)
    , c0(get_user_inputs().user_constants.get_double("c0"))
    , c_ref(get_user_inputs().user_constants.get_double("c_ref"))
    , offset(get_user_inputs().user_constants.get_double("offset"))
    , diffusivity(get_user_inputs().user_constants.get_double("diffusivity"))
    , diff_scale(get_user_inputs().user_constants.get_double("diff_scale"))
    , eig_scale(get_user_inputs().user_constants.get_double("eig_scale"))
    , i_0(get_user_inputs().user_constants.get_double("i_0"))
    , del_phi(get_user_inputs().user_constants.get_double("del_phi"))
    , vegard(get_user_inputs().user_constants.get_double("vegard"))
    , site_vol(get_user_inputs().user_constants.get_double("site_vol"))
    , mol_vol(get_user_inputs().user_constants.get_double("mol_vol"))
    , RT(get_user_inputs().user_constants.get_double("RT"))
    , F(get_user_inputs().user_constants.get_double("F"))
    , stress_scale(get_user_inputs().user_constants.get_double("stress_scale"))
    , V_ref(get_user_inputs().user_constants.get_double("V_ref"))
    , i_target(get_user_inputs().user_constants.get_double("i_target"))
    , V_step(get_user_inputs().user_constants.get_double("V_step"))
    , V_max(get_user_inputs().user_constants.get_double("V_max"))
    , V_min(get_user_inputs().user_constants.get_double("V_min"))
    , stiffness_const(get_user_inputs().user_constants.get_elasticity_tensor("stiffness"))
  {}

private:
  // Chebyshev21Ocv<ScalarValue> ocv_model_;
  Chebyshev21Ocv<ScalarValue> ocv_model_;

  // PiecewiseCubicSplineOcv<ScalarValue> ocv_model_;

  void
  post_solve_block([[maybe_unused]] SolveContext<dim, degree, number> &solve_context,
                   [[maybe_unused]] unsigned int                       solver_id) override
  {
    if (solver_id == 2)
      {
        const auto &psi =
          solve_context.get_solution_indexer().get_solution_vector(Fields::psi.index);
        const auto &rxn =
          solve_context.get_solution_indexer().get_solution_vector(Fields::rxn.index);
        number integrated =
          rxn * solve_context.get_invm_manager().get_jxw(TensorRank::Scalar);
        number dt            = solve_context.get_simulation_timer().get_timestep();
        number total_current = integrated / (mol_vol) *F *
                               1.0e-6; // conversion of site fraction to total current
                                       // density in a one micron thick slice
        // std::cout << "Integrated concentration: " << total_current <<
        // std::endl;
        number increment = solve_context.get_simulation_timer().get_increment();
        bool   is_output_increment =
          solve_context.get_user_inputs().output_parameters.should_output(increment);
        if (is_output_increment == true)
          {
            LogStream(0) << "Total Current is:" << total_current << "\n";
            LogStream(0) << "Cell potential is:" << del_phi << "\n";
          }

        number i_proportion = (i_target - total_current) / i_target;
        number V_inc =
          V_step * dt * i_proportion; // voltage increment based on current density
        del_phi = del_phi + V_inc;
        del_phi = std::max(std::min(del_phi, V_max), V_min);
      }
  }

  void
  set_initial_condition([[maybe_unused]] const unsigned int       &index,
                        [[maybe_unused]] const unsigned int       &component,
                        [[maybe_unused]] const dealii::Point<dim> &point,
                        [[maybe_unused]] number                   &scalar_value,
                        [[maybe_unused]] number &vector_component_value) const override
  {
    const dealii::Tensor<1, dim> &mesh_size =
      get_user_inputs().spatial_discretization.rectangular_mesh.size;
    dealii::Point<dim> center(mesh_size / 2.0);
    double             rad  = mesh_size[0] * (2.0 / 5.0);
    double             sdf  = ((point - center).norm_square() - rad * rad) / (2.0 * rad);
    double domain_parameter = 0.5 * ((1.0 + offset) - (1.0 - offset) * std::tanh(sdf));
    if (index == Fields::c.index)
      scalar_value = c0;
    if (index == Fields::mu.index) // mu
      scalar_value = -F / RT * ocv_model_.eval_U_ocv(ScalarValue(c0))[0];
    // if (index == Fields::psi.index) scalar_value = domain_parameter;
    // if (index == Fields::D1.index) vector_component_value = 1e-4;
    // if (index == Fields::D2.index) scalar_value = 0.0;
    // if (index == Fields::Cel1.index)  vector_component_value = 70.0;
    // if ((index == Fields::Cel2.index) && (component == 0))
    // vector_component_value = 20.0; if ((index == Fields::Cel2.index) &&
    // (component == 1)) vector_component_value = 30.0; if (index ==
    // Fields::Cel3.index) vector_component_value = 0.0; if (index ==
    // Fields::eig1.index) vector_component_value = 0.01; if (index ==
    // Fields::eig2.index) scalar_value = 0.0;
  }

  void
  set_dirichlet([[maybe_unused]] const unsigned int       &index,
                [[maybe_unused]] const unsigned int       &boundary_id,
                [[maybe_unused]] const unsigned int       &component,
                [[maybe_unused]] const dealii::Point<dim> &point,
                [[maybe_unused]] const SimulationTimer    &sim_timer,
                [[maybe_unused]] number                   &scalar_value,
                [[maybe_unused]] number &vector_component_value) const override
  {
    scalar_value           = 0.0;
    vector_component_value = 0.0;
  }

  void
  compute_rhs([[maybe_unused]] FieldContainer<dim, degree, number> &variable_list,
              [[maybe_unused]] const SimulationTimer               &sim_timer,
              [[maybe_unused]] unsigned int solve_block_id) const override
  {
    using std::exp;
    using std::log;
    using std::pow;
    using std::sqrt;
    if (solve_block_id == 0) // c
      {
        // Calling variables
        /*
        ScalarValue psi =
          variable_list.template get_value<Scalar, Current>(Fields::psi.index);
        ScalarValue solve_limit =
          0.010; // Value of psi, beyond which no calculations occur
        if (double(psi) <= double(solve_limit))
          {
          }
          */
        ScalarValue c_val =
          variable_list.template get_value<Scalar, OldOne>(Fields::c.index);
        ScalarValue mu_val =
          variable_list.template get_value<Scalar, OldOne>(Fields::mu.index);
        ScalarGrad mu_grad =
          variable_list.template get_gradient<Scalar, OldOne>(Fields::mu.index);
        ScalarValue psi =
          variable_list.template get_value<Scalar, Current>(Fields::psi.index);
        ScalarGrad psi_grad =
          variable_list.template get_gradient<Scalar, Current>(Fields::psi.index);
        ScalarValue psi_grad_mag = psi_grad.norm();
        ScalarValue dt           = sim_timer.get_timestep();

        // Functions
        // ScalarValue mobility = (diffusivity * c_val) / RT;
        ScalarValue mobility_factor = -(RT / F) * (1.0 / ocv_model_.eval_dU_ocv(c_val));
        ScalarValue app_pot_energy  = F * del_phi;
        ScalarValue eta             = app_pot_energy + RT * mu_val;
        // Diffusion and Reaction functions
        ScalarValue react = -2.0 * (i_0 / F) * std::sinh(eta / (2.0 * RT));
        VectorGrad  D =
          diff_scale *
          TensorHelper::extract_rank2_symm(variable_list, Subsets::diffusion());
        VectorValue flux        = D * mu_grad;
        ScalarValue func_c      = (psi_grad / psi) * mobility_factor * RT * flux;
        ScalarGrad  func_c_grad = -mobility_factor * RT * flux;

        // Forward Euler time stepping
        ScalarValue rxn   = psi_grad_mag * react;
        ScalarValue eq_c  = c_val + dt * (func_c + rxn / psi);
        ScalarGrad  eqx_c = dt * func_c_grad;
        variable_list.set_value_term(Fields::rxn.index, rxn);
        variable_list.set_value_term(Fields::c.index, eq_c);
        variable_list.set_gradient_term(Fields::c.index, eqx_c);
      }
    if (solve_block_id == 1) // u
      {
        ScalarValue c_val =
          variable_list.template get_value<Scalar, OldOne>(Fields::c.index);
        ScalarValue psi =
          variable_list.template get_value<Scalar, Current>(Fields::psi.index);
        VectorGrad eigenstrain0 =
          eig_scale *
          TensorHelper::extract_rank2_symm(variable_list, Subsets::eigenstrain());
        VectorGrad eigenstrain = (c_val - c_ref) * eigenstrain0;
        dealii::Tensor<2, Mechanics::voigt_tensor_size<dim>, ScalarValue> stiffness =
          TensorHelper::extract_stiffness(variable_list, Subsets::stiffness());
        VectorGrad stress;
        Mechanics::compute_stress<dim, ScalarValue>(stiffness, psi * eigenstrain, stress);
        variable_list.set_gradient_term(Fields::u.index, stress);
      }
    if (solve_block_id == 2) // mu
      {
        ScalarValue c_val =
          variable_list.template get_value<Scalar, Current>(Fields::c.index);
        VectorGrad ux =
          variable_list.template get_symmetric_gradient<Vector, Current>(Fields::u.index);
        ScalarValue psi =
          variable_list.template get_value<Scalar, Current>(Fields::psi.index);
        VectorGrad eigenstrain0 =
          eig_scale *
          TensorHelper::extract_rank2_symm(variable_list, Subsets::eigenstrain());
        VectorGrad eigenstrain = (c_val - c_ref) * eigenstrain0;
        VectorGrad stress;
        dealii::Tensor<2, Mechanics::voigt_tensor_size<dim>, ScalarValue> stiffness =
          TensorHelper::extract_stiffness(variable_list, Subsets::stiffness());
        Mechanics::compute_stress<dim, ScalarValue>(stiffness,
                                                    psi * (ux - eigenstrain),
                                                    stress);
        ScalarValue elastic_diffpot =
          dealii::scalar_product<2, dim, ScalarValue>(stress, eigenstrain0);
        elastic_diffpot *= site_vol * stress_scale;
        ScalarValue U_ocv  = ocv_model_.eval_U_ocv(c_val);
        ScalarValue mu_val = (-(F / RT) * U_ocv) - elastic_diffpot / RT;
        variable_list.set_value_term(Fields::mu.index, mu_val);
      }
    if (solve_block_id == 3) // post-processing
      {
        ScalarValue c_val =
          variable_list.template get_value<Scalar, Current>(Fields::c.index);
        VectorGrad ux =
          variable_list.template get_symmetric_gradient<Vector, Current>(Fields::u.index);
        ScalarValue psi =
          variable_list.template get_value<Scalar, Current>(Fields::psi.index);
        ScalarValue mu_val =
          variable_list.template get_value<Scalar, Current>(Fields::mu.index);

        // pp fields directly from solution fields
        variable_list.set_value_term(Fields::particle_concentration.index, psi * c_val);
        variable_list.set_value_term(Fields::overpotential.index,
                                     del_phi + RT / F * mu_val);

        // stress calculations
        VectorGrad eigenstrain0 =
          eig_scale *
          TensorHelper::extract_rank2_symm(variable_list, Subsets::eigenstrain());
        VectorGrad eigenstrain = (c_val - c_ref) * eigenstrain0;
        VectorGrad stress;
        dealii::Tensor<2, Mechanics::voigt_tensor_size<dim>, ScalarValue> stiffness =
          TensorHelper::extract_stiffness(variable_list, Subsets::stiffness());
        Mechanics::compute_stress<dim, ScalarValue>(stiffness,
                                                    psi * (ux - eigenstrain),
                                                    stress);

        ScalarGrad stress_diag;
        ScalarGrad stress_offdiag;
        for (unsigned int i = 0; i < dim; ++i)
          {
            stress_diag[i] = stress[i][i];
          }
        if constexpr (dim == 2)
          {
            stress_offdiag[0] = stress[0][1];
          }
        else
          {
            stress_offdiag[0] = stress[0][1];
            stress_offdiag[1] = stress[0][2];
            stress_offdiag[2] = stress[1][2];
          }

        ScalarGrad stress_eigs;
        for (unsigned int k = 0; k < stress[0][0].size(); ++k)
          {
            dealii::SymmetricTensor<2, dim, number> stress_sym;
            for (unsigned int i = 0; i < dim; ++i)
              {
                for (unsigned int j = 0; j < dim; ++j)
                  {
                    stress_sym[i][j] = stress[i][j][k];
                  }
              }
            std::array<number, dim> stress_eigvals = dealii::eigenvalues(stress_sym);
            for (unsigned int i = 0; i < dim; ++i)
              {
                stress_eigs[i][k] = stress_eigvals[i];
              }
          }
        variable_list.set_value_term(Fields::stress_diag.index, stress_diag);
        variable_list.set_value_term(Fields::stress_off_diag.index, stress_offdiag);
        variable_list.set_value_term(Fields::stress_principal.index, stress_eigs);

        // components of the Li-vacancy diffusion potential (expressed in volts)
        ScalarValue elastic_diffpot =
          dealii::scalar_product<2, dim, ScalarValue>(stress, eigenstrain0);
        elastic_diffpot *= site_vol * stress_scale;
        variable_list.set_value_term(Fields::mu_elastic.index, elastic_diffpot / F);
        variable_list.set_value_term(Fields::mu_chem.index, ocv_model_.eval_U_ocv(c_val));
      }
  }

  void
  compute_lhs([[maybe_unused]] FieldContainer<dim, degree, number> &variable_list,
              [[maybe_unused]] const SimulationTimer               &sim_timer,
              [[maybe_unused]] unsigned int solve_block_id) const override
  {
    if (solve_block_id == 1) // mechanics - lhs
      {
        VectorGrad ux =
          variable_list.template get_symmetric_gradient<Vector, LHS>(Fields::u.index);
        ScalarValue psi =
          variable_list.template get_value<Scalar, Current>(Fields::psi.index);
        VectorGrad                                                        stress;
        dealii::Tensor<2, Mechanics::voigt_tensor_size<dim>, ScalarValue> stiffness =
          TensorHelper::extract_stiffness(variable_list, Subsets::stiffness());
        Mechanics::compute_stress<dim, ScalarValue>(stiffness, psi * ux, stress);
        variable_list.set_gradient_term(Fields::u.index, stress);
      }
  }

  number                                                       i_0;
  number                                                       del_phi;
  number                                                       offset;
  number                                                       c0;
  number                                                       c_ref;
  number                                                       RT;
  number                                                       F;
  number                                                       diffusivity;
  number                                                       diff_scale;
  number                                                       eig_scale;
  number                                                       vegard;
  number                                                       site_vol;
  number                                                       mol_vol;
  number                                                       stress_scale;
  number                                                       V_ref;
  number                                                       i_target;
  number                                                       V_step;
  number                                                       V_min;
  number                                                       V_max;
  dealii::Tensor<2, Mechanics::voigt_tensor_size<dim>, number> stiffness_const;
};

PRISMS_PF_END_NAMESPACE
