// SPDX-FileCopyrightText: © 2025 PRISMS Center at the University of Michigan
// SPDX-License-Identifier: GNU Lesser General Public Version 2.1

#include <prismspf/core/pde_operator_base.h>
#include <prismspf/core/type_enums.h>

#include <prismspf/utilities/mechanics.h>

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

  /**
   * @brief Constructor.
   */
  CustomPDE(const UserInputParameters<dim> &_user_inputs, PhaseFieldTools<dim> &_pf_tools)
    : PDEOperatorBase<dim, degree, number>(_user_inputs, _pf_tools)
    , c0(get_user_inputs().user_constants.get_double("c0"))
    , c_ref(get_user_inputs().user_constants.get_double("c_ref"))
    , offset(get_user_inputs().user_constants.get_double("offset"))
    , diffusivity(get_user_inputs().user_constants.get_double("diffusivity"))
    , i_0(get_user_inputs().user_constants.get_double("i_0"))
    , del_phi(get_user_inputs().user_constants.get_double("del_phi"))
    , vegard(get_user_inputs().user_constants.get_double("vegard"))
    , site_vol(get_user_inputs().user_constants.get_double("site_vol"))
    , mol_vol(get_user_inputs().user_constants.get_double("mol_vol"))
    , RT(get_user_inputs().user_constants.get_double("RT"))
    , F(get_user_inputs().user_constants.get_double("F"))
    , stiffness(get_user_inputs().user_constants.get_elasticity_tensor("stiffness"))
  {}

private:
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
    double             rad  = 10.0;
    double             sdf  = ((point - center).norm_square() - rad * rad) / (2.0 * rad);
    double domain_parameter = 0.5 * ((1.0 + offset) - (1.0 - offset) * std::tanh(sdf));
    if (index == 2) // c
      {
        scalar_value = c0 * domain_parameter;
      }
    if (index == 3) // psi
      {
        scalar_value = domain_parameter;
      }
  }

  void
  set_dirichlet([[maybe_unused]] const unsigned int       &index,
                [[maybe_unused]] const unsigned int       &boundary_id,
                [[maybe_unused]] const unsigned int       &component,
                [[maybe_unused]] const dealii::Point<dim> &point,
                [[maybe_unused]] number                   &scalar_value,
                [[maybe_unused]] number &vector_component_value) const override
  {
    if (index == 0)
      {
        scalar_value           = 0.0;
        vector_component_value = 0.0;
      }
  }

  void
  compute_rhs([[maybe_unused]] FieldContainer<dim, degree, number> &variable_list,
              [[maybe_unused]] const SimulationTimer               &sim_timer,
              [[maybe_unused]] unsigned int solve_block_id) const override

  {
    // static const ScalarValue alpha(0.5);
    using std::exp;
    using std::pow;
    using std::sqrt;
    if (solve_block_id == 0) // concentration and mu
      {
        // Displacement
        VectorGrad u_grad =
          variable_list.template get_symmetric_gradient<Vector, Current>(0);

        // Diffusion Potential terms
        ScalarValue mu_val  = variable_list.template get_value<Scalar, Current>(1);
        ScalarGrad  mu_grad = variable_list.template get_gradient<Scalar, Current>(1);

        // Concentration Terms
        ScalarValue c_val = variable_list.template get_value<Scalar, Current>(2);
        ScalarValue c_old = variable_list.template get_value<Scalar, OldOne>(2);

        // concentration offset
        ScalarValue epsilon = 1.0e-8;

        // Order Parameter
        ScalarValue psi      = variable_list.template get_value<Scalar, Current>(3);
        ScalarGrad  psi_grad = variable_list.template get_gradient<Scalar, Current>(3);
        ScalarValue psi_grad_mag = psi_grad.norm() + offset;

        // Time step
        ScalarValue dt = sim_timer.get_timestep();

        // Mechanics Calculation
        ScalarValue eigenstrain = (vegard / 3.0) * (c_val - c_ref);

        for (unsigned int i = 0; i < dim; i++)
          {
            u_grad[i][i] -= eigenstrain;
          }
        VectorGrad stress;
        Mechanics::compute_stress<dim, ScalarValue>(stiffness, psi * u_grad, stress);
        ScalarValue hydrostatic_stress = dealii::trace(stress) / 3.0;

        // Additional terms
        ScalarValue mobility       = (diffusivity * c_val) / RT;
        ScalarValue app_pot_energy = F * del_phi;
        ScalarValue eta            = RT * mu_val + app_pot_energy;
        ScalarValue react          = -2.0 * (i_0 / F) * std::sinh(eta / (2.0 * RT));
        ScalarValue c_func_val     = psi_grad_mag * dt * react;
        ScalarGrad  c_func_grad    = mobility * RT * mu_grad;
        ScalarValue stress_func    = (site_vol * vegard * hydrostatic_stress) / RT;

        // Residuals
        ScalarValue r_c_val  = psi * (c_old - c_val) + dt * c_func_val;
        ScalarGrad  r_c_grad = dt * c_func_grad;
        ScalarValue r_mu_val = c_val - exp(mu_val + stress_func);

        variable_list.set_value_term(1, r_mu_val);
        variable_list.set_value_term(2, r_c_val);
        variable_list.set_gradient_term(2, -r_c_grad);
      }
    else if (solve_block_id == 1) // displacement
      {
        // Calling Variables
        ScalarValue c_val = variable_list.template get_value<Scalar, Current>(2);
        ScalarValue psi   = variable_list.template get_value<Scalar, Current>(3);

        // Solving for stress
        VectorGrad  transformation_strain;
        ScalarValue eigenstrain = (vegard / 3.0) * (c_val - c_ref);

        for (unsigned int i = 0; i < dim; i++)
          {
            transformation_strain[i][i] = -eigenstrain;
          }
        VectorGrad stress;
        Mechanics::compute_stress<dim, ScalarValue>(stiffness,
                                                    psi * transformation_strain,
                                                    stress);

        // Updating displacement and hydrostatic stress residuals
        variable_list.set_gradient_term(0, -stress);
      }
    else if (solve_block_id == 2) // pp
      {
        // Concentration
        ScalarValue c_val = variable_list.template get_value<Scalar, Current>(2);
        // Order Parameter
        ScalarValue psi = variable_list.template get_value<Scalar, Current>(3);

        // Concentration inside the particle
        variable_list.set_value_term(4, c_val * psi);
      }
  }

  void
  compute_lhs([[maybe_unused]] FieldContainer<dim, degree, number> &variable_list,
              [[maybe_unused]] const SimulationTimer               &sim_timer,
              [[maybe_unused]] unsigned int solve_block_id) const override
  {
    // static const ScalarValue alpha(0.5);
    using std::exp;
    using std::pow;
    using std::sqrt;
    if (solve_block_id == 0) // concentration
      {
        // Value terms
        VectorGrad u_grad =
          variable_list.template get_symmetric_gradient<Vector, Current>(0);
        ScalarValue mu_val  = variable_list.template get_value<Scalar, Current>(1);
        ScalarGrad  mu_grad = variable_list.template get_gradient<Scalar, Current>(1);
        ScalarValue c_val   = variable_list.template get_value<Scalar, Current>(2);

        // concentration offset
        ScalarValue epsilon = 1.0e-8;

        // Domain Parameter
        ScalarValue psi      = variable_list.template get_value<Scalar, Current>(3);
        ScalarGrad  psi_grad = variable_list.template get_gradient<Scalar, Current>(3);
        ScalarValue psi_grad_mag = psi_grad.norm() + offset;

        // Change terms
        VectorGrad del_u_grad =
          variable_list.template get_symmetric_gradient<Vector, LHS>(0);
        ScalarValue del_mu      = variable_list.template get_value<Scalar, LHS>(1);
        ScalarGrad  del_mu_grad = variable_list.template get_gradient<Scalar, LHS>(1);
        ScalarValue del_c       = variable_list.template get_value<Scalar, LHS>(2);

        // Time Step
        ScalarValue dt = sim_timer.get_timestep();

        // Mechanics calculations
        ScalarValue eigenstrain = (vegard / 3.0) * (c_val - c_ref);

        for (unsigned int i = 0; i < dim; i++)
          {
            u_grad[i][i] -= eigenstrain;
          }
        VectorGrad stress;
        Mechanics::compute_stress<dim, ScalarValue>(stiffness, psi * u_grad, stress);
        ScalarValue hydrostatic_stress = dealii::trace(stress) / 3.0;

        VectorGrad stress_del_c; // variation on c
        VectorGrad transformation_strain;
        for (unsigned int i = 0; i < dim; i++)
          {
            transformation_strain[i][i] = -(vegard / 3.0) * del_c;
          }
        Mechanics::compute_stress<dim, ScalarValue>(stiffness,
                                                    psi * transformation_strain,
                                                    stress_del_c);
        ScalarValue s_del_c = dealii::trace(stress_del_c) / 3.0;

        // Reaction Rate, same as rhs
        ScalarValue app_pot_energy = F * del_phi;
        ScalarValue eta            = app_pot_energy + RT * mu_val;
        ScalarValue react_d_mu     = -(i_0 / F) * std::cosh(eta / (2.0 * RT));

        // Additional functions
        ScalarValue mobility = (diffusivity * c_val) / RT;
        ScalarValue mu_exp_term =
          exp((vegard * site_vol) / RT * hydrostatic_stress + mu_val);
        ScalarValue thermo_factor = (vegard * site_vol) / RT;

        // LHS comprised of 3x3 Jacobi

        ScalarValue j_c_c_val   = -psi * del_c; // TODO check this and the vector
        ScalarGrad  j_c_c_grad  = dt * (diffusivity * del_c) * mu_grad;
        ScalarValue j_c_mu_val  = dt * (psi_grad_mag * react_d_mu * del_mu);
        ScalarGrad  j_c_mu_grad = dt * mobility * RT * del_mu_grad;

        ScalarValue j_mu_c_val  = del_c - mu_exp_term * s_del_c * thermo_factor;
        ScalarValue j_mu_mu_val = -mu_exp_term * del_mu;

        // Update fields
        variable_list.set_value_term(1, -j_mu_c_val - j_mu_mu_val);
        variable_list.set_value_term(2, -j_c_c_val - j_c_mu_val);
        variable_list.set_gradient_term(2, j_c_mu_grad + j_c_c_grad);
      }
    if (solve_block_id == 1) // displacement
      {
        // Calling Variables
        VectorGrad del_u_grad =
          variable_list.template get_symmetric_gradient<Vector, LHS>(0);
        ScalarValue psi = variable_list.template get_value<Scalar, Current>(3);

        // Mechanics - lhs
        VectorGrad stress;
        Mechanics::compute_stress<dim, ScalarValue>(stiffness, psi * del_u_grad, stress);

        variable_list.set_gradient_term(0, stress);
      }
  }

  // number alpha;
  number                                                       i_0;
  number                                                       del_phi;
  number                                                       offset;
  number                                                       c0;
  number                                                       c_ref;
  number                                                       RT;
  number                                                       F;
  number                                                       diffusivity;
  number                                                       vegard;
  number                                                       site_vol;
  number                                                       mol_vol;
  dealii::Tensor<2, Mechanics::voigt_tensor_size<dim>, number> stiffness;
};

PRISMS_PF_END_NAMESPACE
