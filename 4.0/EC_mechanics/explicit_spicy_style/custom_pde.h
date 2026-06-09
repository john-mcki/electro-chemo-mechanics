// SPDX-FileCopyrightText: © 2025 PRISMS Center at the University of Michigan
// SPDX-License-Identifier: GNU Lesser General Public Version 2.1

#include <prismspf/core/pde_operator_base.h>

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
    , alpha(get_user_inputs().user_constants.get_double("alpha"))
    , vegard(get_user_inputs().user_constants.get_double("vegard"))
    , site_vol(get_user_inputs().user_constants.get_double("site_vol"))
    , mol_vol(get_user_inputs().user_constants.get_double("mol_vol"))
    , RT(get_user_inputs().user_constants.get_double("RT"))
    , F(get_user_inputs().user_constants.get_double("F"))
    , stiffness(get_user_inputs().user_constants.get_elasticity_tensor("stiffness"))
  {}

  /*
    void
    post_solve_block([[maybe_unused]] SolveContext<dim, degree, number> &solve_context,
                     [[maybe_unused]] unsigned int                       solver_id)
    override
    {
      if (solver_id == 0)
        {
          if (user_inputs.spatial_discretization.has_adaptivity && increment == 0)
            {
              grid_refiner.do_initial_refinement(solvers);
            }
        }
    }
  */

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
  } // redudant, will not be called

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
    using std::exp;
    using std::pow;
    using std::sqrt;
    if (solve_block_id == 0) // concentration and mechanics
      {
        // Dsiplacement
        VectorGrad u_grad =
          variable_list.template get_symmetric_gradient<Vector, Current>(0);
        // Hydrostatic Stress
        ScalarValue s_val  = variable_list.template get_value<Scalar, Current>(1);
        ScalarGrad  s_grad = variable_list.template get_gradient<Scalar, Current>(1);
        // Concentration Terms
        ScalarValue c_val  = variable_list.template get_value<Scalar, Current>(2);
        ScalarValue c_old  = variable_list.template get_value<Scalar, OldOne>(2);
        ScalarGrad  c_grad = variable_list.template get_gradient<Scalar, Current>(2);
        // Order Parameter
        ScalarValue psi      = variable_list.template get_value<Scalar, Current>(3);
        ScalarGrad  psi_grad = variable_list.template get_gradient<Scalar, Current>(3);
        ScalarValue psi_grad_mag = psi_grad.norm() + offset;
        // Time step
        ScalarValue dt = sim_timer.get_timestep();

        // Mechanics update
        ScalarValue eigenstrain = (vegard / 3.0) * (c_val - c_ref);

        for (unsigned int i = 0; i < dim; i++)
          {
            u_grad[i][i] = -eigenstrain;
          }
        VectorGrad stress;
        Mechanics::compute_stress<dim, ScalarValue>(stiffness, psi * u_grad, stress);

        // Concentration update
        ScalarValue c_term1 = (diffusivity / psi) * (psi_grad * c_grad);
        ScalarValue c_term2 =
          -psi_grad / psi * s_grad * diffusivity * (site_vol * vegard * c_val) / (RT);

        // Rate Terms
        ScalarValue app_pot_energy = F * del_phi;
        ScalarValue mech_energy    = site_vol * s_val;
        ScalarValue conf_energy    = RT * std::log(c_val / c_ref);
        ScalarValue eta =
          app_pot_energy + mech_energy + conf_energy; // overpotential term
        ScalarValue BV_exp_term = exp(-eta / (RT));
        ScalarValue c_term3  = (psi_grad_mag / psi) * i_0 / F *
                               (pow(BV_exp_term, alpha) -
                                pow(BV_exp_term, -alpha)); // Full BV reaction rate term
        ScalarGrad  cx_term1 = -diffusivity * c_grad;
        ScalarGrad  cx_term2 = diffusivity * (site_vol * c_val) / RT * s_grad;

        // residual update
        ScalarValue r_c_val  = c_old - dt * (c_term1 + c_term2 + c_term3);
        ScalarGrad  r_c_grad = -dt * (cx_term1 + cx_term2);

        // Update fields
        variable_list.set_gradient_term(0, -stress);
        variable_list.set_value_term(1, 0.0);
        if (sim_timer.get_increment() == 0)
          {
            // variable_list.set_value_term(2, c_val - (c0 * psi));
            variable_list.set_value_term(2, 0.0);
          }
        else
          {
            variable_list.set_value_term(2, r_c_val);
          }
        variable_list.set_gradient_term(2, r_c_grad);
      }
    else if (solve_block_id == 1) // pp
      {
        // Dsiplacement
        VectorGrad u_grad =
          variable_list.template get_symmetric_gradient<Vector, Current>(0);
        // Concentration
        ScalarValue c_val = variable_list.template get_value<Scalar, Current>(2);
        // Order Parameter
        ScalarValue psi = variable_list.template get_value<Scalar, Current>(3);

        // Concentration inside the particle
        variable_list.set_value_term(4, c_val * psi);

        // Solution Free Energy (J/vol)
        ScalarValue conf_energy =
          (RT * c_val * std::log(c_val)) / mol_vol; // check mol_vol vs site_vol
        variable_list.set_value_term(5, psi * conf_energy);

        // Mechanical Free Energy (J/vol)
        VectorGrad  transformation_strain;
        ScalarValue eigenstrain = (vegard / 3.0) * (c_val - c_ref);
        for (unsigned int i = 0; i < dim; i++)
          {
            transformation_strain[i][i] = -eigenstrain;
          }
        VectorGrad stress;
        Mechanics::compute_stress<dim, ScalarValue>(stiffness,
                                                    u_grad - transformation_strain,
                                                    stress);
        ScalarValue mech_energy =
          0.5 * dealii::scalar_product(stress, (u_grad - transformation_strain)) *
          (1.0e-9); // Conversion to J/vol in terms of microns
        variable_list.set_value_term(6, psi * mech_energy);
      }
  }

  void
  compute_lhs([[maybe_unused]] FieldContainer<dim, degree, number> &variable_list,
              [[maybe_unused]] const SimulationTimer               &sim_timer,
              [[maybe_unused]] unsigned int solve_block_id) const override
  {
    static const ScalarValue alpha(0.5);
    using std::exp;
    using std::pow;
    using std::sqrt;
    if (solve_block_id == 0)
      {
        // Calling Variables
        ScalarValue s_val  = variable_list.template get_value<Scalar, Current>(1);
        ScalarGrad  s_grad = variable_list.template get_gradient<Scalar, Current>(1);
        ScalarValue c_val  = variable_list.template get_value<Scalar, Current>(2);

        VectorGrad del_u_grad =
          variable_list.template get_symmetric_gradient<Vector, LHS>(0);
        ScalarValue del_s      = variable_list.template get_value<Scalar, LHS>(1);
        ScalarValue del_c      = variable_list.template get_value<Scalar, LHS>(2);
        ScalarGrad  del_c_grad = variable_list.template get_gradient<Scalar, LHS>(2);

        ScalarValue psi       = variable_list.template get_value<Scalar, Current>(3);
        ScalarGrad  psi_x     = variable_list.template get_gradient<Scalar, Current>(3);
        ScalarValue psi_x_mag = psi_x.norm() + offset;
        ScalarValue dt        = sim_timer.get_timestep();

        // Mechanics - lhs
        VectorGrad stress;
        Mechanics::compute_stress<dim, ScalarValue>(stiffness, psi * del_u_grad, stress);

        // Concentration - lhs
        ScalarValue LHS_c_term1 = -(diffusivity / psi) * (psi_x * del_c_grad);
        ScalarValue LHS_c_term2 =
          (site_vol * diffusivity * del_c) / (RT * psi) * (psi_x * s_grad);
        // Rate Terms
        ScalarValue app_pot_energy = F * del_phi;
        ScalarValue mech_energy    = site_vol * s_val;
        ScalarValue conf_energy    = RT * std::log(c_val / c_ref);
        ScalarValue eta = app_pot_energy + mech_energy +
                          conf_energy; // overpotential term, check conf_energy later
        ScalarValue BV_exp_term = exp(-eta / (RT));
        ScalarValue LHS_c_term3 =
          -(psi_x_mag / psi) * i_0 / F * del_c / c_val *
          (alpha * pow(BV_exp_term, alpha) - (1.0 - alpha) * pow(BV_exp_term, -alpha));

        ScalarGrad  LHS_cx_term1 = diffusivity * del_c_grad;
        ScalarGrad  LHS_cx_term2 = (diffusivity * site_vol) / (RT) * (s_grad * del_c);
        ScalarValue eq_change_c  = del_c + dt * (LHS_c_term1 + LHS_c_term2 + LHS_c_term3);
        ScalarGrad  eq_change_cx = dt * (LHS_cx_term1 + LHS_cx_term2);

        // Update fields
        variable_list.set_gradient_term(0, stress);
        variable_list.set_value_term(1, s_val - dealii::trace(stress) / 3.0);
        variable_list.set_value_term(2, eq_change_c);
        variable_list.set_gradient_term(2, eq_change_cx);
      }
  }

  number alpha;
  number i_0;
  number del_phi;
  number offset;
  number c0;
  number c_ref;
  number RT;
  number F;
  number diffusivity;
  number vegard;
  number site_vol;
  number mol_vol;
  // number                                                       poisson;
  // number                                                       youngs_modulus;
  dealii::Tensor<2, Mechanics::voigt_tensor_size<dim>, number> stiffness;
};

PRISMS_PF_END_NAMESPACE
