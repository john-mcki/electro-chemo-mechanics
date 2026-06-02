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
    , poisson(get_user_inputs().user_constants.get_double("poisson"))
    , youngs_modulus(get_user_inputs().user_constants.get_double("youngs_modulus"))
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
    double             rad = 10.0;
    double             sdf = (rad * rad - (point - center).norm_square()) / (2.0 * rad);
    double             domain_parameter = 0.5 * (1.0 + std::tanh(sdf)) + offset;
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
    using std::exp;
    using std::pow;
    using std::sqrt;
    if (solve_block_id == 1) // c
      {
        // Hydrostatic Stress
        ScalarValue s  = variable_list.template get_value<Scalar, OldOne>(1);
        ScalarGrad  sx = variable_list.template get_gradient<Scalar, OldOne>(1);
        // Concentration Terms
        ScalarValue c  = variable_list.template get_value<Scalar, OldOne>(2);
        ScalarGrad  cx = variable_list.template get_gradient<Scalar, OldOne>(2);
        // Order Parameter
        ScalarValue psi       = variable_list.template get_value<Scalar, Current>(3);
        ScalarGrad  psi_x     = variable_list.template get_gradient<Scalar, Current>(3);
        ScalarValue psi_x_mag = psi_x.norm() + offset;
        ScalarValue dt        = sim_timer.get_timestep();

        // Transport Terms
        ScalarValue c_term1 = (diffusivity / psi) * (psi_x * cx);
        ScalarValue c_term2 =
          -psi_x / psi * sx * diffusivity * (site_vol * vegard * c) / (RT);
        // Rate Terms
        ScalarValue app_pot_energy = F * del_phi;
        ScalarValue mech_energy    = site_vol * s;
        ScalarValue conf_energy    = RT * std::log(c / c_ref);
        ScalarValue eta            = app_pot_energy + mech_energy +
                          conf_energy; // overpotential term, check conf_energy later
        ScalarValue BV_exp_term = exp(-eta / (RT));
        ScalarValue c_term3     = (psi_x_mag / psi) * i_0 / F *
                              (pow(BV_exp_term, alpha) -
                               pow(BV_exp_term, -alpha)); // Full BV reaction rate term
        ScalarGrad cx_term1 = -diffusivity * cx;
        ScalarGrad cx_term2 = diffusivity * (site_vol * c) / RT * sx;

        ScalarValue eq_c  = c + dt * (c_term1 + c_term2 + c_term3);
        ScalarGrad  eqx_c = dt * (cx_term1 + cx_term2);

        variable_list.set_value_term(2, eq_c);
        variable_list.set_gradient_term(2, eqx_c);
      }
    if (solve_block_id == 0) // u, s
      {
        ScalarValue c   = variable_list.template get_value<Scalar, Current>(2);
        ScalarValue psi = variable_list.template get_value<Scalar, Current>(3);
        VectorGrad  transformation_strain;
        ScalarValue eigenstrain = vegard * (c - c_ref);

        for (unsigned int i = 0; i < dim; i++)
          {
            transformation_strain[i][i] = -eigenstrain;
          }
        VectorGrad stress;
        Mechanics::compute_stress<dim, ScalarValue>(stiffness,
                                                    psi * transformation_strain,
                                                    stress);

        variable_list.set_gradient_term(0, -stress);
        variable_list.set_value_term(1, 0.0);
      }
    else if (solve_block_id == 2) // pp
      {
        ScalarValue c   = variable_list.template get_value<Scalar, Current>(2);
        ScalarValue psi = variable_list.template get_value<Scalar, Current>(3);
        variable_list.set_value_term(4, c * psi);
      }
  }

  void
  compute_lhs([[maybe_unused]] FieldContainer<dim, degree, number> &variable_list,
              [[maybe_unused]] const SimulationTimer               &sim_timer,
              [[maybe_unused]] unsigned int solve_block_id) const override
  {
    if (solve_block_id == 0) // mechanics - lhs
      {
        VectorGrad  ux  = variable_list.template get_symmetric_gradient<Vector, LHS>(0);
        ScalarValue s   = variable_list.template get_value<Scalar, LHS>(1);
        ScalarValue psi = variable_list.template get_value<Scalar, Current>(3);
        VectorGrad  stress;
        Mechanics::compute_stress<dim, ScalarValue>(stiffness, psi * ux, stress);
        variable_list.set_gradient_term(0, stress);
        variable_list.set_value_term(1, s - dealii::trace(stress) / 3.0);
      }
    /*
      {
        ScalarValue psi       = variable_list.template get_value<ScalarValue>(3);
        ScalarGrad  psi_x     = variable_list.template get_gradient<ScalarGrad>(3);
        ScalarValue psi_x_mag = psi_x.norm() + offset;

        ScalarGrad  cx      = variable_list.template get_gradient<Scalar, LHS>(2);
        ScalarValue c_term1 = diffusivity * cx;
        ScalarValue c_term2 = (diffusivity * omega * psi_x * sx) / (RT * psi);
        ScalarValue c_term3 = (diffusivity * psi_x_mag * kc) / psi;
        variable_list.set_gradient_term(2, c_term1 + c_term2 + c_term3);
      }
    */
  }

  number                                                       alpha;
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
  number                                                       poisson;
  number                                                       youngs_modulus;
  dealii::Tensor<2, Mechanics::voigt_tensor_size<dim>, number> stiffness;
};

PRISMS_PF_END_NAMESPACE
