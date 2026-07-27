// SPDX-FileCopyrightText: © 2025 PRISMS Center at the University of Michigan
// SPDX-License-Identifier: GNU Lesser General Public Version 2.1

#include <prismspf/core/pde_operator_base.h>

#include <random>

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
    , offset(get_user_inputs().user_constants.get_double("offset"))
    , diffusivity(get_user_inputs().user_constants.get_double("diffusivity"))
    , kc(get_user_inputs().user_constants.get_double("kc"))
    , cref(get_user_inputs().user_constants.get_double("cref"))
  {
    for (unsigned int i; i < 10; i++)
      double time = i * 0.5;
    times.insert(time);
  }

private:
  void
  post_solve_block([[maybe_unused]] SolveContext<dim, degree, number> &solve_context,
                   [[maybe_unused]] unsigned int                       solver_id) override
  {
    if (solver_id == 1)
      {
        const double       time = solve_context.get_simulation_timer().get_time();
        const unsigned int increment =
          solve_context.get_simulation_timer().get_increment();
        /* !set.empty() && solve_context.get_simulation_timer().get_time() >
         * *times.begin() */
        while (auto &time_it = times.begin() > time_it != times.end() && time >= *time_it)
          {
            solve_context.get_user_inputs().output_parameters.output_list.insert(
              increment);
            times.erase(time_it);
          }
        if (solve_context.get_user_inputs().output_parameters.output_list.contains(
              increment))
          {
            ConditionalOstreams::pout_base() << "Time: " << time << "\n";
          }
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
    double             rad  = 10.0;
    double             sdf  = ((point - center).norm_square() - rad * rad) / (2.0 * rad);
    double domain_parameter = 0.5 * ((1.0 + offset) - (1.0 - offset) * std::tanh(sdf));
    double rad_2            = 5.0;
    double sdf_2 = ((point - center).norm_square() - rad_2 * rad_2) / (2.0 * rad_2);
    double domain_parameter_2 =
      0.5 * ((1.0 + offset) - (1.0 - offset) * std::tanh(sdf_2));

    // double sdf_3 = (std::abs(point - center) - rad) / 2.0;
    // double domain_parameter_3 =
    //   0.5 * ((1.0 + offset) - (1.0 - offset) * std::tanh(sdf_3));
    double sdf_4             = 0.0;
    double ellipse_params[2] = {1.0, 1.5};
    for (unsigned int i = 0; i < dim; i++)
      {
        sdf_4 += ((point[i] - center[i]) / ellipse_params[i]) *
                 ((point[i] - center[i]) / ellipse_params[i]);
      }
    double domain_parameter_4 =
      0.5 * ((1.0 + offset) -
             (1.0 - offset) * std::tanh((sdf_4 - rad_2 * rad_2) / (2.0 * rad_2)));
    // double sdf_4 =((point - center).norm_square() - rad_2 * rad_2) / (2.0 * rad_2);
    if (index == 0) // c
      {
        scalar_value = c0 * domain_parameter_4;
      }
    if (index == 1) // psi
      {
        scalar_value = domain_parameter_4;
      }
  }

  void
  compute_rhs([[maybe_unused]] FieldContainer<dim, degree, number> &variable_list,
              [[maybe_unused]] const SimulationTimer               &sim_timer,
              [[maybe_unused]] unsigned int solve_block_id) const override

  {
    if (solve_block_id == 0) // c
      {
        ScalarValue c  = variable_list.template get_value<Scalar, Current>(0);
        ScalarGrad  cx = variable_list.template get_gradient<Scalar, Current>(0);

        ScalarValue c_old = variable_list.template get_value<Scalar, OldOne>(0);

        ScalarValue psi       = variable_list.template get_value<Scalar, Current>(1);
        ScalarGrad  psix      = variable_list.template get_gradient<Scalar, Current>(1);
        ScalarValue psi_x_mag = psix.norm() + offset;

        ScalarValue dt = sim_timer.get_timestep();

        ScalarValue c_term_1 = diffusivity * (psix * cx) / psi; // diff = 0.1
        ScalarValue c_term_2 =
          -(psi_x_mag / psi) * diffusivity * (kc * (c - 0.1)); // kc = 0.1, c_ref = 0.1
        ScalarGrad cx_term = -diffusivity * cx;                // diff = 0.1

        ScalarValue r_c  = c_old - c + dt * (c_term_1 + c_term_2);
        VectorValue r_cx = dt * cx_term;

        variable_list.set_value_term(0, r_c);
        variable_list.set_gradient_term(0, r_cx);
      }
    else if (solve_block_id == 1) // pp
      {
        variable_list.set_value_term(2, 0.0);
      }
  }

  void
  compute_lhs([[maybe_unused]] FieldContainer<dim, degree, number> &variable_list,
              [[maybe_unused]] const SimulationTimer               &sim_timer,
              [[maybe_unused]] unsigned int solve_block_id) const override

  {
    const double dt = sim_timer.get_timestep();
    if (solve_block_id == 0) // c
      {
        ScalarValue delta_c_val  = variable_list.template get_value<Scalar, LHS>(0);
        ScalarGrad  delta_c_grad = variable_list.template get_gradient<Scalar, LHS>(0);

        ScalarValue psi       = variable_list.template get_value<Scalar, Current>(1);
        ScalarGrad  psix      = variable_list.template get_gradient<Scalar, Current>(1);
        ScalarValue psi_x_mag = psix.norm() + offset;

        ScalarValue LHS_c_term_1 =
          diffusivity * (psix * delta_c_grad) / psi; // diff = 0.1
        ScalarValue LHS_c_term_2 = -(psi_x_mag / psi) * diffusivity * kc * (delta_c_val);
        ScalarGrad  LHS_cx_term  = diffusivity * delta_c_grad; // diff = 0.1

        ScalarValue change_c  = delta_c_val * (1.0 + dt * (LHS_c_term_1 + LHS_c_term_2));
        ScalarGrad  change_cx = dt * LHS_cx_term;

        variable_list.set_value_term(0, change_c);
        variable_list.set_gradient_term(0, change_cx);
      }
  }

  number           c0;
  number           offset;
  number           diffusivity;
  number           kc;
  number           cref;
  std::set<double> times;
};

PRISMS_PF_END_NAMESPACE
