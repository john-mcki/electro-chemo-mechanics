// SPDX-FileCopyrightText: © 2025 PRISMS Center at the University of Michigan
// SPDX-License-Identifier: GNU Lesser General Public Version 2.1

#include <prismspf/core/pde_operator_base.h>

#include <prismspf/utilities/mechanics.h>

PRISMS_PF_BEGIN_NAMESPACE

template <unsigned int dim, unsigned int degree, typename number>
class CustomPDE : public PDEOperatorBase<dim, degree, number> {
public:
  using ScalarValue = dealii::VectorizedArray<number>;
  using ScalarGrad = dealii::Tensor<1, dim, ScalarValue>;
  using ScalarHess = dealii::Tensor<2, dim, ScalarValue>;
  using VectorValue = dealii::Tensor<1, dim, ScalarValue>;
  using VectorGrad = dealii::Tensor<2, dim, ScalarValue>;
  using VectorHess = dealii::Tensor<3, dim, ScalarValue>;
  using PDEOperatorBase<dim, degree, number>::get_user_inputs;
  using PDEOperatorBase<dim, degree, number>::get_pf_tools;

  /**
   * @brief Constructor.
   */
  CustomPDE(const UserInputParameters<dim> &_user_inputs,
            PhaseFieldTools<dim> &_pf_tools)
      : PDEOperatorBase<dim, degree, number>(_user_inputs, _pf_tools),
        c0(get_user_inputs().user_constants.get_double("c0")),
        c_ref(get_user_inputs().user_constants.get_double("c_ref")),
        offset(get_user_inputs().user_constants.get_double("offset")),
        diffusivity(get_user_inputs().user_constants.get_double("diffusivity")),
        diff_scale(get_user_inputs().user_constants.get_double("diff_scale")),
        i_0(get_user_inputs().user_constants.get_double("i_0")),
        del_phi(get_user_inputs().user_constants.get_double("del_phi")),
        vegard(get_user_inputs().user_constants.get_double("vegard")),
        site_vol(get_user_inputs().user_constants.get_double("site_vol")),
        mol_vol(get_user_inputs().user_constants.get_double("mol_vol")),
        RT(get_user_inputs().user_constants.get_double("RT")),
        F(get_user_inputs().user_constants.get_double("F")),
        stress_scale(
            get_user_inputs().user_constants.get_double("stress_scale")),
        ocv_q0(get_user_inputs().user_constants.get_double("ocv_q0")),
        ocv_q1(get_user_inputs().user_constants.get_double("ocv_q1")),
        ocv_q2(get_user_inputs().user_constants.get_double("ocv_q2")),
        ocv_q3(get_user_inputs().user_constants.get_double("ocv_q3")),
        ocv_q4(get_user_inputs().user_constants.get_double("ocv_q4")),
        ocv_q5(get_user_inputs().user_constants.get_double("ocv_q5")),
        ocv_q6(get_user_inputs().user_constants.get_double("ocv_q6")),
        ocv_q7(get_user_inputs().user_constants.get_double("ocv_q7")),
        ocv_q8(get_user_inputs().user_constants.get_double("ocv_q8")),
        ocv_q9(get_user_inputs().user_constants.get_double("ocv_q9")),
        ocv_q10(get_user_inputs().user_constants.get_double("ocv_q10")),
        ocv_q11(get_user_inputs().user_constants.get_double("ocv_q11")),
        ocv_q12(get_user_inputs().user_constants.get_double("ocv_q12")),
        ocv_q13(get_user_inputs().user_constants.get_double("ocv_q13")),
        ocv_q14(get_user_inputs().user_constants.get_double("ocv_q14")),
        ocv_q15(get_user_inputs().user_constants.get_double("ocv_q15")),
        ocv_q16(get_user_inputs().user_constants.get_double("ocv_q16")),
        ocv_q17(get_user_inputs().user_constants.get_double("ocv_q17")),
        ocv_q18(get_user_inputs().user_constants.get_double("ocv_q18")),
        ocv_q19(get_user_inputs().user_constants.get_double("ocv_q19")),
        ocv_q20(get_user_inputs().user_constants.get_double("ocv_q20")),
        ocv_q21(get_user_inputs().user_constants.get_double("ocv_q21")),
        V_ref(get_user_inputs().user_constants.get_double("V_ref")),
        i_target(get_user_inputs().user_constants.get_double("i_target")),
        V_step(get_user_inputs().user_constants.get_double("V_step")),
        V_max(get_user_inputs().user_constants.get_double("V_max")),
        V_min(get_user_inputs().user_constants.get_double("V_min")) {}

private:
  // OCV polynomial in normalized variable t = 2*(c - 0.30)/0.69 - 1.
  // Polynomial fit (degree 21) valid over c in [0.30, 0.99].

  ScalarValue eval_U_ocv(ScalarValue c_val) const {
    constexpr double ocv_soc_min = 0.30;
    constexpr double ocv_soc_span = 0.69;
    constexpr double cheb_map_scale = 2.0;

    ScalarValue t_cheb =
        ((cheb_map_scale * (c_val - ocv_soc_min)) / ocv_soc_span) - 1.0;

    ScalarValue ocv_val = ocv_q21;
    ocv_val = (ocv_val * t_cheb) + ocv_q20;
    ocv_val = (ocv_val * t_cheb) + ocv_q19;
    ocv_val = (ocv_val * t_cheb) + ocv_q18;
    ocv_val = (ocv_val * t_cheb) + ocv_q17;
    ocv_val = (ocv_val * t_cheb) + ocv_q16;
    ocv_val = (ocv_val * t_cheb) + ocv_q15;
    ocv_val = (ocv_val * t_cheb) + ocv_q14;
    ocv_val = (ocv_val * t_cheb) + ocv_q13;
    ocv_val = (ocv_val * t_cheb) + ocv_q12;
    ocv_val = (ocv_val * t_cheb) + ocv_q11;
    ocv_val = (ocv_val * t_cheb) + ocv_q10;
    ocv_val = (ocv_val * t_cheb) + ocv_q9;
    ocv_val = (ocv_val * t_cheb) + ocv_q8;
    ocv_val = (ocv_val * t_cheb) + ocv_q7;
    ocv_val = (ocv_val * t_cheb) + ocv_q6;
    ocv_val = (ocv_val * t_cheb) + ocv_q5;
    ocv_val = (ocv_val * t_cheb) + ocv_q4;
    ocv_val = (ocv_val * t_cheb) + ocv_q3;
    ocv_val = (ocv_val * t_cheb) + ocv_q2;
    ocv_val = (ocv_val * t_cheb) + ocv_q1;
    ocv_val = (ocv_val * t_cheb) + ocv_q0;

    return ocv_val;
  }

  ScalarValue eval_dU_ocv(ScalarValue c_val) const {
    constexpr double ocv_soc_min = 0.30;
    constexpr double ocv_soc_span = 0.69;
    constexpr double cheb_map_scale = 2.0;

    ScalarValue t_cheb =
        ((cheb_map_scale * (c_val - ocv_soc_min)) / ocv_soc_span) - 1.0;

    // Chain rule: dt_cheb / dc_val
    constexpr double dt_cheb_dc_val = cheb_map_scale / ocv_soc_span;

    ScalarValue docv_val = 21.0 * ocv_q21;
    docv_val = (docv_val * t_cheb) + 20.0 * ocv_q20;
    docv_val = (docv_val * t_cheb) + 19.0 * ocv_q19;
    docv_val = (docv_val * t_cheb) + 18.0 * ocv_q18;
    docv_val = (docv_val * t_cheb) + 17.0 * ocv_q17;
    docv_val = (docv_val * t_cheb) + 16.0 * ocv_q16;
    docv_val = (docv_val * t_cheb) + 15.0 * ocv_q15;
    docv_val = (docv_val * t_cheb) + 14.0 * ocv_q14;
    docv_val = (docv_val * t_cheb) + 13.0 * ocv_q13;
    docv_val = (docv_val * t_cheb) + 12.0 * ocv_q12;
    docv_val = (docv_val * t_cheb) + 11.0 * ocv_q11;
    docv_val = (docv_val * t_cheb) + 10.0 * ocv_q10;
    docv_val = (docv_val * t_cheb) + 9.0 * ocv_q9;
    docv_val = (docv_val * t_cheb) + 8.0 * ocv_q8;
    docv_val = (docv_val * t_cheb) + 7.0 * ocv_q7;
    docv_val = (docv_val * t_cheb) + 6.0 * ocv_q6;
    docv_val = (docv_val * t_cheb) + 5.0 * ocv_q5;
    docv_val = (docv_val * t_cheb) + 4.0 * ocv_q4;
    docv_val = (docv_val * t_cheb) + 3.0 * ocv_q3;
    docv_val = (docv_val * t_cheb) + 2.0 * ocv_q2;
    docv_val = (docv_val * t_cheb) + ocv_q1;

    return docv_val * dt_cheb_dc_val;
  }

  ScalarValue get_elastic_mu2D(const VectorGrad &stress,
                               const VectorValue &eig1,
                               const ScalarValue &eig2) const {
    ScalarValue energy(0.0);
    if (dim == 2) {
      energy = eig1[0] * stress[0][0] + 2.0 * eig2[0] * stress[0][1] +
               eig1[1] * stress[1][1];
    } else {
      dealii::ExcMessage("dim should be 2 for this code");
    }
    return energy;
  }

  VectorGrad eigenstrain_helper(const VectorValue &eig1,
                                const ScalarValue &eig2) const {
    VectorGrad eigenstrain;
    if (dim == 2) {
      eigenstrain[0][0] = eig1[0];
      eigenstrain[1][1] = eig1[1];
      eigenstrain[0][1] = eig2;
      eigenstrain[1][0] = eig2;
    } else {
      dealii::ExcMessage("dim should be 2 for this code");
    }
    return eigenstrain;
  }

  dealii::Tensor<2, Mechanics::voigt_tensor_size<dim>, ScalarValue>
  get_voigt2D(const VectorValue &Cel1, const VectorValue &Cel2,
              const VectorValue &Cel3) const {
    dealii::Tensor<2, Mechanics::voigt_tensor_size<dim>, ScalarValue> Cij;
    if (dim == 2) {
      Cij[0][0] = Cel1[0];
      Cij[1][1] = Cel1[1];
      Cij[2][2] = Cel2[0];
      Cij[0][1] = Cel2[1];
      Cij[1][0] = Cel2[1];
      Cij[0][2] = Cel3[0];
      Cij[2][0] = Cel3[0];
      Cij[1][2] = Cel3[1];
      Cij[2][1] = Cel3[1];
    } else {
      dealii::ExcMessage("dim should be 2 for this code");
    }
    return Cij;
  }

  VectorValue get_flux2D(const VectorValue &cx, const VectorValue &D1,
                         const ScalarValue &D2) const {
    VectorValue flux;
    if (dim == 2) {
      flux[0] = D1[0] * cx[0] + D2 * cx[1];
      flux[1] = D1[1] * cx[1] + D2 * cx[0];
    } else {
      dealii::ExcMessage("dim should be 2 for this code");
    }
    return flux;
  }

  /*
  ScalarValue eval_U_ocv(ScalarValue c_val) const {
    constexpr double a = -0.6;
    constexpr double b = -0.1;
    constexpr double c = 0.3;
    constexpr double d = 4.4;

    ScalarValue ocv_val =
        a * (c_val * c_val * c_val) + b * (c_val * c_val) + c * c_val + d;
    return ocv_val;
  }

  ScalarValue eval_dU_ocv(ScalarValue c_val) const {
    const double a = -0.6;
    const double b = -0.1;
    const double c = 0.3;

    ScalarValue ocv_val = 3.0 * a * c_val * c_val + 2.0 * b * c_val + c;
    return ocv_val;
  }
  */

  void post_solve_block(
      [[maybe_unused]] SolveContext<dim, degree, number> &solve_context,
      [[maybe_unused]] unsigned int solver_id) override {
    if (solver_id == 2) {
      const auto &psi =
          solve_context.get_solution_indexer().get_solution_vector(3);
      const auto &rxn =
          solve_context.get_solution_indexer().get_solution_vector(14);
      number integrated =
          rxn * solve_context.get_invm_manager().get_jxw(TensorRank::Scalar);
      number dt = solve_context.get_simulation_timer().get_timestep();
      number total_current =
          integrated / (mol_vol)*F *
          1.0e-6; // conversion of site fraction to total current density in a
                  // one micron thick slice
      // std::cout << "Integrated concentration: " << total_current <<
      // std::endl;
      number increment = solve_context.get_simulation_timer().get_increment();
      bool is_output_increment =
          solve_context.get_user_inputs().output_parameters.should_output(
              increment);
      if (is_output_increment == true) {
        ConditionalOStreams::pout_base()
            << "Total Current is:" << total_current << "\n";
        ConditionalOStreams::pout_base()
            << "Cell potential is:" << del_phi << "\n";
      }

      number i_proportion = (i_target - total_current) / i_target;
      number V_inc = V_step * dt *
                     i_proportion; // voltage increment based on current density
      del_phi = del_phi + V_inc;
      del_phi = std::max(std::min(del_phi, V_max), V_min);
    }
  }

  void set_initial_condition(
      [[maybe_unused]] const unsigned int &index,
      [[maybe_unused]] const unsigned int &component,
      [[maybe_unused]] const dealii::Point<dim> &point,
      [[maybe_unused]] number &scalar_value,
      [[maybe_unused]] number &vector_component_value) const override {
    const dealii::Tensor<1, dim> &mesh_size =
        get_user_inputs().spatial_discretization.rectangular_mesh.size;
    dealii::Point<dim> center(mesh_size / 2.0);
    double rad = mesh_size[0] * (2.0 / 5.0);
    double sdf = ((point - center).norm_square() - rad * rad) / (2.0 * rad);
    double domain_parameter =
        0.5 * ((1.0 + offset) - (1.0 - offset) * std::tanh(sdf));
    // double rad_2 = 5.0;
    // double sdf_2 =
    //     ((point - center).norm_square() - rad_2 * rad_2) / (2.0 * rad_2);
    // double domain_parameter_2 =
    //     0.5 * ((1.0 + offset) - (1.0 - offset) * std::tanh(sdf_2));
    if (index == 0) // c
    {
      scalar_value = c0;
    }
    if (index == 1) // c
    {
      vector_component_value = 0.1;
    }
    if (index == 2) // mu
    {
      // scalar_value = log(c0);
      scalar_value = -F / RT * eval_U_ocv(ScalarValue(c0))[0];
    }
    if (index == 3) // psi
    {
      scalar_value = domain_parameter;
    }
  }

  void set_dirichlet(
      [[maybe_unused]] const unsigned int &index,
      [[maybe_unused]] const unsigned int &boundary_id,
      [[maybe_unused]] const unsigned int &component,
      [[maybe_unused]] const dealii::Point<dim> &point,
      [[maybe_unused]] const SimulationTimer &sim_timer,
      [[maybe_unused]] number &scalar_value,
      [[maybe_unused]] number &vector_component_value) const override {
    scalar_value = 0.0;
    vector_component_value = 0.0;
  }

  void compute_rhs(
      [[maybe_unused]] FieldContainer<dim, degree, number> &variable_list,
      [[maybe_unused]] const SimulationTimer &sim_timer,
      [[maybe_unused]] unsigned int solve_block_id) const override

  {
    using std::exp;
    using std::log;
    using std::pow;
    using std::sqrt;
    if (solve_block_id == 0) // c
    {
      // Calling variables
      ScalarValue c_val = variable_list.template get_value<Scalar, OldOne>(0);
      ScalarValue mu_val = variable_list.template get_value<Scalar, OldOne>(2);
      ScalarGrad mu_grad =
          variable_list.template get_gradient<Scalar, OldOne>(2);

      ScalarValue psi = variable_list.template get_value<Scalar, Current>(3);
      psi = std::max(psi, ScalarValue(offset));
      ScalarGrad psi_grad =
          variable_list.template get_gradient<Scalar, Current>(3);
      ScalarValue psi_grad_mag = psi_grad.norm();

      ScalarValue dt = sim_timer.get_timestep();

      VectorValue D1 =
          diff_scale * variable_list.template get_value<Vector, Current>(21);
      ScalarValue D2 =
          diff_scale * variable_list.template get_value<Scalar, Current>(22);

      // Functions
      // ScalarValue mobility = (diffusivity * c_val) / RT;
      ScalarValue mobility_factor = -(RT / F) * (1.0 / eval_dU_ocv(c_val));
      ScalarValue app_pot_energy = F * del_phi;
      ScalarValue eta = app_pot_energy + RT * mu_val;
      // Diffusion and Reaction functions
      ScalarValue react = -2.0 * (i_0 / F) * std::sinh(eta / (2.0 * RT));

      VectorValue flux = get_flux2D(mu_grad, D1, D2);
      ScalarValue func_c = (psi_grad / psi) * mobility_factor * RT * flux;
      ScalarGrad func_c_grad = -mobility_factor * RT * flux;

      // Forward Euler time stepping
      ScalarValue rxn = psi_grad_mag * react;
      ScalarValue eq_c = c_val + dt * (func_c + rxn / psi);
      ScalarGrad eqx_c = dt * func_c_grad;

      variable_list.set_value_term(0, eq_c);
      variable_list.set_gradient_term(0, eqx_c);
      variable_list.set_value_term(14, rxn);
    }
    if (solve_block_id == 1) // u
    {
      ScalarValue c_val = variable_list.template get_value<Scalar, Current>(0);
      ScalarValue psi = variable_list.template get_value<Scalar, Current>(3);
      psi = std::max(psi, ScalarValue(offset));

      VectorValue Cel1 = variable_list.template get_value<Vector, Current>(16);
      VectorValue Cel2 = variable_list.template get_value<Vector, Current>(17);
      VectorValue Cel3 = variable_list.template get_value<Vector, Current>(18);
      VectorValue eig1 = variable_list.template get_value<Vector, Current>(19);
      ScalarValue eig2 = variable_list.template get_value<Scalar, Current>(20);
      VectorGrad eigenstrain0 = eigenstrain_helper(eig1, eig2);
      VectorGrad eigenstrain = (c_val - c_ref) * eigenstrain0;
      dealii::Tensor<2, Mechanics::voigt_tensor_size<dim>, ScalarValue>
          stiffness = get_voigt2D(Cel1, Cel2, Cel3);
      VectorGrad stress;
      Mechanics::compute_stress<dim, ScalarValue>(stiffness,
                                                  psi * (eigenstrain), stress);
      variable_list.set_gradient_term(1, -stress);
    }
    if (solve_block_id == 2) // mu
    {
      ScalarValue c_val = variable_list.template get_value<Scalar, Current>(0);
      VectorGrad ux =
          variable_list.template get_symmetric_gradient<Vector, Current>(1);
      ScalarValue psi = variable_list.template get_value<Scalar, Current>(3);
      psi = std::max(psi, ScalarValue(offset));

      VectorValue Cel1 = variable_list.template get_value<Vector, Current>(16);
      VectorValue Cel2 = variable_list.template get_value<Vector, Current>(17);
      VectorValue Cel3 = variable_list.template get_value<Vector, Current>(18);
      VectorValue eig1 = variable_list.template get_value<Vector, Current>(19);
      ScalarValue eig2 = variable_list.template get_value<Scalar, Current>(20);

      VectorGrad eigenstrain0 = eigenstrain_helper(eig1, eig2);
      VectorGrad eigenstrain = (c_val - c_ref) * eigenstrain0;
      dealii::Tensor<2, Mechanics::voigt_tensor_size<dim>, ScalarValue>
          stiffness = get_voigt2D(Cel1, Cel2, Cel3);
      VectorGrad stress;
      Mechanics::compute_stress<dim, ScalarValue>(
          stiffness, psi * (ux - eigenstrain), stress);
      ScalarValue elastic_diffpot = get_elastic_mu2D(stress, eig1, eig2);
      elastic_diffpot *= site_vol * stress_scale;
      ScalarValue epsilon = 1.0e-8;
      ScalarValue U_ocv = eval_U_ocv(c_val);
      ScalarValue mu_val = (-(F / RT) * (U_ocv)) - elastic_diffpot / RT;
      variable_list.set_value_term(2, mu_val);
    }
    if (solve_block_id == 3) // pp
    {
      ScalarValue c_val = variable_list.template get_value<Scalar, Current>(0);
      VectorGrad ux =
          variable_list.template get_symmetric_gradient<Vector, Current>(1);
      ScalarValue psi = variable_list.template get_value<Scalar, Current>(3);
      psi = std::max(psi, ScalarValue(offset));

      ScalarValue mu_val = variable_list.template get_value<Scalar, Current>(2);

      VectorValue Cel1 = variable_list.template get_value<Vector, Current>(16);
      VectorValue Cel2 = variable_list.template get_value<Vector, Current>(17);
      VectorValue Cel3 = variable_list.template get_value<Vector, Current>(18);
      VectorValue eig1 = variable_list.template get_value<Vector, Current>(19);
      ScalarValue eig2 = variable_list.template get_value<Scalar, Current>(20);

      VectorGrad eigenstrain0 = eigenstrain_helper(eig1, eig2);
      VectorGrad eigenstrain = (c_val - c_ref) * eigenstrain0;
      dealii::Tensor<2, Mechanics::voigt_tensor_size<dim>, ScalarValue>
          stiffness = get_voigt2D(Cel1, Cel2, Cel3);
      VectorGrad stress;
      Mechanics::compute_stress<dim, ScalarValue>(
          stiffness, psi * (ux - eigenstrain), stress);
      ScalarValue elastic_diffpot = get_elastic_mu2D(stress, eig1, eig2);
      elastic_diffpot *= site_vol * stress_scale;

      variable_list.set_value_term(4, psi * c_val);
      variable_list.set_value_term(5, del_phi + RT / F * mu_val);
      variable_list.set_value_term(6, elastic_diffpot / F);
      variable_list.set_value_term(12, stress[1][1]);
      variable_list.set_value_term(11, stress[0][0]);

      variable_list.set_value_term(10, stress[0][1]);
      variable_list.set_value_term(9, stress[1][1]);
      variable_list.set_value_term(8, stress[0][0]);
      variable_list.set_value_term(7, stress[1][1]);
    }
  }

  void compute_lhs(
      [[maybe_unused]] FieldContainer<dim, degree, number> &variable_list,
      [[maybe_unused]] const SimulationTimer &sim_timer,
      [[maybe_unused]] unsigned int solve_block_id) const override {
    if (solve_block_id == 1) // mechanics - lhs
    {
      VectorGrad ux =
          variable_list.template get_symmetric_gradient<Vector, LHS>(1);
      ScalarValue psi = variable_list.template get_value<Scalar, Current>(3);
      psi = std::max(psi, ScalarValue(offset));
      VectorValue Cel1 = variable_list.template get_value<Vector, Current>(16);
      VectorValue Cel2 = variable_list.template get_value<Vector, Current>(17);
      VectorValue Cel3 = variable_list.template get_value<Vector, Current>(18);
      dealii::Tensor<2, Mechanics::voigt_tensor_size<dim>, ScalarValue>
          stiffness = get_voigt2D(Cel1, Cel2, Cel3);
      VectorGrad stress;
      Mechanics::compute_stress<dim, ScalarValue>(stiffness, psi * ux, stress);
      variable_list.set_gradient_term(1, stress);
      // std::cout << "  " << stress << "   ";
    }
  }

  number i_0;
  number del_phi;
  number offset;
  number c0;
  number c_ref;
  number RT;
  number F;
  number diffusivity;
  number diff_scale;
  number vegard;
  number site_vol;
  number mol_vol;
  number stress_scale;
  number ocv_q0, ocv_q1, ocv_q2, ocv_q3, ocv_q4, ocv_q5, ocv_q6, ocv_q7;
  number ocv_q8, ocv_q9, ocv_q10, ocv_q11, ocv_q12, ocv_q13, ocv_q14, ocv_q15;
  number ocv_q16, ocv_q17, ocv_q18, ocv_q19, ocv_q20, ocv_q21;
  number V_ref;
  number i_target;
  number V_step;
  number V_min;
  number V_max;
};

PRISMS_PF_END_NAMESPACE
