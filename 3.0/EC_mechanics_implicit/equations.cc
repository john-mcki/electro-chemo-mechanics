// SPDX-FileCopyrightText: © 2025 PRISMS Center at the University of Michigan
// SPDX-License-Identifier: GNU Lesser General Public Version 2.1

#include "custom_pde.h"

#include <prismspf/core/type_enums.h>
#include <prismspf/core/variable_attribute_loader.h>
#include <prismspf/core/variable_container.h>

#include <prismspf/config.h>

PRISMS_PF_BEGIN_NAMESPACE

void
CustomAttributeLoader::load_variable_attributes()
{
  set_variable_name(0, "u");
  set_variable_type(0, FieldInfo::TensorRank::Vector);
  set_variable_equation_type(0, TimeIndependent);
  set_dependencies_gradient_term_rhs(0, "grad(u),C,p1");
  set_dependencies_gradient_term_lhs(0, "grad(change(u)),p1");

  set_variable_name(1, "S");
  set_variable_type(1, FieldInfo::TensorRank::Scalar);
  set_variable_equation_type(1, Auxiliary);
  set_dependencies_value_term_rhs(1, "grad(u),C,p1");

  set_variable_name(2, "C");
  set_variable_type(2, FieldInfo::TensorRank::Scalar);
  set_variable_equation_type(2, ImplicitTimeDependent);
  set_dependencies_value_term_rhs(2, "C,old_1(C),S,grad(S),p1,grad(p1)");
  set_dependencies_gradient_term_rhs(2, "C,grad(C),grad(S),grad(p1)");
  set_dependencies_value_term_lhs(2, "C,change(C),S,grad(S),p1,grad(p1)");
  set_dependencies_gradient_term_lhs(2, "grad(change(C)),grad(S),grad(p1)");

  set_variable_name(3, "p1");
  set_variable_type(3, FieldInfo::TensorRank::Scalar);
  set_variable_equation_type(3, Constant);

  set_variable_name(4, "p2");
  set_variable_type(4, FieldInfo::TensorRank::Scalar);
  set_variable_equation_type(4, Constant);

  set_variable_name(5, "particle_concentration");
  set_variable_type(5, FieldInfo::TensorRank::Scalar);
  set_variable_equation_type(5, ExplicitTimeDependent);
  set_dependencies_value_term_rhs(5, "C,p1");
  set_is_postprocessed_field(5, true);
}

template <unsigned int dim, unsigned int degree, typename number>
void
CustomPDE<dim, degree, number>::compute_explicit_rhs(
  [[maybe_unused]] VariableContainer<dim, degree, number> &variable_list,
  [[maybe_unused]] const dealii::Point<dim, dealii::VectorizedArray<number>> &q_point_loc,
  [[maybe_unused]] const dealii::VectorizedArray<number> &element_volume,
  [[maybe_unused]] Types::Index                           solve_block) const
{}

template <unsigned int dim, unsigned int degree, typename number>
void
CustomPDE<dim, degree, number>::compute_nonexplicit_rhs(
  [[maybe_unused]] VariableContainer<dim, degree, number> &variable_list,
  [[maybe_unused]] const dealii::Point<dim, dealii::VectorizedArray<number>> &q_point_loc,
  [[maybe_unused]] const dealii::VectorizedArray<number> &element_volume,
  [[maybe_unused]] Types::Index                           solve_block,
  [[maybe_unused]] Types::Index                           index) const
{
  using std::exp;
  using std::pow;
  using std::sqrt;
  static const ScalarValue alpha(0.5);
  if (index == 0)
    {
      // Update the displacement
      VectorGrad  ux = variable_list.template get_symmetric_gradient<VectorGrad>(0);
      ScalarValue C  = variable_list.template get_value<ScalarValue>(2);
      ScalarValue p  = variable_list.template get_value<ScalarValue>(3);
      for (unsigned int i = 0; i < dim; i++)
        {
          ux[i][i] -= omega / (3 * (youngs_modulus * 100)) * (C - C_ref);
        }
      VectorGrad stress;
      compute_stress<dim, ScalarValue>(stiffness, p * ux, stress);
      variable_list.set_gradient_term(0,
                                      -stress); // check sign if results are weird
    }
  if (index == 1)
    {
      // Store the hydrostatic stress
      VectorGrad  ux = variable_list.template get_symmetric_gradient<VectorGrad>(0);
      ScalarValue C  = variable_list.template get_value<ScalarValue>(2);
      ScalarValue p  = variable_list.template get_value<ScalarValue>(3);
      for (unsigned int i = 0; i < dim; i++)
        {
          ux[i][i] -= omega / (3.0 * (youngs_modulus * 100.0)) * (C - C_ref);
        }
      VectorGrad stress;
      compute_stress<dim, ScalarValue>(stiffness, p * ux, stress);
      ScalarValue hydrostatic_stress(0.0);
      for (unsigned int i = 0; i < dim; i++)
        {
          hydrostatic_stress += 1.0 / 3.0 * stress[i][i];
        }
      variable_list.set_value_term(1, hydrostatic_stress);
    }
  if (index == 2)
    {
      // Gradient of Hydrostatic Stress
      ScalarValue S  = variable_list.template get_value<ScalarValue>(1);
      ScalarGrad  Sx = variable_list.template get_gradient<ScalarGrad>(1);
      // Concentration
      ScalarValue C     = variable_list.template get_value<ScalarValue>(2);
      ScalarGrad  Cx    = variable_list.template get_gradient<ScalarGrad>(2);
      ScalarValue C_old = variable_list.template get_value<ScalarValue>(2, OldOne);
      // Order Parameter
      ScalarValue p      = variable_list.template get_value<ScalarValue>(3);
      ScalarGrad  px     = variable_list.template get_gradient<ScalarGrad>(3);
      ScalarValue px_mag = px.norm() + offset;
      ScalarValue dt     = this->get_timestep();
      // Transport Terms
      ScalarValue C_term1 = (diffusivity / p) * (px * Cx);
      ScalarValue C_term2 = -px / p * Sx * diffusivity * (omega * C) / (R * Temp);
      // Rate Terms
      ScalarValue app_pot_energy = F * del_phi;
      ScalarValue mech_energy    = omega * S;
      ScalarValue conf_energy    = R * Temp * std::log(C / C_ref);
      ScalarValue eta = app_pot_energy + mech_energy +
                        conf_energy; // overpotential term, check conf_energy later
      ScalarValue BV_exp_term = exp(-eta / (R * Temp));
      ScalarValue C_term3     = (px_mag / p) * i_0 / F *
                                (pow(BV_exp_term, alpha) -
                                 pow(BV_exp_term, -alpha)); // Full BV reaction rate term
      ScalarGrad  Cx_term1    = -diffusivity * Cx;
      ScalarGrad  Cx_term2    = diffusivity * (omega * C) / (R * Temp) * Sx;
      ScalarValue eq_C        = C_old - C + (dt * (C_term1 + C_term2 + C_term3));
      ScalarGrad  eq_Cx       = dt * (Cx_term1 + Cx_term2);
      variable_list.set_value_term(2, eq_C);
      variable_list.set_gradient_term(2, eq_Cx);
    }
}

template <unsigned int dim, unsigned int degree, typename number>
void
CustomPDE<dim, degree, number>::compute_nonexplicit_lhs(
  [[maybe_unused]] VariableContainer<dim, degree, number> &variable_list,
  [[maybe_unused]] const dealii::Point<dim, dealii::VectorizedArray<number>> &q_point_loc,
  [[maybe_unused]] const dealii::VectorizedArray<number> &element_volume,
  [[maybe_unused]] Types::Index                           solve_block,
  [[maybe_unused]] Types::Index                           index) const
{
  static const ScalarValue alpha(0.5);
  using std::exp;
  using std::pow;
  using std::sqrt;
  if (index == 0)
    {
      VectorGrad change_ux =
        variable_list.template get_symmetric_gradient<VectorGrad>(0, Change);
      ScalarValue p = variable_list.template get_value<ScalarValue>(3);
      VectorGrad  stress;
      compute_stress<dim, ScalarValue>(stiffness, p * change_ux, stress);
      variable_list.set_gradient_term(0, stress, Change);
    }
  if (index == 2)
    {
      // Concentration to be changed
      ScalarValue C         = variable_list.template get_value<ScalarValue>(2);
      ScalarValue change_C  = variable_list.template get_value<ScalarValue>(2, Change);
      ScalarGrad  change_Cx = variable_list.template get_gradient<ScalarGrad>(2, Change);
      // Gradient of Hydrostatic Stress
      ScalarValue S  = variable_list.template get_value<ScalarValue>(1);
      ScalarGrad  Sx = variable_list.template get_gradient<ScalarGrad>(1);
      // Order Parameter, needed but not changed
      ScalarValue p      = variable_list.template get_value<ScalarValue>(3);
      ScalarGrad  px     = variable_list.template get_gradient<ScalarGrad>(3);
      ScalarValue px_mag = px.norm() + offset;
      ScalarValue dt     = get_timestep();
      // Transport Terms
      ScalarValue LHS_C_term1 = -(diffusivity / p) * (px * change_Cx);
      ScalarValue LHS_C_term2 =
        (omega * diffusivity * change_C) / (R * Temp * p) * (px * Sx);
      // Rate Terms
      ScalarValue app_pot_energy = F * del_phi;
      ScalarValue mech_energy    = omega * S;
      ScalarValue conf_energy_x  = (1.0 / (C * C_ref)) * change_C;
      ScalarValue eta =
        app_pot_energy + mech_energy; // overpotential term, check conf_energy later
      // ScalarValue LHS_C_term3 = (px_mag/p) * kc * diffusivity * change_C; //No
      // longer included due to change in boudnary condition ScalarValue
      // LHS_C_term3 = -(px_mag/p) * i_0/F * std::sqrt(C_ref/C) * (2.0 *
      // std::exp(-0.5 * del_phi) * std::exp((-0.5 * omega * S)/(R * Temp)) + 0.5
      // * std::exp(0.5 * del_phi) * std::exp((0.5 * omega * S)/(R * Temp))) *
      // change_C;

      ScalarValue BV_exp_term  = std::exp(-eta / (R * Temp));
      ScalarValue LHS_C_term3  = -(px_mag / p) * i_0 / F *
                                 (pow(BV_exp_term, alpha) * pow(conf_energy_x, -alpha) -
                                  pow(BV_exp_term, -alpha) * pow(conf_energy_x, alpha));
      ScalarGrad  LHS_Cx_term1 = diffusivity * change_Cx;
      ScalarGrad  LHS_Cx_term2 = (diffusivity * omega) / (R * Temp) * (Sx * change_C);
      ScalarValue eq_change_C = change_C + dt * (LHS_C_term1 + LHS_C_term2 + LHS_C_term3);
      ScalarGrad  eq_change_Cx = dt * (LHS_Cx_term1 + LHS_Cx_term2);

      variable_list.set_value_term(2, eq_change_C, Change);
      variable_list.set_gradient_term(2, eq_change_Cx, Change);
    }
}

template <unsigned int dim, unsigned int degree, typename number>
void
CustomPDE<dim, degree, number>::compute_postprocess_explicit_rhs(
  [[maybe_unused]] VariableContainer<dim, degree, number> &variable_list,
  [[maybe_unused]] const dealii::Point<dim, dealii::VectorizedArray<number>> &q_point_loc,
  [[maybe_unused]] const dealii::VectorizedArray<number> &element_volume,
  [[maybe_unused]] Types::Index                           solve_block) const
{
  ScalarValue particle_concentration =
    variable_list.template get_value<ScalarValue>(2) *
    variable_list.template get_value<ScalarValue>(3); // Concentration * domain parameter
  variable_list.set_value_term(5, particle_concentration);
}

#include "custom_pde.inst"

PRISMS_PF_END_NAMESPACE