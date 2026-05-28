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
  set_dependencies_gradient_term_rhs(0, "grad(u),c,psi_1");
  set_dependencies_gradient_term_lhs(0, "grad(change(u)),psi_1");

  set_variable_name(1, "s");
  set_variable_type(1, FieldInfo::TensorRank::Scalar);
  set_variable_equation_type(1, Auxiliary);
  set_dependencies_value_term_rhs(1, "grad(u),c,psi_1");

  set_variable_name(2, "c");
  set_variable_type(2, FieldInfo::TensorRank::Scalar);
  set_variable_equation_type(2, ImplicitTimeDependent);
  set_dependencies_value_term_rhs(2, "c,old_1(c),s,grad(s),psi_1,grad(psi_1)");
  set_dependencies_gradient_term_rhs(2, "c,grad(c),grad(s),grad(psi_1)");
  set_dependencies_value_term_lhs(2, "c,change(c),s,grad(s),psi_1,grad(psi_1)");
  set_dependencies_gradient_term_lhs(2, "grad(change(c)),grad(s),grad(psi_1)");

  set_variable_name(3, "psi_1");
  set_variable_type(3, FieldInfo::TensorRank::Scalar);
  set_variable_equation_type(3, Constant);

  set_variable_name(4, "psi_2");
  set_variable_type(4, FieldInfo::TensorRank::Scalar);
  set_variable_equation_type(4, Constant);

  set_variable_name(5, "particle_concentration");
  set_variable_type(5, FieldInfo::TensorRank::Scalar);
  set_variable_equation_type(5, ExplicitTimeDependent);
  set_dependencies_value_term_rhs(5, "c,psi_1");
  set_is_postprocessed_field(5, true);

  set_variable_name(6, "f_config");
  set_variable_type(6, FieldInfo::TensorRank::Scalar);
  set_variable_equation_type(6, ExplicitTimeDependent);
  set_dependencies_value_term_rhs(6, "c,psi_1");
  set_is_postprocessed_field(6, true);

  set_variable_name(7, "f_elec");
  set_variable_type(7, FieldInfo::TensorRank::Scalar);
  set_variable_equation_type(7, ExplicitTimeDependent);
  set_dependencies_value_term_rhs(7, "psi_1");
  set_is_postprocessed_field(7, true);

  set_variable_name(8, "f_mech");
  set_variable_type(8, FieldInfo::TensorRank::Scalar);
  set_variable_equation_type(8, ExplicitTimeDependent);
  set_dependencies_value_term_rhs(8, "s,psi_1");
  set_is_postprocessed_field(8, true);

  set_variable_name(9, "f_tot");
  set_variable_type(9, FieldInfo::TensorRank::Scalar);
  set_variable_equation_type(9, ExplicitTimeDependent);
  // set_dependencies_value_term_rhs(9, "c,psi_1");
  set_is_postprocessed_field(9, true);
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
      VectorGrad  ux  = variable_list.template get_symmetric_gradient<VectorGrad>(0);
      ScalarValue c   = variable_list.template get_value<ScalarValue>(2);
      ScalarValue psi = variable_list.template get_value<ScalarValue>(3);
      for (unsigned int i = 0; i < dim; i++)
        {
          ux[i][i] -= omega / (3.0 * (youngs_modulus * 100.0)) * (c - c_ref);
        }
      VectorGrad stress;
      compute_stress<dim, ScalarValue>(stiffness, psi * ux, stress);
      variable_list.set_gradient_term(0,
                                      -stress); // check sign if results are weird
    }
  if (index == 1)
    {
      // Store the hydrostatic stress
      VectorGrad  ux  = variable_list.template get_symmetric_gradient<VectorGrad>(0);
      ScalarValue c   = variable_list.template get_value<ScalarValue>(2);
      ScalarValue psi = variable_list.template get_value<ScalarValue>(3);
      for (unsigned int i = 0; i < dim; i++)
        {
          ux[i][i] -= omega / (3.0 * (youngs_modulus * 100.0)) * (c - c_ref);
        }
      VectorGrad stress;
      compute_stress<dim, ScalarValue>(stiffness, psi * ux, stress);
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
      ScalarValue s  = variable_list.template get_value<ScalarValue>(1);
      ScalarGrad  sx = variable_list.template get_gradient<ScalarGrad>(1);
      // Concentration
      ScalarValue c     = variable_list.template get_value<ScalarValue>(2);
      ScalarGrad  cx    = variable_list.template get_gradient<ScalarGrad>(2);
      ScalarValue c_old = variable_list.template get_value<ScalarValue>(2, OldOne);
      // Order Parameter
      ScalarValue psi       = variable_list.template get_value<ScalarValue>(3);
      ScalarGrad  psi_x     = variable_list.template get_gradient<ScalarGrad>(3);
      ScalarValue psi_x_mag = psi_x.norm() + offset;
      ScalarValue dt        = this->get_timestep();
      // Transport Terms
      ScalarValue c_term1 = (diffusivity / psi) * (psi_x * cx);
      ScalarValue c_term2 = -psi_x / psi * sx * diffusivity * (omega * c) / RT;
      // Rate Terms
      ScalarValue app_pot_energy = F * del_phi;
      ScalarValue mech_energy    = omega * s;
      ScalarValue conf_energy    = RT * std::log(c / c_ref);
      ScalarValue eta = app_pot_energy + mech_energy +
                        conf_energy; // overpotential term, check conf_energy later
      ScalarValue BV_exp_term = exp(-eta / RT);
      ScalarValue c_term3     = (psi_x_mag / psi) * i_0 / F *
                                (pow(BV_exp_term, alpha) -
                                 pow(BV_exp_term, -alpha)); // Full BV reaction rate term
      ScalarGrad  cx_term1    = -diffusivity * cx;
      ScalarGrad  cx_term2    = diffusivity * (omega * c) / RT * sx;
      ScalarValue eq_c        = c_old - c + (dt * (c_term1 + c_term2 + c_term3));
      ScalarGrad  eq_cx       = dt * (cx_term1 + cx_term2);
      variable_list.set_value_term(2, eq_c);
      variable_list.set_gradient_term(2, eq_cx);
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
      ScalarValue psi = variable_list.template get_value<ScalarValue>(3);
      VectorGrad  stress;
      compute_stress<dim, ScalarValue>(stiffness, psi * change_ux, stress);
      variable_list.set_gradient_term(0, stress, Change);
    }
  if (index == 2)
    {
      // Concentration to be changed
      ScalarValue c         = variable_list.template get_value<ScalarValue>(2);
      ScalarValue change_c  = variable_list.template get_value<ScalarValue>(2, Change);
      ScalarGrad  change_cx = variable_list.template get_gradient<ScalarGrad>(2, Change);
      // Gradient of Hydrostatic Stress
      ScalarValue s  = variable_list.template get_value<ScalarValue>(1);
      ScalarGrad  sx = variable_list.template get_gradient<ScalarGrad>(1);
      // Order Parameter, needed but not changed
      ScalarValue psi       = variable_list.template get_value<ScalarValue>(3);
      ScalarGrad  psi_x     = variable_list.template get_gradient<ScalarGrad>(3);
      ScalarValue psi_x_mag = psi_x.norm() + offset;
      ScalarValue dt        = get_timestep();
      // Transport Terms
      ScalarValue LHS_c_term1 = -(diffusivity / psi) * (psi_x * change_cx);
      ScalarValue LHS_c_term2 =
        (omega * diffusivity * change_c) / (RT * psi) * (psi_x * sx);
      // Rate Terms
      ScalarValue app_pot_energy = F * del_phi;
      ScalarValue mech_energy    = omega * s;
      ScalarValue conf_energy    = RT * std::log(c / c_ref);
      ScalarValue eta = app_pot_energy + mech_energy +
                        conf_energy; // overpotential term, check conf_energy later
      ScalarValue BV_exp_term = exp(-eta / (RT));
      ScalarValue LHS_c_term3 =
        -(psi_x_mag / psi) * i_0 / F * change_c / c *
        (alpha * pow(BV_exp_term, alpha) - (1.0 - alpha) * pow(BV_exp_term, -alpha));
      ScalarGrad  LHS_cx_term1 = diffusivity * change_cx;
      ScalarGrad  LHS_cx_term2 = (diffusivity * omega) / (RT) * (sx * change_c);
      ScalarValue eq_change_c = change_c + dt * (LHS_c_term1 + LHS_c_term2 + LHS_c_term3);
      ScalarGrad  eq_change_cx = dt * (LHS_cx_term1 + LHS_cx_term2);

      variable_list.set_value_term(2, eq_change_c, Change);
      variable_list.set_gradient_term(2, eq_change_cx, Change);
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
  ScalarValue s   = variable_list.template get_value<ScalarValue>(1);
  ScalarValue c   = variable_list.template get_value<ScalarValue>(2);
  ScalarValue psi = variable_list.template get_value<ScalarValue>(3);

  ScalarValue particle_concentration = c * psi; // Concentration * domain parameter

  ScalarValue f_config = RT * std::log(c / c_ref) * psi;
  ScalarValue f_elec   = F * del_phi * psi;
  ScalarValue f_mech   = omega * s * psi;
  ScalarValue f_tot    = f_config + f_elec + f_mech;

  variable_list.set_value_term(5, particle_concentration);
  variable_list.set_value_term(6, f_config);
  variable_list.set_value_term(7, f_elec);
  variable_list.set_value_term(8, f_mech);
  variable_list.set_value_term(9, f_tot);
}

#include "custom_pde.inst"

PRISMS_PF_END_NAMESPACE