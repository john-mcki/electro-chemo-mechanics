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
  // set_dependencies_gradient_term_rhs(1,"grad(u),C,p"); //Double check gradient
  // dependencies

  set_variable_name(2, "c");
  set_variable_type(2, FieldInfo::TensorRank::Scalar);
  set_variable_equation_type(2, ExplicitTimeDependent);
  set_dependencies_value_term_rhs(2, "c,s,grad(s),psi_1,grad(psi_1)");
  set_dependencies_gradient_term_rhs(2, "c,grad(c),grad(s),grad(psi_1)");

  set_variable_name(3, "psi_1");
  set_variable_type(3, FieldInfo::TensorRank::Scalar);
  set_variable_equation_type(3, Constant);

  set_variable_name(4, "psi_2");
  set_variable_type(4, FieldInfo::TensorRank::Scalar);
  set_variable_equation_type(4, Constant);
}

template <unsigned int dim, unsigned int degree, typename number>
void
CustomPDE<dim, degree, number>::compute_explicit_rhs(
  [[maybe_unused]] VariableContainer<dim, degree, number> &variable_list,
  [[maybe_unused]] const dealii::Point<dim, dealii::VectorizedArray<number>> &q_point_loc,
  [[maybe_unused]] const dealii::VectorizedArray<number> &element_volume,
  [[maybe_unused]] Types::Index                           solve_block) const
{
  using std::exp;
  using std::pow;
  using std::sqrt;
  static const ScalarValue alpha(0.5);
  // Gradient of Hydrostatic Stress
  ScalarValue s  = variable_list.template get_value<ScalarValue>(1);
  ScalarGrad  sx = variable_list.template get_gradient<ScalarGrad>(1);
  // Concentration
  ScalarValue c  = variable_list.template get_value<ScalarValue>(2);
  ScalarGrad  cx = variable_list.template get_gradient<ScalarGrad>(2);
  // Order Parameter
  ScalarValue psi       = variable_list.template get_value<ScalarValue>(3);
  ScalarGrad  psi_x     = variable_list.template get_gradient<ScalarGrad>(3);
  ScalarValue psi_x_mag = psi_x.norm() + offset; // Initial value is equal to offset
  ScalarValue dt        = this->get_timestep();
  // Rate Terms
  ScalarValue app_pot_energy = F * del_phi;
  ScalarValue mech_energy    = omega * s;
  ScalarValue conf_energy    = RT * std::log(c / c_ref);
  ScalarValue eta            = app_pot_energy + mech_energy +
                               conf_energy; // overpotential term, check conf_energy later
  ScalarValue BV_exp_term    = exp(-eta / RT);
  ScalarValue c_term1        = (diffusivity / psi) * (psi_x * cx);
  ScalarValue c_term2        = -psi_x / psi * sx * diffusivity * (omega * c) / RT;
  ScalarValue c_term3 =
    (psi_x_mag / psi) * i_0 / F *
    (pow(BV_exp_term, alpha) - pow(BV_exp_term, -alpha)); // Full BV reaction rate term
  ScalarGrad  cx_term1 = -diffusivity * cx;
  ScalarGrad  cx_term2 = diffusivity * (omega * c) / (RT) *sx;
  ScalarValue eq_c     = c + (dt * (c_term1 + c_term2 + c_term3));
  ScalarGrad  eq_cx    = dt * (cx_term1 + cx_term2);
  variable_list.set_value_term(2, eq_c);
  variable_list.set_gradient_term(2, eq_cx);
}

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
  if (index == 0)
    {
      VectorGrad  ux  = variable_list.template get_symmetric_gradient<VectorGrad>(0);
      ScalarValue c   = variable_list.template get_value<ScalarValue>(2);
      ScalarValue psi = variable_list.template get_value<ScalarValue>(3);
      for (unsigned int i = 0; i < dim; i++)
        {
          ux[i][i] -= omega / (3.0 * (youngs_modulus * 100.0)) * (c - c_ref);
        }
      VectorGrad stress;
      compute_stress<dim, ScalarValue>(stiffness, psi * ux, stress);
      variable_list.set_gradient_term(0, -stress); // check sign if results are weird
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
  if (index == 0)
    {
      VectorGrad change_ux =
        variable_list.template get_symmetric_gradient<VectorGrad>(0, Change);
      ScalarValue p = variable_list.template get_value<ScalarValue>(3);
      VectorGrad  stress;
      compute_stress<dim, ScalarValue>(stiffness, p * change_ux, stress);
      variable_list.set_gradient_term(0, stress, Change);
    }
}

template <unsigned int dim, unsigned int degree, typename number>
void
CustomPDE<dim, degree, number>::compute_postprocess_explicit_rhs(
  [[maybe_unused]] VariableContainer<dim, degree, number> &variable_list,
  [[maybe_unused]] const dealii::Point<dim, dealii::VectorizedArray<number>> &q_point_loc,
  [[maybe_unused]] const dealii::VectorizedArray<number> &element_volume,
  [[maybe_unused]] Types::Index                           solve_block) const
{}

#include "custom_pde.inst"

PRISMS_PF_END_NAMESPACE