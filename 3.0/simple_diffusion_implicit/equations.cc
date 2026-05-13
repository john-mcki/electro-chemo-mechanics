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
  set_variable_name(0, "c");
  set_variable_type(0, FieldInfo::TensorRank::Scalar);
  set_variable_equation_type(0, ImplicitTimeDependent);
  set_dependencies_value_term_rhs(0, "c, old_1(c), p1");
  set_dependencies_gradient_term_rhs(0, "grad(c), grad(p1)");
  set_dependencies_value_term_lhs(0, "change(c),p1");
  set_dependencies_gradient_term_lhs(0, "grad(change(c)),grad(p1)");

  set_variable_name(1, "p1");
  set_variable_type(1, FieldInfo::TensorRank::Scalar);
  set_variable_equation_type(1, Constant);

  set_variable_name(2, "p2");
  set_variable_type(2, FieldInfo::TensorRank::Scalar);
  set_variable_equation_type(2, Constant);
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
  if (index == 0)
    {
      ScalarValue c = variable_list.template get_value<ScalarValue>(0);
      ScalarGrad cx = variable_list.template get_gradient<ScalarGrad>(0);
      ScalarValue c_old = variable_list.template get_value<ScalarValue>(0,OldOne);
      ScalarValue p = variable_list.template get_value<ScalarValue>(1);
      ScalarGrad px = variable_list.template get_gradient<ScalarGrad>(1);
      ScalarValue px_mag = px.norm() + offset;
      ScalarValue dt = get_timestep();
      ScalarValue B_Neu = -0.1 * (c - c_ref);
      ScalarValue c_term1 = (diffusivity/p) * (px * cx);
      ScalarValue c_term2 = (px_mag/p) * diffusivity * B_Neu;
      ScalarValue eq_c = c_old - c + (dt * (c_term1 + c_term2));
      ScalarGrad eq_cx = -diffusivity * dt * cx;
      variable_list.set_value_term(0, eq_c);
      variable_list.set_gradient_term(0, eq_cx);
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
  ScalarValue change_c = variable_list.template get_value<ScalarValue>(0, Change);
  ScalarGrad change_cx = variable_list.template get_gradient<ScalarGrad>(0, Change);
  ScalarValue p = variable_list.template get_value<ScalarValue>(1);
  ScalarGrad px = variable_list.template get_gradient<ScalarGrad>(1);
  ScalarValue px_mag = px.norm() + offset;
  ScalarValue dt = get_timestep();
  ScalarValue LHS_c_term1 = -1.0 * (diffusivity/p) * (px * change_cx);
  ScalarValue LHS_c_term2 = (px_mag/p) * 0.1 * diffusivity * change_c;
  ScalarValue eq_change_c = change_c + dt * (LHS_c_term1 + LHS_c_term2);
  ScalarGrad eq_change_cx = change_cx * diffusivity * dt;
  variable_list.set_value_term(0, eq_change_c, Change);
  variable_list.set_gradient_term(0, eq_change_cx, Change);
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