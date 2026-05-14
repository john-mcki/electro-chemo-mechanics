// SPDX-FileCopyrightText: © 2025 PRISMS Center at the University of Michigan
// SPDX-License-Identifier: GNU Lesser General Public Version 2.1

#include "custom_pde.h"

#include <prismspf/core/initial_conditions.h>
#include <prismspf/core/nonuniform_dirichlet.h>

#include <prismspf/user_inputs/user_input_parameters.h>

#include <prismspf/utilities/utilities.h>

#include <prismspf/config.h>

#include <cmath>

PRISMS_PF_BEGIN_NAMESPACE

template <unsigned int dim, unsigned int degree, typename number>
void
CustomPDE<dim, degree, number>::set_initial_condition(
  [[maybe_unused]] const unsigned int       &index,
  [[maybe_unused]] const unsigned int       &component,
  [[maybe_unused]] const dealii::Point<dim> &point,
  [[maybe_unused]] number                   &scalar_value,
  [[maybe_unused]] number                   &vector_component_value) const
{
  const double dim_size =
    this->get_user_inputs().get_spatial_discretization().get_size()[0];
  std::vector<double> center = {dim_size / 2, dim_size / 2, dim_size / 2};
  double              dist   = 0.0;
  for (unsigned int i = 0; i < dim; i++)
    {
      dist += (point[i] - center[i]) * (point[i] - center[i]);
    }
  dist = std::sqrt(dist);
  double domain_parameter =
    0.5 * (1.0 + offset) -
    0.5 * (1 - offset) *
      std::tanh((dist * dist - radius * radius) / (interface_width * radius));
  if (index == 0)
    {
      vector_component_value = 0.0; // initial displacement u
    }
  if (index == 2)
    {
      scalar_value = c_init * domain_parameter + (c_ref) * (1.0 - domain_parameter);
    }
  if (index == 3)
    {
      scalar_value = domain_parameter;
    }
  if (index == 4)
    {
      scalar_value = 1.0 - domain_parameter;
    }
}

template <unsigned int dim, unsigned int degree, typename number>
void
CustomPDE<dim, degree, number>::set_nonuniform_dirichlet(
  [[maybe_unused]] const unsigned int       &index,
  [[maybe_unused]] const unsigned int       &boundary_id,
  [[maybe_unused]] const unsigned int       &component,
  [[maybe_unused]] const dealii::Point<dim> &point,
  [[maybe_unused]] number                   &scalar_value,
  [[maybe_unused]] number                   &vector_component_value) const
{}

#include "custom_pde.inst"

PRISMS_PF_END_NAMESPACE