#pragma once

#include <deal.II/base/tensor.h>
#include <prismspf/utilities/mechanics.h>
#include <prismspf/core/field_container.h>
#include "field_registry.h"

PRISMS_PF_BEGIN_NAMESPACE
template <unsigned int dim, typename VectorizedType>
struct TensorHelper {
  // Constructs a symmetric Tensor<2, dim, VectorizedType> from evaluation fields
  static dealii::Tensor<2, dim, VectorizedType> extract_rank2_symm(
      const auto &vars, 
      const std::array<FieldSpec, 2> &fields) {
    
    dealii::Tensor<2, dim, VectorizedType> tensor;

    // fields[0] is diagonal component vector field (Dxx, Dyy, [Dzz])
    // fields[1] is off-diagonal scalar/vector field (Dxy, [Dyz, Dxz])
    const auto &diag = vars.template get_value<Vector, Current>(fields[0].index);
    
    tensor[0][0] = diag[0]; // Dxx
    tensor[1][1] = diag[1]; // Dyy
    
    if constexpr (dim == 2) {
      const auto &xy = vars.template get_value<Scalar, Current>(fields[1].index);
      tensor[0][1] = xy;      // Dxy
      tensor[1][0] = xy;      // Dyx (Symmetric alias—stored directly in SIMD register)
    } else {
      tensor[2][2] = diag[2]; // Dzz
      const auto &off_diag = vars.template get_value<Vector, Current>(fields[1].index);
      tensor[0][1] = off_diag[2]; tensor[1][0] = off_diag[2]; // Dxy
      tensor[0][2] = off_diag[1]; tensor[2][0] = off_diag[1]; // Dxz
      tensor[1][2] = off_diag[0]; tensor[2][1] = off_diag[0]; // Dyz
    }

    return tensor;
  }

  static dealii::Tensor<2, Mechanics::voigt_tensor_size<dim>, VectorizedType> extract_stiffness(
      const auto &vars, 
      const std::array<FieldSpec, FieldSubsets<dim>::stiffness_size()>  &fields) {
    
    dealii::Tensor<2, Mechanics::voigt_tensor_size<dim>, VectorizedType> elasticity_tensor;

    VectorizedType safe_value(1e-6);

    auto set_sym = [&](int i, int j, VectorizedType val)
      {
          elasticity_tensor[i][j] = val;
          elasticity_tensor[j][i] = val;
      };

    if constexpr (dim == 2)
      {
        const auto &c1 = vars.template get_value<Vector, Current>(fields[0].index);
        const auto &c2 = vars.template get_value<Vector, Current>(fields[1].index);
        const auto &c3 = vars.template get_value<Vector, Current>(fields[2].index);

        elasticity_tensor[0][0] = std::max(c1[0], safe_value); // C11
        elasticity_tensor[1][1] = std::max(c1[1], safe_value); // C22
        elasticity_tensor[2][2] = std::max(c2[0], safe_value); // C66
        set_sym(0, 1, c2[1]); // C12
        set_sym(0, 2, c3[0]); // C16
        set_sym(1, 2, c3[1]); // C26
      }
    if constexpr (dim == 3)
      {
        const auto &c1 = vars.template get_value<Vector, Current>(fields[0].index);
        const auto &c2 = vars.template get_value<Vector, Current>(fields[1].index);
        const auto &c3 = vars.template get_value<Vector, Current>(fields[2].index);
        const auto &c4 = vars.template get_value<Vector, Current>(fields[3].index);
        const auto &c5 = vars.template get_value<Vector, Current>(fields[4].index);
        const auto &c6 = vars.template get_value<Vector, Current>(fields[5].index);
        const auto &c7 = vars.template get_value<Vector, Current>(fields[6].index);
        elasticity_tensor[0][0] = std::max(c1[0], safe_value); // C11
        elasticity_tensor[1][1] = std::max(c1[1], safe_value); // C22
        elasticity_tensor[2][2] = std::max(c1[2], safe_value); // C33
        elasticity_tensor[3][3] = std::max(c2[0], safe_value); // C44
        elasticity_tensor[4][4] = std::max(c2[1], safe_value); // C55
        elasticity_tensor[5][5] = std::max(c2[2], safe_value); // C66
        set_sym(0, 1, c3[0]); // C12
        set_sym(0, 2, c3[1]); // C13
        set_sym(0, 3, c3[2]); // C14
        set_sym(0, 4, c4[0]); // C15
        set_sym(0, 5, c4[1]); // C16
        set_sym(1, 2, c4[2]); // C23
        set_sym(1, 3, c5[0]); // C24
        set_sym(1, 4, c5[1]); // C25
        set_sym(1, 5, c5[2]); // C26
        set_sym(2, 3, c6[0]); // C34
        set_sym(2, 4, c6[1]); // C35
        set_sym(2, 5, c6[2]); // C36
        set_sym(3, 4, c7[0]); // C45
        set_sym(3, 5, c7[1]); // C46
        set_sym(4, 5, c7[2]); // C56
      }
    return elasticity_tensor;
  }
};

PRISMS_PF_END_NAMESPACE
