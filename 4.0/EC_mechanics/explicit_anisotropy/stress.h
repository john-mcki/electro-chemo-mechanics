#pragma once

#include <deal.II/base/tensor.h>

#include <prismspf/config.h>

PRISMS_PF_BEGIN_NAMESPACE
  /**
   * @brief Compute the stress with a given displacement and elasticity tensor.
   *
   * @note This function internally converts to Voigt notation.
   */
  template <unsigned int dim, typename T>
  inline DEAL_II_ALWAYS_INLINE void compute_stress2D(
  const dealii::Tensor<2, dim, T> &strain,
  const dealii::Tensor<1, dim, T> &Cel1,
  const dealii::Tensor<1, dim, T> &Cel2,
  const dealii::Tensor<1, dim, T> &Cel3,
  dealii::Tensor<2, dim, T> &stress)
  {
    if(dim == 2)
      {
        stress[0][0] = Cel1[0]*strain[0][0] + Cel2[1]*strain[1][1] + Cel3[0]*(strain[0][1] + strain[1][0]);
        stress[1][1] = Cel1[1]*strain[1][1] + Cel2[1]*strain[0][0] + Cel3[1]*(strain[0][1] + strain[1][0]);
        stress[0][1] = Cel2[0]*(strain[0][1] + strain[1][0])  + Cel3[0]*strain[0][0] + Cel3[1]*strain[1][1];
        stress[1][0] = stress[0][1];
      }
    else
      {
        dealii::ExcMessage("dim should be 2 for this code");
      }
  }

PRISMS_PF_END_NAMESPACE
