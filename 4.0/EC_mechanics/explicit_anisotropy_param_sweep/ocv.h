// chebyshev_ocv.h
#ifndef OCV_H
#define OCV_H

#include <array>
#include <cstddef>

template <typename ScalarValue>
class OcvBase
{
public:
  virtual ~OcvBase() = default;

  virtual ScalarValue
  eval_U_ocv(const ScalarValue &c_val) const = 0;
  virtual ScalarValue
  eval_dU_ocv(const ScalarValue &c_val) const = 0;
};

template <typename ScalarValue>
class Chebyshev21Ocv : public OcvBase<ScalarValue>
{
public:
  static constexpr size_t degree = 21;
  // Compile-time coefficients (q0 through q21)
  static constexpr std::array<double, degree + 1> coefficients = {
    /* q0  */ 3.760060570358124e+00,
    /* q1  */ -2.024259589873655e-01,
    /* q2  */ 2.691682752135670e-01,
    /* q3  */ -6.352149288213646e-02,
    /* q4  */ -2.243592647577209e+00,
    /* q5  */ -1.293875959613897e+01,
    /* q6  */ 3.836767727237708e+01,
    /* q7  */ 1.604735968307998e+02,
    /* q8  */ -3.077229112935632e+02,
    /* q9  */ -9.936780252696260e+02,
    /* q10 */ 1.345660647298912e+03,
    /* q11 */ 3.646097875951269e+03,
    /* q12 */ -3.498541621367946e+03,
    /* q13 */ -8.349228920373598e+03,
    /* q14 */ 5.559342157468553e+03,
    /* q15 */ 1.203400233562777e+04,
    /* q16 */ -5.296608096504102e+03,
    /* q17 */ -1.060177043792399e+04,
    /* q18 */ 2.778078192261170e+03,
    /* q19 */ 5.211929686346203e+03,
    /* q20 */ -6.165671379287322e+02,
    /* q21 */ -1.095113531608059e+03};
  static constexpr double ocv_soc_min    = 0.30;
  static constexpr double ocv_soc_span   = 0.69;
  static constexpr double cheb_map_scale = 2.0;

  ScalarValue
  eval_U_ocv(const ScalarValue &c_val) const override
  {
    ScalarValue t_cheb = ((cheb_map_scale * (c_val - ocv_soc_min)) / ocv_soc_span) - 1.0;

    // Horner's method using compile-time coefficients array
    ScalarValue ocv_val = coefficients[degree];
    for (int i = degree - 1; i >= 0; --i)
      {
        ocv_val = (ocv_val * t_cheb) + coefficients[i];
      }
    return ocv_val;
  }

  ScalarValue
  eval_dU_ocv(const ScalarValue &c_val) const override
  {
    ScalarValue t_cheb = ((cheb_map_scale * (c_val - ocv_soc_min)) / ocv_soc_span) - 1.0;
    constexpr double dt_cheb_dc_val = cheb_map_scale / ocv_soc_span;

    // Horner's method for derivative evaluation
    ScalarValue docv_val = static_cast<double>(degree) * coefficients[degree];
    for (int i = degree - 1; i >= 1; --i)
      {
        docv_val = (docv_val * t_cheb) + (static_cast<double>(i) * coefficients[i]);
      }
    return docv_val * dt_cheb_dc_val;
  }
};

template <typename ScalarValue>
class PolynomialOcv : public OcvBase<ScalarValue>
{
public:
  ScalarValue
  eval_U_ocv(const ScalarValue &c_val) const override
  {
    constexpr double a = -0.6;
    constexpr double b = -0.1;
    constexpr double c = 0.3;
    constexpr double d = 4.4;

    ScalarValue ocv_val =
      a * (c_val * c_val * c_val) + b * (c_val * c_val) + c * c_val + d;
    return ocv_val;
  }

  ScalarValue
  eval_dU_ocv(const ScalarValue &c_val) const override
  {
    const double a = -0.6;
    const double b = -0.1;
    const double c = 0.3;

    ScalarValue ocv_val = 3.0 * a * c_val * c_val + 2.0 * b * c_val + c;
    return ocv_val;
  }
};

template <typename ScalarValue>
class PiecewiseCubicSplineOcv : public OcvBase<ScalarValue>
{
public:
  struct SplineSegment
  {
    double a; // Constant (y-offset at knot x0)
    double b; // Linear term coefficient
    double c; // Quadratic term coefficient
    double d; // Cubic term coefficient
  };

  static constexpr size_t num_segments = 5;

  // Knot locations (SOC values along x-axis)
  static constexpr std::array<double, num_segments + 1> knots =
    {0.30, 0.45, 0.60, 0.75, 0.90, 0.99};

  // Pre-calculated polynomial coefficients for each segment:
  // S_i(x) = a + b*(x - x0) + c*(x - x0)^2 + d*(x - x0)^3
  static constexpr std::array<SplineSegment, num_segments> segments = {
    {{3.2000, 0.8500, -0.1200, 0.0400},
     {3.3210, 0.7800, 0.0300, -0.0100},
     {3.4350, 0.8100, -0.0500, 0.0200},
     {3.5520, 1.0500, 0.2100, -0.0300},
     {3.7200, 1.4000, -0.1000, 0.0100}}
  };

  ScalarValue
  eval_U_ocv(const ScalarValue &c_val) const override
  {
    // Default fallback: evaluate final segment (x >= knots[4])
    ScalarValue result = eval_segment_U(num_segments - 1, c_val);

    // Cascade backward through interior knots using SIMD blend masking
    for (int i = static_cast<int>(num_segments) - 2; i >= 0; --i)
      {
        const ScalarValue seg_val = eval_segment_U(static_cast<size_t>(i), c_val);
        const ScalarValue knot_val(knots[i + 1]);

        // If c_val < knot_val, replace lane value with seg_val
        result =
          dealii::compare_and_apply_mask<dealii::SIMDComparison::less_than>(c_val,
                                                                            knot_val,
                                                                            seg_val,
                                                                            result);
      }
    return result;
  }

  ScalarValue
  eval_dU_ocv(const ScalarValue &c_val) const override
  {
    ScalarValue result = eval_segment_dU(num_segments - 1, c_val);

    for (int i = static_cast<int>(num_segments) - 2; i >= 0; --i)
      {
        const ScalarValue seg_val = eval_segment_dU(static_cast<size_t>(i), c_val);
        const ScalarValue knot_val(knots[i + 1]);

        result =
          dealii::compare_and_apply_mask<dealii::SIMDComparison::less_than>(c_val,
                                                                            knot_val,
                                                                            seg_val,
                                                                            result);
      }
    return result;
  }

private:
  ScalarValue
  eval_segment_U(size_t idx, const ScalarValue &c_val) const
  {
    const auto       &seg = segments[idx];
    const ScalarValue dx  = c_val - ScalarValue(knots[idx]);
    return ScalarValue(seg.a) +
           dx *
             (ScalarValue(seg.b) + dx * (ScalarValue(seg.c) + dx * ScalarValue(seg.d)));
  }

  ScalarValue
  eval_segment_dU(size_t idx, const ScalarValue &c_val) const
  {
    const auto       &seg = segments[idx];
    const ScalarValue dx  = c_val - ScalarValue(knots[idx]);
    return ScalarValue(seg.b) +
           dx * (ScalarValue(2.0 * seg.c) + dx * ScalarValue(3.0 * seg.d));
  }
};

#endif // OCV_H