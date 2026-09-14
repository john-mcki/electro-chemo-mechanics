// field_registry.h
#ifndef FIELD_REGISTRY_H
#define FIELD_REGISTRY_H

#include <array>
#include <set>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include <prismspf/core/field_attributes.h>

PRISMS_PF_BEGIN_NAMESPACE

// Helper struct for method chaining dependency string expressions
struct DependencyBuilder {
  std::string expr;

  DependencyBuilder grad()  const { return DependencyBuilder{"grad(" + expr + ")"}; }
  DependencyBuilder old_1() const { return DependencyBuilder{"old_1(" + expr + ")"}; }
  DependencyBuilder lhs()   const { return DependencyBuilder{"lhs(" + expr + ")"}; }

  operator std::string() const { return expr; }
};

// Descriptor struct linking field name, index, type, and dependency builders
struct FieldSpec {
  std::string_view name;
  unsigned int index;
  bool is_vector = false;

  // Implicit conversion to FieldAttributes for vector creation
  operator FieldAttributes() const {
    return is_vector ? FieldAttributes(std::string(name), Vector)
                     : FieldAttributes(std::string(name));
  }

  // Implicit conversion for index extraction
  operator unsigned int() const { return index; }

  // Method chaining entry points
  DependencyBuilder grad()  const { return DependencyBuilder{std::string(name)}.grad(); }
  DependencyBuilder old_1() const { return DependencyBuilder{std::string(name)}.old_1(); }
  DependencyBuilder lhs()   const { return DependencyBuilder{std::string(name)}.lhs(); }

  operator std::string() const { return std::string(name); }
};

template <unsigned int dim, bool HasFracture = false>
struct FieldStruct {
  static constexpr FieldSpec c                     { "c",                     0  };
  static constexpr FieldSpec u                     { "u",                     1,  true };
  static constexpr FieldSpec mu                    { "mu",                    2  };
  static constexpr FieldSpec psi                   { "psi",                   3  };
  static constexpr FieldSpec rxn                   { "rxn",                   4  };
  static constexpr unsigned int pp_base = 5;
  static constexpr FieldSpec particle_concentration { "particle_concentration", pp_base  };
  static constexpr FieldSpec mu_chem                { "mu_chem",                pp_base + 1  };
  static constexpr FieldSpec mu_elastic             { "mu_elastic",             pp_base + 2  };
  static constexpr FieldSpec overpotential          { "overpotential",          pp_base + 3  };
  static constexpr FieldSpec stress_diag           { "stress_diag",           pp_base + 4, true };
  static constexpr FieldSpec stress_off_diag       { "stress_off_diag",       pp_base + 5, true };
  static constexpr FieldSpec stress_principal      { "stress_principal",      pp_base + 6, true };
  static constexpr unsigned int tensor_base = pp_base + 7;
  static constexpr FieldSpec Cel1                  { "Cel1",       tensor_base, true };
  static constexpr FieldSpec Cel2                  { "Cel2",       tensor_base + 1, true };
  static constexpr FieldSpec Cel3                  { "Cel3",       tensor_base + 2, true };
  static constexpr FieldSpec eig1                  { "eig1",       tensor_base + 3, true };
  static constexpr FieldSpec D1                    { "D1",         tensor_base + 4, true };
  static constexpr unsigned int dim_conditional_offset = (dim == 3 ? 0 : 100); // index is big if fields aren't used
  static constexpr unsigned int dim_conditional_isvector = (dim == 3 ? true : false);
  static constexpr FieldSpec eig2              { "eig2",  tensor_base + 5 ,     dim_conditional_isvector };
  static constexpr FieldSpec D2                { "D2",    tensor_base + 6 ,     dim_conditional_isvector };
  static constexpr FieldSpec Cel4              { "Cel4",  tensor_base + 7  + dim_conditional_offset, true };
  static constexpr FieldSpec Cel5              { "Cel5",  tensor_base + 8  + dim_conditional_offset, true };
  static constexpr FieldSpec Cel6              { "Cel6",  tensor_base + 9  + dim_conditional_offset, true };
  static constexpr FieldSpec Cel7              { "Cel7",  tensor_base + 10 + dim_conditional_offset, true };
  static constexpr unsigned int fracture_base = tensor_base + (dim == 3 ? 6 : 2);
  static constexpr unsigned int fracture_conditional_offset = (HasFracture ? 0 : 200); // index is big if fields aren't used
  static constexpr FieldSpec fracture_field      { "fracture_field",      fracture_base };
  static constexpr FieldSpec gc_field            { "gc_field",            fracture_base + fracture_conditional_offset + 1};
  static constexpr FieldSpec strain_energy_plus  { "strain_energy_plus",  fracture_base + fracture_conditional_offset + 2};
  static constexpr FieldSpec strain_energy_minus { "strain_energy_minus", fracture_base + fracture_conditional_offset + 3};

  static auto get_field_attributes() {
    // Collect all field specs
    std::vector<FieldSpec> fields = {psi, rxn, u, c, mu, particle_concentration,
       overpotential, mu_chem, mu_elastic, stress_diag, stress_off_diag, stress_principal,
       Cel1, Cel2, Cel3, eig1, eig2, D1, D2};
    if constexpr (dim == 3) {
      fields.insert(fields.end(), {Cel4, Cel5, Cel6, Cel7});
    }
    if constexpr (HasFracture) {
      fields.insert(fields.end(), {fracture_field, gc_field, strain_energy_plus, strain_energy_minus});
    }
    // Sort by index so ordering relies entirely on the FieldSpec index definitions
    std::sort(fields.begin(), fields.end(), [](const FieldSpec& a, const FieldSpec& b) {
      return a.index < b.index;
    });
    // Convert to std::vector<FieldAttributes>
    std::vector<FieldAttributes> attributes;
    attributes.reserve(fields.size());
    for (const auto& field : fields) {
      attributes.push_back(field); // Uses implicit conversion to FieldAttributes
    }
    return attributes;
  }
};

// Dimension-aware subset helpers for group dependencies
template <unsigned int dim, bool HasFracture = false>
struct FieldSubsets {
  using Fields = FieldStruct<dim, HasFracture>;
  static constexpr auto stiffness() {
    if constexpr (dim == 2) {
      return std::array{Fields::Cel1, Fields::Cel2, Fields::Cel3};
    } else {
      return std::array{Fields::Cel1, Fields::Cel2, Fields::Cel3,
         Fields::Cel4, Fields::Cel5, Fields::Cel6, Fields::Cel7};
    }
  }

  static constexpr unsigned int stiffness_size() {
    if constexpr (dim == 2) {
      return 3;
    } else {
      return 7;
    }
  }

  static constexpr auto eigenstrain() {
    return std::array{Fields::eig1, Fields::eig2};
  }

  static constexpr auto diffusion() {
    return std::array{Fields::D1, Fields::D2};
  }

  static constexpr auto postprocess() {
    if constexpr (HasFracture) {
      return std::array{Fields::stress_diag, Fields::stress_off_diag, Fields::stress_principal,
            Fields::mu_chem, Fields::mu_elastic, Fields::overpotential, Fields::particle_concentration,
            Fields::strain_energy_plus, Fields::strain_energy_minus};
    } else {
      return std::array{Fields::stress_diag, Fields::stress_off_diag, Fields::stress_principal,
            Fields::overpotential, Fields::mu_chem, Fields::mu_elastic, Fields::particle_concentration};
    }
  }
};

// Variadic helper to unpack and flatten single items, chains, and sets into std::set<unsigned int>
template <typename... Args>
std::set<unsigned int> collect_indices(Args&&... args) {
  std::set<unsigned int> result;

  auto unpack = [&](auto&& item) {
    using T = std::decay_t<decltype(item)>;
    if constexpr (std::is_same_v<T, FieldSpec>) {
      result.insert(item.index);
    } else if constexpr (std::is_integral_v<T>) {
      result.insert(static_cast<unsigned int>(item));
    } else {
      // Iterates through arrays, spans, or vectors of FieldSpec
      for (const auto& elem : item) {
        if constexpr (std::is_same_v<std::decay_t<decltype(elem)>, FieldSpec>) {
          result.insert(elem.index);
        } else {
          result.insert(static_cast<unsigned int>(elem));
        }
      }
    }
  };

  (unpack(std::forward<Args>(args)), ...);
  return result;
}

// Variadic helper to unpack and flatten single items, chains, and sets into std::set<std::string>
template <typename... Args>
std::set<std::string> collect_deps(Args&&... args) {
  std::set<std::string> result;

  auto unpack = [&](auto&& item) {
    using T = std::decay_t<decltype(item)>;
    if constexpr (std::is_same_v<T, FieldSpec>) {
      result.insert(std::string(item.name));
    } else if constexpr (std::is_same_v<T, DependencyBuilder> ||
                         std::is_same_v<T, std::string>) {
      result.insert(std::string(item));
    } else if constexpr (std::is_convertible_v<T, std::string_view>) {
      result.insert(std::string(item));
    } else {
      // Handles std::array, std::vector, std::set, std::span of FieldSpec/DependencyBuilder/string
      for (const auto& elem : item) {
        using ElemT = std::decay_t<decltype(elem)>;
        if constexpr (std::is_same_v<ElemT, FieldSpec>) {
          result.insert(std::string(elem.name));
        } else {
          result.insert(std::string(elem));
        }
      }
    }
  };

  (unpack(std::forward<Args>(args)), ...);
  return result;
}

PRISMS_PF_END_NAMESPACE

#endif // FIELD_REGISTRY_H