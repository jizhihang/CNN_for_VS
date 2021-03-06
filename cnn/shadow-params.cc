#include "shadow-params.h"
#include "tensor.h"
#include "aligned-mem-pool.h"
#include "model.h"

using namespace std;

namespace cnn {

ShadowParameters::ShadowParameters(const Parameters& p) : h(p.values) {
  h.v = (float*)cnn_mm_malloc(h.d.size() * sizeof(float), 256);
  TensorTools::Zero(h);
}

ShadowLookupParameters::ShadowLookupParameters(const LookupParameters& lp) : h(lp.values) {
  for (auto& t : h) {
    t.v = (float*)cnn_mm_malloc(t.d.size() * sizeof(float), 256);
    TensorTools::Zero(t);
  }
}

vector<ShadowParameters> AllocateShadowParameters(const Model& m) {
  vector<ShadowParameters> v;
  v.reserve(m.parameters_list().size());
  for (auto& p : m.parameters_list())
    v.emplace_back(*p);
  return v;
}

vector<ShadowLookupParameters> AllocateShadowLookupParameters(const Model& m) {
  vector<ShadowLookupParameters> v;
  v.reserve(m.lookup_parameters_list().size());
  for (auto& p : m.lookup_parameters_list())
    v.emplace_back(*p);
  return v;
}

} // namespace cnn

