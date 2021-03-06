#ifndef CNN_HSMBUILDER_H
#define CNN_HSMBUILDER_H

#include <vector>
#include <string>
#include "cnn.h"
#include "expr.h"
#include "dict.h"

namespace cnn {

struct Parameters;

// helps with implementation of hierarchical softmax
// read a file with lines of the following format
// CLASSID   word    [freq]
class ClassFactoredSoftmaxBuilder {
 public:
  ClassFactoredSoftmaxBuilder(unsigned rep_dim,
                              const std::string& cluster_file,
                              Dict* word_dict,
                              Model* model);

  // call this once per ComputationGraph
  void new_graph(ComputationGraph& cg);

  // -log(p(c | rep) * p(w | c, rep))
  expr::Expression neg_log_softmax(const expr::Expression& rep, unsigned wordidx);

  // samples a word from p(w,c | rep)
  unsigned sample(const expr::Expression& rep);

 private:
  void ReadClusterFile(const std::string& cluster_file, Dict* word_dict);
  Dict cdict;
  std::vector<int> widx2cidx; // will be -1 if not present
  std::vector<unsigned> widx2cwidx; // word index to word index inside of cluster
  std::vector<std::vector<unsigned>> cidx2words;
  std::vector<bool> singleton_cluster; // does cluster contain a single word type?

  // parameters
  Parameters* p_r2c;
  Parameters* p_cbias;
  std::vector<Parameters*> p_rc2ws;     // len = number of classes
  std::vector<Parameters*> p_rcwbiases; // len = number of classes

  // Expressions for current graph
  inline expr::Expression& get_rc2w(unsigned cluster_idx) {
    expr::Expression& e = rc2ws[cluster_idx];
    if (!e.pg)
      e = expr::parameter(*pcg, p_rc2ws[cluster_idx]);
    return e;
  }
  inline expr::Expression& get_rc2wbias(unsigned cluster_idx) {
    expr::Expression& e = rc2biases[cluster_idx];
    if (!e.pg)
      e = expr::parameter(*pcg, p_rcwbiases[cluster_idx]);
    return e;
  }
  ComputationGraph* pcg;
  expr::Expression r2c;
  expr::Expression cbias;
  std::vector<expr::Expression> rc2ws;
  std::vector<expr::Expression> rc2biases;
};

}  // namespace cnn

#endif
