// This file is MACHINE GENERATED! Do not edit.


#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/string_ops_internal.h"

namespace tensorflow {
namespace ops {
namespace internal {
// NOTE: This namespace has internal TensorFlow details that
// are not part of TensorFlow's public API.

StaticRegexFullMatch::StaticRegexFullMatch(const ::tensorflow::Scope& scope,
                                           ::tensorflow::Input input,
                                           StringPiece pattern) {
  if (!scope.ok()) return;
  auto _input = ::tensorflow::ops::AsNodeOut(scope, input);
  if (!scope.ok()) return;
  ::tensorflow::Node* ret;
  const auto unique_name = scope.GetUniqueNameForOp("StaticRegexFullMatch");
  auto builder = ::tensorflow::NodeBuilder(unique_name, "StaticRegexFullMatch")
                     .Input(_input)
                     .Attr("pattern", pattern)
  ;
  scope.UpdateBuilder(&builder);
  scope.UpdateStatus(builder.Finalize(scope.graph(), &ret));
  if (!scope.ok()) return;
  scope.UpdateStatus(scope.DoShapeInference(ret));
  this->operation = Operation(ret);
  this->output = Output(ret, 0);
}

StaticRegexReplace::StaticRegexReplace(const ::tensorflow::Scope& scope,
                                       ::tensorflow::Input input, StringPiece
                                       pattern, StringPiece rewrite, const
                                       StaticRegexReplace::Attrs& attrs) {
  if (!scope.ok()) return;
  auto _input = ::tensorflow::ops::AsNodeOut(scope, input);
  if (!scope.ok()) return;
  ::tensorflow::Node* ret;
  const auto unique_name = scope.GetUniqueNameForOp("StaticRegexReplace");
  auto builder = ::tensorflow::NodeBuilder(unique_name, "StaticRegexReplace")
                     .Input(_input)
                     .Attr("pattern", pattern)
                     .Attr("rewrite", rewrite)
                     .Attr("replace_global", attrs.replace_global_)
  ;
  scope.UpdateBuilder(&builder);
  scope.UpdateStatus(builder.Finalize(scope.graph(), &ret));
  if (!scope.ok()) return;
  scope.UpdateStatus(scope.DoShapeInference(ret));
  this->operation = Operation(ret);
  this->output = Output(ret, 0);
}

StaticRegexReplace::StaticRegexReplace(const ::tensorflow::Scope& scope,
                                       ::tensorflow::Input input, StringPiece
                                       pattern, StringPiece rewrite)
  : StaticRegexReplace(scope, input, pattern, rewrite, StaticRegexReplace::Attrs()) {}

}  // namespace internal
}  // namespace ops
}  // namespace tensorflow
