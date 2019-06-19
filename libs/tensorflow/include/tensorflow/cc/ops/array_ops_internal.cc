// This file is MACHINE GENERATED! Do not edit.


#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/array_ops_internal.h"

namespace tensorflow {
namespace ops {
namespace internal {
// NOTE: This namespace has internal TensorFlow details that
// are not part of TensorFlow's public API.

BroadcastGradientArgs::BroadcastGradientArgs(const ::tensorflow::Scope& scope,
                                             ::tensorflow::Input s0,
                                             ::tensorflow::Input s1) {
  if (!scope.ok()) return;
  auto _s0 = ::tensorflow::ops::AsNodeOut(scope, s0);
  if (!scope.ok()) return;
  auto _s1 = ::tensorflow::ops::AsNodeOut(scope, s1);
  if (!scope.ok()) return;
  ::tensorflow::Node* ret;
  const auto unique_name = scope.GetUniqueNameForOp("BroadcastGradientArgs");
  auto builder = ::tensorflow::NodeBuilder(unique_name, "BroadcastGradientArgs")
                     .Input(_s0)
                     .Input(_s1)
  ;
  scope.UpdateBuilder(&builder);
  scope.UpdateStatus(builder.Finalize(scope.graph(), &ret));
  if (!scope.ok()) return;
  scope.UpdateStatus(scope.DoShapeInference(ret));
  this->operation = Operation(ret);
  ::tensorflow::NameRangeMap _outputs_range;
  ::tensorflow::Status _status_ = ::tensorflow::NameRangesForNode(*ret, ret->op_def(), nullptr, &_outputs_range);
  if (!_status_.ok()) {
    scope.UpdateStatus(_status_);
    return;
  }

  this->r0 = Output(ret, _outputs_range["r0"].first);
  this->r1 = Output(ret, _outputs_range["r1"].first);
}

LowerBound::LowerBound(const ::tensorflow::Scope& scope, ::tensorflow::Input
                       sorted_inputs, ::tensorflow::Input values, const
                       LowerBound::Attrs& attrs) {
  if (!scope.ok()) return;
  auto _sorted_inputs = ::tensorflow::ops::AsNodeOut(scope, sorted_inputs);
  if (!scope.ok()) return;
  auto _values = ::tensorflow::ops::AsNodeOut(scope, values);
  if (!scope.ok()) return;
  ::tensorflow::Node* ret;
  const auto unique_name = scope.GetUniqueNameForOp("LowerBound");
  auto builder = ::tensorflow::NodeBuilder(unique_name, "LowerBound")
                     .Input(_sorted_inputs)
                     .Input(_values)
                     .Attr("out_type", attrs.out_type_)
  ;
  scope.UpdateBuilder(&builder);
  scope.UpdateStatus(builder.Finalize(scope.graph(), &ret));
  if (!scope.ok()) return;
  scope.UpdateStatus(scope.DoShapeInference(ret));
  this->operation = Operation(ret);
  this->output = Output(ret, 0);
}

LowerBound::LowerBound(const ::tensorflow::Scope& scope, ::tensorflow::Input
                       sorted_inputs, ::tensorflow::Input values)
  : LowerBound(scope, sorted_inputs, values, LowerBound::Attrs()) {}

MirrorPadGrad::MirrorPadGrad(const ::tensorflow::Scope& scope,
                             ::tensorflow::Input input, ::tensorflow::Input
                             paddings, StringPiece mode) {
  if (!scope.ok()) return;
  auto _input = ::tensorflow::ops::AsNodeOut(scope, input);
  if (!scope.ok()) return;
  auto _paddings = ::tensorflow::ops::AsNodeOut(scope, paddings);
  if (!scope.ok()) return;
  ::tensorflow::Node* ret;
  const auto unique_name = scope.GetUniqueNameForOp("MirrorPadGrad");
  auto builder = ::tensorflow::NodeBuilder(unique_name, "MirrorPadGrad")
                     .Input(_input)
                     .Input(_paddings)
                     .Attr("mode", mode)
  ;
  scope.UpdateBuilder(&builder);
  scope.UpdateStatus(builder.Finalize(scope.graph(), &ret));
  if (!scope.ok()) return;
  scope.UpdateStatus(scope.DoShapeInference(ret));
  this->operation = Operation(ret);
  this->output = Output(ret, 0);
}

RefIdentity::RefIdentity(const ::tensorflow::Scope& scope, ::tensorflow::Input
                         input) {
  if (!scope.ok()) return;
  auto _input = ::tensorflow::ops::AsNodeOut(scope, input);
  if (!scope.ok()) return;
  ::tensorflow::Node* ret;
  const auto unique_name = scope.GetUniqueNameForOp("RefIdentity");
  auto builder = ::tensorflow::NodeBuilder(unique_name, "RefIdentity")
                     .Input(_input)
  ;
  scope.UpdateBuilder(&builder);
  scope.UpdateStatus(builder.Finalize(scope.graph(), &ret));
  if (!scope.ok()) return;
  scope.UpdateStatus(scope.DoShapeInference(ret));
  this->operation = Operation(ret);
  this->output = Output(ret, 0);
}

UpperBound::UpperBound(const ::tensorflow::Scope& scope, ::tensorflow::Input
                       sorted_inputs, ::tensorflow::Input values, const
                       UpperBound::Attrs& attrs) {
  if (!scope.ok()) return;
  auto _sorted_inputs = ::tensorflow::ops::AsNodeOut(scope, sorted_inputs);
  if (!scope.ok()) return;
  auto _values = ::tensorflow::ops::AsNodeOut(scope, values);
  if (!scope.ok()) return;
  ::tensorflow::Node* ret;
  const auto unique_name = scope.GetUniqueNameForOp("UpperBound");
  auto builder = ::tensorflow::NodeBuilder(unique_name, "UpperBound")
                     .Input(_sorted_inputs)
                     .Input(_values)
                     .Attr("out_type", attrs.out_type_)
  ;
  scope.UpdateBuilder(&builder);
  scope.UpdateStatus(builder.Finalize(scope.graph(), &ret));
  if (!scope.ok()) return;
  scope.UpdateStatus(scope.DoShapeInference(ret));
  this->operation = Operation(ret);
  this->output = Output(ret, 0);
}

UpperBound::UpperBound(const ::tensorflow::Scope& scope, ::tensorflow::Input
                       sorted_inputs, ::tensorflow::Input values)
  : UpperBound(scope, sorted_inputs, values, UpperBound::Attrs()) {}

}  // namespace internal
}  // namespace ops
}  // namespace tensorflow
