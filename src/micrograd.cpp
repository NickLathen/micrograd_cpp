#include "micrograd.hh"
#include <cmath>
#include <functional>
#include <iostream>
#include <random>
#include <unordered_set>

Value::Value(double _data) : data{_data} {};
Value::Value(double _data, ValueVec _children, Op op)
    : data{_data},
      grad{0.0},
      _prev{_children},
      _op{op} {

      };
ValuePtr Value::pow(double y) {
  ValuePtr exponent{new Value(y)};
  ValueVec children = {shared_from_this(), exponent};
  ValuePtr out{new Value{std::pow(data, y), children, Op_pow}};
  return out;
}

ValuePtr Value::exp() {
  ValueVec children = {shared_from_this()};
  ValuePtr out{new Value{std::exp(data), children, Op_exp}};
  return out;
}

ValuePtr Value::tanh() {
  ValueVec children = {shared_from_this()};
  ValuePtr out{new Value{std::tanh(data), children, Op_tanh}};
  return out;
}

inline void Value::_backward_add() {
  _prev[0]->grad += grad;
  _prev[1]->grad += grad;
}
inline void Value::_backward_mul() {
  _prev[0]->grad += _prev[1]->data * grad;
  _prev[1]->grad += _prev[0]->data * grad;
}
inline void Value::_backward_exp() {
  _prev[0]->grad += data * grad;
}
inline void Value::_backward_pow() {
  double exponent = _prev[1]->data;
  _prev[0]->grad += exponent * std::pow(_prev[0]->data, exponent - 1) * grad;
}
inline void Value::_backward_tanh() {
  _prev[0]->grad += (1 - std::pow(data, 2)) * grad;
}
void Value::_backward() {
  switch (_op) {
    case Op_add:
      return _backward_add();
    case Op_mul:
      return _backward_mul();
    case Op_exp:
      return _backward_exp();
    case Op_pow:
      return _backward_pow();
    case Op_tanh:
      return _backward_tanh();
    case Op_no:
      return;
  }
}

void Value::backward() {
  ValueVec topo{};
  std::unordered_set<ValuePtr> visited{};
  std::function<void(ValuePtr)> build_topo = [&topo, &visited,
                                              &build_topo](ValuePtr v) {
    if (visited.count(v) == 0) {
      visited.insert(v);
      for (ValuePtr child : v->_prev) {
        build_topo(child);
      }
      topo.push_back(v);
    }
  };
  grad = 1.0;
  build_topo(shared_from_this());
  for (ValueVec::reverse_iterator v = topo.rbegin(); v != topo.rend(); ++v) {
    (*v)->_backward();
  }
}

ValuePtr operator+(ValuePtr lhs, ValuePtr rhs) {
  ValueVec children = {lhs, rhs};
  ValuePtr out{new Value{lhs->data + rhs->data, children, Op_add}};
  return out;
}

ValuePtr operator+(ValuePtr lhs, double r) {
  ValuePtr rhs{new Value{r}};
  return lhs + rhs;
}

ValuePtr operator+(double l, ValuePtr rhs) {
  return rhs + l;
}

ValuePtr operator*(ValuePtr lhs, ValuePtr rhs) {
  ValueVec children = {lhs, rhs};
  ValuePtr out{new Value{lhs->data * rhs->data, children, Op_mul}};
  return out;
}

ValuePtr operator*(ValuePtr lhs, double r) {
  ValuePtr rhs{new Value{r}};
  return lhs * rhs;
}

ValuePtr operator*(double l, ValuePtr rhs) {
  return rhs * l;
}

ValuePtr operator-(ValuePtr lhs) {
  return lhs * -1.0;
}

ValuePtr operator-(ValuePtr lhs, ValuePtr rhs) {
  return lhs + (-rhs);
}

ValuePtr operator-(ValuePtr lhs, double r) {
  return lhs + (-r);
}

ValuePtr operator-(double l, ValuePtr rhs) {
  return l + (-rhs);
}

ValuePtr operator/(ValuePtr lhs, ValuePtr rhs) {
  return lhs * (rhs->pow(-1.0));
}

ValuePtr operator/(ValuePtr lhs, double r) {
  return lhs * (1 / r);
}

ValuePtr operator/(double l, ValuePtr rhs) {
  return l * (rhs->pow(-1));
}

Neuron::Neuron(int nin) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dis(-1.0, 1.0);
  for (int i = 0; i < nin; i++) {
    ValuePtr wi{new Value(dis(gen))};
    w.push_back(wi);
  }
  ValuePtr _b{new Value(dis(gen))};
  b = _b;
}
ValuePtr Neuron::operator()(ValueVec x) {
  ValuePtr act{b};
  for (u_int64_t i = 0; i < x.size(); i++) {
    ValuePtr wi = w[i];
    ValuePtr xi = x[i];
    ValuePtr p = wi * xi;
    act = act + p;
  }
  return act->tanh();
}
ValueVec Neuron::parameters() {
  ValueVec params{w};
  params.push_back(b);
  return params;
}

Layer::Layer(int nin, int nout) {
  for (int i = 0; i < nout; i++) {
    neurons.push_back(Neuron{nin});
  }
}
ValueVec Layer::operator()(ValueVec x) {
  ValueVec outs{};
  for (Neuron n : neurons) {
    outs.push_back(n(x));
  }
  return outs;
}
ValueVec Layer::parameters() {
  ValueVec params{};
  for (Neuron n : neurons) {
    ValueVec nParams = n.parameters();
    params.insert(params.end(), nParams.begin(), nParams.end());
  }
  return params;
}

MLP::MLP(int nin, std::vector<int> nouts) {
  std::vector<int> sz{};
  sz.push_back(nin);
  sz.insert(sz.end(), nouts.begin(), nouts.end());
  for (uint64_t i = 0; i < nouts.size(); i++) {
    layers.push_back(Layer(sz[i], sz[i + 1]));
  }
}
ValueVec MLP::operator()(ValueVec x) {
  for (Layer l : layers) {
    x = l(x);
  }
  return x;
}
ValueVec MLP::parameters() {
  ValueVec params{};
  for (Layer l : layers) {
    ValueVec lParams = l.parameters();
    params.insert(params.end(), lParams.begin(), lParams.end());
  }
  return params;
}