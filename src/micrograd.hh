#ifndef MICROGRAD_HH
#define MICROGRAD_HH

#include <memory>
#include <vector>

class Value;
using ValuePtr = std::shared_ptr<Value>;
using ValueVec = std::vector<ValuePtr>;

enum Op { Op_add, Op_mul, Op_pow, Op_exp, Op_tanh, Op_no };

class Value : public std::enable_shared_from_this<Value> {
 public:
  Value(double _data);
  Value(double _data, ValueVec _children, Op op);

  double data{0.0};
  double grad{0.0};
  ValueVec _prev{};
  Op _op{Op_no};

  ValuePtr pow(double y);
  ValuePtr exp();
  ValuePtr tanh();

  void backward();

 private:
  void _backward();
  void _backward_add();
  void _backward_mul();
  void _backward_exp();
  void _backward_pow();
  void _backward_tanh();
};

ValuePtr operator+(ValuePtr lhs, ValuePtr rhs);
ValuePtr operator+(ValuePtr lhs, double r);
ValuePtr operator+(double l, ValuePtr rhs);
ValuePtr operator*(ValuePtr lhs, ValuePtr rhs);
ValuePtr operator*(ValuePtr lhs, double r);
ValuePtr operator*(double l, ValuePtr rhs);
ValuePtr operator-(ValuePtr lhs);
ValuePtr operator-(ValuePtr lhs, ValuePtr rhs);
ValuePtr operator-(ValuePtr lhs, double r);
ValuePtr operator-(double l, ValuePtr rhs);
ValuePtr operator/(ValuePtr lhs, ValuePtr rhs);
ValuePtr operator/(ValuePtr lhs, double r);
ValuePtr operator/(double l, ValuePtr rhs);

class Neuron {
 public:
  Neuron(int nin);

  ValuePtr operator()(ValueVec x);
  ValueVec parameters();

 private:
  ValueVec w{};
  ValuePtr b;
};

class Layer {
 public:
  Layer(int nin, int nout);

  ValueVec operator()(ValueVec x);
  ValueVec parameters();

 private:
  std::vector<Neuron> neurons{};
};

class MLP {
 public:
  MLP(int nin, std::vector<int> nouts);

  ValueVec operator()(ValueVec x);
  ValueVec parameters();

 private:
  std::vector<Layer> layers{};
};

#endif
