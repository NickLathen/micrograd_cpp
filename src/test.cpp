#include <iostream>
#include "micrograd.hh"

void testNeuron() {
  Neuron n{3};

  ValueVec p = n.parameters();
  std::cout << "Parameters=";
  for (ValuePtr v : p) {
    std::cout << v->data << ",";
  }
  std::cout << std::endl;

  ValuePtr a{new Value{.4}};
  ValuePtr b{new Value{.7}};
  ValuePtr c{new Value{.8}};
  ValueVec x{a, b, c};
  ValuePtr d = n(x);
  d->backward();
  std::cout << "a->data=" << a->data << std::endl;
  std::cout << "a->grad=" << a->grad << std::endl;
  std::cout << "b->data=" << b->data << std::endl;
  std::cout << "b->grad=" << b->grad << std::endl;
  std::cout << "c->data=" << c->data << std::endl;
  std::cout << "c->grad=" << c->grad << std::endl;
  std::cout << "d->data=" << d->data << std::endl;
  std::cout << "d->grad=" << d->grad << std::endl;

  std::cout << "ParametersGrad=";
  for (ValuePtr v : p) {
    std::cout << v->grad << ",";
  }
  std::cout << std::endl;
}

void testLayer() {
  Layer l{3, 3};

  ValueVec p = l.parameters();
  std::cout << "Parameters=";
  for (ValuePtr v : p) {
    std::cout << v->data << ",";
  }
  std::cout << std::endl;

  ValuePtr a{new Value{.4}};
  ValuePtr b{new Value{.7}};
  ValuePtr c{new Value{.8}};
  ValueVec x{a, b, c};
  ValueVec outs = l(x);

  std::cout << "Outs=";
  for (ValuePtr o : outs) {
    std::cout << o->data << ",";
  }
  std::cout << std::endl;

  ValuePtr out = outs[0] + outs[1] + outs[2];
  out->backward();

  std::cout << "ParametersGrad=";
  for (ValuePtr v : p) {
    std::cout << v->grad << ",";
  }
  std::cout << std::endl;
}

void testMLP() {
  MLP m{3, {4, 4, 1}};

  ValueVec p = m.parameters();
  std::cout << "Parameters=";
  for (ValuePtr v : p) {
    std::cout << v->data << ",";
  }
  std::cout << std::endl;

  ValuePtr a{new Value{.4}};
  ValuePtr b{new Value{.7}};
  ValuePtr c{new Value{.8}};
  ValueVec x{a, b, c};
  ValueVec outs = m(x);

  std::cout << "Outs=";
  for (ValuePtr o : outs) {
    std::cout << o->data << ",";
  }
  std::cout << std::endl;

  outs[0]->backward();

  std::cout << "ParametersGrad=";
  for (ValuePtr v : p) {
    std::cout << v->grad << ",";
  }
  std::cout << std::endl;
}

void testTrain() {
  MLP m{3, {4, 4, 1}};

  // build input vector
  std::vector<std::vector<double>> xs_raw{
      {2.0, 3.0, -1.0}, {3.0, -1.0, .5}, {.5, 1.0, 1.0}, {1.0, 1.0, -1.0}};
  std::vector<ValueVec> xs{};
  for (std::vector<double> x_raw : xs_raw) {
    ValueVec x{};
    for (double d : x_raw) {
      ValuePtr v{new Value{d}};
      x.push_back(v);
    }
    xs.push_back(x);
  }

  // build real output vector
  ValueVec ys{};
  std::vector<double> ys_raw{1.0, -1.0, -1.0, 1.0};
  for (double y_raw : ys_raw) {
    ValuePtr y{new Value{y_raw}};
    ys.push_back(y);
  }
  for (int k = 0; k < 100; k++) {
    // build predicted output vector
    ValueVec ypred{};
    for (ValueVec x : xs) {
      ypred.push_back(m(x)[0]);
    }
    // build loss vector
    ValuePtr loss{new Value{0.0}};
    for (uint64_t i = 0; i < ys.size(); i++) {
      ValuePtr yloss = (ys[i] - ypred[i])->pow(2);
      loss = loss + yloss;
    }
    // zero grads
    for (ValuePtr p : m.parameters()) {
      p->grad = 0.0;
    }
    loss->backward();
    // gradient descent
    for (ValuePtr p : m.parameters()) {
      p->data += -.1 * p->grad;
    }
    std::cout << k << " " << loss->data << std::endl;
  }
}

int main() {
  std::cout << "start" << std::endl;
  testNeuron();
  testLayer();
  testMLP();
  testTrain();
  std::cout << "end" << std::endl;
}