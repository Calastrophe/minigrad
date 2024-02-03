use num_traits::Pow;
use std::ops::{Add, Div, Mul, Neg, Sub};

#[derive(Debug, Clone)]
enum Op {
    Add(Box<Value>, Box<Value>),
    Mul(Box<Value>, Box<Value>),
    Relu(Box<Value>),
    Pow(Box<Value>, f64),
}

#[derive(Debug, Clone)]
pub struct Value {
    pub data: f64,
    pub grad: f64,
    op: Option<Op>,
}

impl Value {
    pub fn new(data: f64) -> Value {
        Value {
            data,
            grad: 0.0,
            op: None,
        }
    }

    fn from_op(data: f64, op: Op) -> Self {
        Value {
            data,
            grad: 0.0,
            op: Some(op),
        }
    }

    pub fn backward(&mut self) {
        self.grad = 1.0;

        self.dfs();
    }

    fn dfs(&mut self) {
        self.gradient_fn();

        if let Some(op) = &mut self.op {
            match op {
                Op::Add(lhs, rhs) | Op::Mul(lhs, rhs) => {
                    rhs.dfs();
                    lhs.dfs();
                }
                Op::Relu(lhs) | Op::Pow(lhs, ..) => {
                    lhs.dfs();
                }
            }
        }
    }

    fn gradient_fn(&mut self) {
        if let Some(op) = &mut self.op {
            match op {
                Op::Add(lhs, rhs) => {
                    lhs.grad += self.grad;
                    rhs.grad += self.grad;
                }
                Op::Mul(lhs, rhs) => {
                    lhs.grad += rhs.data * self.grad;
                    rhs.grad += lhs.data * self.grad;
                }
                Op::Relu(lhs) => {
                    if self.data > 0.0 {
                        lhs.grad += self.grad;
                    }
                }
                Op::Pow(lhs, exp) => lhs.grad += *exp * lhs.data.pow(*exp - 1.0) * self.grad,
            }
        }
    }

    pub fn relu(self) -> Self {
        let value = if self.data < 0.0 { 0.0 } else { self.data };

        Value::from_op(value, Op::Relu(Box::new(self)))
    }
}

impl Add for Value {
    type Output = Value;

    fn add(self, rhs: Self) -> Self::Output {
        Value::from_op(self.data + rhs.data, Op::Add(Box::new(self), Box::new(rhs)))
    }
}

impl Mul for Value {
    type Output = Value;

    fn mul(self, rhs: Self) -> Self::Output {
        Value::from_op(self.data * rhs.data, Op::Mul(Box::new(self), Box::new(rhs)))
    }
}

impl<T: Into<f64>> Pow<T> for Value {
    type Output = Value;

    fn pow(self, rhs: T) -> Self::Output {
        let rhs = rhs.into();
        Value::from_op(self.data.pow(rhs), Op::Pow(Box::new(self), rhs))
    }
}

impl Neg for Value {
    type Output = Value;

    fn neg(self) -> Self::Output {
        self * Value::new(-1.0)
    }
}

impl Sub for Value {
    type Output = Value;

    fn sub(self, rhs: Self) -> Self::Output {
        self + (-rhs)
    }
}

impl Div for Value {
    type Output = Value;

    fn div(self, rhs: Self) -> Self::Output {
        self * rhs.pow(-1.0)
    }
}
