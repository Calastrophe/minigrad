use num_traits::{NumAssign, Pow};
use rand::distributions::uniform::SampleUniform;
use std::cmp::PartialOrd;
use std::iter::Sum;
use std::ops::{Add, Div, Mul, Neg, Sub};

pub trait MathOps<T>:
    NumAssign + Copy + PartialOrd + Pow<T, Output = T> + Neg<Output = T> + SampleUniform
{
}

impl<T> MathOps<T> for T where
    T: NumAssign + Copy + PartialOrd + Pow<T, Output = T> + Neg<Output = T> + SampleUniform
{
}

#[derive(Debug, Clone, Copy)]
enum Op<T: MathOps<T>> {
    Add,
    Mul,
    Relu,
    Pow(T),
}

type BxValue<T> = Box<Value<T>>;

#[derive(Debug, Clone)]
pub struct Value<T: MathOps<T>> {
    pub data: T,
    pub grad: T,
    prev: Option<(BxValue<T>, Option<BxValue<T>>)>,
    op: Option<Op<T>>,
}

impl<T: MathOps<T>> Value<T> {
    pub fn new(data: T) -> Value<T> {
        Value {
            data,
            grad: T::zero(),
            prev: None,
            op: None,
        }
    }

    fn from_op(data: T, prev: (BxValue<T>, Option<BxValue<T>>), op: Op<T>) -> Self {
        Value {
            data,
            grad: T::zero(),
            prev: Some(prev),
            op: Some(op),
        }
    }

    pub fn backprop(&mut self) {
        self.grad = T::one();

        self.explore();
    }

    fn explore(&mut self) {
        self.backward();

        if let Some(prev) = &mut self.prev {
            match prev {
                (ref mut lhs, Some(ref mut rhs)) => {
                    rhs.explore();
                    lhs.explore();
                }
                (ref mut lhs, None) => {
                    lhs.explore();
                }
            }
        }
    }

    fn backward(&mut self) {
        if let Some(op) = self.op {
            match op {
                Op::Add => {
                    if let Some((ref mut left, Some(ref mut right))) = &mut self.prev {
                        left.grad += self.grad;
                        right.grad += self.grad;
                    }
                }
                Op::Mul => {
                    if let Some((ref mut left, Some(ref mut right))) = &mut self.prev {
                        left.grad += right.data * self.grad;
                        right.grad += left.data * self.grad;
                    }
                }
                Op::Relu => {
                    if let Some((ref mut left, None)) = &mut self.prev {
                        if self.data > T::zero() {
                            left.grad += self.grad;
                        }
                    }
                }
                Op::Pow(exp) => {
                    if let Some((ref mut left, None)) = &mut self.prev {
                        left.grad += exp * left.data.pow(exp - T::one()) * self.grad
                    }
                }
            }
        }
    }

    pub fn relu(self) -> Self {
        let value = if self.data < T::zero() {
            T::zero()
        } else {
            self.data
        };

        Value::from_op(value, (Box::new(self), None), Op::Relu)
    }
}

impl<T: MathOps<T>> Add for Value<T> {
    type Output = Value<T>;

    fn add(self, rhs: Self) -> Self::Output {
        Value::from_op(
            self.data + rhs.data,
            (Box::new(self), Some(Box::new(rhs))),
            Op::Add,
        )
    }
}

impl<T: MathOps<T>> Mul for Value<T> {
    type Output = Value<T>;

    fn mul(self, rhs: Self) -> Self::Output {
        Value::from_op(
            self.data * rhs.data,
            (Box::new(self), Some(Box::new(rhs))),
            Op::Mul,
        )
    }
}

impl<T: MathOps<T>> Pow<T> for Value<T> {
    type Output = Value<T>;

    fn pow(self, rhs: T) -> Self::Output {
        Value::from_op(self.data.pow(rhs), (Box::new(self), None), Op::Pow(rhs))
    }
}

impl<T: MathOps<T>> Neg for Value<T> {
    type Output = Value<T>;

    fn neg(self) -> Self::Output {
        let neg = Value::new(T::one().neg());

        self * neg
    }
}

impl<T: MathOps<T>> Sub for Value<T> {
    type Output = Value<T>;

    fn sub(self, rhs: Self) -> Self::Output {
        self + (-rhs)
    }
}

impl<T: MathOps<T>> Div for Value<T> {
    type Output = Value<T>;

    fn div(self, rhs: Self) -> Self::Output {
        let neg = T::one().neg();

        self * rhs.pow(neg)
    }
}
