use num_traits::{NumAssign, Pow};
use std::cmp::PartialOrd;
use std::ops::{Add, Div, Mul, Neg, Sub};

#[derive(Debug, Clone, Copy)]
enum Op<T>
where
    T: NumAssign + Copy + PartialOrd + Pow<T, Output = T> + Neg<Output = T>,
{
    Add,
    Mul,
    Relu,
    Pow(T),
}

type BxValue<T> = Box<Value<T>>;

pub struct Value<T>
where
    T: NumAssign + Copy + PartialOrd + Pow<T, Output = T> + Neg<Output = T>,
{
    pub data: T,
    pub grad: T,
    prev: Option<(BxValue<T>, Option<BxValue<T>>)>,
    op: Option<Op<T>>,
}

impl<T> Value<T>
where
    T: NumAssign + Copy + PartialOrd + Pow<T, Output = T> + Neg<Output = T>,
{
    pub fn new(data: T) -> Value<T> {
        Value {
            data,
            grad: T::zero(),
            prev: None,
            op: None,
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

    fn from_op(data: T, prev: (BxValue<T>, Option<BxValue<T>>), op: Op<T>) -> Self {
        Value {
            data,
            grad: T::zero(),
            prev: Some(prev),
            op: Some(op),
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
}

impl<T> Add for Value<T>
where
    T: NumAssign + Copy + PartialOrd + Pow<T, Output = T> + Neg<Output = T>,
{
    type Output = Value<T>;

    fn add(self, rhs: Self) -> Self::Output {
        Value::from_op(
            self.data + rhs.data,
            (Box::new(self), Some(Box::new(rhs))),
            Op::Add,
        )
    }
}

impl<T> Mul for Value<T>
where
    T: NumAssign + Copy + PartialOrd + Pow<T, Output = T> + Neg<Output = T>,
{
    type Output = Value<T>;

    fn mul(self, rhs: Self) -> Self::Output {
        Value::from_op(
            self.data * rhs.data,
            (Box::new(self), Some(Box::new(rhs))),
            Op::Mul,
        )
    }
}

impl<T> Pow<T> for Value<T>
where
    T: NumAssign + Copy + PartialOrd + Pow<T, Output = T> + Neg<Output = T>,
{
    type Output = Value<T>;

    fn pow(self, rhs: T) -> Self::Output {
        Value::from_op(self.data.pow(rhs), (Box::new(self), None), Op::Pow(rhs))
    }
}

impl<T> Neg for Value<T>
where
    T: NumAssign + Copy + PartialOrd + Pow<T, Output = T> + Neg<Output = T>,
{
    type Output = Value<T>;

    fn neg(self) -> Self::Output {
        let neg = Value::new(T::one().neg());

        self * neg
    }
}

impl<T> Sub for Value<T>
where
    T: NumAssign + Copy + PartialOrd + Pow<T, Output = T> + Neg<Output = T>,
{
    type Output = Value<T>;

    fn sub(self, rhs: Self) -> Self::Output {
        self + (-rhs)
    }
}

impl<T> Div for Value<T>
where
    T: NumAssign + Copy + PartialOrd + Pow<T, Output = T> + Neg<Output = T>,
{
    type Output = Value<T>;

    fn div(self, rhs: Self) -> Self::Output {
        let neg = T::one().neg();

        self * rhs.pow(neg)
    }
}
