# minigrad

This is a reimplementation of [micrograd](https://github.com/karpathy/micrograd/) in Rust.

Essentially, it is a small scalar-valued auto gradient engine, but thanks to Rust's trait system it is highly generic.

There is a small *opinionated* neural network library adapted from [micrograd](https://github.com/karpathy/micrograd/) on top of it.
