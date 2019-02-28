pub mod core;
pub mod dataset;
pub mod math;
pub mod misc;

// Test dependencies

#[cfg(test)]
#[macro_use]
extern crate quickcheck;

#[cfg(test)]
#[macro_use]
extern crate approx;
