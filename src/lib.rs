//! # Visual Odometry in Rust (vors)
//!
//! `visual-odometry-rs` is a library providing implementation
//! of visual odometry algorithms fully in Rust.

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
