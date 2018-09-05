extern crate byteorder;
extern crate image;
extern crate nalgebra;
extern crate num_traits;
extern crate png;
extern crate rand;

pub mod camera;
pub mod candidates;
pub mod colormap;
pub mod helper;
pub mod icl_nuim;
pub mod interop;
pub mod inverse_depth;
pub mod multires;
pub mod multires_float;
pub mod optimization;
pub mod se3;
pub mod so3;
pub mod view;

// Test dependencies

#[cfg(test)]
#[macro_use]
extern crate quickcheck;

#[cfg(test)]
#[macro_use]
extern crate approx;
