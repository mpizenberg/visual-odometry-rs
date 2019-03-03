//! Type aliases for common types used all over the code base.

use nalgebra as na;

pub type Float = f32;

pub type Point2 = na::Point2<Float>;
pub type Point3 = na::Point3<Float>;

pub type Vec3 = na::Vector3<Float>;
pub type Vec6 = na::Vector6<Float>;

pub type Mat3 = na::Matrix3<Float>;
pub type Mat4 = na::Matrix4<Float>;
pub type Mat6 = na::Matrix6<Float>;

pub type Iso3 = na::Isometry3<Float>;
