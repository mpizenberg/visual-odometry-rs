// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Type aliases for common types used all over the code base.

use nalgebra as na;

/// At the moment, the library is focused on f32 computation.
pub type Float = f32;

/// A point with two Float coordinates.
pub type Point2 = na::Point2<Float>;
/// A point with three Float coordinates.
pub type Point3 = na::Point3<Float>;

/// A vector with three Float coordinates.
pub type Vec3 = na::Vector3<Float>;
/// A vector with six Float coordinates.
pub type Vec6 = na::Vector6<Float>;

/// A 3x3 matrix of Floats.
pub type Mat3 = na::Matrix3<Float>;
/// A 4x4 matrix of Floats.
pub type Mat4 = na::Matrix4<Float>;
/// A 6x6 matrix of Floats.
pub type Mat6 = na::Matrix6<Float>;

/// A direct 3D isometry, also known as rigid body motion.
pub type Iso3 = na::Isometry3<Float>;
