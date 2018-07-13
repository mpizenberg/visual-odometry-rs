use nalgebra::{Matrix3, Quaternion, UnitQuaternion};
use std::f32::consts::PI;

pub type Float = f32;

const EPSILON: Float = 1e-10;
const _1_8: Float = 0.125;
const _1_48: Float = 1.0 / 48.0;
const _1_384: Float = 1.0 / 384.0;
const _1_3840: Float = 1.0 / 3840.0;

pub struct Element {
    w1: Float,
    w2: Float,
    w3: Float,
}

impl Element {
    // Hat operator.
    // Goes from so3 parameterization to so3 element (skew-symmetric matrix).
    pub fn hat(&self) -> Matrix3<Float> {
        Matrix3::from_column_slice(&[
            0.0, self.w3, -self.w2, -self.w3, 0.0, self.w1, self.w2, -self.w1, 0.0
        ])
    }

    // Vee operator.
    // Inverse of hat operator.
    // Warning! does not check that the given matrix is skew-symmetric.
    pub fn vee(mat: Matrix3<Float>) -> Self {
        // shall be improved for performance.
        Self {
            w1: mat[(2, 1)],
            w2: mat[(0, 2)],
            w3: mat[(1, 0)],
        }
    }

    // Compute the exponential map from Lie algebra so3 to Lie group SO3.
    // Goes from so3 parameterization to SO3 element (rotation matrix).
    // Also returns the norm of the Lie algebra element.
    pub fn exp(&self) -> (UnitQuaternion<Float>, Float) {
        let w1 = self.w1;
        let w2 = self.w2;
        let w3 = self.w3;
        let theta_2 = w1 * w1 + w2 * w2 + w3 * w3;
        let theta = theta_2.sqrt();
        let real_factor;
        let imag_factor;
        if theta < EPSILON {
            real_factor = 1.0 - theta_2 * (_1_8 - theta_2 * _1_384);
            imag_factor = 0.5 - theta_2 * (_1_48 - theta_2 * _1_3840);
        } else {
            let half_theta = 0.5 * theta;
            real_factor = half_theta.cos();
            imag_factor = half_theta.sin() / theta;
        }
        (
            UnitQuaternion::from_quaternion(Quaternion::new(
                real_factor,
                w1 * imag_factor,
                w2 * imag_factor,
                w3 * imag_factor,
            )),
            theta,
        )
    }

    // Compute the logarithm map from the Lie group SO3 to the Lie algebra so3.
    // Inverse of the exponential map.
    // Also returns the form of the Lie algebra element.
    //
    // Computation taken from the Sophus library.
    //
    // Atan-based log thanks to:
    //
    // C. Hertzberg et al.
    // "Integrating Generic Sensor Fusion Algorithms with Sound State
    // Representation through Encapsulation of Manifolds"
    // Information Fusion, 2011
    pub fn log(rotation: UnitQuaternion<Float>) -> (Self, Float) {
        let imag_vector = rotation.vector();
        let imag_norm_2 = imag_vector.norm_squared();
        let imag_norm = imag_norm_2.sqrt();
        let scalar = rotation.scalar();
        let theta;
        let tangent;
        if imag_norm < EPSILON {
            let scalar_2 = scalar * scalar;
            let atan_coef = 2.0 * (1.0 - imag_norm_2 / scalar_2) / scalar;
            theta = atan_coef * imag_norm;
            tangent = atan_coef * imag_vector;
        } else if scalar.abs() < EPSILON {
            theta = if scalar > 0.0 { PI } else { -PI };
            tangent = (theta / imag_norm) * imag_vector;
        } else {
            theta = 2.0 * (imag_norm / scalar).atan();
            tangent = (theta / imag_norm) * imag_vector;
        }
        // TODO: improve efficiency here.
        // TODO: add the norm theta to element type?
        let w1 = tangent[0];
        let w2 = tangent[1];
        let w3 = tangent[2];
        (Self { w1, w2, w3 }, theta)
    }
}
