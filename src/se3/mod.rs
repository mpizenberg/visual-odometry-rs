// Interesting reads
// * Sophus c++ library: https://github.com/strasdat/Sophus
// * Ethan Eade course on Lie Groups for 2D and 3D transformations:
//     * details: http://ethaneade.com/lie.pdf
//     * summary: http://ethaneade.com/lie_groups.pdf

use nalgebra::{Isometry3, Matrix3, Matrix4, Translation3, Vector3};
use so3;

pub type Float = f32;

const EPSILON_TAYLOR: Float = 1e-2;
const _1_6: Float = 1.0 / 6.0;
const _1_12: Float = 1.0 / 12.0;
const _1_24: Float = 1.0 / 24.0;
const _1_120: Float = 1.0 / 120.0;

#[derive(PartialEq, Debug, Copy, Clone)]
pub struct Twist {
    v: Vector3<Float>,
    w: so3::Element,
}

// Hat operator.
// Goes from se3 parameters to se3 element (4x4 matrix).
pub fn hat(xi: Twist) -> Matrix4<Float> {
    let w1 = xi.w[0];
    let w2 = xi.w[1];
    let w3 = xi.w[2];
    let v1 = xi.v[0];
    let v2 = xi.v[1];
    let v3 = xi.v[2];
    Matrix4::from_column_slice(&[
        0.0, w3, -w2, 0.0, -w3, 0.0, w1, 0.0, w2, -w1, 0.0, 0.0, v1, v2, v3, 0.0
    ])
}

// Vee operator.
// Inverse of hat operator.
// Warning! does not check that the given
// top left 3x3 sub-matrix is skew-symmetric.
pub fn vee(mat: Matrix4<Float>) -> Twist {
    // TODO: improve performance.
    Twist {
        w: Vector3::from_column_slice(&[mat[(2, 1)], mat[(0, 2)], mat[(1, 0)]]),
        v: Vector3::from_column_slice(&[mat[(0, 3)], mat[(1, 3)], mat[(1, 0)]]),
    }
}

// Compute the exponential map from Lie algebra se3 to Lie group SE3.
// Goes from se3 parameterization to SE3 element (rigid body motion).
pub fn exp(xi: Twist) -> Isometry3<Float> {
    let (rotation, theta) = so3::exp(xi.w);
    let theta_2 = theta * theta;
    let (omega, omega_2) = (so3::hat(xi.w), so3::hat_2(xi.w));
    let v = if theta < EPSILON_TAYLOR {
        Matrix3::identity() + (0.5 - _1_24 * theta_2) * omega + (_1_6 - _1_120 * theta_2) * omega_2
    } else {
        Matrix3::identity() + (1.0 - theta.cos()) / theta_2 * omega
            + (theta - theta.sin()) / (theta * theta_2) * omega_2
    };
    Isometry3::from_parts(Translation3::from_vector(v * xi.v), rotation)
}

// Compute the logarithm map from the Lie group SE3 to the Lie algebra se3.
// Inverse of the exponential map.
pub fn log(iso: Isometry3<Float>) -> Twist {
    let (w, theta) = so3::log(iso.rotation);
    let (omega, omega_2) = (so3::hat(w), so3::hat_2(w));
    let v_inv = if theta < EPSILON_TAYLOR {
        Matrix3::identity() - 0.5 * omega + _1_12 * omega_2
    } else {
        let half_theta = 0.5 * theta;
        Matrix3::identity() - 0.5 * omega + (1.0 - half_theta / half_theta.tan()) * omega_2
    };
    Twist {
        v: v_inv * iso.translation.vector,
        w: w,
    }
}

// TESTS #############################################################

#[cfg(test)]
mod tests {

    use super::*;
    use nalgebra::UnitQuaternion;

    // The best precision I get for a round trip
    // with exact trigonometric computations ("else" branches)
    // is around 1e-6.
    const EPSILON_ROUNDTRIP_APPROX: Float = 1e-1;

    #[test]
    fn exp_log_round_trip() {
        let v = Vector3::zeros();
        let w = Vector3::zeros();
        let xi = Twist { v, w };
        assert_eq!(xi, round_trip_from_algebra(xi));
    }

    #[test]
    // Unit test with a case that doesn't go better than 1e-6 on round trip error.
    // Even in exact computation branches (set EPSILON_TAYLOR = 1e-30 for example).
    fn log_exp_round_trip_1() {
        let translation = &[0.0, 0.0, 1.0];
        let rotation = &[0.0, 2.0, 0.0];
        let rigid_motion = gen_rigid_motion(translation, rotation);
        assert_abs_diff_eq!(
            rigid_motion,
            round_trip_from_group(rigid_motion),
            epsilon = EPSILON_ROUNDTRIP_APPROX
        )
    }

    // PROPERTY TESTS ################################################

    // quickcheck! {
    //     fn log_exp_round_trip(t1: Float, t2: Float, t3:Float, a1: Float, a2: Float, a3: Float) -> bool {
    //         let rigid_motion = gen_rigid_motion(&[t1,t2,t3], &[a1,a2,a3]);
    //         abs_diff_eq!(
    //             rigid_motion,
    //             round_trip_from_group(rigid_motion),
    //             epsilon = EPSILON_ROUNDTRIP_APPROX
    //         )
    //     }
    // }

    // GENERATORS ####################################################

    fn gen_rigid_motion(translation_slice: &[Float; 3], angles: &[Float; 3]) -> Isometry3<Float> {
        let translation = Translation3::from_vector(Vector3::from_column_slice(translation_slice));
        let rotation = UnitQuaternion::from_euler_angles(angles[0], angles[1], angles[2]);
        Isometry3::from_parts(translation, rotation)
    }

    // HELPERS #######################################################

    fn round_trip_from_algebra(xi: Twist) -> Twist {
        log(exp(xi))
    }

    fn round_trip_from_group(rigid_motion: Isometry3<Float>) -> Isometry3<Float> {
        exp(log(rigid_motion))
    }
}
