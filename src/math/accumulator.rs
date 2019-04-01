// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Helper types to accumulate (sum) a lot of values).

use nalgebra::{MatrixMN, U1, U21};

use crate::misc::type_aliases::{Float, Mat6, Vec6};

const SYM_MAT_6_THRESHOLD: u32 = 100;

/// Symmetric matrix accumulator.
#[derive(Clone)]
pub struct SymMat6 {
    nb_data: u32,
    data: MatrixMN<Float, U21, U1>,
    data_hundreds: MatrixMN<Float, U21, U1>,
}

impl SymMat6 {
    /// Initialize with zeros.
    #[inline]
    pub fn new() -> Self {
        Self {
            nb_data: 0,
            data: MatrixMN::<Float, U21, U1>::zeros(),
            data_hundreds: MatrixMN::<Float, U21, U1>::zeros(),
        }
    }

    /// Add another `SymMat6`.
    #[inline]
    pub fn add(&mut self, other: &Self) {
        let nb_data = self.nb_data + other.nb_data;
        if nb_data < SYM_MAT_6_THRESHOLD {
            self.nb_data = nb_data;
            self.data += other.data;
            self.data_hundreds += other.data_hundreds;
        } else {
            self.nb_data = 0;
            self.data_hundreds += self.data + other.data + other.data_hundreds;
            self.data = MatrixMN::<Float, U21, U1>::zeros();
        }
    }

    /// Add a term vv^t to the symmetric matrix accumulator.
    #[inline]
    pub fn add_vec(&mut self, vec: &Vec6) {
        let mut vec_data = unsafe { MatrixMN::<Float, U21, U1>::new_uninitialized() };
        let mut index = 0_usize;
        for j in 0..6 {
            let data_j = unsafe { vec.vget_unchecked(j) };
            for i in j..6 {
                unsafe {
                    *vec_data.vget_unchecked_mut(index) = data_j * vec.vget_unchecked(i);
                }
                index += 1;
            }
        }
        if self.nb_data < SYM_MAT_6_THRESHOLD {
            self.nb_data += 1;
            self.data += vec_data;
        } else {
            self.nb_data = 1;
            self.data_hundreds += self.data;
            self.data = vec_data;
        }
    }

    /// Add a term w * vv^t to the symmetric matrix accumulator.
    #[inline]
    pub fn add_vec_weighted(&mut self, weight: Float, vec: &Vec6) {
        let mut vec_data = unsafe { MatrixMN::<Float, U21, U1>::new_uninitialized() };
        let mut index = 0_usize;
        for j in 0..6 {
            let data_j = weight * unsafe { vec.vget_unchecked(j) };
            for i in j..6 {
                unsafe {
                    *vec_data.vget_unchecked_mut(index) = data_j * vec.vget_unchecked(i);
                }
                index += 1;
            }
        }
        if self.nb_data < SYM_MAT_6_THRESHOLD {
            self.nb_data += 1;
            self.data += vec_data;
        } else {
            self.nb_data = 1;
            self.data_hundreds += self.data;
            self.data = vec_data;
        }
    }

    /// Accumulate all values into the field used in the `to_mat` function.
    /// Clear the other fields.
    #[inline]
    pub fn flush(&mut self) {
        if self.nb_data > 0 {
            self.data_hundreds += self.data;
            self.data = MatrixMN::<Float, U21, U1>::zeros();
            self.nb_data = 0;
        }
    }

    /// Convert the `SymMat6` into a normal matrix `Mat6`.
    /// Requires the use of `flush()` before.
    #[inline]
    pub fn to_mat(&self) -> Mat6 {
        let mut mat = Mat6::zeros();
        let mut index = 0_usize;
        for j in 0..6 {
            for i in j..6 {
                unsafe {
                    let data_ij = self.data_hundreds.vget_unchecked(index);
                    *mat.get_unchecked_mut((i, j)) = *data_ij;
                    *mat.get_unchecked_mut((j, i)) = *data_ij;
                }
                index += 1;
            }
        }
        mat
    }
}

// TESTS #############################################################

#[cfg(test)]
mod tests {

    use super::*;
    use approx;
    use quickcheck_macros;

    const EPSILON: Float = 1e-5;

    /// Test used to temporary reproduce a quickcheck failed test.
    // #[test]
    // fn temp() {
    // }

    // PROPERTY TESTS ################################################

    #[quickcheck_macros::quickcheck]
    fn new() -> bool {
        let accum = SymMat6::new();
        accum.nb_data == 0
    }

    #[quickcheck_macros::quickcheck]
    fn add_vec_101_nb_0(a: Float, b: Float, c: Float, d: Float, e: Float, f: Float) -> bool {
        let vec = Vec6::new(a, b, c, d, e, f);
        let mut accum = SymMat6::new();
        for _ in 0..101 {
            accum.add_vec(&vec);
        }
        accum.nb_data == 1
    }

    #[quickcheck_macros::quickcheck]
    fn add_vec_101_sum(a: Float, b: Float, c: Float, d: Float, e: Float, f: Float) -> bool {
        let vec = Vec6::new(a, b, c, d, e, f);
        let mut accum = SymMat6::new();
        for _ in 0..101 {
            accum.add_vec(&vec);
        }
        let mut accum_sum = SymMat6::new();
        accum_sum.add_vec(&(vec));
        approx::relative_eq!(
            accum.data_hundreds,
            100.0 * accum_sum.data,
            max_relative = EPSILON
        )
    }

    #[quickcheck_macros::quickcheck]
    fn add_vec_201_sum(a: Float, b: Float, c: Float, d: Float, e: Float, f: Float) -> bool {
        let vec = Vec6::new(a, b, c, d, e, f);
        let mut accum = SymMat6::new();
        for _ in 0..201 {
            accum.add_vec(&vec);
        }
        let mut accum_sum = SymMat6::new();
        accum_sum.add_vec(&(vec));
        approx::relative_eq!(
            accum.data_hundreds,
            200.0 * accum_sum.data,
            max_relative = EPSILON
        )
    }

    #[quickcheck_macros::quickcheck]
    fn to_mat(a: Float, b: Float, c: Float, d: Float, e: Float, f: Float) -> bool {
        let vec = Vec6::new(a, b, c, d, e, f);
        let mut accum = SymMat6::new();
        accum.add_vec(&vec);
        accum.flush();
        approx::relative_eq!(
            accum.to_mat(),
            vec * vec.transpose(),
            max_relative = EPSILON
        )
    }

    #[quickcheck_macros::quickcheck]
    fn add_vec_1000_better(a: Float, b: Float, c: Float, d: Float, e: Float, f: Float) -> bool {
        let vec = Vec6::new(a, b, c, d, e, f);
        let base = vec * vec.transpose();
        let ground_truth = 1000.0 * base;

        // Compute the sum without the accumulator.
        let mut normal_sum = Mat6::zeros();
        for _ in 0..1000 {
            normal_sum += base;
        }

        // Compute the sum with the accumulator.
        let mut accum = SymMat6::new();
        for _ in 0..1000 {
            accum.add_vec(&vec);
        }
        accum.flush();

        // Compare both versions.
        let accum_error = (accum.to_mat() - ground_truth).norm();
        let sum_error = (normal_sum - ground_truth).norm();
        accum_error.min(sum_error) == accum_error
    }
}
