use nalgebra::{Affine2, Isometry3, Matrix3, Point2, Point3, Vector3};

pub type Float = f32;

#[derive(PartialEq, Debug, Clone)]
pub struct Camera {
    pub intrinsics: Intrinsics,
    pub extrinsics: Extrinsics,
}

impl Camera {
    pub fn new(intrinsics: Intrinsics, extrinsics: Extrinsics) -> Camera {
        Camera {
            intrinsics,
            extrinsics,
        }
    }

    pub fn project(&self, point: Point3<Float>) -> Vector3<Float> {
        self.intrinsics
            .project(extrinsics::project(&self.extrinsics, point))
    }

    pub fn back_project(&self, point: Point2<Float>, depth: Float) -> Point3<Float> {
        extrinsics::back_project(&self.extrinsics, self.intrinsics.back_project(point, depth))
    }

    pub fn multi_res(self, n: usize) -> Vec<Camera> {
        self.intrinsics
            .clone()
            .multi_res(n)
            .into_iter()
            .map(|intrinsics| Self::new(intrinsics, self.extrinsics.clone()))
            .collect()
    }

    pub fn half_res(&self) -> Camera {
        Self::new(self.intrinsics.half_res(), self.extrinsics.clone())
    }
}

// EXTRINSICS ##############################################

pub type Extrinsics = Isometry3<Float>;

pub mod extrinsics {
    use super::{Extrinsics, Float};
    use nalgebra::{Isometry3, Point3, Translation3, UnitQuaternion};

    pub fn from_parts(
        translation: Translation3<Float>,
        rotation: UnitQuaternion<Float>,
    ) -> Extrinsics {
        Isometry3::from_parts(translation, rotation)
    }

    pub fn project(motion: &Extrinsics, point: Point3<Float>) -> Point3<Float> {
        motion.rotation.inverse() * (motion.translation.inverse() * point)
    }

    pub fn back_project(motion: &Extrinsics, point: Point3<Float>) -> Point3<Float> {
        motion * point
    }
}

// INTRINSICS ##############################################

#[derive(PartialEq, Debug, Clone)]
pub struct Intrinsics {
    pub principal_point: (Float, Float),
    pub focal_length: Float,
    pub scaling: (Float, Float),
    pub skew: Float,
}

impl Intrinsics {
    pub fn matrix(&self) -> Affine2<Float> {
        #[cfg_attr(rustfmt, rustfmt_skip)]
        Affine2::from_matrix_unchecked(Matrix3::new(
            self.focal_length * self.scaling.0, self.skew, self.principal_point.0,
            0.0, self.focal_length * self.scaling.1,       self.principal_point.1,
            0.0, 0.0, 1.0,
        ))
    }

    pub fn multi_res(self, n: usize) -> Vec<Intrinsics> {
        let mut intrinsics = Vec::new();
        if n > 0 {
            intrinsics.push(self);
            for _ in 1..n {
                let new = { intrinsics.last().unwrap().half_res() };
                intrinsics.push(new);
            }
        }
        intrinsics
    }

    pub fn half_res(&self) -> Intrinsics {
        let (cx, cy) = self.principal_point;
        let (sx, sy) = self.scaling;
        Intrinsics {
            principal_point: ((cx + 0.5) / 2.0 - 0.5, (cy + 0.5) / 2.0 - 0.5),
            focal_length: self.focal_length,
            scaling: (sx / 2.0, sy / 2.0),
            skew: self.skew,
        }
    }

    pub fn project(&self, point: Point3<Float>) -> Vector3<Float> {
        Vector3::new(
            self.focal_length * self.scaling.0 * point[0]
                + self.skew * point[1]
                + self.principal_point.0 * point[2],
            self.focal_length * self.scaling.1 * point[1] + self.principal_point.1 * point[2],
            point[2],
        )
    }

    pub fn back_project(&self, point: Point2<Float>, depth: Float) -> Point3<Float> {
        let z = depth;
        let y = (point[1] - self.principal_point.1) * z / (self.focal_length * self.scaling.1);
        let x = ((point[0] - self.principal_point.0) * z - self.skew * y)
            / (self.focal_length * self.scaling.0);
        Point3::new(x, y, z)
    }
}
