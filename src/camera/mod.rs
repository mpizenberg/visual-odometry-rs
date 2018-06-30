use nalgebra::{Affine2, Matrix3, Point2, Point3, Translation3, UnitQuaternion, Vector3};

type Float = f32;

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
        self.intrinsics.project(self.extrinsics.project(point))
    }

    pub fn back_project(&self, point: Point2<Float>, depth: Float) -> Point3<Float> {
        self.extrinsics
            .back_project(self.intrinsics.back_project(point, depth))
    }
}

// EXTRINSICS ##############################################

#[derive(PartialEq, Debug, Clone)]
pub struct Extrinsics {
    pub translation: Translation3<Float>,
    pub rotation: UnitQuaternion<Float>,
}

impl Extrinsics {
    pub fn new(translation: Translation3<Float>, rotation: UnitQuaternion<Float>) -> Extrinsics {
        Extrinsics {
            translation,
            rotation,
        }
    }

    pub fn project(&self, point: Point3<Float>) -> Point3<Float> {
        self.translation * (self.rotation * point)
    }

    pub fn back_project(&self, point: Point3<Float>) -> Point3<Float> {
        self.rotation.inverse() * (self.translation.inverse() * point)
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
        Affine2::from_matrix_unchecked(Matrix3::new(
            self.focal_length * self.scaling.0,
            self.skew,
            self.principal_point.0,
            0.0,
            self.focal_length * self.scaling.1,
            self.principal_point.1,
            0.0,
            0.0,
            1.0,
        ))
    }

    pub fn project(&self, point: Point3<Float>) -> Vector3<Float> {
        Vector3::new(
            self.focal_length * self.scaling.0 * point[0] + self.skew * point[1]
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

// FUNCTIONS ###############################################
