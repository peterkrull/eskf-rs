//! # Error State Kalman Filter (ESKF)
//! An [Error State Kalman Filter](https://arxiv.org/abs/1711.02508) is a navigation filter based
//! on regular Kalman filters, more specifically [Extended Kalman
//! Filters](https://en.wikipedia.org/wiki/Extended_Kalman_filter), that model the "error state" of
//! the system instead of modelling the movement of the system.
//!
//! The navigation filter is used to track `position`, `velocity` and `orientation` of an object
//! which is sensing its state through an [Inertial Measurement Unit
//! (IMU)](https://en.wikipedia.org/wiki/Inertial_measurement_unit) and some means of observing the
//! true state of the filter such as GPS, LIDAR or visual odometry.
//!
//! ## Usage
//! ```
//! use eskf;
//! use nalgebra::{Vector3, Point3};
//! use std::time::Duration;
//!
//! // Create a default filter, modelling a perfect IMU without drift
//! let mut filter = eskf::Builder::new().build();
//! // Read measurements from IMU
//! let imu_acceleration = Vector3::new(0.0, 0.0, -9.81);
//! let imu_rotation = Vector3::zeros();
//! // Tell the filter what we just measured
//! filter.predict(imu_acceleration, imu_rotation, Duration::from_millis(1000));
//! // Check the new state of the filter
//! // filter.position or filter.velocity...
//! // ...
//! // After some time we get an observation of the actual state
//! filter.observe_position(
//!     Point3::new(0.0, 0.0, 0.0),
//!     eskf::ESKF::variance_from_element(0.1))
//!         .expect("Filter update failed");
//! // Since we have supplied an observation of the actual state of the filter the states have now
//! // been updated. The uncertainty of the filter is also updated to reflect this new information.
//! ```

#![deny(missing_docs)]
#![deny(unsafe_code)]
#![cfg_attr(not(feature = "std"), no_std)]

use core::ops::{AddAssign, SubAssign};

use nalgebra::{
    Matrix, Matrix2, Matrix3, Point3, SMatrix, SVector, UnitQuaternion, Vector, Vector2, Vector3, U15, U5
};
#[cfg(feature = "no_std")]
use num_traits::float::Float;

#[cfg(any(
    all(feature = "std", feature = "no_std"),
    not(any(feature = "std", feature = "no_std"))
))]
compile_error!("Exactly one of features `std` and `no_std` must be enabled");

/// Potential errors raised during operations
#[derive(Copy, Clone, Debug)]
pub enum Error {
    /// It is not always [the case that a matrix is
    /// invertible](https://en.wikipedia.org/wiki/Invertible_matrix) which can lead to errors. It
    /// is difficult to handle this both for the library and for the users. In the case of the
    /// [`ESKF`], if this happens, it may be caused by an irregular shaped variance matrix for the
    /// update step. In such cases, inspect the variance matrix. If this happens irregularly it can
    /// be a sign that the uncertainty update is not stable, if possible try one of `cov-symmetric`
    /// or `cov-joseph` features for a more stable update.
    InversionError,
}

/// Helper definition to make it easier to work with errors
pub type Result<T> = core::result::Result<T, Error>;
/// Time delta as a duration, used when `std` is available

/// Builder for [`ESKF`]
#[derive(Copy, Clone, Debug)]
pub struct Builder {
    var_acc: Vector3<f32>,
    var_rot: Vector3<f32>,
    var_acc_bias: Vector3<f32>,
    var_rot_bias: Vector3<f32>,
    process_covariance: f32,
    gravity: f32,
}

impl Builder {
    /// Create a new `ESKF` builder with which to configure an `ESKF`
    pub fn new() -> Self {
        Builder::default()
    }

    /// Set the acceleration variance of the IMU system being modeled
    ///
    /// The variance should be `m/s²`
    pub fn acceleration_variance(mut self, var: f32) -> Self {
        self.var_acc = Vector3::from_element(var.powi(2));
        self
    }

    /// Set the acceleration variance of the IMU system being modeled
    ///
    /// The variance should be a vector in [`m/s²`, 3]
    pub fn acceleration_variance_from_vec(mut self, var: Vector3<f32>) -> Self {
        self.var_acc = var.map(|e| e.powi(2));
        self
    }

    /// Set the rotation variance of the IMU system being modeled
    ///
    /// The variance should be `rad/s`
    pub fn rotation_variance(mut self, var: f32) -> Self {
        self.var_rot = Vector3::from_element(var.powi(2));
        self
    }

    /// Set the rotation variance of the IMU system being modeled
    ///
    /// The variance should a vector in [`rad/s`, 3]
    pub fn rotation_variance_from_vec(mut self, var: Vector3<f32>) -> Self {
        self.var_rot = var.map(|e| e.powi(2));
        self
    }

    /// Set the acceleration bias of the IMU system being modeled
    ///
    /// The bias should be `m/(s²sqrt(s))`
    pub fn acceleration_bias(mut self, bias: f32) -> Self {
        self.var_acc_bias = Vector3::from_element(bias.powi(2));
        self
    }

    /// Set the acceleration bias of the IMU system being modeled
    ///
    /// The bias should be a vector in [`m/(s²sqrt(s))`, 3]
    pub fn acceleration_bias_from_vec(mut self, bias: Vector3<f32>) -> Self {
        self.var_acc_bias = bias.map(|e| e.powi(2));
        self
    }

    /// Set the rotation bias of the IMU system being modeled
    ///
    /// The bias should be `rad/(s sqrt(s))`
    pub fn rotation_bias(mut self, bias: f32) -> Self {
        self.var_rot_bias = Vector3::from_element(bias.powi(2));
        self
    }

    /// Set the rotation bias of the IMU system being modeled
    ///
    /// The bias should a vector in [`rad/(s sqrt(s))`, 3]
    pub fn rotation_bias_from_vec(mut self, bias: Vector3<f32>) -> Self {
        self.var_rot_bias = bias.map(|e| e.powi(2));
        self
    }

    /// Set the initial covariance for the process matrix
    ///
    /// The covariance value should be a small process value so tha
    /// t the covariance of the filter
    /// quickly converges to the correct value. Too small values could lead to the filter taking a
    /// long time to converge and report a lower covariance than what it should.
    pub fn initial_covariance(mut self, covar: f32) -> Self {
        self.process_covariance = covar;
        self
    }

    /// Set the used gravity in m/s².
    ///
    /// The default value is 9.81 m/s².
    pub fn gravity(mut self, gravity: f32) -> Self {
        self.gravity = gravity;
        self
    }

    /// Convert the builder into a new filter
    pub fn build(self) -> ESKF {
        ESKF {
            position: Point3::origin(),
            velocity: Vector3::zeros(),
            orientation: UnitQuaternion::identity(),
            acc_bias: Vector3::zeros(),
            gyr_bias: Vector3::zeros(),
            gravity: Vector3::new(0.0, 0.0, self.gravity),
            covariance: SMatrix::<f32, 15, 15>::identity() * self.process_covariance,
            var_acc: self.var_acc,
            var_gyr: self.var_rot,
            var_acc_bias: self.var_acc_bias,
            var_gyr_bias: self.var_rot_bias,
        }
    }
}

impl Default for Builder {
    fn default() -> Self {
        Self {
            var_acc: Default::default(),
            var_rot: Default::default(),
            var_acc_bias: Default::default(),
            var_rot_bias: Default::default(),
            process_covariance: Default::default(),
            gravity: 9.81,
        }
    }
}

/// Error State Kalman Filter
///
/// The filter works by calling [`predict`](ESKF::predict) and one or more of the
/// [`observe_`](ESKF::observe_position) methods when data is available. It is expected that
/// several calls to [`predict`](ESKF::predict) will be in between calls to `observe_`.
///
/// The [`predict`](ESKF::predict) step updates the internal state of the filter based on measured
/// acceleration and rotation coming from an IMU. This step updates the states in the filter based
/// on kinematic equations while increasing the uncertainty of the filter. When one of the
/// `observe_` methods are called, the filter updates the internal state based on this observation,
/// which exposes the error state to the filter, which we can then use to correct the internal
/// state. The uncertainty of the filter is also updated to reflect the variance of the observation
/// and the updated state.
#[derive(Copy, Clone, Debug)]
pub struct ESKF {
    /// Estimated position in filter
    pub position: Point3<f32>,
    /// Estimated velocity in filter
    pub velocity: Vector3<f32>,
    /// Estimated orientation in filter
    pub orientation: UnitQuaternion<f32>,
    /// Estimated acceleration bias
    pub acc_bias: Vector3<f32>,
    /// Estimated rotation bias
    pub gyr_bias: Vector3<f32>,
    /// Gravity vector.
    pub gravity: Vector3<f32>,
    /// Covariance of filter state
    covariance: SMatrix<f32, 15, 15>,
    /// Acceleration variance
    var_acc: Vector3<f32>,
    /// Angular velocity (gyro) variance
    var_gyr: Vector3<f32>,
    /// Acceleration variance bias
    var_acc_bias: Vector3<f32>,
    /// Angular velocity (gyro) variance bias
    var_gyr_bias: Vector3<f32>,
}

impl ESKF {
    /// Create a symmetric variance matrix based on a single variance element
    ///
    /// This helper method can be used when the sensor being modelled has a symmetric variance
    /// around its three axis. Or if only an estimate of the variance is known.
    pub fn variance_from_element(var: f32) -> Matrix3<f32> {
        Matrix3::from_diagonal_element(var)
    }

    /// Create a symmetric variance matrix based on the diagonal vector
    ///
    /// This helper method can be used when the sensor being modelled has a independent variance
    /// around its three axis.
    pub fn variance_from_diagonal(var: Vector3<f32>) -> Matrix3<f32> {
        Matrix3::from_diagonal(&var)
    }

    /// Internal helper method to extract 3 dimensional uncertainty from the covariance state
    fn uncertainty3(&self, start: usize) -> Vector3<f32> {
        self.covariance
            .diagonal()
            .fixed_view_mut::<3, 1>(start, 0)
            .map(|var| var.sqrt())
    }

    /// Get the uncertainty of the position estimate
    pub fn position_uncertainty(&self) -> Vector3<f32> {
        self.uncertainty3(0)
    }

    /// Get the uncertainty of the velocity estimate
    pub fn velocity_uncertainty(&self) -> Vector3<f32> {
        self.uncertainty3(3)
    }

    /// Get the uncertainty of the orientation estimate
    pub fn orientation_uncertainty(&self) -> Vector3<f32> {
        self.uncertainty3(6)
    }

    /// Update the filter, predicting the new state, based on measured acceleration and angular velocity
    /// from an `IMU`. The accelerometer readings must be m/s^2, and the gyroscope reading must be rad/s.
    pub fn predict(&mut self, acc_meas: Vector3<f32>, gyr_meas: Vector3<f32>, delta_t: f32) {

        let rot_acc_grav = self
            .orientation
            .transform_vector(&(acc_meas - self.acc_bias))
            + self.gravity;
        let norm_rot = UnitQuaternion::from_scaled_axis((gyr_meas - self.gyr_bias) * delta_t);
        let orient_mat = self.orientation.to_rotation_matrix().into_inner();
        // Update internal state kinematics
        self.position += self.velocity * delta_t + 0.5 * rot_acc_grav * delta_t.powi(2);
        self.velocity += rot_acc_grav * delta_t;
        self.orientation *= norm_rot;

        // Propagate uncertainty, since we have not observed any new information about the state of
        // the filter we need to update our estimate of the uncertainty of the filer
        let ident_delta = Matrix3::<f32>::identity() * delta_t;
        let mut error_jacobian = SMatrix::<f32, 15, 15>::identity();
        error_jacobian
            .fixed_view_mut::<3, 3>(0, 3)
            .copy_from(&ident_delta);
        error_jacobian
            .fixed_view_mut::<3, 3>(3, 6)
            .copy_from(&(-orient_mat * skew(&(acc_meas - self.acc_bias)) * delta_t));
        error_jacobian
            .fixed_view_mut::<3, 3>(3, 9)
            .copy_from(&(-orient_mat * delta_t));
        error_jacobian
            .fixed_view_mut::<3, 3>(6, 6)
            .copy_from(&norm_rot.to_rotation_matrix().into_inner().transpose());
        error_jacobian
            .fixed_view_mut::<3, 3>(6, 12)
            .copy_from(&-ident_delta);

        // Add noise variance
        self.covariance = error_jacobian * self.covariance * error_jacobian.transpose();
        let mut diagonal = self.covariance.diagonal();
        diagonal
            .fixed_view_mut::<3, 1>(3, 0)
            .add_assign(self.var_acc * delta_t.powi(2));
        diagonal
            .fixed_view_mut::<3, 1>(6, 0)
            .add_assign(self.var_gyr * delta_t.powi(2));
        diagonal
            .fixed_view_mut::<3, 1>(9, 0)
            .add_assign(self.var_acc_bias * delta_t);
        diagonal
            .fixed_view_mut::<3, 1>(12, 0)
            .add_assign(self.var_gyr_bias * delta_t);
        self.covariance.set_diagonal(&diagonal);
    }

    /// Update the filter with a generic observation
    ///
    /// # Arguments
    /// - `jacobian` is the measurement Jacobian matrix
    /// - `difference` is the difference between the measured and the estimated state
    /// - `variance` is the uncertainty of the observation
    pub fn update< const R: usize>(
        &mut self,
        jacobian: SMatrix<f32, R, 15>,
        difference: SVector<f32, R>,
        variance: SMatrix<f32, R, R>,
    ) -> Result<()> {
        // Correct filter based on Kalman gain
        let kalman_gain = self.covariance
            * &jacobian.transpose()
            * (&jacobian * self.covariance * &jacobian.transpose() + &variance)
                .try_inverse()
                .ok_or(Error::InversionError)?;
        let error_state = &kalman_gain * difference;
        // Update the covariance based on the observed filter state
        if cfg!(feature = "cov-symmetric") {
            self.covariance -= &kalman_gain
                * (&jacobian * self.covariance * &jacobian.transpose() + &variance)
                * &kalman_gain.transpose();
        } else if cfg!(feature = "cov-joseph") {
            let step1 = SMatrix::<f32, 15, 15>::identity() - &kalman_gain * &jacobian;
            let step2 = &kalman_gain * &variance * &kalman_gain.transpose();
            self.covariance = step1 * self.covariance * step1.transpose() + step2;
        } else {
            self.covariance =
                (SMatrix::<f32, 15, 15>::identity() - &kalman_gain * &jacobian) * self.covariance;
        }

        self.update_the_sequel(error_state)
    }

    /// Update the filter with a generic observation
    ///
    /// # Arguments
    /// - `error_state` is the error state calculated by the [`ESKF::update`] function
    fn update_the_sequel(
        &mut self,
        error_state: SVector<f32, 15>,
    )  -> Result<()>
    {
        // Inject error state into nominal
        self.position += error_state.fixed_view::<3, 1>(0, 0);
        self.velocity += error_state.fixed_view::<3, 1>(3, 0);
        self.orientation *= UnitQuaternion::from_scaled_axis(error_state.fixed_view::<3, 1>(6, 0));
        self.acc_bias += error_state.fixed_view::<3, 1>(9, 0);
        self.gyr_bias += error_state.fixed_view::<3, 1>(12, 0);
        // Perform full ESKF reset
        //
        // Since the orientation error is usually relatively small this step can be skipped, but
        // the full formulation can lead to better stability of the filter
        if cfg!(feature = "full-reset") {
            let mut g = SMatrix::<f32, 15, 15>::identity();
            g.fixed_view_mut::<3, 3>(6, 6)
                .sub_assign(0.5 * skew(&error_state.fixed_view::<3, 1>(6, 0).clone_owned()));
            self.covariance = g * self.covariance * g.transpose();
        }
        Ok(())
    }

    /// Observe the position and velocity in the X and Y axis
    ///
    /// Most GPS units are capable of observing both position and velocity, by combining these two
    /// measurements into one update we should be able to reduce the computational complexity. Also
    /// note that GPS velocity tends to be more precise than position.
    pub fn observe_position_velocity2d(
        &mut self,
        position: Point3<f32>,
        position_var: Matrix3<f32>,
        velocity: Vector2<f32>,
        velocity_var: Matrix2<f32>,
    ) -> Result<()> {
        let mut jacobian = Matrix::<f32, U5, U15, _>::zeros();
        jacobian.fixed_view_mut::<5, 5>(0, 0).fill_with_identity();

        let mut diff = Vector::<f32, U5, _>::zeros();
        diff.fixed_view_mut::<3, 1>(0, 0)
            .copy_from(&(position - self.position));
        diff.fixed_view_mut::<2, 1>(3, 0)
            .copy_from(&(velocity - self.velocity.xy()));

        let mut var = Matrix::<f32, U5, U5, _>::zeros();
        var.fixed_view_mut::<3, 3>(0, 0).copy_from(&position_var);
        var.fixed_view_mut::<2, 2>(3, 3).copy_from(&velocity_var);

        self.update(jacobian, diff, var)
    }

    /// Observe the position and velocity
    ///
    /// Most GPS units are capable of observing both position and velocity, by combining these two
    /// measurements into one update we should be able to reduce the computational complexity. Also
    /// note that GPS velocity tends to be more precise than position.
    pub fn observe_position_velocity(
        &mut self,
        position: Point3<f32>,
        position_var: Matrix3<f32>,
        velocity: Vector3<f32>,
        velocity_var: Matrix3<f32>,
    ) -> Result<()> {
        let mut jacobian = SMatrix::<f32, 6, 15>::zeros();
        jacobian.fixed_view_mut::<6, 6>(0, 0).fill_with_identity();

        let mut diff = SVector::<f32, 6>::zeros();
        diff.fixed_view_mut::<3, 1>(0, 0)
            .copy_from(&(position - self.position));
        diff.fixed_view_mut::<3, 1>(3, 0)
            .copy_from(&(velocity - self.velocity));

        let mut var = SMatrix::<f32, 6, 6>::zeros();
        var.fixed_view_mut::<3, 3>(0, 0).copy_from(&position_var);
        var.fixed_view_mut::<3, 3>(3, 3).copy_from(&velocity_var);

        self.update(jacobian, diff, var)
    }

    /// Update the filter with an observation of the position
    pub fn observe_position(
        &mut self,
        measurement: Point3<f32>,
        variance: Matrix3<f32>,
    ) -> Result<()> {
        let mut jacobian = SMatrix::<f32, 3, 15>::zeros();
        jacobian.fixed_view_mut::<3, 3>(0, 0).fill_with_identity();
        let diff = measurement - self.position;
        self.update(jacobian, diff, variance)
    }

    /// Update the filter with an observation of the height alone
    pub fn observe_height(&mut self, measured: f32, variance: f32) -> Result<()> {
        let mut jacobian = SMatrix::<f32, 1, 15>::zeros();
        jacobian.fixed_view_mut::<1, 1>(0, 2).fill_with_identity();
        let diff = SVector::<f32, 1>::new(measured - self.position.z);
        let var = SMatrix::<f32, 1, 1>::new(variance);
        self.update(jacobian, diff, var)
    }

    /// Update the filter with an observation of the velocity
    ///
    /// # Note
    /// If the observation comes from a sensor relative to the filter, e.g. an optical flow sensor
    /// that turns with the UAV, the sensor values **needs** to be rotated into the same frame as
    /// the filter, e.g. `filter.orientation.transform_vector(&relative_measurement)`.
    pub fn observe_velocity(
        &mut self,
        measurement: Vector3<f32>,
        variance: Matrix3<f32>,
    ) -> Result<()> {
        let mut jacobian = SMatrix::<f32, 3, 15>::zeros();
        jacobian.fixed_view_mut::<3, 3>(0, 3).fill_with_identity();
        let diff = measurement - self.velocity;
        self.update(jacobian, diff, variance)
    }

    /// Update the filter with an observation of the velocity in only the `[X, Y]` axis
    ///
    /// # Note
    /// If the observation comes from a sensor relative to the filter, e.g. an optical flow sensor
    /// that turns with the UAV, the sensor values **needs** to be rotated into the same frame as
    /// the filter, e.g. `filter.orientation.transform_vector(&relative_measurement)`.
    pub fn observe_velocity2d(
        &mut self,
        measurement: Vector2<f32>,
        variance: Matrix2<f32>,
    ) -> Result<()> {
        let mut jacobian = SMatrix::<f32, 2, 15>::zeros();
        jacobian.fixed_view_mut::<2, 2>(0, 3).fill_with_identity();
        let diff = Vector2::new(
            measurement.x - self.velocity.x,
            measurement.y - self.velocity.y,
        );
        self.update(jacobian, diff, variance)
    }

    /// Update the filter with an observation of the orientation
    pub fn observe_orientation(
        &mut self,
        measurement: UnitQuaternion<f32>,
        variance: Matrix3<f32>,
    ) -> Result<()> {
        let mut jacobian = SMatrix::<f32, 3, 15>::zeros();
        jacobian.fixed_view_mut::<3, 3>(0, 6).fill_with_identity();
        let diff = measurement * self.orientation;
        self.update(jacobian, diff.scaled_axis(), variance)
    }
}

/// Create the skew-symmetric matrix from a vector
#[rustfmt::skip]
fn skew(v: &Vector3<f32>) -> Matrix3<f32> {
    Matrix3::new(0., -v.z, v.y,
                 v.z, 0., -v.x,
                 -v.y, v.x, 0.)
}

#[cfg(test)]
mod test {
    use super::Builder;
    use approx::assert_relative_eq;
    use nalgebra::{Point3, UnitQuaternion, Vector3};
    use std::f32::consts::FRAC_PI_2;
    use std::time::Duration;

    #[test]
    fn creation() {
        let filter = Builder::new().build();
        assert_relative_eq!(filter.position, Point3::origin());
        assert_relative_eq!(filter.velocity, Vector3::zeros());
    }

    #[test]
    fn linear_motion() {
        let mut filter = Builder::new().build();
        // Some initial motion to move the filter
        filter.predict(
            Vector3::new(1.0, 0.0, -9.81),
            Vector3::zeros(),
            Duration::from_millis(1000).as_secs_f32(),
        );
        assert_relative_eq!(filter.position, Point3::new(0.5, 0.0, 0.0));
        assert_relative_eq!(filter.velocity, Vector3::new(1.0, 0.0, 0.0));
        // There should be no orientation change from the above motion
        assert_relative_eq!(filter.orientation, UnitQuaternion::identity());
        // Acceleration has stopped, but there will still be inertia in the filter
        filter.predict(
            Vector3::new(0.0, 0.0, -9.81),
            Vector3::zeros(),
            Duration::from_millis(500).as_secs_f32(),
        );
        assert_relative_eq!(filter.position, Point3::new(1.0, 0.0, 0.0));
        assert_relative_eq!(filter.velocity, Vector3::new(1.0, 0.0, 0.0));
        assert_relative_eq!(filter.orientation, UnitQuaternion::identity());
        filter.predict(
            Vector3::new(-1.0, -1.0, -9.81),
            Vector3::zeros(),
            Duration::from_millis(1000).as_secs_f32(),
        );
        assert_relative_eq!(filter.position, Point3::new(1.5, -0.5, 0.0));
        assert_relative_eq!(filter.velocity, Vector3::new(0.0, -1.0, 0.0));
        assert_relative_eq!(filter.orientation, UnitQuaternion::identity());
    }

    #[test]
    fn rotational_motion() {
        let mut filter = Builder::new().build();
        // Note that this motion is a free fall rotation
        filter.predict(
            Vector3::zeros(),
            Vector3::new(FRAC_PI_2, 0.0, 0.0),
            Duration::from_millis(1000).as_secs_f32(),
        );
        assert_relative_eq!(
            filter.orientation,
            UnitQuaternion::from_euler_angles(FRAC_PI_2, 0.0, 0.0)
        );
        filter.predict(
            Vector3::zeros(),
            Vector3::new(-FRAC_PI_2, 0.0, 0.0),
            Duration::from_millis(1000).as_secs_f32(),
        );
        assert_relative_eq!(
            filter.orientation,
            UnitQuaternion::from_euler_angles(0.0, 0.0, 0.0)
        );
        // We reset the filter here so that the following equalities are not affected by existing
        // motion in the filter
        let mut filter = Builder::new().build();
        filter.predict(
            Vector3::zeros(),
            Vector3::new(0.0, -FRAC_PI_2, 0.0),
            Duration::from_millis(1000).as_secs_f32(),
        );
        assert_relative_eq!(
            filter.orientation,
            UnitQuaternion::from_euler_angles(0.0, -FRAC_PI_2, 0.0)
        );
    }
}
