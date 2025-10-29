//! # Error State Kalman Filter (ESKF)
//! An [Error State Kalman Filter](https://arxiv.org/abs/1711.02508) is a navigation filter based
//! on regular Kalman filters, more specifically [Extended Kalman
//! Filters](https://en.wikipedia.org/wiki/Extended_Kalman_filter), that model the "error state" of
//! the system instead of modelling the movement of the system.
//!
//! The navigation filter is used to track `position`, `velocity` and `rotation` of an object
//! which is sensing its state through an [Inertial Measurement Unit
//! (IMU)](https://en.wikipedia.org/wiki/Inertial_measurement_unit) and some means of observing the
//! true state of the filter such as GPS, LIDAR or visual odometry.
//! ```

#![deny(missing_docs)]
#![deny(unsafe_code)]
#![cfg_attr(feature = "no_std", no_std)]

use core::ops::{AddAssign, SubAssign};

use nalgebra::{Matrix2, Matrix3, Point3, SMatrix, SVector, UnitQuaternion, Vector2, Vector3};

#[cfg(feature = "no_std")]
use num_traits::float::Float as _;

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
    /// Estimated rotation in filter
    pub rotation: UnitQuaternion<f32>,
    /// Estimated acceleromter bias
    pub acc_bias: Vector3<f32>,
    /// Estimated gyroscope bias
    pub gyr_bias: Vector3<f32>,
    /// Gravity vector.
    pub gravity: Vector3<f32>,
    /// Covariance of filter state
    covariance: SMatrix<f32, 15, 15>,
    /// Power Spectral Density of the accelerometer's white noise
    acc_noise_var: Vector3<f32>,
    /// Power Spectral Density of the gyroscope's white noise
    gyr_noise_var: Vector3<f32>,
    /// Power Spectral Density of the accelerometer bias random walk
    acc_bias_var: Vector3<f32>,
    /// Power Spectral Density of the gyroscope bias random walk
    gyr_bias_var: Vector3<f32>,
}


impl Default for ESKF {
    fn default() -> Self {
        ESKF {
            position: Point3::origin(),
            velocity: Vector3::zeros(),
            rotation: UnitQuaternion::identity(),
            acc_bias: Vector3::zeros(),
            gyr_bias: Vector3::zeros(),
            gravity: Vector3::new(0.0, 0.0, 9.81),
            covariance: SMatrix::identity() * 0.1,
            acc_noise_var: SVector::from_element(1e-3),
            gyr_noise_var: SVector::from_element(1e-3),
            acc_bias_var: SVector::from_element(1e-4),
            gyr_bias_var: SVector::from_element(1e-4),
        }
    }
}

impl ESKF {

    /// Construct a new filter with default values.
    pub fn new() -> ESKF {
        Self::default()
    }

    /// Borrow the [`ESKF`] filter mutably, returning the filter as a value.
    pub fn with_mut(mut self, func: impl Fn(&mut Self)) -> Self {
        func(&mut self);
        self
    }

    /// Set the accelerometer measurement noise standard deviation
    ///
    /// This value represents the standard deviation of the white noise
    /// affecting a single accelerometer reading. The unit should be `(m/s²)`.
    pub fn acc_noise_std(&mut self, std: f32) -> &mut Self {
        self.acc_noise_var = Vector3::from_element(std.powi(2));
        self
    }

    /// Set the gyroscope measurement noise standard deviation
    ///
    ///
    /// This value represents the standard deviation of the white noise
    /// affecting a single accelerometer reading. The unit should be `(rad/s)`.
    pub fn gyr_noise_std(&mut self, std: f32) -> &mut Self {
        self.gyr_noise_var = Vector3::from_element(std.powi(2));
        self
    }

    /// Set the accelerometer bias drift random walk standard deviation.
    ///
    /// This represents the standard deviation of the bias drift accumulated
    /// over a unit time interval. The unit should be `(m/s²)/s/√Hz`.
    pub fn acc_bias_std(&mut self, std: f32) -> &mut Self {
        self.acc_bias_var = Vector3::from_element(std.powi(2));
        self
    }

    /// Set the gyroscope bias drift random walk standard deviation.
    ///
    /// This represents the standard deviation of the bias drift accumulated
    /// over a unit time interval. The unit should be `(rad/s)/s/√Hz`.
    pub fn gyr_bias_std(&mut self, std: f32) -> &mut Self {
        self.gyr_bias_var = Vector3::from_element(std.powi(2));
        self
    }

    /// Set the diagonal elements of the covariance for the process matrix.
    ///
    /// The covariance value should be a small process value so that the
    /// covariance of the filter quickly converges to the correct value.
    /// Too small values could lead to the filter taking a long time to
    /// converge and report a lower covariance than what it should.
    pub fn covariance_diag(&mut self, cov: f32) -> &mut Self {
        self.covariance = SMatrix::identity() * cov.abs();
        self
    }

    /// Set the used gravity in m/s².
    ///
    /// The default value is (positive) 9.81 m/s² in the z direction.
    /// This is fitting for a NED (north east down) reference frame.
    pub fn with_gravity(&mut self, gravity: Vector3<f32>) -> &mut Self {
        self.gravity = gravity;
        self
    }

    /// Get the uncertainty of the position estimate
    pub fn position_variance(&self) -> Vector3<f32> {
        self.variance_3(0)
    }

    /// Get the uncertainty of the velocity estimate
    pub fn velocity_variance(&self) -> Vector3<f32> {
        self.variance_3(3)
    }

    /// Get the uncertainty of the rotation estimate
    pub fn rotation_variance(&self) -> Vector3<f32> {
        self.variance_3(6)
    }

    /// Internal helper method to extract 3 dimensional uncertainty from the covariance state
    fn variance_3(&self, start: usize) -> Vector3<f32> {
        self.covariance
            .diagonal()
            .fixed_view_mut::<3, 1>(start, 0)
            .map(|var| var.sqrt())
    }

    /// Get the full covariance matrix. See `*_uncertainty` methods for
    pub fn full_covariance(&self) -> &SMatrix<f32, 15, 15> {
        &self.covariance
    }

    /// Update the filter, predicting the new state, based on measured acceleration and angular velocity
    /// from an `IMU`. The accelerometer readings must be m/s^2, and the gyroscope reading must be rad/s.
    pub fn predict(&mut self, acc_meas: Vector3<f32>, gyr_meas: Vector3<f32>, dt: f32) {
        let rot_acc_grav =
            self.rotation.transform_vector(&(acc_meas - self.acc_bias)) + self.gravity;
        let delta_rotation = UnitQuaternion::from_scaled_axis((gyr_meas - self.gyr_bias) * dt);
        let orient_mat = self.rotation.to_rotation_matrix().into_inner();
        
        // Update internal state kinematics
        self.position += self.velocity * dt + 0.5 * rot_acc_grav * dt.powi(2);
        self.velocity += rot_acc_grav * dt;
        self.rotation *= delta_rotation;

        // Propagate uncertainty, since we have not observed any new information about the state of
        // the filter we need to update our estimate of the uncertainty of the filer
        let ident_delta = Matrix3::<f32>::identity() * dt;
        let mut error_jacobian = SMatrix::<f32, 15, 15>::identity();
        error_jacobian
            .fixed_view_mut::<3, 3>(0, 3)
            .copy_from(&ident_delta);
        error_jacobian
            .fixed_view_mut::<3, 3>(3, 6)
            .copy_from(&(-orient_mat * skew(&(acc_meas - self.acc_bias)) * dt));
        error_jacobian
            .fixed_view_mut::<3, 3>(3, 9)
            .copy_from(&(-orient_mat * dt));
        error_jacobian
            .fixed_view_mut::<3, 3>(6, 6)
            .copy_from(&delta_rotation.to_rotation_matrix().into_inner().transpose());
        error_jacobian
            .fixed_view_mut::<3, 3>(6, 12)
            .copy_from(&-ident_delta);

        // Add noise variance
        self.covariance = error_jacobian * self.covariance * error_jacobian.transpose();
        let mut diagonal = self.covariance.diagonal();
        diagonal
            .fixed_view_mut::<3, 1>(3, 0)
            .add_assign(self.acc_noise_var * dt.powi(2));
        diagonal
            .fixed_view_mut::<3, 1>(6, 0)
            .add_assign(self.gyr_noise_var * dt.powi(2));
        diagonal
            .fixed_view_mut::<3, 1>(9, 0)
            .add_assign(self.acc_bias_var * dt);
        diagonal
            .fixed_view_mut::<3, 1>(12, 0)
            .add_assign(self.gyr_bias_var * dt);
        self.covariance.set_diagonal(&diagonal);
    }

    /// Update the filter, predicting the new state, based on measured acceleration and angular velocity
    /// from an `IMU`. The accelerometer readings must be m/s^2, and the gyroscope reading must be rad/s.
    ///
    /// This is the optimized implementation that uses matrix symmetry to minimize memory operations.
    pub fn predict_optimized(&mut self, acc_meas: Vector3<f32>, gyr_meas: Vector3<f32>, dt: f32) {
        let rot_acc_grav =
            self.rotation.transform_vector(&(acc_meas - self.acc_bias)) + self.gravity;
        let norm_rot = UnitQuaternion::from_scaled_axis((gyr_meas - self.gyr_bias) * dt);
        let orient_mat = self.rotation.to_rotation_matrix().into_inner();

        // Update internal state kinematics
        self.position += self.velocity * dt + 0.5 * rot_acc_grav * dt.powi(2);
        self.velocity += rot_acc_grav * dt;
        self.rotation *= norm_rot;

        // Block-entries of error jacobian and their transposes
        let f36 = -orient_mat * skew(&(acc_meas - self.acc_bias)) * dt;
        let f39 = -orient_mat * dt;
        let f66 = norm_rot.to_rotation_matrix().into_inner().transpose();

        // Extract the 15 upper-triangle blocks from the current covariance matrix
        let p_1_1 = self.covariance.fixed_view::<3, 3>(0, 0).clone_owned();
        let p_1_2 = self.covariance.fixed_view::<3, 3>(0, 3).clone_owned();
        let p_1_3 = self.covariance.fixed_view::<3, 3>(0, 6).clone_owned();
        let p_1_4 = self.covariance.fixed_view::<3, 3>(0, 9).clone_owned();
        let p_1_5 = self.covariance.fixed_view::<3, 3>(0, 12).clone_owned();

        let p_2_2 = self.covariance.fixed_view::<3, 3>(3, 3).clone_owned();
        let p_2_3 = self.covariance.fixed_view::<3, 3>(3, 6).clone_owned();
        let p_2_4 = self.covariance.fixed_view::<3, 3>(3, 9).clone_owned();
        let p_2_5 = self.covariance.fixed_view::<3, 3>(3, 12).clone_owned();

        let p_3_3 = self.covariance.fixed_view::<3, 3>(6, 6).clone_owned();
        let p_3_4 = self.covariance.fixed_view::<3, 3>(6, 9).clone_owned();
        let p_3_5 = self.covariance.fixed_view::<3, 3>(6, 12).clone_owned();

        let p_4_4 = self.covariance.fixed_view::<3, 3>(9, 9).clone_owned();
        let p_4_5 = self.covariance.fixed_view::<3, 3>(9, 12).clone_owned();

        let p_5_5 = self.covariance.fixed_view::<3, 3>(12, 12).clone_owned();

        // Row 1
        self.covariance
            .fixed_view_mut::<3, 3>(0, 0)
            .copy_from(&(p_1_1 + dt * (p_1_2.transpose() + p_1_2 + dt * p_2_2)));
        self.covariance.fixed_view_mut::<3, 3>(0, 3).copy_from(
            &(p_1_2
                + dt * p_2_2
                + (p_1_3 + dt * p_2_3) * f36.transpose()
                + (p_1_4 + dt * p_2_4) * f39.transpose()),
        );
        self.covariance
            .fixed_view_mut::<3, 3>(0, 6)
            .copy_from(&((p_1_3 + dt * p_2_3) * f66.transpose() - dt * (p_1_5 + dt * p_2_5)));
        self.covariance
            .fixed_view_mut::<3, 3>(0, 9)
            .copy_from(&(p_1_4 + dt * p_2_4));
        self.covariance
            .fixed_view_mut::<3, 3>(0, 12)
            .copy_from(&(p_1_5 + dt * p_2_5));

        // Row 2
        self.covariance.fixed_view_mut::<3, 3>(3, 3).copy_from(
            &(p_2_2
                + f36 * p_2_3.transpose()
                + f39 * p_2_4.transpose()
                + (p_2_3 + f36 * p_3_3 + f39 * p_3_4.transpose()) * f36.transpose()
                + (p_2_4 + f36 * p_3_4 + f39 * p_4_4) * f39.transpose()),
        );
        self.covariance.fixed_view_mut::<3, 3>(3, 6).copy_from(
            &((p_2_3 + f36 * p_3_3 + f39 * p_3_4.transpose()) * f66.transpose()
                - dt * (p_2_5 + f36 * p_3_5 + f39 * p_4_5)),
        );
        self.covariance
            .fixed_view_mut::<3, 3>(3, 9)
            .copy_from(&(p_2_4 + f36 * p_3_4 + f39 * p_4_4));
        self.covariance
            .fixed_view_mut::<3, 3>(3, 12)
            .copy_from(&(p_2_5 + f36 * p_3_5 + f39 * p_4_5));

        // Row 3
        self.covariance.fixed_view_mut::<3, 3>(6, 6).copy_from(
            &((f66 * p_3_3 - dt * p_3_5.transpose()) * f66.transpose()
                - dt * (f66 * p_3_5 - dt * p_5_5)),
        );
        self.covariance
            .fixed_view_mut::<3, 3>(6, 9)
            .copy_from(&(f66 * p_3_4 - dt * p_4_5.transpose()));
        self.covariance
            .fixed_view_mut::<3, 3>(6, 12)
            .copy_from(&(f66 * p_3_5 - dt * p_5_5));

        // Row 4 and 5 can be omitted since they are not changed..
        // Fill elements into lower part, since covariance are symmetric.
        self.covariance.fill_lower_triangle_with_upper_triangle();

        // Add noise variance
        let mut diagonal = self.covariance.diagonal();
        diagonal
            .fixed_view_mut::<3, 1>(3, 0)
            .add_assign(self.acc_noise_var * dt.powi(2));
        diagonal
            .fixed_view_mut::<3, 1>(6, 0)
            .add_assign(self.gyr_noise_var * dt.powi(2));
        diagonal
            .fixed_view_mut::<3, 1>(9, 0)
            .add_assign(self.acc_bias_var * dt);
        diagonal
            .fixed_view_mut::<3, 1>(12, 0)
            .add_assign(self.gyr_bias_var * dt);
        self.covariance.set_diagonal(&diagonal);
    }

    /// Update the filter with a generic observation
    ///
    /// # Arguments
    /// - `jacobian` is the measurement Jacobian matrix
    /// - `difference` is the difference between the measured and the estimated state
    /// - `variance` is the uncertainty of the observation
    pub fn update<const R: usize>(
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

        self.update_finalize(error_state)
    }

    /// Outlined finalization of [`ESKF::update`] function to reduce monomorphization impact.
    ///
    /// # Arguments
    /// - `error_state` is the error state calculated by the [`ESKF::update`] function
    #[inline(never)]
    fn update_finalize(&mut self, error_state: SVector<f32, 15>) -> Result<()> {
        // Inject error state into nominal
        self.position += error_state.fixed_view::<3, 1>(0, 0);
        self.velocity += error_state.fixed_view::<3, 1>(3, 0);
        self.rotation *= UnitQuaternion::from_scaled_axis(error_state.fixed_view::<3, 1>(6, 0));
        self.acc_bias += error_state.fixed_view::<3, 1>(9, 0);
        self.gyr_bias += error_state.fixed_view::<3, 1>(12, 0);

        // Perform full ESKF reset
        //
        // Since the rotation error is usually relatively small this step can be skipped, but
        // the full formulation can lead to better stability of the filter
        if cfg!(feature = "full-reset") {
            let mut g = SMatrix::<f32, 15, 15>::identity();
            g.fixed_view_mut::<3, 3>(6, 6)
                .sub_assign(0.5 * skew(&error_state.fixed_view::<3, 1>(6, 0).clone_owned()));
            self.covariance = g * self.covariance * g.transpose();
        }

        // Ensure rotation stays consistent
        self.rotation.renormalize_fast();

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
        let mut jacobian = SMatrix::<f32, 5, 15>::zeros();
        jacobian.fixed_view_mut::<5, 5>(0, 0).fill_with_identity();

        let mut diff = SVector::<f32, 5>::zeros();
        diff.fixed_view_mut::<3, 1>(0, 0)
            .copy_from(&(position - self.position));
        diff.fixed_view_mut::<2, 1>(3, 0)
            .copy_from(&(velocity - self.velocity.xy()));

        let mut var = SMatrix::<f32, 5, 5>::zeros();
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
        position: Point3<f32>,
        position_var: Matrix3<f32>,
    ) -> Result<()> {
        let mut jacobian = SMatrix::<f32, 3, 15>::zeros();
        jacobian.fixed_view_mut::<3, 3>(0, 0).fill_with_identity();
        let diff = position - self.position;
        self.update(jacobian, diff, position_var)
    }

    /// Update the filter with an observation of the height alone
    pub fn observe_height(&mut self, height: f32, height_var: f32) -> Result<()> {
        let mut jacobian = SMatrix::<f32, 1, 15>::zeros();
        jacobian.fixed_view_mut::<1, 1>(0, 2).fill_with_identity();
        let diff = SVector::<f32, 1>::new(height - self.position.z);
        let var = SMatrix::<f32, 1, 1>::new(height_var);
        self.update(jacobian, diff, var)
    }

    /// Update the filter with an observation of the position and rotation
    pub fn observe_position_rotation(
        &mut self,
        position: Point3<f32>,
        position_var: Matrix3<f32>,
        rotation: UnitQuaternion<f32>,
        rotation_var: Matrix3<f32>,
    ) -> Result<()> {
        let mut jacobian = SMatrix::<f32, 6, 15>::zeros();
        jacobian.fixed_view_mut::<3, 3>(0, 0).fill_with_identity();
        jacobian.fixed_view_mut::<3, 3>(3, 6).fill_with_identity();

        let mut diff = SVector::<f32, 6>::zeros();
        diff.fixed_view_mut::<3, 1>(0, 0)
            .copy_from(&(position - self.position));
        diff.fixed_view_mut::<3, 1>(3, 0)
            .copy_from(&(self.rotation.inverse() * rotation).scaled_axis());

        let mut var = SMatrix::<f32, 6, 6>::zeros();
        var.fixed_view_mut::<3, 3>(0, 0).copy_from(&position_var);
        var.fixed_view_mut::<3, 3>(3, 3).copy_from(&rotation_var);

        self.update(jacobian, diff, var)
    }

    /// Update the filter with an observation of the velocity
    ///
    /// # Note
    /// If the observation comes from a sensor relative to the filter, e.g. an optical flow sensor
    /// that turns with the UAV, the sensor values **needs** to be rotated into the same frame as
    /// the filter, e.g. `filter.rotation.transform_vector(&relative_measurement)`.
    pub fn observe_velocity(
        &mut self,
        velocity: Vector3<f32>,
        velocity_var: Matrix3<f32>,
    ) -> Result<()> {
        let mut jacobian = SMatrix::<f32, 3, 15>::zeros();
        jacobian.fixed_view_mut::<3, 3>(0, 3).fill_with_identity();
        let diff = velocity - self.velocity;
        self.update(jacobian, diff, velocity_var)
    }

    /// Update the filter with an observation of the velocity in only the `[X, Y]` axis
    ///
    /// # Note
    /// If the observation comes from a sensor relative to the filter, e.g. an optical flow sensor
    /// that turns with the UAV, the sensor values **needs** to be rotated into the same frame as
    /// the filter, e.g. `filter.rotation.transform_vector(&relative_measurement)`.
    pub fn observe_velocity2d(
        &mut self,
        velocity: Vector2<f32>,
        velocity_var: Matrix2<f32>,
    ) -> Result<()> {
        let mut jacobian = SMatrix::<f32, 2, 15>::zeros();
        jacobian.fixed_view_mut::<2, 2>(0, 3).fill_with_identity();
        let diff = Vector2::new(velocity.x - self.velocity.x, velocity.y - self.velocity.y);
        self.update(jacobian, diff, velocity_var)
    }

    /// Update the filter with an observation of the rotation
    pub fn observe_rotation(
        &mut self,
        rotation: UnitQuaternion<f32>,
        rotation_var: Matrix3<f32>,
    ) -> Result<()> {
        let mut jacobian = SMatrix::<f32, 3, 15>::zeros();
        jacobian.fixed_view_mut::<3, 3>(0, 6).fill_with_identity();
        let diff = (self.rotation.inverse() * rotation).scaled_axis();
        self.update(jacobian, diff, rotation_var)
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
    use approx::assert_relative_eq;
    use nalgebra::{Point3, UnitQuaternion, Vector3};
    use std::f32::consts::FRAC_PI_2;
    use std::time::Duration;

    use crate::ESKF;

    #[test]
    fn creation() {
        let filter = ESKF::new();
        assert_relative_eq!(filter.position, Point3::origin());
        assert_relative_eq!(filter.velocity, Vector3::zeros());
    }

    #[test]
    fn linear_motion() {
        let mut filter = ESKF::new();
        // Some initial motion to move the filter
        filter.predict(
            Vector3::new(1.0, 0.0, -9.81),
            Vector3::zeros(),
            Duration::from_millis(1000).as_secs_f32(),
        );
        assert_relative_eq!(filter.position, Point3::new(0.5, 0.0, 0.0));
        assert_relative_eq!(filter.velocity, Vector3::new(1.0, 0.0, 0.0));
        // There should be no rotation change from the above motion
        assert_relative_eq!(filter.rotation, UnitQuaternion::identity());
        // Acceleration has stopped, but there will still be inertia in the filter
        filter.predict(
            Vector3::new(0.0, 0.0, -9.81),
            Vector3::zeros(),
            Duration::from_millis(500).as_secs_f32(),
        );
        assert_relative_eq!(filter.position, Point3::new(1.0, 0.0, 0.0));
        assert_relative_eq!(filter.velocity, Vector3::new(1.0, 0.0, 0.0));
        assert_relative_eq!(filter.rotation, UnitQuaternion::identity());
        filter.predict(
            Vector3::new(-1.0, -1.0, -9.81),
            Vector3::zeros(),
            Duration::from_millis(1000).as_secs_f32(),
        );
        assert_relative_eq!(filter.position, Point3::new(1.5, -0.5, 0.0));
        assert_relative_eq!(filter.velocity, Vector3::new(0.0, -1.0, 0.0));
        assert_relative_eq!(filter.rotation, UnitQuaternion::identity());
    }

    #[test]
    fn rotational_motion() {
        let mut filter = ESKF::new();
        // Note that this motion is a free fall rotation
        filter.predict(
            Vector3::zeros(),
            Vector3::new(FRAC_PI_2, 0.0, 0.0),
            Duration::from_millis(1000).as_secs_f32(),
        );
        assert_relative_eq!(
            filter.rotation,
            UnitQuaternion::from_euler_angles(FRAC_PI_2, 0.0, 0.0)
        );
        filter.predict(
            Vector3::zeros(),
            Vector3::new(-FRAC_PI_2, 0.0, 0.0),
            Duration::from_millis(1000).as_secs_f32(),
        );
        assert_relative_eq!(
            filter.rotation,
            UnitQuaternion::from_euler_angles(0.0, 0.0, 0.0)
        );
        // We reset the filter here so that the following equalities are not affected by existing
        // motion in the filter
        let mut filter = ESKF::new();
        filter.predict(
            Vector3::zeros(),
            Vector3::new(0.0, -FRAC_PI_2, 0.0),
            Duration::from_millis(1000).as_secs_f32(),
        );
        assert_relative_eq!(
            filter.rotation,
            UnitQuaternion::from_euler_angles(0.0, -FRAC_PI_2, 0.0)
        );
    }
}
