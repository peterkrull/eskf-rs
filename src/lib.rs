//! # 15-state navigation filter for embedded systems
//!
//! Based on the [Error State Kalman Filter](https://arxiv.org/abs/1711.02508)
//! formulation, [`NavigationFilter`] is able to track the `position`,
//! `velocity` and `rotation` of an object, sensing its state through an
//! Inertial Measurement Unit (IMU) and some means of observing the true state
//! of the filter such as GPS, LIDAR or visual odometry.

#![deny(missing_docs)]
#![deny(unsafe_code)]
#![cfg_attr(feature = "no_std", no_std)]

use core::ops::{AddAssign, SubAssign};

use nalgebra::{SMatrix, SVector, UnitQuaternion};

#[cfg(feature = "no_std")]
use num_traits::float::Float as _;

/// Potential errors raised during operations
#[derive(Copy, Clone, Debug)]
pub enum Error {
    /// It is not always the case that a matrix is invertible which can lead to
    /// errors. It is difficult to handle this both for the library and for the
    /// users. In the case of the [`ESKF`], if this happens, it may be caused by
    /// an irregular shaped variance matrix for the update step. In such cases,
    /// inspect the variance matrix. If this happens irregularly it can be a
    /// sign that the uncertainty update is not stable, if possible try one of
    /// `cov-symmetric` or `cov-joseph` features for a more stable update.
    InversionError,
}

/// Helper definition to make it easier to work with errors
pub type Res<T> = core::result::Result<T, Error>;

/// Error State Kalman Filter
///
/// The filter works by calling [`predict`](NavigationFilter::predict) and one
/// or more of the [`observe_`](NavigationFilter::observe_position) methods when
/// data is available..
///
/// The [`predict`](NavigationFilter::predict) step updates the internal state
/// of the filter based on measured acceleration and angular rate coming from an
/// IMU. This step updates the states in the filter based on kinematic equations
/// while increasing the uncertainty of the filter. When one of the `observe_`
/// methods are called, the filter updates the internal state based on this
/// observation, which exposes the error state to the filter, which we can then
/// use to correct the internal state. The uncertainty of the filter is also
/// updated to reflect the variance of the observation and the updated state.
#[derive(Copy, Clone, Debug)]
pub struct NavigationFilter {
    /// Estimated position in filter
    pub position: SVector<f32, 3>,
    /// Estimated velocity in filter
    pub velocity: SVector<f32, 3>,
    /// Estimated rotation in filter
    pub rotation: UnitQuaternion<f32>,
    /// Estimated acceleromter bias
    pub acc_bias: SVector<f32, 3>,
    /// Estimated gyroscope bias
    pub gyr_bias: SVector<f32, 3>,
    /// Gravity vector.
    pub gravity: SVector<f32, 3>,
    /// Covariance of filter state
    covariance: SMatrix<f32, 15, 15>,
    /// Power Spectral Density of the accelerometer's white noise.
    acc_noise_psd: SVector<f32, 3>,
    /// Power Spectral Density of the gyroscope's white noise.
    gyr_noise_psd: SVector<f32, 3>,
    /// Power Spectral Density of the accelerometer bias random walk.
    acc_bias_psd: SVector<f32, 3>,
    /// Power Spectral Density of the gyroscope bias random walk.
    gyr_bias_psd: SVector<f32, 3>,
}

impl Default for NavigationFilter {
    fn default() -> Self {
        NavigationFilter {
            position: SVector::zeros(),
            velocity: SVector::zeros(),
            rotation: UnitQuaternion::identity(),
            acc_bias: SVector::zeros(),
            gyr_bias: SVector::zeros(),
            gravity: SVector::z() * 9.81,
            covariance: SMatrix::from_diagonal(
                &[
                    0.2, 0.2, 0.2,
                    0.02, 0.02, 0.02,
                    0.05, 0.05, 0.05,
                    1e-4, 1e-4, 1e-4, 
                    1e-4, 1e-4, 1e-4,
                ]
                .into(),
            ),
            acc_noise_psd: SVector::from_element(1e-3),
            gyr_noise_psd: SVector::from_element(1e-3),
            acc_bias_psd: SVector::from_element(1e-5),
            gyr_bias_psd: SVector::from_element(1e-5),
        }
    }
}

impl NavigationFilter {
    /// Construct a new filter with default values, with gravity pointing in
    /// in the positive z-direction, good for north-east-down (NED) coordinates.
    pub fn new() -> NavigationFilter {
        Self::default()
    }

    /// Set the accelerometer's **white noise density** (standard deviation).
    ///
    /// This value is typically found on the sensor's datasheet, often called
    /// "Noise Density" or "Velocity Random Walk" (VRW).
    ///
    /// The unit must be `(m/s²) / √Hz`.
    pub fn acc_noise_density(mut self, noise_density: f32) -> Self {
        self.acc_noise_psd = SVector::from_element(noise_density.powi(2));
        self
    }

    /// Set the gyroscope's **white noise density** (standard deviation).
    ///
    /// This value is typically found on the sensor's datasheet, often called
    /// "Noise Density" or "Angle Random Walk" (ARW).
    ///
    /// The unit must be `(rad/s) / √Hz`.
    pub fn gyr_noise_density(mut self, noise_density: f32) -> Self {
        self.gyr_noise_psd = SVector::from_element(noise_density.powi(2));
        self
    }

    /// Set the accelerometer's **bias random walk** (standard deviation).
    ///
    /// This value represents the standard deviation of the bias drift,
    /// modeling it as a continuous random walk.
    ///
    /// The unit must be `(m/s²) / √s` (or `(m/s²) / s / √Hz`).
    pub fn acc_bias_random_walk(mut self, random_walk: f32) -> Self {
        self.acc_bias_psd = SVector::from_element(random_walk.powi(2));
        self
    }

    /// Set the gyroscope's **bias random walk** (standard deviation).
    ///
    /// This value represents the standard deviation of the bias drift,
    /// modeling it as a continuous random walk.
    ///
    /// The unit must be `(rad/s) / √s` (or `(rad/s) / s / √Hz`).
    pub fn gyr_bias_random_walk(mut self, random_walk: f32) -> Self {
        self.gyr_bias_psd = SVector::from_element(random_walk.powi(2));
        self
    }

    /// Set all diagonal elements of the covariance for the process matrix.
    ///
    /// The covariance value should be a small process value so that the
    /// covariance of the filter quickly converges to the correct value. Too
    /// small values could lead to the filter taking a long time to converge and
    /// report a lower covariance than what it should.
    ///
    /// Note: Values smaller than `1e-9` will be clamped to this value.
    pub fn covariance_diag_element(mut self, cov: f32) -> Self {
        self.covariance = SMatrix::identity() * cov.max(1e-9);
        self
    }

    /// Set the diagonal elements of the covariance for the process matrix.
    ///
    /// The covariance value should be a small process value so that the
    /// covariance of the filter quickly converges to the correct value. Too
    /// small values could lead to the filter taking a long time to converge and
    /// report a lower covariance than what it should.
    ///
    /// Note: Values smaller than 1e-9 will be clamped to this value.
    pub fn covariance_diag(mut self, cov: impl Into<SVector<f32, 15>>) -> Self {
        self.covariance = SMatrix::from_diagonal(&cov.into().map(|x| x.max(1e-9)));
        self
    }

    /// Set the used gravity in m/s².
    ///
    /// The default value is (positive) 9.81 m/s² in the z direction.
    /// This is fitting for a NED (north east down) reference frame.
    pub fn with_gravity(mut self, gravity: impl Into<SVector<f32, 3>>) -> Self {
        self.gravity = gravity.into();
        self
    }

    /// Get the uncertainty of the position estimate
    pub fn position_uncertainty(&self) -> SVector<f32, 3> {
        self.uncertainty_3(0)
    }

    /// Get the uncertainty of the velocity estimate
    pub fn velocity_uncertainty(&self) -> SVector<f32, 3> {
        self.uncertainty_3(3)
    }

    /// Get the uncertainty of the rotation estimate
    pub fn rotation_uncertainty(&self) -> SVector<f32, 3> {
        self.uncertainty_3(6)
    }

    /// Get the uncertainty accelerometer bias estimate
    pub fn acc_bias_uncertainty(&self) -> SVector<f32, 3> {
        self.uncertainty_3(9)
    }

    /// Get the uncertainty gyroscope bias estimate
    pub fn gyr_bias_uncertainty(&self) -> SVector<f32, 3> {
        self.uncertainty_3(12)
    }

    /// Internal helper method to extract 3 dimensional uncertainty from the covariance state
    fn uncertainty_3(&self, start: usize) -> SVector<f32, 3> {
        self.covariance
            .fixed_view::<3, 3>(start, start)
            .diagonal()
            .map(|var| var.sqrt())
    }

    /// Get the full covariance matrix. See `*_uncertainty` methods for the
    /// variance of specific estimates.
    pub fn covariance_matrix(&self) -> &SMatrix<f32, 15, 15> {
        &self.covariance
    }

    fn cov3_copy_from(&mut self, row: usize, col: usize, other: &SMatrix<f32, 3, 3>) {
        self.covariance
            .fixed_view_mut::<3, 3>(row, col)
            .copy_from(other);
    }

    fn cov3_clone(&mut self, row: usize, col: usize) -> SMatrix<f32, 3, 3> {
        self.covariance.fixed_view::<3, 3>(row, col).clone_owned()
    }

    /// Update the filter, predicting the new state, based on measured
    /// acceleration and angular velocity from an `IMU`. The accelerometer
    /// readings must be m/s^2, and the gyroscope reading must be rad/s.
    pub fn predict(&mut self, acc_meas: SVector<f32, 3>, gyr_meas: SVector<f32, 3>, dt: f32) {
        // Adjust measurement using predicted bias
        let acc_corrected = acc_meas - self.acc_bias;
        let gyr_corrected = gyr_meas - self.gyr_bias;

        // Rotate acceleration into world frame and compensate for gravity
        let rot_acc_grav = self.rotation.transform_vector(&acc_corrected) + self.gravity;

        // Change in rotation according to gyroscope measurements
        let delta_rot = UnitQuaternion::from_scaled_axis(gyr_corrected * dt);

        // Save the apriori rotation matrix for covariance propagation step
        let rot_mat = self.rotation.to_rotation_matrix().into_inner();

        // Update internal state kinematics
        self.position += self.velocity * dt + 0.5 * rot_acc_grav * dt.powi(2);
        self.velocity += rot_acc_grav * dt;
        self.rotation *= delta_rot;

        // Ensure rotation stays consistent
        self.rotation.renormalize_fast();

        // Block-entries of error jacobian and their transposes
        let f_1_2 = -rot_mat * skew(acc_corrected) * dt;
        let f_1_3 = -rot_mat * dt;
        let f_2_2_t = delta_rot.to_rotation_matrix().into_inner();

        let f_1_2_t = f_1_2.transpose();
        let f_1_3_t = f_1_3.transpose();
        let f_2_2 = f_2_2_t.transpose();

        // Extract the 15 upper-triangle blocks from the covariance matrix
        let p_1_1 = self.cov3_clone(0, 0);
        let p_1_2 = self.cov3_clone(0, 3);
        let p_1_3 = self.cov3_clone(0, 6);
        let p_1_4 = self.cov3_clone(0, 9);
        let p_1_5 = self.cov3_clone(0, 12);

        let p_2_2 = self.cov3_clone(3, 3);
        let p_2_3 = self.cov3_clone(3, 6);
        let p_2_4 = self.cov3_clone(3, 9);
        let p_2_5 = self.cov3_clone(3, 12);

        let p_3_3 = self.cov3_clone(6, 6);
        let p_3_4 = self.cov3_clone(6, 9);
        let p_3_5 = self.cov3_clone(6, 12);

        let p_4_4 = self.cov3_clone(9, 9);
        let p_4_5 = self.cov3_clone(9, 12);

        let p_5_5 = self.cov3_clone(12, 12);

        // Block-row 1
        let temp_2 = &p_1_2 + dt * &p_2_2;
        let temp_3 = &p_1_3 + dt * &p_2_3;
        let temp_4 = &p_1_4 + dt * &p_2_4;
        let temp_5 = &p_1_5 + dt * &p_2_5;
        self.cov3_copy_from(0, 0, &(p_1_1 + dt * (p_1_2.transpose() + temp_2)));
        self.cov3_copy_from(0, 3, &(temp_2 + &temp_3 * &f_1_2_t + &temp_4 * &f_1_3_t));
        self.cov3_copy_from(0, 6, &(temp_3 * &f_2_2_t - dt * &temp_5));
        self.cov3_copy_from(0, 9, &temp_4);
        self.cov3_copy_from(0, 12, &temp_5);

        // Block-row 2
        let temp_3 = &p_2_3 + &f_1_2 * &p_3_3 + &f_1_3 * &p_3_4.transpose();
        let temp_4 = &p_2_4 + &f_1_2 * &p_3_4 + &f_1_3 * &p_4_4;
        let temp_5 = &p_2_5 + &f_1_2 * &p_3_5 + &f_1_3 * &p_4_5;
        self.cov3_copy_from(
            3,
            3,
            &(p_2_2
                + &f_1_2 * p_2_3.transpose()
                + &f_1_3 * p_2_4.transpose()
                + &temp_3 * &f_1_2_t
                + &temp_4 * &f_1_3_t),
        );
        self.cov3_copy_from(3, 6, &(temp_3 * &f_2_2_t - dt * &temp_5));
        self.cov3_copy_from(3, 9, &temp_4);
        self.cov3_copy_from(3, 12, &temp_5);

        // Block-row 3
        let temp_3 = &f_2_2 * &p_3_3 - dt * &p_3_5.transpose();
        let temp_4 = &f_2_2 * &p_3_4 - dt * &p_4_5.transpose();
        let temp_5 = &f_2_2 * &p_3_5 - dt * &p_5_5;
        self.cov3_copy_from(6, 6, &(temp_3 * f_2_2_t - dt * temp_5));
        self.cov3_copy_from(6, 9, &temp_4);
        self.cov3_copy_from(6, 12, &temp_5);

        // Row 4 and 5 can be omitted since they are not changed.
        // This also makes sense, since we need observations in order to
        // say anything about how the bias estimates should be updated.

        // Fill elements into lower part, since covariance are symmetric.
        self.covariance.fill_lower_triangle_with_upper_triangle();

        // Add process noise based on the continuous-time model
        let dt2 = dt * dt;
        let dt3 = dt2 * dt;
        let q_pos = self.acc_noise_psd * (1.0/3.0 * dt3);
        let q_pos_vel = self.acc_noise_psd * (1.0/2.0 * dt2);
        let q_vel = self.acc_noise_psd * dt;

        // Add primary diagonal terms
        let mut diagonal = self.covariance.diagonal();
        diagonal
            .fixed_view_mut::<3, 1>(0, 0)
            .add_assign(q_pos);
        diagonal
            .fixed_view_mut::<3, 1>(3, 0)
            .add_assign(q_vel);
        diagonal
            .fixed_view_mut::<3, 1>(6, 0)
            .add_assign(self.gyr_noise_psd * dt);
        diagonal
            .fixed_view_mut::<3, 1>(9, 0)
            .add_assign(self.acc_bias_psd * dt);
        diagonal
            .fixed_view_mut::<3, 1>(12, 0)
            .add_assign(self.gyr_bias_psd * dt);

        self.covariance.set_diagonal(&diagonal);

        // Add off-diagonal position-velocity noise
        let q_pv_diag = SMatrix::from_diagonal(&q_pos_vel);
        self.covariance
            .fixed_view_mut::<3, 3>(0, 3)
            .add_assign(&q_pv_diag);
        self.covariance
            .fixed_view_mut::<3, 3>(3, 0)
            .add_assign(&q_pv_diag);
    }

    /// Update the filter with a generic observation
    ///
    /// # Arguments
    /// - `jacobian` is the measurement Jacobian matrix
    /// - `residual` is the error between the measured and the estimated state
    /// - `variance` is the uncertainty of the observation
    pub fn update<const R: usize>(
        &mut self,
        jacobian: SMatrix<f32, R, 15>,
        residual: SVector<f32, R>,
        variance: SMatrix<f32, R, R>,
    ) -> Res<()> {
        // Correct filter based on Kalman gain
        let cov_x_jacob = self.covariance * &jacobian.transpose();
        let innovation_cov = &jacobian * cov_x_jacob + &variance;
        let innovation_cov_inv = innovation_cov.try_inverse().ok_or(Error::InversionError)?;
        let kalman_gain = cov_x_jacob * innovation_cov_inv;

        let error_state = &kalman_gain * residual;

        // Update the covariance based on the observed filter state
        if cfg!(feature = "cov-joseph") {
            let step1 = SMatrix::identity() - &kalman_gain * &jacobian;
            let step2 = &kalman_gain * &variance * &kalman_gain.transpose();
            self.covariance = step1 * self.covariance * step1.transpose() + step2;
        } else {
            self.covariance -= &kalman_gain * &innovation_cov * &kalman_gain.transpose();
        }

        self.update_finalize(error_state)
    }

    /// Update the filter with a generic observation, where the jacobian is
    /// identity. If this is not the case, see [`NavigationFilter::update`].
    ///
    /// # Arguments
    /// - `index` is the start index of the identity jacobian
    /// - `residual` is the error between the measured and the estimated state
    /// - `variance` is the uncertainty of the observation
    pub fn update_identity<const R: usize>(
        &mut self,
        index: usize,
        residual: SVector<f32, R>,
        variance: SMatrix<f32, R, R>,
    ) -> Res<()> {
        // When the jacobian is identity, many matrix-matrix multiplications
        // will simplify to just extracting a sub-matrix view, so we might as
        // well do that directly here.
        let cov_x_jacob = self.covariance.fixed_view::<15, R>(0, index);
        let innovation_cov = self.covariance.fixed_view::<R, R>(index, index) + &variance;
        let innovation_cov_inv = innovation_cov.try_inverse().ok_or(Error::InversionError)?;
        let kalman_gain = cov_x_jacob * innovation_cov_inv;

        let error_state = &kalman_gain * residual;

        // Update the covariance based on the observed filter state
        if cfg!(feature = "cov-joseph") {
            let mut step1 = SMatrix::<f32, 15, 15>::identity();
            step1
                .fixed_view_mut::<15, R>(0, index)
                .sub_assign(&kalman_gain);
            let step2 = &kalman_gain * &variance * &kalman_gain.transpose();
            self.covariance = step1 * self.covariance * step1.transpose() + step2;
        } else {
            self.covariance -= &kalman_gain * &innovation_cov * &kalman_gain.transpose();
        }

        self.update_finalize(error_state)
    }

    /// Outlined finalization of [`ESKF::update`] function to reduce monomorphization impact.
    ///
    /// # Arguments
    /// - `error_state` is the error state calculated by the [`ESKF::update`] function
    #[inline(never)]
    fn update_finalize(&mut self, error_state: SVector<f32, 15>) -> Res<()> {
        // Inject error state into nominal state
        self.position += error_state.fixed_view::<3, 1>(0, 0);
        self.velocity += error_state.fixed_view::<3, 1>(3, 0);
        self.rotation *= UnitQuaternion::from_scaled_axis(error_state.fixed_view::<3, 1>(6, 0));
        self.acc_bias += error_state.fixed_view::<3, 1>(9, 0);
        self.gyr_bias += error_state.fixed_view::<3, 1>(12, 0);

        if cfg!(feature = "full-reset") {
            // Get the 3x3 rotation-reset block
            let error_rot = error_state.fixed_view::<3, 1>(6, 0).clone_owned();
            let g_rot = SMatrix::<f32, 3, 3>::identity() - skew(error_rot) * 0.5;
            let g_rot_t = g_rot.transpose();

            // Clone the blocks we need from the upper triangle
            let p_pos_rot = self.covariance.fixed_view::<3, 3>(0, 6);
            let p_vel_rot = self.covariance.fixed_view::<3, 3>(3, 6);
            let p_rot_rot = self.covariance.fixed_view::<3, 3>(6, 6);
            let p_rot_acc = self.covariance.fixed_view::<3, 3>(6, 9);
            let p_rot_gyr = self.covariance.fixed_view::<3, 3>(6, 12);

            // Perform the cheaper 3x3 multiplications
            let p_pos_rot_new = p_pos_rot * g_rot_t;
            let p_vel_rot_new = p_vel_rot * g_rot_t;
            let p_rot_rot_new = g_rot * p_rot_rot * g_rot_t;
            let p_rot_acc_new = g_rot * p_rot_acc;
            let p_rot_gyr_new = g_rot * p_rot_gyr;

            // Write the new values back to the covariance matrix
            self.covariance
                .fixed_view_mut::<3, 3>(0, 6)
                .copy_from(&p_pos_rot_new);
            self.covariance
                .fixed_view_mut::<3, 3>(3, 6)
                .copy_from(&p_vel_rot_new);
            self.covariance
                .fixed_view_mut::<3, 3>(6, 6)
                .copy_from(&p_rot_rot_new);
            self.covariance
                .fixed_view_mut::<3, 3>(6, 9)
                .copy_from(&p_rot_acc_new);
            self.covariance
                .fixed_view_mut::<3, 3>(6, 12)
                .copy_from(&p_rot_gyr_new);

            // And the transposed go into the lower triangle
            self.covariance
                .fixed_view_mut::<3, 3>(6, 0)
                .copy_from(&p_pos_rot_new.transpose());
            self.covariance
                .fixed_view_mut::<3, 3>(6, 3)
                .copy_from(&p_vel_rot_new.transpose());
            self.covariance
                .fixed_view_mut::<3, 3>(9, 6)
                .copy_from(&p_rot_acc_new.transpose());
            self.covariance
                .fixed_view_mut::<3, 3>(12, 6)
                .copy_from(&p_rot_gyr_new.transpose());
        }

        // Ensure rotation stays consistent
        self.rotation.renormalize_fast();

        Ok(())
    }

    /// Update the filter with an observation of the position.
    pub fn observe_position(
        &mut self,
        position: SVector<f32, 3>,
        position_var: SMatrix<f32, 3, 3>,
    ) -> Res<()> {
        let diff = position - self.position;
        self.update_identity(0, diff, position_var)
    }

    /// Update the filter with an observation of the position in the x-y plane.
    pub fn observe_position_xy(
        &mut self,
        position_xy: SVector<f32, 2>,
        position_xy_var: SMatrix<f32, 2, 2>,
    ) -> Res<()> {
        let diff = position_xy - self.position.xy();
        self.update_identity(0, diff, position_xy_var)
    }

    /// Update the filter with an observation of the z-position only.
    pub fn observe_position_z(&mut self, position_z: f32, position_z_var: f32) -> Res<()> {
        let diff = position_z - self.position.z;
        self.update_identity(2, [diff].into(), [position_z_var].into())
    }

    /// Update the filter with an observation of the velocity
    ///
    /// # Note
    /// If the observation comes from a sensor relative to the filter, e.g. an optical flow sensor
    /// that turns with the UAV, the sensor values **needs** to be rotated into the same frame as
    /// the filter, e.g. `filter.rotation.transform_vector(&relative_measurement)`.
    pub fn observe_velocity(
        &mut self,
        velocity: SVector<f32, 3>,
        velocity_var: SMatrix<f32, 3, 3>,
    ) -> Res<()> {
        let diff = velocity - self.velocity;
        self.update_identity(3, diff, velocity_var)
    }

    /// Update the filter with an observation of the velocity in only the `[X, Y]` axis
    ///
    /// # Note
    /// If the observation comes from a sensor relative to the filter, e.g. an optical flow sensor
    /// that turns with the UAV, the sensor values **needs** to be rotated into the same frame as
    /// the filter, e.g. `filter.rotation.transform_vector(&relative_measurement)`.
    pub fn observe_velocity_xy(
        &mut self,
        velocity: SVector<f32, 2>,
        velocity_var: SMatrix<f32, 2, 2>,
    ) -> Res<()> {
        let diff = velocity - self.velocity.xy();
        self.update_identity(3, diff, velocity_var)
    }

    /// Update the filter with an observation of the velocity in the x-y axis
    pub fn observe_velocity_z(&mut self, velocity: f32, velocity_var: f32) -> Res<()> {
        let diff = velocity - self.velocity.z;
        self.update_identity(5, [diff].into(), [velocity_var].into())
    }

    /// Update the filter with an observation of the rotation
    pub fn observe_rotation(
        &mut self,
        rotation: UnitQuaternion<f32>,
        rotation_var: SMatrix<f32, 3, 3>,
    ) -> Res<()> {
        let diff = (self.rotation.inverse() * rotation).scaled_axis();
        self.update_identity(6, diff, rotation_var)
    }
}

/// Create the skew-symmetric matrix from a vector
#[rustfmt::skip]
fn skew(v: SVector<f32, 3>) -> SMatrix<f32, 3, 3> {
    SMatrix::<f32, 3, 3>::new(0., -v.z, v.y,
                 v.z, 0., -v.x,
                 -v.y, v.x, 0.)
}

#[cfg(test)]
mod test {
    use approx::assert_relative_eq;
    use nalgebra::{UnitQuaternion, Vector3};
    use std::f32::consts::FRAC_PI_2;
    use std::time::Duration;

    use crate::NavigationFilter;

    #[test]
    fn creation() {
        let filter = NavigationFilter::new();
        assert_relative_eq!(filter.position, Vector3::zeros());
        assert_relative_eq!(filter.velocity, Vector3::zeros());
    }

    #[test]
    fn linear_motion() {
        let mut filter = NavigationFilter::new();
        // Some initial motion to move the filter
        filter.predict(
            Vector3::new(1.0, 0.0, -9.81),
            Vector3::zeros(),
            Duration::from_millis(1000).as_secs_f32(),
        );
        assert_relative_eq!(filter.position, Vector3::new(0.5, 0.0, 0.0));
        assert_relative_eq!(filter.velocity, Vector3::new(1.0, 0.0, 0.0));
        // There should be no rotation change from the above motion
        assert_relative_eq!(filter.rotation, UnitQuaternion::identity());
        // Acceleration has stopped, but there will still be inertia in the filter
        filter.predict(
            Vector3::new(0.0, 0.0, -9.81),
            Vector3::zeros(),
            Duration::from_millis(500).as_secs_f32(),
        );
        assert_relative_eq!(filter.position, Vector3::new(1.0, 0.0, 0.0));
        assert_relative_eq!(filter.velocity, Vector3::new(1.0, 0.0, 0.0));
        assert_relative_eq!(filter.rotation, UnitQuaternion::identity());
        filter.predict(
            Vector3::new(-1.0, -1.0, -9.81),
            Vector3::zeros(),
            Duration::from_millis(1000).as_secs_f32(),
        );
        assert_relative_eq!(filter.position, Vector3::new(1.5, -0.5, 0.0));
        assert_relative_eq!(filter.velocity, Vector3::new(0.0, -1.0, 0.0));
        assert_relative_eq!(filter.rotation, UnitQuaternion::identity());
    }

    #[test]
    fn rotational_motion() {
        let mut filter = NavigationFilter::new();
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
        let mut filter = NavigationFilter::new();
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
