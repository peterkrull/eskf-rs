use nalgebra::{SMatrix, UnitQuaternion, Vector3};
use core::ops::AddAssign;

/// Stores pre-integrated IMU measurements between Kalman filter updates.
///
/// # Example
/// ```ignore
/// let mut preint = ImuPreintegration::new();
/// 
/// // Integrate multiple IMU measurements (e.g., at 1000 Hz)
/// for _ in 0..10 {
///     preint.integrate(acc_reading, gyr_reading, 0.001, acc_bias, gyr_bias);
/// }
/// 
/// // Apply accumulated measurement to filter (e.g., at 100 Hz)
/// filter.predict_preintegrated(&preint, gravity);
/// 
/// // Reset for next integration period
/// preint.reset();
/// ```
#[derive(Clone, Debug)]
pub struct PreIntegrator {
    /// Integrated position change (in body frame at start of integration)
    pub delta_position: Vector3<f32>,
    /// Integrated velocity change (in body frame at start of integration)
    pub delta_velocity: Vector3<f32>,
    /// Integrated orientation change
    pub delta_orientation: UnitQuaternion<f32>,
    /// Total accumulated integration time
    pub delta_time: f32,

    /// 9x9 Covariance of the pre-integrated measurements. Order: (orientation, velocity, position)
    pub covariance: SMatrix<f32, 9, 9>,
    /// 9x6 Jacobian of pre-integrated measurements w.r.t. biases. Order: d(orientation, velocity, position)/d(acc_bias, gyr_bias)
    pub jacobian: SMatrix<f32, 9, 6>,
}

impl PreIntegrator {
    /// Create a new preintegrator with identity values
    pub fn new() -> Self {
        Self {
            delta_position: Vector3::zeros(),
            delta_velocity: Vector3::zeros(),
            delta_orientation: UnitQuaternion::identity(),
            delta_time: 0.0,
            covariance: SMatrix::zeros(),
            jacobian: SMatrix::zeros(),
        }
    }

    /// Integrate a single IMU measurement into the accumulated state.
    ///
    /// # Arguments
    /// - `acc_meas`: Measured acceleration from accelerometer (m/s²)
    /// - `gyr_meas`: Measured angular velocity from gyroscope (rad/s)
    /// - `dt`: Time step since last integration (seconds)
    /// - `acc_bias`: Current estimate of accelerometer bias (m/s²)
    /// - `gyr_bias`: Current estimate of gyroscope bias (rad/s)
    ///
    /// # Note
    /// The measurements should be in the same coordinate frame as the filter.
    /// Bias estimates should be the current filter estimates.
    pub fn integrate(
        &mut self,
        acc_meas: Vector3<f32>,
        gyr_meas: Vector3<f32>,
        acc_bias: Vector3<f32>,
        gyr_bias: Vector3<f32>,
        acc_var: Vector3<f32>,
        gyr_var: Vector3<f32>,
        dt: f32,
    ) {
        
        let acc_corrected = acc_meas - acc_bias;
        let gyr_corrected = gyr_meas - gyr_bias;

        let delta_rot = UnitQuaternion::from_scaled_axis(gyr_corrected * dt);
        let delta_rot_mat = delta_rot.to_rotation_matrix().into_inner();
        let rot_mat = self.delta_orientation.to_rotation_matrix().into_inner();

        // --- 1. Update Covariance and Jacobians ---
        let acc_skew = super::skew(&acc_corrected);

        // Discrete-time state transition matrix for the error state [d_theta, d_vel, d_pos]
        let mut a = SMatrix::<f32, 9, 9>::identity();
        a.fixed_view_mut::<3, 3>(0, 0).copy_from(&delta_rot_mat.transpose());
        a.fixed_view_mut::<3, 3>(3, 0).copy_from(&(-rot_mat * acc_skew * dt));
        a.fixed_view_mut::<3, 3>(6, 0).copy_from(&(-0.5 * rot_mat * acc_skew * dt.powi(2)));
        a.fixed_view_mut::<3, 3>(6, 3).copy_from(&(SMatrix::identity() * dt));

        // Propagate covariance
        self.covariance = a * self.covariance * a.transpose();
        
        let rot_var_acc = rot_mat * SMatrix::from_diagonal(&acc_var) * rot_mat.transpose();
        self.covariance.fixed_view_mut::<3, 3>(0,0).add_assign(&SMatrix::from_diagonal(&(gyr_var * dt)));
        self.covariance.fixed_view_mut::<3, 3>(3,3).add_assign(&(rot_var_acc * dt));
        self.covariance.fixed_view_mut::<3, 3>(3,6).add_assign(&(rot_var_acc * dt.powi(2) * 0.5));
        self.covariance.fixed_view_mut::<3, 3>(6,3).add_assign(&(rot_var_acc * dt.powi(2) * 0.5));
        self.covariance.fixed_view_mut::<3, 3>(6,6).add_assign(&(rot_var_acc * dt.powi(3) * 0.25));


        // Propagate bias jacobian: J_k+1 = A * J_k + C
        // Where C maps bias changes to error state changes in this step
        let mut c = SMatrix::<f32, 9, 6>::zeros();
        c.fixed_view_mut::<3, 3>(0, 3).copy_from(&(-SMatrix::identity() * dt)); // d_theta / d_bg
        c.fixed_view_mut::<3, 3>(3, 0).copy_from(&(-rot_mat * dt)); // d_vel / d_ba
        c.fixed_view_mut::<3, 3>(6, 0).copy_from(&(-0.5 * rot_mat * dt.powi(2))); // d_pos / d_ba
        
        self.jacobian = a * self.jacobian + c;

        // --- 2. Update Mean Pre-integrated Values ---
        let d_r_acc_dt = rot_mat * acc_corrected * dt;
        self.delta_position += self.delta_velocity * dt + 0.5 * d_r_acc_dt * dt;
        self.delta_velocity += d_r_acc_dt;
        self.delta_orientation *= delta_rot;
        
        self.delta_time += dt;
    }

    /// Reset the preintegrator to identity values, ready for a new integration period.
    pub fn reset(&mut self) {
        self.delta_position.fill(0.0);
        self.delta_velocity.fill(0.0);
        self.delta_orientation = UnitQuaternion::identity();
        self.delta_time = 0.0;
        self.covariance.fill(0.0);
        self.jacobian.fill(0.0);
    }

    /// Check if any measurements have been integrated
    pub fn is_empty(&self) -> bool {
        self.delta_time == 0.0
    }

    /// Get the total accumulated time
    pub fn total_time(&self) -> f32 {
        self.delta_time
    }
}

impl Default for PreIntegrator {
    fn default() -> Self {
        Self::new()
    }
}