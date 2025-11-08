# Embeddable navigation filter

This is a fork of [eskf-rs](https://github.com/nordmoen/eskf-rs/) tailored for `no_std` embedded environments with all statically allocated matrices, and an unrolled prediction function for much better performance. An STM32F405 is able to do a full integration of IMU measurements and prediction, including the 15x15 covariance propagation, in approximately 38 µs. This implementation is thus suitable embedded for real-time estimation.

## Error State Kalman Filter (ESKF)
An [Error State Kalman Filter](https://arxiv.org/abs/1711.02508) is a navigation
filter based on regular Kalman filters, more specifically [Extended Kalman
Filters](https://en.wikipedia.org/wiki/Extended_Kalman_filter), that model the
"error state" of the system instead of modelling the movement of the system
explicitly.

The navigation filter is used to track `position`, `velocity` and `orientation`
of an object which is sensing its state through an [Inertial Measurement Unit
(IMU)](https://en.wikipedia.org/wiki/Inertial_measurement_unit) and some means
of observing the true state of the filter such as GPS, LIDAR or visual odometry. 
Additionally, the bias of the accelerometer and gyroscope belonging to the IMU
is estimated for even more accurate estimation.

## Usage
```rust
use eskf::ESKF;
use nalgebra::{SMatrix, Vector3, Point3};

// Create the filter and configure the parameters.
let mut filter = ESKF::new();
filt.acc_noise_std(0.05);
filt.gyr_noise_std(0.01);
filt.acc_bias_std(0.001);
filt.acc_bias_std(0.001);
filt.covariance_diag(0.2);
filt.with_gravity(Vector3::z() * 9.81);

loop {

    // We are likely fusing multiple types of sensors, so in an async framework
    // we might just select over some futures which yields the measurements.
    match select(
        imu_sensor.read_6dof(),
        pos_sensor.read_position()
    ).await {
        First(imu_data) => {
            // The IMU measurements are automatically adjusted using the
            // estimated bias, and rotated into the global reference frame
            // to update the position, velocity and rotation estimates.
            filter.predict_optimized(
                imu_data.acc.into(),
                imu_data.gyr.into(),
            );

            // Now we can use the estimates (see "Usage tip" below)
            let pos = filter.position;
            let vel = filter.velocity;
            let rot = filter.rotation;
        },
        Second(pos_data) => {
            if filter.observe_position(
                Point3::new(pos_data.x, pos_data.y, pos_data.z),
                SMatrix::from_diagonal_element(pos_data.var),
            ).is_err() {
                // This is a basic way to handle the error case. It might also
                // be fine to initially skip the observation, and let the
                // covariance evolve a bit until the next observation.
                defmt::error!("[eskf] Could not compute, resetting covariance");
                filt.covariance_diag(0.2);
            }
        }
    }
}
```

### Usage tip

While the Kalman filter is optimal in the least-squares sense, it might not be ideal to use some of its values directily. Specifically, the position estimate is likely to "jump" slightly whenever a new observation makes a correction. This could result in large derivative spikes for a position controller. However, complimentary filtering the estimated position and velocity can yield a much smoother and less jumpy position estimate, without losing any useful high frequency information.
