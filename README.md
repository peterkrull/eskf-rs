# Embeddable navigation filter

This is a fork of [eskf-rs](https://github.com/nordmoen/eskf-rs/) tailored for `no_std` and embedded environments with all statically allocated matrices, and unrolled sparse matrix operations for much better performance. An STM32F405 is able to do a full integration of IMU measurements and prediction, including the 15x15 covariance propagation, in approximately 38 µs. This implementation is thus suitable for embedded real-time estimation.

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
use eskf_rs::NavigationFilter;
use nalgebra::{SMatrix, Vector3};

// Create the filter and configure the parameters.
let mut filter = NavigationFilter::new()
    .acc_noise_density(0.005);
    .gyr_noise_density(0.002);
    .acc_bias_random_walk(0.0001);
    .acc_bias_random_walk(0.0001);
    .with_gravity(Vector3::z() * 9.81);

// Alternatively, this can be calculated on the go
const IMU_SAMPLE_TIME: f32 = 1. / 1000.;

loop {

    // We are likely fusing multiple types of sensors, so in an async framework
    // we might just select over some futures which yields the measurements.
    match select(
        imu_sensor.read_6dof(),
        motion_sensor.read(),
    ).await {
        First(imu_data) => {
            // The IMU measurements are automatically adjusted using the
            // estimated bias, and rotated into the global reference frame
            // to update the position, velocity and rotation estimates.
            filter.predict(
                imu_data.acc.into(),
                imu_data.gyr.into(),
                IMU_SAMPLE_TIME as f32,
            );

            // Now we can use the estimates (see "Usage tip" below)
            let pos = filter.position;
            let vel = filter.velocity;
            let rot = filter.rotation;
        },
        Second(motion_data) => {

            // We may make multiple observations sequentially. As long as we assume little to no cross-covariance between the observations this is fine.
            if filter.observe_position(
                Vector3::from(motion_data.position),
                SMatrix::from_diagonal_element(motion_data.pos_var),
            ).is_err() {
                defmt::error!("[eskf] Failure during velocity observation");
            }

            if filter.observe_velocity(
                Vector3::from(motion_data.velocity),
                SMatrix::from_diagonal_element(motion_data.vel_var),
            ).is_err() {
                defmt::error!("[eskf] Failure during velocity observation");
            }
        }
    }
}
```

### Usage tip

While the Kalman filter is optimal in the least-squares sense, it might not be ideal to use some of its values directily. Specifically, the position estimate is likely to "jump" slightly whenever a new observation makes a correction. This could result in large derivative spikes for a position controller. However, complimentary filtering the estimated position and velocity can yield a much smoother and less jumpy position estimate, without losing any useful high frequency information.
