use approx::assert_relative_eq;
use eskf::Builder;
use nalgebra::Vector3;

#[test]
fn test_predict_equivalence() {
    // Create three identical filters
    let mut filter_original = Builder::new()
        .acceleration_variance(0.01)
        .rotation_variance(0.001)
        .acceleration_bias(0.0001)
        .rotation_bias(0.0001)
        .initial_covariance(1e-6)
        .build();

    let mut filter_optimized = Builder::new()
        .acceleration_variance(0.01)
        .rotation_variance(0.001)
        .acceleration_bias(0.0001)
        .rotation_bias(0.0001)
        .initial_covariance(1e-6)
        .build();

    let mut filter_optimized_tr = Builder::new()
        .acceleration_variance(0.01)
        .rotation_variance(0.001)
        .acceleration_bias(0.0001)
        .rotation_bias(0.0001)
        .initial_covariance(1e-6)
        .build();

    // Apply the same sequence of predictions
    let predictions = vec![
        (
            Vector3::new(0.1, 0.2, -9.81),
            Vector3::new(0.01, -0.02, 0.005),
            0.01,
        ),
        (
            Vector3::new(0.15, 0.1, -9.82),
            Vector3::new(0.02, -0.01, 0.003),
            0.01,
        ),
        (
            Vector3::new(0.05, 0.3, -9.80),
            Vector3::new(-0.01, 0.02, -0.002),
            0.01,
        ),
        (
            Vector3::new(0.2, 0.0, -9.81),
            Vector3::new(0.0, 0.0, 0.01),
            0.01,
        ),
    ];

    for (acc, gyr, dt) in predictions {
        filter_original.predict_original(acc, gyr, dt);
        filter_optimized.predict_optimized(acc, gyr, dt);
        filter_optimized_tr.predict_optimized(acc, gyr, dt);

        // Verify position, velocity, and orientation are identical
        assert_relative_eq!(
            filter_original.position,
            filter_optimized.position,
            epsilon = 1e-6
        );
        assert_relative_eq!(
            filter_original.position,
            filter_optimized_tr.position,
            epsilon = 1e-6
        );
        assert_relative_eq!(
            filter_original.velocity,
            filter_optimized.velocity,
            epsilon = 1e-6
        );
        assert_relative_eq!(
            filter_original.velocity,
            filter_optimized_tr.velocity,
            epsilon = 1e-6
        );
        assert_relative_eq!(
            filter_original.orientation.as_vector(),
            filter_optimized.orientation.as_vector(),
            epsilon = 1e-6
        );
        assert_relative_eq!(
            filter_original.orientation.as_vector(),
            filter_optimized_tr.orientation.as_vector(),
            epsilon = 1e-6
        );
        assert_relative_eq!(
            filter_original.acc_bias,
            filter_optimized.acc_bias,
            epsilon = 1e-6
        );
        assert_relative_eq!(
            filter_original.acc_bias,
            filter_optimized_tr.acc_bias,
            epsilon = 1e-6
        );
        assert_relative_eq!(
            filter_original.gyr_bias,
            filter_optimized.gyr_bias,
            epsilon = 1e-6
        );
        assert_relative_eq!(
            filter_original.gyr_bias,
            filter_optimized_tr.gyr_bias,
            epsilon = 1e-6
        );

        // Verify uncertainty estimates are identical
        assert_relative_eq!(
            filter_original.position_uncertainty(),
            filter_optimized.position_uncertainty(),
            epsilon = 1e-6
        );
        assert_relative_eq!(
            filter_original.position_uncertainty(),
            filter_optimized_tr.position_uncertainty(),
            epsilon = 1e-6
        );
        assert_relative_eq!(
            filter_original.velocity_uncertainty(),
            filter_optimized.velocity_uncertainty(),
            epsilon = 1e-6
        );
        assert_relative_eq!(
            filter_original.velocity_uncertainty(),
            filter_optimized_tr.velocity_uncertainty(),
            epsilon = 1e-6
        );
        assert_relative_eq!(
            filter_original.orientation_uncertainty(),
            filter_optimized.orientation_uncertainty(),
            epsilon = 1e-6
        );
        assert_relative_eq!(
            filter_original.orientation_uncertainty(),
            filter_optimized_tr.orientation_uncertainty(),
            epsilon = 1e-6
        );
    }
}

#[test]
fn test_predict_equivalence_long_sequence() {
    // Test with a longer sequence to catch accumulation errors
    let mut filter_original = Builder::new()
        .acceleration_variance(0.05)
        .rotation_variance(0.005)
        .acceleration_bias(0.001)
        .rotation_bias(0.001)
        .initial_covariance(1e-5)
        .build();

    let mut filter_optimized = Builder::new()
        .acceleration_variance(0.05)
        .rotation_variance(0.005)
        .acceleration_bias(0.001)
        .rotation_bias(0.001)
        .initial_covariance(1e-5)
        .build();

    let mut filter_optimized_tr = Builder::new()
        .acceleration_variance(0.05)
        .rotation_variance(0.005)
        .acceleration_bias(0.001)
        .rotation_bias(0.001)
        .initial_covariance(1e-5)
        .build();

    // Run 100 prediction steps
    for i in 0..100 {
        let t = i as f32 * 0.01;
        let acc = Vector3::new(
            0.1 * (t * 2.0).sin(),
            0.1 * (t * 3.0).cos(),
            -9.81 + 0.05 * (t * 1.5).sin(),
        );
        let gyr = Vector3::new(
            0.01 * (t * 1.0).sin(),
            0.01 * (t * 2.0).cos(),
            0.005 * (t * 3.0).sin(),
        );

        filter_original.predict_original(acc, gyr, 0.01);
        filter_optimized.predict_optimized(acc, gyr, 0.01);
        filter_optimized_tr.predict_optimized(acc, gyr, 0.01);
    }

    // After 100 steps, results should still be identical
    assert_relative_eq!(
        filter_original.position,
        filter_optimized.position,
        epsilon = 1e-5
    );
    assert_relative_eq!(
        filter_original.position,
        filter_optimized_tr.position,
        epsilon = 1e-5
    );
    assert_relative_eq!(
        filter_original.velocity,
        filter_optimized.velocity,
        epsilon = 1e-5
    );
    assert_relative_eq!(
        filter_original.velocity,
        filter_optimized_tr.velocity,
        epsilon = 1e-5
    );
    assert_relative_eq!(
        filter_original.orientation.as_vector(),
        filter_optimized.orientation.as_vector(),
        epsilon = 1e-5
    );
    assert_relative_eq!(
        filter_original.orientation.as_vector(),
        filter_optimized_tr.orientation.as_vector(),
        epsilon = 1e-5
    );
    assert_relative_eq!(
        filter_original.position_uncertainty(),
        filter_optimized.position_uncertainty(),
        epsilon = 1e-5
    );
    assert_relative_eq!(
        filter_original.position_uncertainty(),
        filter_optimized_tr.position_uncertainty(),
        epsilon = 1e-5
    );
    assert_relative_eq!(
        filter_original.velocity_uncertainty(),
        filter_optimized.velocity_uncertainty(),
        epsilon = 1e-5
    );
    assert_relative_eq!(
        filter_original.velocity_uncertainty(),
        filter_optimized_tr.velocity_uncertainty(),
        epsilon = 1e-5
    );
    assert_relative_eq!(
        filter_original.orientation_uncertainty(),
        filter_optimized.orientation_uncertainty(),
        epsilon = 1e-5
    );
    assert_relative_eq!(
        filter_original.orientation_uncertainty(),
        filter_optimized_tr.orientation_uncertainty(),
        epsilon = 1e-5
    );
}
