#include "FusionEKF.h"
#include <iostream>
#include "Eigen/Dense"
#include "tools.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::cout;
using std::endl;
using std::vector;

/**
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);
  Hj_      = MatrixXd(3, 4);

  //measurement covariance matrix - laser
  R_laser_ << 0.0225,      0,
                   0, 0.0225;

  //measurement covariance matrix - radar
  R_radar_ << 0.09,      0,    0,
                 0, 0.0009,    0,
                 0,      0, 0.09;

  // create a 4D state vector, we don't know yet the values of the x state
  // x = [px, py, vx, vy]T
  ekf_.x_ = VectorXd(4);

  // state covariance matrix P
  ekf_.P_ = MatrixXd(4, 4);
  ekf_.P_ << 1, 0,    0,    0,
             0, 1,    0,    0,
             0, 0, 1000,    0,
             0, 0,    0, 1000;


  // measurement covariance
  ekf_.R_ = MatrixXd(2, 2);
  ekf_.R_ << 0.0225,      0,
                  0, 0.0225;

  // measurement matrix
  ekf_.H_ = MatrixXd(2, 4);
  ekf_.H_ << 1, 0, 0, 0,
             0, 1, 0, 0;

  // the initial transition matrix F_
  ekf_.F_ = MatrixXd(4, 4);
  ekf_.F_ << 1, 0, 1, 0,
             0, 1, 0, 1,
             0, 0, 1, 0,
             0, 0, 0, 1;

}

/**
 * Destructor.
 */
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {
  /**
   * Initialization
   */
  if (!is_initialized_) {
    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      const auto rho = measurement_pack.raw_measurements_[0];
      const auto phi = measurement_pack.raw_measurements_[1];
      const auto px  =  rho * cos(phi);
      const auto py  =  rho * sin(phi);

      ekf_.x_ << px,
                 py,
                  0,
                  0;
    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      const auto px = measurement_pack.raw_measurements_[0];
      const auto py = measurement_pack.raw_measurements_[1];

      ekf_.x_ << px,
                 py,
                  0,
                  0;
    }

    // remember previous timestamp
    previous_timestamp_ = measurement_pack.timestamp_;

    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }

  /**
   * Prediction
   */
  // Compure dt in seconds
  const auto dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;
  previous_timestamp_ = measurement_pack.timestamp_;

  // Compute variable
  const auto dt_2 = dt * dt;
  const auto dt_3 = dt_2 * dt;
  const auto dt_4 = dt_3 * dt;

  // Update the state transition matrix F according to the new elapsed time.
  ekf_.F_(0, 2) = dt;
  ekf_.F_(1, 3) = dt;

  // Update the process noise covariance matrix.
  ekf_.Q_ = MatrixXd(4, 4);
  {
    // set the acceleration noise components
    constexpr auto n_ax = 9;
    constexpr auto n_ay = 9;
    ekf_.Q_ << dt_4/4*n_ax,           0, dt_3/2*n_ax,           0,
                         0, dt_4/4*n_ay,           0, dt_3/2*n_ay,
               dt_3/2*n_ax,           0,   dt_2*n_ax,           0,
                         0, dt_3/2*n_ay,           0,   dt_2*n_ay;
  }

  ekf_.Predict();

  /**
   * Update
   */
  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    Tools tools;
    ekf_.H_ = tools.CalculateJacobian(ekf_.x_);
    ekf_.R_ = R_radar_;
    ekf_.UpdateEKF(measurement_pack.raw_measurements_);
  } else {
    ekf_.H_ = MatrixXd(2, 4);
    ekf_.H_ << 1, 0, 0, 0,
               0, 1, 0, 0;

    ekf_.R_ = R_laser_;
    ekf_.Update(measurement_pack.raw_measurements_);
   }

  // print the output
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
}
