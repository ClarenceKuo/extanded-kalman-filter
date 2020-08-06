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
  Hj_ = MatrixXd(3, 4);

  //measurement covariance matrix - laser
  R_laser_ << 0.0225, 0,
              0, 0.0225;

  //measurement covariance matrix - radar
  R_radar_ << 0.09, 0, 0,
              0, 0.0009, 0,
              0, 0, 0.09;

  /**
   * TODO: Finish initializing the FusionEKF.
   * TODO: Set the process and measurement noises
   */
  // Measurement matrix - laser
  H_laser_ << 1, 0, 0, 0,
              0, 1, 0, 0;

  // State covariance matrix
  ekf_.P_ = MatrixXd(4, 4);
  ekf_.P_ << 1, 0, 0, 0,
             0, 1, 0, 0,
             0, 0, 1000, 0,
             0, 0, 0, 1000;

  // Init state transition matrix
  ekf_.F_ = MatrixXd(4, 4);
  ekf_.F_ << 1, 0, 1, 0,
            0, 1, 0, 1,
            0, 0, 1, 0,
            0, 0, 0, 1;

  // Init noise covariance matrix
  ekf_.Q_ = MatrixXd(4, 4);
  ekf_.Q_ << 1, 0, 1, 0,
         0, 1, 0, 1,
         1, 0, 1, 0,
         0, 1, 0, 1;


}

/**
 * Destructor.
 */
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {
  if (!is_initialized_) {
 
    // first measurement
    cout << "Init EKF" << endl;
    ekf_.x_ = VectorXd(4);
    double px;
    double py;
    double vx;
    double vy;
    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      double rho = measurement_pack.raw_measurements_[0];
      double phi = measurement_pack.raw_measurements_[1];
      double d_rho = measurement_pack.raw_measurements_[2];
      px = rho * cos(phi);
      py = rho * sin(phi);
      vx = d_rho * cos(phi);
      vy = d_rho * sin(phi);
    }else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      px = measurement_pack.raw_measurements_[0];
      py = measurement_pack.raw_measurements_[1];
      vx = 0.0;
      vy = 0.0;
    }
    ekf_.x_ << px, py, vx, vy;
    previous_timestamp_  = measurement_pack.timestamp_;
    is_initialized_ = true;
    return;
  }

  /**
   * Prediction
   */

  //Get new elapsed time
  float d_t = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;
  previous_timestamp_  = measurement_pack.timestamp_;
  
  // Update the process noise covariance matrix.
  ekf_.F_ <<  1, 0, d_t,   0,
              0, 1,   0, d_t,
              0, 0,   1,   0,
              0, 0,   0,   1;
  
  // Use noise_ax = 9 and noise_ay = 9 for your Q matrix.
  double noise_ax = 9.0;
  double noise_ay = 9.0;
  double d_t_2 = d_t * d_t;
  double d_t_3 = d_t_2 * d_t;
  double d_t_4 = d_t_3 * d_t;

  ekf_.Q_ <<  (d_t_4/4) * noise_ax,     0, (d_t_3/2) * noise_ax, 0,
              0, (d_t_4/4) * noise_ay,    0,  (d_t_3/2) * noise_ay,
              (d_t_3/2) * noise_ax,     0,  d_t_2 * noise_ax,    0,
              0, (d_t_3/2) * noise_ay,    0,   d_t_2 * noise_ay;
  
  // Predict function
  ekf_.Predict();

  /**
   * Update
   */

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    ekf_.H_ = tools.CalculateJacobian(ekf_.x_);
    ekf_.R_ = R_radar_;
    ekf_.UpdateEKF(measurement_pack.raw_measurements_);
  } else {
    ekf_.H_ = H_laser_;
    ekf_.R_ = R_laser_;
    ekf_.Update(measurement_pack.raw_measurements_);
  }
}
