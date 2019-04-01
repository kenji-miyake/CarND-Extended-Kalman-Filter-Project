#include "kalman_filter.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

/*
 * Please note that the Eigen library does not initialize
 *   VectorXd or MatrixXd objects with zeros upon creation.
 */

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  MatrixXd Ft = F_.transpose();

  x_ = F_ * x_;
  P_ = F_ * P_ * F_.transpose() + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  MatrixXd I = MatrixXd::Identity(x_.size(), x_.size());

  MatrixXd S = H_ * P_ * H_.transpose() + R_;
  MatrixXd K = P_ * H_.transpose() * S.inverse();
  VectorXd y = z - H_ * x_;

  x_ = x_ + (K * y);
  P_ = (I - K * H_) * P_;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  MatrixXd I = MatrixXd::Identity(x_.size(), x_.size());

  const auto px = x_(0);
  const auto py = x_(1);
  const auto vx = x_(2);
  const auto vy = x_(3);

  const auto rho = sqrt(px * px + py * py);

  VectorXd h_x(3);
  h_x << rho,
        atan2(py, px),
        (px * vx + py * vy) / rho;

  MatrixXd S = H_ * P_ * H_.transpose() + R_;
  MatrixXd K = P_ * H_.transpose() * S.inverse();
  VectorXd y = z - h_x;

  // normalize angle
  if(y(1) > M_PI)
  {
    y(1) = 2 * M_PI - y(1);
  }
  else if(y(1) < -M_PI)
  {
    y(1) = 2 * M_PI + y(1);
  }

  x_ = x_ + (K * y);
  P_ = (I - K * H_) * P_;
}
