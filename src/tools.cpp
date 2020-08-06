#include "tools.h"
#include <iostream>

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  VectorXd rmse(4);
  rmse << 0,0,0,0;

  // Input validation
  if(estimations.size() == 0 || estimations.size() != ground_truth.size()){
    std::cout << "Input format error" << std::endl;
    return rmse;
  }
  
  // Residual Summation
  for(unsigned int i=0; i < estimations.size(); ++i){
    VectorXd residual = estimations[i] - ground_truth[i];
    residual = residual.array()*residual.array();
    rmse += residual;
  } 

  rmse = rmse / estimations.size();
  rmse = rmse.array().sqrt();
  return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  MatrixXd Hj_ = MatrixXd(3, 4);
  Hj_ << 0,0,0,0,0,0,0,0,0,0,0,0;
  double px = x_state(0);
  double py = x_state(1);
  double vx = x_state(2);
  double vy = x_state(3);
  double divider = sqrt(px * px + py * py);
  double divider_2 = divider * divider;
  double divider_3 = divider * divider * divider;
  
  if(divider < 0.0001){
    divider = 0.0001;
  }
  if(divider_2 < 0.0001){
    divider_2 = 0.0001;
  }
  if(divider_3 < 0.0001){
    divider_3 = 0.0001;
  }
  

  Hj_ <<  px / divider,    py / divider,   0,   0,
          -py / divider_2, px / divider_2, 0,   0,
          py * (vx * py - vy * px) / divider_3,    px * (vy * px - vx * py) / divider_3,   px / divider,   py / divider;
  return Hj_;
}
