/**
 * activation.hpp
 * 
 * In this header file, we define
 * all neuron activation functions.
 * Specifically, there is an 
 * implementation of the sigmoid
 * activation function. There is  
 * also the corresponding derivative 
 * of the sigmoid function to be used 
 * for the back propagation algorithm.
 */

#pragma once

#include "common.hpp"

double sigmoid(double x);
double sig_derivative(double x);
