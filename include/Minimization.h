/*
 * Minimization.h
 *
 *      Author: Sebastian Lutz
 *  University: Trinity College Dublin
 *      School: Computer Science and Statistics
 *     Project: V-SENSE
 */

#ifndef MINIMIZATION_H_
#define MINIMIZATION_H_

#include <vector>
#include <tuple>
#include <opencv2/opencv.hpp>
#include <numeric>
#include "constants.h"

typedef double (* vFunctionCall)(std::vector<double> v, void* params); // pointer to function returning double
typedef std::vector<double> (* vFunctionCall2)(std::vector<double> v, void* params);

//sta