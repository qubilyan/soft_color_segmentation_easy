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

//static variables for debugging (which is reached: maxIterLineSeach, cg_max_iter or isMin)
extern int reach_ls_iter;
extern int total_line_search;
extern int reach_cg_iter;
extern int reach_isMin_iter;

double energy(std::vector<double> v, std::vector<cv::Vec3d> &means, s