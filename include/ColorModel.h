/*
 * ColorModel.h
 *
 *      Author: Sebastian Lutz & Mairéad Grogan & Johanna Barbier
 *  University: Trinity College Dublin
 *      School: Computer Science and Statistics
 *     Project: V-SENSE
 */

#ifndef COLORMODEL_H_
#define COLORMODEL_H_

#include <opencv2/opencv.hpp>
#include <vector>
#include "constants.h"
#include <tuple>
#include "guidedfilter.h"
#include "Pixel.h"
#include "Minimization.h"

void getGlobalColorModel(cv::Mat &imag