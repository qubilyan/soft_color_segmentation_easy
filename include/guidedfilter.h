/*
 * guidedfilter.h
 *
 *  Created on: 9 Mar 2017
 *      Author: https://github.com/atilimcetin/guided-filter
 */

#ifndef GUIDED_FILTER_H
#define GUIDED_FILTER_H

#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc/edge_filter.hpp>

class GuidedFilterImpl;

class GuidedFilter
{
public:
    GuidedFilter(const cv