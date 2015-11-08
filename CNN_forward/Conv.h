#pragma once
#ifndef __Conv__
#define __Conv__
#include"CnnLayer.h"

class Conv :public CnnLayer{
public:
	cv::Mat bias;
	std::vector<std::vector<cv::Mat>> weight;
	int dim[4];
	void forward(const std::vector<CnnLayer*>& structure);
};

#endif