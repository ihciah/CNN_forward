#pragma once
#ifndef __Dense__
#define __Dense__

#include"CnnLayer.h"
class Dense :public CnnLayer{
public:
	cv::Mat bias;
	std::vector<std::vector<cv::Mat>> weight;
	int dim[4];
	void forward(const std::vector<CnnLayer*>& structure);
};

#endif