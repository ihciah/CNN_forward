/*
Copyright 2015 By ihciah
https://github.com/ihciah/CNN_forward
*/
#pragma once
#ifndef __Input__
#define __Input__

#include"CnnLayer.h"
class Input :public CnnLayer{
public:
	int w, h;
	void forward(const std::vector<CnnLayer*>& structure);
	void load_image(cv::Mat im);
};

#endif