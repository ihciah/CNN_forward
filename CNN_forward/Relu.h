/*
Copyright 2015 By ihciah
https://github.com/ihciah/CNN_forward
*/
#pragma once
#ifndef __Relu__
#define __Relu__

#include"CnnLayer.h"

class Relu :public CnnLayer{
public:
	void forward(const std::vector<CnnLayer*>& structure);
};


#endif