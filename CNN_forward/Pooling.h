/*
Copyright 2015 By ihciah
https://github.com/ihciah/CNN_forward
*/
#pragma once
#ifndef __Pooling__
#define __Pooling__

#include"CnnLayer.h"

#define MAX_POOLING 1
#define AVG_POOLING 2

class Pooling :public CnnLayer{
public:
	int poolsize;
	void forward(const std::vector<CnnLayer*>& structure);
};

#endif