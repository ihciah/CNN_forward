#pragma once
#ifndef __Relu__
#define __Relu__

#include"CnnLayer.h"

class Relu :public CnnLayer{
public:
	void forward(const std::vector<CnnLayer*>& structure);
};


#endif