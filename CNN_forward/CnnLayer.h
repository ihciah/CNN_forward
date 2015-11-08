#pragma once
#ifndef __CnnLayer__
#define __CnnLayer__
#define HAVE_TBB
#include<vector>
#include<opencv2\core\core.hpp>
#include<opencv2\imgproc\imgproc.hpp>
#include<opencv2\highgui\highgui.hpp> 
#include "tbb/parallel_for.h"
#include "tbb/blocked_range.h"

class CnnLayer{
public:
	std::vector<int> parents;//该层的上一层序号
	std::vector<cv::Mat> result;//该层的输出结果
	virtual void forward(const std::vector<CnnLayer*>& structure) = 0;//由子类实现前向计算，将输入计算得到结果存入result
};
#endif