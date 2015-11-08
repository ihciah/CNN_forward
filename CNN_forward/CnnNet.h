#pragma once
#ifndef __CnnNet__
#define __CnnNet__

#include<vector>
#include<string.h>
#include<opencv2\core\core.hpp>

#include "CnnAllLayers.h"
#include "ModelReader.h"
#include "LayerConfig.h"

#define COLOR 1
#define GRAY 0

class CnnNet{
public:
	std::vector<CnnLayer*> structure;//存放CnnLayer子类指针
	Model* model;
	void forward(const cv::Mat&);
	void forward(const std::string path,int mode);
	void init(std::string FilePath,std::string Key);//创建Layer并加载模型参数到Layer里
	std::vector<int> argmax(const std::vector<int>& layer_nums);//从指定层号读取结果并返回argmax
	std::vector<int> argmax();//从net中定义的结果层读取结果并返回argmax
private:
	void proc_layers(std::vector<LayerConfig*>);//由layer_config完成structure的初始化操作
	std::vector<int> result_layer;//存放产生结果的层的序号
};

#endif