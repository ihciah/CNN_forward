#pragma once
#ifndef __ModelReader__
#define __ModelReader__

#include<string.h>
#include<vector>
#include<opencv2\core\core.hpp>

class DataBlock{
public:
	std::string name;
	char type;
	int dim[4];
	std::vector<char*> data_bin;
	std::vector<std::vector<char>> data_vec;
};

class Model{
public:
	Model(std::string path);//直接加载指定路径的Model
	Model(std::string path,std::string key);//解密并加载Model
	Model();
	void load(std::string model_path);
	void load(std::string model_path,std::string key);
	std::vector<std::vector<cv::Mat>> get_weight(std::string layer_name);//返回指定名称的卷积层的Weight
	cv::Mat get_bias(std::string layer_name);//返回指定名称的层的bias
	std::vector<std::vector<cv::Mat>> Model::get_fc_weight(std::string layer_name);//返回指定名称的全连接层的Weight
	std::vector<int> Model::get_dim(std::string layer_name);//返回指定名称的层的dim
private:
	std::vector<DataBlock*> data;
};

#endif