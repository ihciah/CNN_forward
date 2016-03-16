/*
Copyright 2015 By ihciah
https://github.com/ihciah/CNN_forward
*/
#pragma once
#ifndef __LayerConfig__
#define __LayerConfig__

#include"CnnNet.h"

#define CONV 'c'
#define INPUT 'i'
#define POOLING 'p'
#define RELU 'r'
#define DENSE 'd'


class LayerConfig{
public:
	std::string name;
	std::vector<std::string> parent_names;
	char type;//c:conv;p:pooling;d:dense;r:relu;i:input
	int method;//For pooling
	int poolsize;//For pooling
	bool is_output;//是否算最后结果
	int w, h;//resize的长宽
	LayerConfig(std::string name, char type, bool is_output);
	LayerConfig(std::string name, char type);
	LayerConfig(std::string name, char type, std::vector<std::string> parent_names);
	LayerConfig(std::string name, char type, std::vector<std::string> parent_names, bool is_output);
	LayerConfig(std::string name, char type, std::string parent_name);
	LayerConfig(std::string name, char type, std::string parent_name, bool is_output);
	LayerConfig(std::string name, char type, int w, int h);
private:
	void init_layer(std::string name, char type, std::vector<std::string> parent_names, bool is_output);
};

#endif

