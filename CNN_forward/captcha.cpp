#include "captcha.h"

#include<iostream>
#include<string>
#include<vector>


void Cap_rec::init(char* model){
	std::string smodel = model;
	this->net.init(smodel,"");
}

void Cap_rec::rec(char* path, char* result){
	std::string spath = path;
	this->net.forward(path, GRAY);
	std::vector<int> labels = this->net.argmax();
	num_to_label_cstring(labels, result);
}