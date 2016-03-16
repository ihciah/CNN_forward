/*
Copyright 2015 By ihciah
https://github.com/ihciah/CNN_forward
*/
#pragma once
#ifndef __UTILS__
#define __UTILS__

#include<vector>

std::string num_to_label_string(const std::vector<int>& res);
void num_to_label_cstring(const std::vector<int>& res, char*);


#endif