/*
Copyright 2015 By ihciah
https://github.com/ihciah/CNN_forward
*/
#include "utils.h"

using namespace std;

string num_to_label_string(const std::vector<int>& res){
	string result="";
	char tc;
	for (vector<int>::const_iterator it = res.begin(); it != res.end(); it++){
		if (*it < 10){
			tc = *it + '0';
		}
		else if (*it == 36){
			continue;
		}
		else{
			tc = *it - 10 + 'A';
		}
		result.push_back(tc);
	}
	return result;
}

void num_to_label_cstring(const std::vector<int>& res,char* dst){
	const char* cresult;
	string result = num_to_label_string(res);
	result.copy(dst, result.length());
	dst[result.length()] = '\0';
}