/*
Copyright 2015 By ihciah
https://github.com/ihciah/CNN_forward
*/
#include"Dense.h"

using namespace std;
using namespace cv;

void Dense::forward(const std::vector<CnnLayer*>& structure){
	this->result.clear();
	vector<Mat> input = structure[this->parents[0]]->result;
	Mat input_vec(this->dim[1], 1, CV_32F);
	MatIterator_<float> input_vec_it = input_vec.begin<float>();
	for (int i = 0; i < input.size(); i++){
		for (MatConstIterator_<float> j = input[i].begin<float>(); j != input[i].end<float>(); j++, input_vec_it++){
			*input_vec_it = *j;
		}
	}
	for (int output_num = 0; output_num < this->dim[0]; output_num++){
		Mat tmp(1, 1, CV_32F);
		tmp.setTo(this->bias.at<float>(output_num, 0));
		cv::add(tmp, this->weight[output_num][0] * input_vec, tmp);
		//dim = [output, input_sum_count, 1, 1]; weight=dim[0]*1*Mat(1*dim[1]); bias=Mat(dim[0]*1)
		this->result.push_back(tmp);
	}
}