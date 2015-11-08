#include"Relu.h"

using namespace std;
using namespace cv;
void Relu::forward(const std::vector<CnnLayer*>& structure){
	this->result.clear();
	vector<Mat> input = structure[this->parents[0]]->result;
	for (vector<Mat>::const_iterator it_v = input.begin(); it_v != input.end(); it_v++){
		Mat outImage(it_v->size(), it_v->type());
		threshold(*it_v, outImage, 0.0, 255.0, CV_THRESH_TOZERO);
		this->result.push_back(outImage);
	}
}