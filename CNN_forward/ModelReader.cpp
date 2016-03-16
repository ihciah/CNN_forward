/*
Copyright 2015 By ihciah
https://github.com/ihciah/CNN_forward
*/
#include <fstream>

#include"ModelReader.h"
#include"blowfish.h"

using namespace std;
using namespace cv;

/*
File format:
layer_name[16]|char type|int dim[4]|int size|char original_data[size]|int size2|char original_data2[size]|int size3....until sizen=0
if layer is dense, dim=[output,input_sum_count,1,1]
if layer is conv, dim=[output,input,kernel_rows,kernel_cols]
*/

Model::Model(string path){
	this->load(path);
}

Model::Model(string path, string key){
	this->load(path, key);
}

Model::Model(){

}

void Model::load(std::string model_path){
	ifstream modelfile(model_path, ios::binary);
	if (!modelfile){
		exit(1);
	}
	char cname[17];//存储层名

	while (modelfile.read(cname, 16*sizeof(char))){
		DataBlock* layer_data = new DataBlock();//每读一层的数据创建一个DataBlock存储读入的数据
		cname[16] = '\0'; //防止溢出
		layer_data->name = string(cname);
		char type;
		modelfile.read(&type, sizeof(char));
		layer_data->type = type;
		for (int i = 0; i <= 3; i++){
			modelfile.read((char*)&(layer_data->dim[i]), sizeof(int));
		}
		int data_size;
		modelfile.read((char*)&data_size, sizeof(int));
		while (data_size){//读入长度，如果长度不为0则读取该长度的数据
			vector<char> original_data;//每次读取数据创建一个vec存储数据
			for (int i = 0; i < data_size*4; i++){
				char x;
				modelfile.read(&x, sizeof(char));//循环读入数据
				original_data.push_back(x);
			}
			layer_data->data_vec.push_back(original_data);
			char* buf = new char[sizeof(char)*original_data.size()+1];
			std::copy(original_data.begin(), original_data.end(), buf);//转换为char*数组
			layer_data->data_bin.push_back(buf);
			modelfile.read((char*)&data_size, sizeof(int));
		}
		this->data.push_back(layer_data);//将创建的DataBlock加入Model的data
		
	}
}
void Model::load(std::string model_path, std::string key){
	this->load(model_path);
	vector<char> v_key(key.begin(), key.end());
	Blowfish bf(v_key);
	for (vector<DataBlock*>::iterator x = this->data.begin(); x != this->data.end(); x++){
		(*x)->data_bin.clear();
		for (vector<vector<char>>::iterator y = (*x)->data_vec.begin(); y != (*x)->data_vec.end(); y++){
			(*y) = bf.Decrypt(*y);//对每个层中的每个数据块解密
			char* buf = new char[sizeof(char)*(*y).size() + 1];
			std::copy((*y).begin(), (*y).end(), buf);//转换为char*数组
			(*x)->data_bin.push_back(buf);
		}
	}
}
vector<vector<Mat>> Model::get_weight(std::string layer_name){
	vector<vector<Mat>> ret;
	int mat_size;
	for (vector<DataBlock*>::iterator x = this->data.begin(); x != this->data.end(); x++){
		if ((*x)->name == layer_name){
			mat_size = (*x)->dim[2] * (*x)->dim[3];
			float* pointer = (float*)(*x)->data_bin[0];
			for (int A = 0; A < (*x)->dim[0]; A++){
				vector<Mat> tmp;
				for (int B = 0; B < (*x)->dim[1]; B++){
					tmp.push_back(Mat((*x)->dim[2], (*x)->dim[3], CV_32F, (void*)pointer));
					pointer += mat_size;
				}
				ret.push_back(tmp);
			}
			break;
		}
	}
	return ret;//dim[0]*dim[1]*Mat(dim[2]*dim[3])
}
Mat Model::get_bias(std::string layer_name){
	Mat ret;
	int mat_size;
	for (vector<DataBlock*>::iterator x = this->data.begin(); x != this->data.end(); x++){
		if ((*x)->name == layer_name){
			mat_size = (*x)->dim[0];
			float* pointer = (float*)(*x)->data_bin[1];
			ret = Mat(mat_size, 1, CV_32F, (void*)pointer);
			break;
		}
	}
	return ret;//Mat(dim[0]*1)
}
vector<vector<Mat>> Model::get_fc_weight(std::string layer_name){
	vector<vector<Mat>> ret;
	int mat_size;
	for (vector<DataBlock*>::iterator x = this->data.begin(); x != this->data.end(); x++){
		if ((*x)->name == layer_name){
			mat_size = (*x)->dim[1];
			float* pointer = (float*)(*x)->data_bin[0];
			for (int A = 0; A < (*x)->dim[0]; A++){
				vector<Mat> tmp;
				tmp.push_back(Mat(1, (*x)->dim[1], CV_32F, (void*)pointer));
				pointer += mat_size;
				ret.push_back(tmp);
			}
			break;
		}
	}
	return ret;//dim[0]*1*Mat(1*dim[1])
}
vector<int> Model::get_dim(std::string layer_name){
	vector<int> ret;
	for (vector<DataBlock*>::iterator x = this->data.begin(); x != this->data.end(); x++){
		if ((*x)->name == layer_name){
			for (int i = 0; i < 4;i++)
				ret.push_back((*x)->dim[i]);
			break;
		}
	}
	return ret;//vector(int4)
}