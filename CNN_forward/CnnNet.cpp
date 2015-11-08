#include"CnnNet.h"
#include"LayerConfig.h"
#include"CnnAllLayers.h"

using namespace std;
using namespace cv;

void LayerConfig::init_layer(std::string name, char type, vector<std::string> parent_names, bool is_output){
	this->name = name;
	this->parent_names = parent_names;
	this->type = type;
	this->poolsize = 2;
	this->method = MAX_POOLING;
	this->is_output = is_output;
}
LayerConfig::LayerConfig(std::string name, char type){
	this->w = 0;
	this->h = 0;
	this->init_layer(name, type, vector<string>(), false);
}
LayerConfig::LayerConfig(std::string name, char type, bool is_output){
	this->init_layer(name, type, vector<string>(), is_output);
}
LayerConfig::LayerConfig(std::string name, char type, vector<std::string> parent_names){
	this->init_layer(name, type, parent_names, false);
}
LayerConfig::LayerConfig(std::string name, char type, vector<std::string> parent_names, bool is_output){
	this->init_layer(name, type, parent_names, is_output);
}
LayerConfig::LayerConfig(std::string name, char type, std::string parent_name){
	vector<string> parent_names;
	parent_names.push_back(parent_name);
	this->init_layer(name, type, parent_names, false);
}
LayerConfig::LayerConfig(std::string name, char type, std::string parent_name, bool is_output){
	vector<string> parent_names;
	parent_names.push_back(parent_name);
	this->init_layer(name, type, parent_names, is_output);
}
LayerConfig::LayerConfig(std::string name, char type, int w, int h){
	this->w = w;
	this->h = h;
	this->init_layer(name, type, vector<string>(), false);
}








void CnnNet::init(string FilePath, string Key=""){
	vector<LayerConfig*> layer_config;

	LayerConfig* input = new LayerConfig("input", INPUT, 128, 40);
	layer_config.push_back(input);

	LayerConfig* conv1 = new LayerConfig("conv1", CONV);
	layer_config.push_back(conv1);

	LayerConfig* relu1 = new LayerConfig("relu1", RELU);
	layer_config.push_back(relu1);

	LayerConfig* pooling1 = new LayerConfig("pooling1", POOLING);
	layer_config.push_back(pooling1);//默认2x2 Max Pooling

	LayerConfig* conv2 = new LayerConfig("conv2", CONV);
	layer_config.push_back(conv2);

	LayerConfig* relu2 = new LayerConfig("relu2", RELU);
	layer_config.push_back(relu2);

	LayerConfig* pooling2 = new LayerConfig("pooling2", POOLING);
	layer_config.push_back(pooling2);//默认2x2 Max Pooling

	LayerConfig* conv3 = new LayerConfig("conv3", CONV);
	layer_config.push_back(conv3);

	LayerConfig* relu3 = new LayerConfig("relu3", RELU);
	layer_config.push_back(relu3);

	LayerConfig* pooling3 = new LayerConfig("pooling3", POOLING);
	layer_config.push_back(pooling3);//默认2x2 Max Pooling

	LayerConfig* densea1 = new LayerConfig("ip2a", DENSE);
	layer_config.push_back(densea1);

	LayerConfig* densea2 = new LayerConfig("ipfinala", DENSE, true);
	layer_config.push_back(densea2);

	const char* last_layer[] = { "pooling3" };
	vector<string> last(last_layer, last_layer + 1);

	LayerConfig* denseb1 = new LayerConfig("ip2b", DENSE, last);//use more than 1 parents
	layer_config.push_back(denseb1);

	LayerConfig* denseb2 = new LayerConfig("ipfinalb", DENSE, true);//true means output
	layer_config.push_back(denseb2);

	LayerConfig* densec1 = new LayerConfig("ip2c", DENSE, (string)"pooling3");//use only one parent
	layer_config.push_back(densec1);

	LayerConfig* densec2 = new LayerConfig("ipfinalc", DENSE ,true);
	layer_config.push_back(densec2);

	LayerConfig* densed1 = new LayerConfig("ip2d", DENSE, (string)"pooling3");//add (string) to prevent being converted to bool
	layer_config.push_back(densed1);

	LayerConfig* densd2 = new LayerConfig("ipfinald", DENSE, true);
	layer_config.push_back(densd2);

	LayerConfig* densee1 = new LayerConfig("ip2e", DENSE, (string)"pooling3");
	layer_config.push_back(densee1);

	LayerConfig* densee2 = new LayerConfig("ipfinale", DENSE, true);
	layer_config.push_back(densee2);

	LayerConfig* densef1 = new LayerConfig("ip2f", DENSE, (string)"pooling3");
	layer_config.push_back(densef1);

	LayerConfig* densef2 = new LayerConfig("ipfinalf", DENSE, true);
	layer_config.push_back(densef2);

	if (Key == "")
		this->model = new Model(FilePath);
	else
		this->model = new Model(FilePath, Key);


	this->proc_layers(layer_config);
	
}

void CnnNet::proc_layers(vector<LayerConfig*> layer_config){
	map<string, int> dic;//存储layer名字和下标的对应表
	int v_counter = 0;
	for (vector<LayerConfig*>::iterator x = layer_config.begin(); x != layer_config.end(); x++, v_counter++){
		//如果创建网络时未指定前一层名字，自动指定为上一层
		if ((*x)->parent_names.size() == 0 && (*x)->type != INPUT){
			(*x)->parent_names.push_back((*(x - 1))->name);
		}
		dic[(*x)->name] = v_counter;
	}
	//用于计算A->B有无链接的图.graph[A][B]=1表示有A->B的连接
	vector<vector<int>> graph;
	int node_size = layer_config.size();
	for (int A = 0; A < node_size; A++){
		vector<int> tmp;
		for (int B = 0; B < node_size; B++){
			tmp.push_back(0);
		}
		graph.push_back(tmp);
	}
	v_counter = 0;
	for (vector<LayerConfig*>::iterator x = layer_config.begin(); x != layer_config.end(); x++, v_counter++){
		for (vector<string>::iterator name = (*x)->parent_names.begin();
			name != (*x)->parent_names.end(); name++){
			graph[dic[*name]][v_counter] = 1;
		}
	}
	vector<int> sorted_list;//保存产生的下标结果，网络前向时以该下标为顺序传播
	for (int num = 0; num < node_size; num++){
		//每个循环向sorted_list push一个结果
		for (int B = 0; B < node_size; B++){
			//找入度为0的结果，如果该点已经在sorted_list中则跳过
			vector<int>::iterator ret;
			ret = std::find(sorted_list.begin(), sorted_list.end(), B);
			if (ret != sorted_list.end())
				continue;
			bool is_zero = true;
			for (int A = 0; A < node_size; A++){
				if (graph[A][B]){
					is_zero = false;
					break;
				}
			}
			if (is_zero){
				sorted_list.push_back(B);
				for (int C = 0; C < node_size; C++)
					graph[B][C] = 0;
			}
		}
	}
	//按名字读取所有模型权值
	for (vector<int>::iterator x = sorted_list.begin(); x != sorted_list.end(); x++){
		//按顺序遍历sorted_list并按照该结果取出layer，处理后push进structure
		LayerConfig* single_layer_config = layer_config[*x];
		switch (single_layer_config->type){
		case INPUT:
		{
			Input* i_la = new Input();
			i_la->w = single_layer_config->w;
			i_la->h = single_layer_config->h;
			this->structure.push_back(i_la);
			break;
		}
		case CONV:
		{
			Conv* c_la = new Conv();
			for (vector<string>::iterator it = single_layer_config->parent_names.begin();
				it != single_layer_config->parent_names.end(); it++){
				c_la->parents.push_back(dic[*it]);//按照string-int对照表将parent_names存入该层parents
			}
			vector<int> _dim = this->model->get_dim(single_layer_config->name);
			for (int ci = 0; ci < 4; ci++)
				//c_la->dim[ci] = single_layer_config->dim[ci];//拷贝dim[4]
				c_la->dim[ci] = _dim[ci];//拷贝dim[4]
			c_la->weight = this->model->get_weight(single_layer_config->name);//拷贝weight
			c_la->bias = this->model->get_bias(single_layer_config->name);//拷贝bias
			this->structure.push_back(c_la);
			break;
		}
		case POOLING:
		{
			Pooling* p_la = new Pooling();
			for (vector<string>::iterator it = single_layer_config->parent_names.begin();
				it != single_layer_config->parent_names.end(); it++){
				p_la->parents.push_back(dic[*it]);//按照string-int对照表将parent_names存入该层parents
			}
			p_la->poolsize = single_layer_config->poolsize;
			this->structure.push_back(p_la);
			break;
		}
		case RELU:
		{
			Relu* r_la = new Relu();
			for (vector<string>::iterator it = single_layer_config->parent_names.begin();
				it != single_layer_config->parent_names.end(); it++){
				r_la->parents.push_back(dic[*it]);//按照string-int对照表将parent_names存入该层parents
			}
			this->structure.push_back(r_la);
			break;
		}
		case DENSE:
		{
			Dense* d_la = new Dense();
			for (vector<string>::iterator it = single_layer_config->parent_names.begin();
				it != single_layer_config->parent_names.end(); it++){
				d_la->parents.push_back(dic[*it]);//按照string-int对照表将parent_names存入该层parents
			}
			vector<int> _dim = this->model->get_dim(single_layer_config->name);
			for (int ci = 0; ci < 4; ci++)
				//c_la->dim[ci] = single_layer_config->dim[ci];//拷贝dim[4]
				d_la->dim[ci] = _dim[ci];//拷贝dim[4]
			d_la->weight = this->model->get_fc_weight(single_layer_config->name);//拷贝weight
			d_la->bias = this->model->get_bias(single_layer_config->name);//拷贝bias
			this->structure.push_back(d_la);
			if (single_layer_config->is_output==true){
				this->result_layer.push_back(this->structure.size() - 1);
			}
			break;
		}
		default:
			break;
		}
	}
}

void CnnNet::forward(const cv::Mat &im){
	((Input*)this->structure[0])->load_image(im);
	for (vector<CnnLayer*>::iterator it = this->structure.begin(); it != this->structure.end(); it++){
		(*it)->forward(this->structure);
	}
}

void CnnNet::forward(const std::string path,int mode){
	Mat im;
	if (mode == GRAY){
		im = imread(path, CV_LOAD_IMAGE_GRAYSCALE);
	}
	else{
		im = imread(path);
	}
	((Input*)this->structure[0])->load_image(im);
	for (vector<CnnLayer*>::iterator it = this->structure.begin(); it != this->structure.end(); it++){
		(*it)->forward(this->structure);
	}
}


std::vector<int> CnnNet::argmax(const std::vector<int>& layer_nums){
	vector<int> result;
	for (vector<int>::const_iterator num = layer_nums.begin(); num != layer_nums.end(); num++){
		float tmp = (this->structure[*num]->result)[0].at<float>(0, 0);
		int label = 0, iter = 0;
		for (vector<Mat>::iterator it = this->structure[*num]->result.begin(); it != this->structure[*num]->result.end(); it++, iter++){
			if ((*it).at<float>(0, 0) > tmp){
				label = iter;
				tmp = (*it).at<float>(0, 0);
			}
		}
		result.push_back(label);
	}
	return result;
}

std::vector<int> CnnNet::argmax(){
	vector<int> result = this->argmax(this->result_layer);
	return result;
}