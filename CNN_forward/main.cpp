/*
Copyright 2015 By ihciah
https://github.com/ihciah/CNN_forward
*/
#include "CnnNet.h"
#include "utils.h"
#include <iostream>
#include <time.h>
using namespace std;

int main(){
	int t_before;

	cout << "Initializing CNN-net...";
	t_before = clock();
	CnnNet net;
	net.init("model", "");
	cout << "Done. " << clock() - t_before << "ms"<<endl;
//======
	t_before = clock();
	cout << "Net forwarding...";
	net.forward("test.jpg",GRAY);
	cout << "Done. " << clock() - t_before << "ms"<<endl;

	t_before = clock();
	cout << "Calculating result..."<<endl;
	vector<int> labels = net.argmax();
	char tmp[20];
	num_to_label_cstring(labels,tmp);
	cout << tmp << endl;
	cout << "Done. " << clock() - t_before << "ms" << endl;
//=====


	return 0;
}
