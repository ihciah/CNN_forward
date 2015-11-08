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
	net.init("test_output", "");
	cout << "Done. " << clock() - t_before << "ms"<<endl;

	t_before = clock();
	cout << "Net forwarding...";
	net.forward("6.jpg",GRAY);
	cout << "Done. " << clock() - t_before << "ms"<<endl;

	t_before = clock();
	cout << "Calculating result..."<<endl;
	vector<int> labels = net.argmax();
	char tmp[20];
	num_to_label_cstring(labels,tmp);
	cout << tmp << endl;
	cout << "Done. " << clock() - t_before << "ms" << endl;

	return 0;
}
