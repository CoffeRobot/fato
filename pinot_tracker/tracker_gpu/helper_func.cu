#include "helper_func.h"
#include <iostream>

using namespace std;

void GetDevices()
{
	int count;
	cudaGetDeviceCount(&count);

	cout << "Number of Devices: " << count << endl;
}