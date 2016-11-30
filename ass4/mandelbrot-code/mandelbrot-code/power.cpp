#include <iostream>
#include <math.h>

using namespace std;
int main(){
	int N;
	cin >> N;
	float p = (pow(3, N/2) - 1)/(float)2;
	cout << 400/p;
	return 0;
}