#include "time.h"
#include <iostream>
#include <math.h>
#include <cstdlib>
#include <stdio.h>
#include "VJ.cpp"

int main(int argc, char **argv) {
	int action = atoi(argv[1]);
    VJ *myVJ = new VJ();
    if (action==0) {
	    myVJ->init_train();
	    myVJ->run();
    }
    if (action==1) {
        myVJ->init_read();
        myVJ->detect();
    }
	puts("Complete");
	
	
	return 0;
}
