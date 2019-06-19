FAST= -fast
CCINTEL=icpc
CCMPI=mpicxx
CCMPIINTEL=mpiicpc
MPIFAST=-0fast

FLAGS= -O3 
DEBUG= -g -p
FAST = -fast
CC = g++
SRCS = main.cpp  

OBJS = $(SRCS)

main: $(OBJS) 
	$(CC)  $(OBJS) $(FLAGS) -o run -w -std=c++11 

intelfast: $(OBJS)
	$(CCINTEL)  $(OBJS) $(FAST) -o run -std=c++11

debug: $(OBJS)
	$(CC)  $(OBJS) $(DEBUG) -o debug -w -std=c++11

%.cpp.o: %.cpp
	$(CC) $(FLAGS) $(DEBUG) -c -o $@ $< 
clean:
	rm *.cpp.o

