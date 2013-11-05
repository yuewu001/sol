test:test.o
	g++ test.o -lpthread -lz -o test
test.o:test.cpp
	#g++ -c test.cpp -g -Wall -o test.o
	g++ -c test.cpp -O2 -s -o test.o
.PHONY: clean
clean:
	-rm test test.o
