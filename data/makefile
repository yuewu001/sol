test:test.o
	g++ test.o -lpthread -o test
test.o:test.cpp
	g++ -c test.cpp -s -O2 -o test.o
.PHONY: clean
clean:
	-rm test test.o
