FLAGS = -g -Wall
#FLAGS = -O2 -s

all: analysis test
analsis:data_analysis.o basic_io.o
	g++ $^ -lpthread -lz -o analysis

test:test.o basic_io.o
	g++ $^ -lpthread -lz -o test

%.o:%.cpp
	g++ -c $^ -o $@ $(FLAGS)

.PHONY: clean
clean:
	-rm test *.o
