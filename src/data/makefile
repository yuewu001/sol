FLAGS = -g -Wall
#FLAGS = -O2 -s

LIBS=basic_io.o zlib_io.o gzip_io.o
TARGETS=analysis test

all: $(TARGETS)
analysis:data_analysis.o $(LIBS)
	g++ $^ -lpthread -lz -o analysis

test:test.o $(LIBS)
	g++ $^ -lpthread -lz -o test

%.o:%.cpp
	g++ -c $^ -o $@ $(FLAGS)

.PHONY: clean
clean:
	-rm $(TARGETS) *.o
