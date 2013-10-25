DEBUG_FLAG=0

DEBUG=-g -Wall
RELEASE=-O2 -s

INCLUDE=-I . -I ./data -I ./loss -I./optimizer
VPATH=.:./data:./loss:./optimizer

CFLAGS=-c $(INCLUDE) -Wno-write-strings
LDFLAGS=-lpthread

ifeq ($(DEBUG_FLAG),1)
	CFLAGS += $(DEBUG)
else
	CFLAGS += $(RELEASE)
endif

OBJS=main.o Params.o
TARGET=SOL

all:$(TARGET) analysis

$(TARGET):$(OBJS)
	g++ $^ -o $@ $(LDFLAGS)

analysis:data_analysis.o
	g++ $< -o $@

%.o:%.cpp
	g++ $< -o $@ $(CFLAGS)

.PHONY:clean
clean:
	-rm -f *.o $(TARGET) $(addsuffix .exe, $(TARGET)) tags cscope*


