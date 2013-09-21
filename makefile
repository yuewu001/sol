DEBUG_FLAG=1

DEBUG=-g -Wall
RELEASE=-O2 -s

INCLUDE=-I . -I ./data
VPATH=.:./data

CFLAGS=-c $(INCLUDE)
LDFLAGS=

ifeq ($(DEBUG_FLAG),1)
	CFLAGS += $(DEBUG)
else
	CFLAGS += $(RELEASE)
endif

OBJS=test.o
TARGET=test

all:$(TARGET)

$(TARGET):$(OBJS)
	g++ $^ -o $@ $(LDFLAGS)

%.o:%.cpp
	g++ $< -o $@ $(CFLAGS)

.PHONY:clean
clean:
	-rm -f *.o $(TARGET) $(addsuffix .exe, $(TARGET))
