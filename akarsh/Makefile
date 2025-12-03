CC=g++
OPTS=-O3
PREFIX=~/bin
APREFIX=bench_overlap_

SOURCES=src/generator_base.cpp src/generator_switch.cpp src/history.cpp

all: switch hide expand birthdeath mergesplit

switch: ${SOURCES} src/*switch*.cpp
	$(CC) $(OPTS) -o $(APREFIX)switch src/bench_switch.cpp

hide: ${SOURCES} src/*hide*.cpp 
	$(CC) $(OPTS) -o $(APREFIX)hide src/bench_hide.cpp

expand: ${SOURCES} src/*expand*.cpp 
	$(CC) $(OPTS) -o $(APREFIX)expand src/bench_expand.cpp
		
birthdeath: ${SOURCES} src/*birth*.cpp
	$(CC) $(OPTS) -o $(APREFIX)birthdeath src/bench_birthdeath.cpp

mergesplit: ${SOURCES} src/*mergesplit*.cpp 
	$(CC) $(OPTS) -o $(APREFIX)mergesplit src/bench_mergesplit.cpp

install:
	cp $(APREFIX)* $(PREFIX)
	
clean:
	rm -f $(APREFIX)switch $(APREFIX)hide $(APREFIX)expand $(APREFIX)birthdeath $(APREFIX)mergesplit

cleandata:
	rm -f *.dat *.comm *.edges
