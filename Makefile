# You may need to edit this file to reflect the type and capabilities of your system.
# The defaults are for a Linux system and may need to be changed for other systems (eg. Mac OS X).


CXX=g++

INPUT = main_nodebug.cpp


INCLUDE_FLAG = -I "C:\Users\Titus\Desktop\Final Year Project\Codes\fast-ica-for-bci\ica_cpp"



CXXFLAGS = $(INCLUDE_FLAG)

test: main_nodebug.cpp
	$(CXX) $(CXXFLAGS) $(INPUT)  -o $@


.PHONY: clean

clean:
	rm -f test

