include ../../../Makefile.comp
include ../../../Makefile.dependheaders


ssblasGemmBatchedEx.o: ssblasGemmBatchedEx.cpp $(HPPSRCS) $(HSRCS)
	$(CXX) -c ssblasGemmBatchedEx.cpp  $(CXX_OPT) $(LIB_OPT) $(OMP)

clean:
	rm *.o

	