include Makefile.comp
include Makefile.dependheaders


SUBDIRS = ./include/Impl/GemmBatchedEx

OBJDPENDS = ./include/Impl/GemmBatchedEx/ssblasGemmBatchedEx.o
#libssblasGemmBatchedEx.so: ssblasGemmBatchedEx.cpp $(HPPSRCS) $(HSRCS)
all:
	@for dir in $(SUBDIRS); do \
		make -C $$dir; \
	done
	$(CXX) $(OBJDPENDS) -o libssblasGemmBatchedEx.so $(CXX_OPT) $(LIB_OPT) $(OMP)

#libssblasGemmBatchedEx.so: $(OBJDPENDS) $(HPPSRCS) $(HSRCS)
#	$(CXX) $(OBJDPENDS) -o libssblasGemmBatchedEx.so $(CXX_OPT) $(LIB_OPT) $(OMP)


#./include/Impl/GemmBatchedEx/ssblasGemmBatchedEx.o: 
#	@for dir in $(SUBDIRS); do \
#		make -C $$dir; \
#	done

clean:
	@for dir in $(SUBDIRS); do \
		make -C $$dir clean; \
	done
	rm libssblasGemmBatchedEx.so