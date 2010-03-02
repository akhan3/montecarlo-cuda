CXX         := 	g++-4.3
CC         	:=	gcc-4.3
LINKER     	:= 	g++-4.3 -fPIC

# CUDA compiler, includes and libraries
NVCC 		:= 	nvcc
CUDA_INSTALL_PATH 	:= 	/usr/local/cuda
CUDA_SDK_PATH 		:= 	$(HOME)/NVIDIA_GPU_Computing_SDK
CUDA_INCLUDES   	:= 	-I$(CUDA_INSTALL_PATH)/include -I$(CUDA_SDK_PATH)/C/common/inc
CUDA_LIBRARIES 		:= 	-L$(CUDA_INSTALL_PATH)/lib64 -lcuda -lcudart \
						-L$(CUDA_SDK_PATH)/C/lib/ -lcutil


# My includes and libraries
INCLUDES   	+= -I. $(MATLAB_INCLUDES)
LIBRARIES 	+= $(MATLAB_LIBRARIES)

ifeq ($(emu),1)
	CXXFLAGS 	+=	-g -D__DEBUG__ -D__DEVICE_EMULATION__
	NVCCFLAGS	+= 	-g -D__DEBUG__ -deviceemu -D__DEVICE_EMULATION__
	OBJDIR		+= 	obj/emu/
	BINDIR		+= 	bin/emu/
else
ifeq ($(dbg),1)
	CXXFLAGS 	+= 	-g -D__DEBUG__
	NVCCFLAGS	+= 	-g -D__DEBUG__
	OBJDIR		+= 	obj/debug/
	BINDIR		+= 	bin/debug/
else
	CXXFLAGS 	+= 	-O3
	#NVCCFLAGS	+= 	-O3
	OBJDIR		+= 	obj/release/
	BINDIR		+= 	bin/release/
endif
endif

ifeq ($(verbose),1)
	VERBOSE	:=
else
	VERBOSE	:=	@
endif

COMMONFLAGS		+=	-Wall -W $(INCLUDES)
CXXFLAGS    	+= 	$(COMMONFLAGS)
NVCCFLAGS   	+= 	--compiler-options "$(COMMONFLAGS) -fno-strict-aliasing" \
					--compiler-bindir=$(HOME)/usr_local/bin/gcc-4.3 \
					$(CUDA_INCLUDES)

OBJS	:=	$(OBJDIR)sim_constants.o \
			$(OBJDIR)save_matfile.o  \
			$(OBJDIR)main.o \
			$(OBJDIR)rk4solver_kernel.o
				
TARGET	:= 	$(BINDIR)nanomagnet_cuda

# ==================================================================
# Rules, target and dependencies
# ==================================================================

$(TARGET):	compile create_bin_dir
	$(VERBOSE)	$(LINKER) -o $@ $(OBJS) $(LIBRARIES) $(CUDA_LIBRARIES)

$(OBJDIR)sim_constants.o	: sim_constants.cpp sim_constants.hpp my_macros.hpp 
	$(VERBOSE)	$(CXX) $(CXXFLAGS) $(CUDA_INCLUDES) -o $@ -c sim_constants.cpp
$(OBJDIR)save_matfile.o		: save_matfile.cpp my_macros.hpp
	$(VERBOSE)	$(CXX) $(CXXFLAGS) $(CUDA_INCLUDES) -o $@ -c save_matfile.cpp
$(OBJDIR)main.o				: main.cpp my_macros.hpp
	$(VERBOSE)	$(CXX) $(CXXFLAGS) $(CUDA_INCLUDES) -o $@ -c main.cpp
$(OBJDIR)rk4solver_kernel.o : rk4solver_kernel.cu Vector3.cpp Vector3.hpp my_macros.hpp 
	$(VERBOSE)	$(NVCC) $(NVCCFLAGS) -o $@ -c rk4solver_kernel.cu


compile:	create_obj_dir $(OBJS)
create_obj_dir: 
	$(VERBOSE)	mkdir -p $(OBJDIR)
create_bin_dir:
	$(VERBOSE)	mkdir -p $(BINDIR)
	
clean:
	$(VERBOSE)	rm -vf $(OBJS)
tidy:	clean
	$(VERBOSE)	rm -vf $(TARGET)
clobber:	tidy
	$(VERBOSE)	rm -rvf $(OBJDIR)
	$(VERBOSE)	rm -rvf $(BINDIR)
