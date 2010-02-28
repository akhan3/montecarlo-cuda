NVCC 		:= 	nvcc
CXX         := 	g++-4.3
CC         	:=	gcc-4.3
LINK      	:= 	g++-4.3 -fPIC

CUDA_INSTALL_PATH 	:= /usr/local/cuda
CUDA_SDK_PATH 		:= $(HOME)/NVIDIA_GPU_Computing_SDK

INCLUDES   	:= -I$(CUDA_INSTALL_PATH)/include -I$(CUDA_SDK_PATH)/C/common/inc
LIBRARIES 	:= -L$(CUDA_INSTALL_PATH)/lib64 -lcuda -lcudart \
			   -L$(CUDA_SDK_PATH)/C/lib/ -lcutil

ifeq ($(dbg),0)
	COMMONFLAGS += -O3
	OBJDIR		:= obj/release
else
	COMMONFLAGS += -g
	OBJDIR		:= obj/debug
endif

INCLUDES   	+= -I. $(MATLAB_INCLUDES)
LIBRARIES 	+= $(MATLAB_LIBRARIES)

CXXFLAGS    := -Wall -W $(INCLUDES) $(COMMONFLAGS)
NVCCFLAGS   += --compiler-options -fno-strict-aliasing \
				--compiler-bindir=$(HOME)/usr_local/bin/gcc-4.3

OBJS        :=	sim_constants.o \
				save_matfile.o  \
				main.o \
				rk4solver_kernel.o
				#Vector3.o Matrix3.o
				
TARGET    	:= main.out

# ==================================================================
# Rules, target and dependencies
# ==================================================================

$(TARGET): 	$(OBJS)
	$(LINK) -o $@ $(OBJS) $(LIBRARIES)

sim_constants.o : sim_constants.cpp sim_constants.hpp my_macros.hpp 
save_matfile.o	: save_matfile.cpp my_macros.hpp
main.o			: main.cpp my_macros.hpp 
rk4solver_kernel.o : rk4solver_kernel.cu Vector3.cpp Vector3.hpp my_macros.hpp 
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $@ -c rk4solver_kernel.cu


compile:	$(OBJS)

clean:
	rm -vf $(OBJS)

tidy: clean
	rm -vf $(TARGET)

none:
# Vector3.o		: Vector3.cu Vector3.hpp my_macros.hpp
# Matrix3.o		: Matrix3.cpp Matrix3.hpp my_macros.hpp
