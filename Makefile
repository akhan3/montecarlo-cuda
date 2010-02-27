CXX         := 	g++-4.3
CC         	:=	gcc-4.3
NVCC 		:= 	nvcc
LINK      	:= 	g++-4.3 -fPIC

CUDA_INSTALL_PATH 	:= /usr/local/cuda
CUDA_SDK_PATH 		:= $(HOME)/NVIDIA_GPU_Computing_SDK

INCLUDES   	:= -I$(CUDA_INSTALL_PATH)/include -I$(CUDA_SDK_PATH)/C/common/inc
LIBRARIES 	:= -L$(CUDA_INSTALL_PATH)/lib64 -lcuda -lcudart \
			   -L$(CUDA_SDK_PATH)/C/lib/ -lcutil

ifeq ($(dbg),1)
	COMMONFLAGS += -g
	OBJDIR		:= obj/debug
else
	COMMONFLAGS += -g
	OBJDIR		:= obj/release
endif

INCLUDES   	+= -I. $(MATLAB_INCLUDES)
LIBRARIES 	+= $(MATLAB_LIBRARIES)

CXXFLAGS    := -Wall -W $(INCLUDES) $(COMMONFLAGS)
NVCCFLAGS   += --compiler-options -fno-strict-aliasing \
				--compiler-bindir=$(HOME)/usr_local/bin/gcc-4.3

#OBJS        := main.o save_matfile.o sim_constants.o Matrix3.o Vector3.o
OBJS        := main.o save_matfile.o sim_constants.o Vector3.o \
				template_gold.o template.o \
				template_kernel.o

TARGET    	:= main.out

# ==================================================================
# Rules, target and dependencies
# ==================================================================
$(TARGET): $(OBJS)
	$(LINK) -o $@ $(OBJS) $(LIBRARIES)

template_kernel.o : template_kernel.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $@ -c template_kernel.cu
template.o				: template.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $@ -c template.cu
template_gold.o			: template_gold.cpp

main.o			: main.cpp my_macros.hpp
save_matfile.o 	: save_matfile.cpp my_macros.hpp
sim_constants.o : sim_constants.cpp sim_constants.hpp my_macros.hpp
Vector3.o		: Vector3.cpp Vector3.hpp my_macros.hpp
#Matrix3.o		: Matrix3.cpp Matrix3.hpp my_macros.hpp

clean:
	rm -vf $(OBJS) $(TARGET)
