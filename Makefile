CXX         := 	g++
CC         	:=	gcc
LINKER     	:= 	g++ -fPIC

# CUDA compiler, includes and libraries
NVCC 		:= 	nvcc
CUDA_INSTALL_PATH 	:= 	/usr/local/cuda
CUDA_SDK_PATH 		:= 	$(HOME)/NVIDIA_GPU_Computing_SDK
CUDA_INCLUDES   	:= 	-I$(CUDA_INSTALL_PATH)/include -I$(CUDA_SDK_PATH)/C/common/inc
CUDA_LIBRARIES 		:= 	-L$(CUDA_INSTALL_PATH)/lib64 -lcuda -lcudart \
						-L$(CUDA_SDK_PATH)/C/lib/ -lcutil_x86_64


# My includes and libraries
INCLUDES   	+= -I. $(MATLAB_INCLUDES)
LIBRARIES 	+= $(MATLAB_LIBRARIES)

ifeq ($(emu),1)
	CXXFLAGS 	+=	-g -D__DEBUG__ -D__DEVICE_EMULATION__
	NVCCFLAGS	+= 	-g -D__DEBUG__ -deviceemu -D__DEVICE_EMULATION__
else
ifeq ($(dbg),1)
	CXXFLAGS 	+= 	-g -D__DEBUG__
	NVCCFLAGS	+= 	-g -D__DEBUG__
else
	CXXFLAGS 	+= 	-O3
	#NVCCFLAGS	+= 	-O3
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
					$(CUDA_INCLUDES)

OBJS	:=	$(OBJDIR)main.o \
			$(OBJDIR)calc_pi_kernel.o
				
TARGET	:= 	$(BINDIR)mc_cuda

# ==================================================================
# Rules, target and dependencies
# ==================================================================

$(TARGET):	$(OBJS)
	$(VERBOSE)	$(LINKER) -o $@ $(OBJS) $(LIBRARIES) $(CUDA_LIBRARIES)

$(OBJDIR)main.o				: main.cpp my_macros.hpp
	$(VERBOSE)	$(CXX) $(CXXFLAGS) $(CUDA_INCLUDES) -o $@ -c main.cpp
$(OBJDIR)calc_pi_kernel.o : calc_pi_kernel.cu my_macros.hpp 
	$(VERBOSE)	$(NVCC) $(NVCCFLAGS) -o $@ -c calc_pi_kernel.cu


	
clean:
	$(VERBOSE)	rm -vf $(OBJS)
tidy:	clean
	$(VERBOSE)	rm -vf $(TARGET)
