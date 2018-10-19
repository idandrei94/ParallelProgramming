# GNUmakefile to build some OpenCL programs.
#
# OpenCL programs can contain a "kernel" as a string or read it from
# a file. You can copy the kernel file to the current directory or
# you can use an environment variable pointing to the directory with
# kernel files, if you want to run the program from any directory
# without copying kernel files. OpenCL programs from this directory
# try to locate kernel files using the environment variable
# KERNEL_FILES. The pathname must start with a drive letter, if you
# use Cygwin or Windows and if the program and kernel files may be
# located on different drives, e.g.,
#   Cygwin:  "setenv KERNEL_FILES c:/cygwin64/${HOME}/kernel_files",
#   Windows: "set KERNEL_FILES=c:\temp\kernel_files".
#
# GNUmakefile includes GNUmakefile.env to determine the operating
# system, processor architecture, used compiler for "mpicc", and
# some flags. The following variables will be initialized accordingly
# to the platform and compiler (variables may be empty, if a compiler
# doesn't support a programming model).
#
# MYOS		    current operating system
#		      (at the moment Cygwin, Darwin, or Linux)
# MYARCH	    current processor architecture
#		      (at the moment always "x86_64")
# USE_CC	    selected / used C compiler
#		      (important for default compiler for "mpicc")
# USE_CC_3	    first 3 characters of USE_CC
#		      (important for "Darwin" where "gcc" is a link to
#		      "clang", while GNU C is "gcc-mp-7", "gcc-7" or
#		      something similar)
# USE_CFLAGS	    favoured / necessary common CFLAGS
# USE_LDFLAGS	    favoured / necessary common LDFLAGS
# AUTOPAR_CFLAGS    CFLAGS to enable automatic parallelization
# OPENACC_CFLAGS    CFLAGS to enable OpenACC
# OPENMP_CFLAGS	    CFLAGS to enable OpenMP
# OPENMP_ACC_CFLAGS CFLAGS to enable OpenMP for accelerators
# CUDA_CAPABLE	    yes / no
# GSL_CAPABLE	    yes / no
# OPENCL_CAPABLE    yes / no
# PTHREAD_CAPABLE   yes / no
# SUNPERF_CAPABLE   yes / no
# MPICC_CMD	    yes / no
#
# "*_CFLAGS" don't contain a flag for the kind of optimization (could
# result in a fatal error for "nvcc") and "USE_CFLAGS" doesn't contain
# a flag for the C standard for "clang" (could result in an error if
# you want to compile CUDA programs with "clang").
#
#
# You can use something like the following statement
#   $(warning CC = $(CC))
# if you want to print messages or values of (environment) variables
# on the screen to make debugging easier if something went wrong.
#
# You can also use the command line arguments "-d" (make -d ...) if
# you want to observe what "make" is doing.
#
# This file can only be used with "GNU make" (gmake), because it
# contains conditional statements and other "GNU make" specific
# stuff. If you don't have "GNU make" add a comment char in front
# of all unnecessary lines.
#
# "make" searches automatically for "Makefile" while "gmake" looks
# for "GNUmakefile" first and then for "Makefile", i.e., you should
# name your file "Makefile" if it contains just "normal" code which
# can be processed by both programs and you should name it
# "GNUmakefile" if it contains special code for "gmake", e.g.,
# conditional expressions. If you choose an arbitrary file name you
# must provide the file name via the command line argument "-f".
#
# "(g)make -f <arbitrary file name>"
#
# You can find more/all information about "Gnu make" using
# http://www.gnu.org/software/make/manual/make.html.
#
# Rules:
#   <target>:	<depends on file 1> <depends on file 2> ...
#      <action to create target>
#
# Some tasks are simpler and easier to adapt if you use variables.
#
# variable = <something>
#
#  "$(variable)" returns the value of the variable.
#
#
# usage:
#   (g)make		make all programs
#   (g)make all		same as "(g)make"
#
#   (g)make clean	remove object files
#   (g)make clean_all	remove object, binary, and library files
#
#   If you call "make DEBUG=1 ..." you get some debugging messages.
#   If you call "make CC=cc ..." you can overwrite the default
#   value for "CC" (using "cc" instead of "gcc")..
#
#
# File: GNUmakefile			Author: S. Gross
# Date: 10.09.2017
#


# common definitions
#
# **********  choose compiler  **********
#
# It's not allowed to add a comment after the next statement or
# similar statements! Otherwise everything between the value and
# the comment character is part of the variable so that comparisons
# with the variable will normally fail in that case!
#
# OpenCL programs can be compiled with any C compiler if OpenCL
# header files and libraries are available.
# Possible values:
#   Linux:   cc, clang, gcc, icc, mpicc, nvcc, pgcc
#   Windows: cl, nvcc
#   Cygwin:  nvcc
#
CC = nvcc


# necessary files
#
# program names: use the source code file name without file name
#                extension ".c"
# header files:  file name including file name extension
# kernel files:  file name including file name extension
#
# A kernel file must be present in the same directory as the program
# itself. The program will not find the file if it resides in a
# different directory.
#
FILE1	 = platformInfo
FILE2	 = saxpy_OpenCL_1
FILE3	 = saxpy_OpenCL_2
FILE3_CL = saxpyKernel.cl
FILE4	 = daxpy_OpenCL_1
FILE5	 = daxpy_OpenCL_2
FILE5_CL = daxpyKernel.cl
FILE6	 = index
FILE6_CL = indexKernel.cl
FILE7	 = hello_OpenCL
FILE7_CL = helloKernel.cl

FILE90	= errorCodes


# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!               Don't change the next statement                   !!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#
# Determine operating system, processor architecture, used compiler for
# "mpicc", and some flags.
#
include GNUmakefile.env
ifeq ($(MYOS), Cygwin)
  ifneq ($(CC), nvcc)
    $(warning *** It is not allowed to change "CC" to "$(CC)".)
    $(error On Cygwin only "CC=nvcc" can build OpenCL programs)
  endif
endif


# **********  choose additional FLAGS and libraries  **********
#
PTHREAD_LIBS = -lpthread
GSL_LIBS     = -lgsl -lgslcblas

ifeq ($(USE_CC), clang)
  AUTOPAR_LIBS	   = -lomp
else
  AUTOPAR_LIBS	   = 
endif

ifeq ($(MYOS), Darwin)
  OPENCL_LIBS = -framework OpenCL
else
  OPENCL_LIBS = -lOpenCL
endif

ifeq ($(MYOS), Linux)
  LIBS += -lm
endif


ifeq ($(USE_CC), cc)
  ifeq ($(MYOS), Linux)
    # Oracle Developer Studio C Compiler. CUDA, OpenACC, and OpenMP
    # for accelerators are not supported.
    USE_CFLAGS	      += -fast
    #AUTOPAR_CFLAGS    += -xloopinfo -xvpara
    #OPENACC_CFLAGS    += 
    #OPENMP_CFLAGS     += -xloopinfo -xvpara
    #OPENMP_ACC_CFLAGS += 
    USE_LDFLAGS	      += 
    LIBS	      += 
  else
    ifeq ($(MYOS), Cygwin)
      # Link to "gcc". CUDA, OpenACC, OpenCL, and OpenMP for
      # accelerators are not supported.
      USE_CFLAGS	+= -O3
      #AUTOPAR_CFLAGS	+= -fopt-info-optimized-missed-optall
      #OPENACC_CFLAGS	+= 
      #OPENMP_CFLAGS	+= -fopt-info-optimized-missed-optall
      #OPENMP_ACC_CFLAGS	+= 
      USE_LDFLAGS	+= 
      LIBS		+= 
    else
      ifeq ($(MYOS), Darwin)
	# Link to "clang". Automatic parallelization, CUDA, OpenACC,
	# and OpenMP for accelerators are not supported.
	USE_CFLAGS	  += -std=c11 -O2
	#AUTOPAR_CFLAGS	  += 
	#OPENACC_CFLAGS	  += 
	#OPENMP_CFLAGS	  += -Rpass=".*" -Rpass-missed=".*"
	#OPENMP_ACC_CFLAGS += 
	USE_LDFLAGS	  += 
	LIBS		  += 
      endif
    endif
  endif
endif

ifeq ($(USE_CC), clang)
  ifeq ($(MYOS), Linux)
    # OpenACC not supported.
    # "-std=c11" is not allowed if you want to compile CUDA programs
    # (would result in an error message "invalid argument '-std=c11'
    # not allowed with 'CUDA'").
    USE_CFLAGS	       += -std=c11 -O2
    #AUTOPAR_CFLAGS     += -Rpass=".*" -Rpass-missed=".*"
    #OPENACC_CFLAGS     += 
    #OPENMP_CFLAGS      += -Rpass=".*" -Rpass-missed=".*"
    #OPENMP_ACC_CFLAGS  += -Rpass=".*" -Rpass-missed=".*"
    USE_LDFLAGS	       += 
    LIBS	       += 
  else
    ifeq ($(MYOS), Cygwin)
      # Automatic parallelization, CUDA, OpenACC, OpenCL, OpenMP,
      # and OpenMP for accelerators are not supported
      USE_CFLAGS	 += -std=c11 -O2
      #AUTOPAR_CFLAGS     += 
      #OPENACC_CFLAGS     += 
      #OPENMP_CFLAGS      += 
      #OPENMP_ACC_CFLAGS  += 
      USE_LDFLAGS	 += 
      LIBS		 += 
    else
      ifeq ($(MYOS), Darwin)
	# Automatic parallelization, CUDA, OpenACC, and OpenMP
	# for accelerators are not supported
	USE_CFLAGS	   += -std=c11 -O2
	#AUTOPAR_CFLAGS     += 
	#OPENACC_CFLAGS     += 
	#OPENMP_CFLAGS      += -Rpass=".*" -Rpass-missed=".*"
	#OPENMP_ACC_CFLAGS  += 
	USE_LDFLAGS	   += 
	LIBS		   += 
      endif
    endif
  endif
endif

ifeq ($(USE_CC_3), gcc)
  ifeq ($(MYOS), Linux)
    # CUDA not supported.
    USE_CFLAGS	       += -O3
    #AUTOPAR_CFLAGS     += -fopt-info-optimized-missed-optall
    #OPENACC_CFLAGS     += -fopt-info-optimized-missed-optall
    #OPENMP_CFLAGS      += -fopt-info-optimized-missed-optall
    #OPENMP_ACC_CFLAGS  += -fopt-info-optimized-missed-optall
    USE_LDFLAGS	       += 
    LIBS	       += 
  else
    ifeq ($(MYOS), Cygwin)
      # CUDA, OpenACC, OpenCL, and OpenMP for accelerators are
      # not supported.
      USE_CFLAGS	 += -O3
      #AUTOPAR_CFLAGS     += -fopt-info-optimized-missed-optall
      #OPENACC_CFLAGS     += 
      #OPENMP_CFLAGS      += -fopt-info-optimized-missed-optall
      #OPENMP_ACC_CFLAGS  += 
      USE_LDFLAGS	 += 
      LIBS		 += 
    else
      ifeq ($(MYOS), Darwin)
	ifeq ($(CC), gcc)
	  # Link to "clang". Automatic parallelization, CUDA, OpenACC
	  # and OpenMP for accelerators are not supported.
	  USE_CFLAGS	    += -std=c11 -O2
	  #AUTOPAR_CFLAGS    += 
	  #OPENACC_CFLAGS    += 
	  #OPENMP_CFLAGS	    += -Rpass=".*" -Rpass-missed=".*"
	  #OPENMP_ACC_CFLAGS += 
	  USE_LDFLAGS	    += 
	  LIBS		    += 
	else
	  # GNU C: gcc-mp-7, gcc-7, etc. CUDA and OpenMP for accelerators
	  # are not supported.
	  USE_CFLAGS	    += -O3
	  #AUTOPAR_CFLAGS    += -fopt-info-optimized-missed-optall
	  #OPENACC_CFLAGS    += -fopt-info-optimized-missed-optall
	  #OPENMP_CFLAGS	    += -fopt-info-optimized-missed-optall
	  #OPENMP_ACC_CFLAGS += 
	  USE_LDFLAGS	    += 
	  LIBS		    += 
	endif
      endif
    endif
  endif
endif

ifeq ($(USE_CC), icc)
  # CUDA, OpenACC and OpenMP for accelerators are not supported.
  USE_CFLAGS	    += -O3
  #AUTOPAR_CFLAGS    += -qopt-report=5 -qopt-report-phase=all
  #AUTOPAR_CFLAGS    += -qopt-report-file=stderr
  #OPENACC_CFLAGS    += 
  #OPENMP_CFLAGS	    += -qopt-report=5 -qopt-report-phase=all
  #OPENMP_CFLAGS	    += -qopt-report-file=stderr
  #OPENMP_ACC_CFLAGS += 
  USE_LDFLAGS	    += 
  LIBS		    += 
endif

ifeq ($(USE_CC), nvcc)
  ifeq ($(MYOS), Linux)
    # Automatic parallelization, OpenACC, and OpenMP for accelerators
    # are not supported.
    # Remember that "nvcc" complains with a fatal error if you duplicate
    # a command line argument.
    #
    USE_CFLAGS	      += -arch=sm_50 -O3
    #AUTOPAR_CFLAGS    += 
    #OPENACC_CFLAGS    += 
    #OPENMP_CFLAGS     += 
    #OPENMP_ACC_CFLAGS += 
    USE_LDFLAGS	      += 
    LIBS	      += 
  else
    ifeq ($(MYOS), Cygwin)
      USE_CFLAGS	+= -arch=sm_30 -O3
      #AUTOPAR_CFLAGS	+= 
      #OPENACC_CFLAGS	+= 
      #OPENMP_CFLAGS	+= 
      #OPENMP_ACC_CFLAGS	+= 
      USE_LDFLAGS	+= 
      LIBS		+= 
    endif
  endif
endif

ifeq ($(USE_CC), pgcc)
  # CUDA and OpenMP for accelerators are not supported.
  USE_CFLAGS	     += -O3
  #AUTOPAR_CFLAGS     += -Minfo=all
  #OPENACC_CFLAGS     += -Minfo=all
  #OPENMP_CFLAGS	     += -Minfo=all
  #OPENMP_ACC_CFLAGS  += 
  USE_LDFLAGS	     += 
  LIBS		     += 
endif


# **********  print some values  **********
ifeq ($(DEBUG), 1)
  $(warning )
  $(warning )
  $(warning MYOS = $(MYOS))
  $(warning )
  $(warning MYARCH = $(MYARCH))
  $(warning )
  $(warning USE_CC = $(USE_CC))
  $(warning )
#  $(warning USE_CC_3 = $(USE_CC_3))
#  $(warning )
  $(warning USE_CFLAGS = $(USE_CFLAGS))
  $(warning )
  $(warning USE_LDFLAGS = $(USE_LDFLAGS))
  $(warning )
  $(warning AUTOPAR_CFLAGS = $(AUTOPAR_CFLAGS))
  $(warning )
  $(warning OPENACC_CFLAGS = $(OPENACC_CFLAGS))
  $(warning )
  $(warning OPENMP_CFLAGS = $(OPENMP_CFLAGS))
  $(warning )
  $(warning OPENMP_ACC_CFLAGS = $(OPENMP_ACC_CFLAGS))
  $(warning )
  $(warning CUDA_CAPABLE = $(CUDA_CAPABLE))
  $(warning )
  $(warning GSL_CAPABLE = $(GSL_CAPABLE))
  $(warning )
  $(warning OPENCL_CAPABLE = $(OPENCL_CAPABLE))
  $(warning )
  $(warning PTHREAD_CAPABLE = $(PTHREAD_CAPABLE))
  $(warning )
  $(warning SUNPERF_CAPABLE = $(SUNPERF_CAPABLE))
  $(warning )
  $(warning AUTOPAR_LIBS = $(AUTOPAR_LIBS))
  $(warning )
  $(warning OPENCL_LIBS = $(OPENCL_LIBS))
  $(warning )
  $(warning OPENMP_ACC_LIBS = $(OPENMP_ACC_LIBS))
  $(warning )
  $(warning PTHREAD_LIBS = $(PTHREAD_LIBS))
  $(warning )
  $(warning LIBS = $(LIBS))
  $(warning )
#  $(warning LD_LIBRARY_PATH = $(LD_LIBRARY_PATH))
#  $(warning )
#  $(warning PATH = $(PATH))
#  $(warning )
  $(warning )
endif


# **********  choose directory for binaries  **********
#
# You can choose the local directory if you have only homogeneous
# systems. You need a different directory for executables for each
# operating system and architecture if you use NFS (network file
# system). Sometimes it is also a good idea to have one directory for
# binaries which you can add to the environment variable PATH so that
# you can execute your binaries from all directories.
#
# "nvcc" adds "the current drive letter" in front of "${HOME}" on
# Windows/Cygwin so that you get a wrong pathname, if your source
# code is located on a different drive than your binary files.
# Therefore, it's necessary to use the full Windows pathname to
# the target directory.
#
#EXE_DIR = .
ifeq ($(MYOS), Cygwin)
  EXE_DIR = c:/cygwin64/$(HOME)/$(MYOS)/$(MYARCH)/bin
else
  EXE_DIR = $(HOME)/$(MYOS)/$(MYARCH)/bin
endif


# **********  choose program name and location  **********
#
ifeq ($(MYOS), Cygwin)
  TARGET1   = $(EXE_DIR)/$(FILE1).exe
  TARGET2   = $(EXE_DIR)/$(FILE2).exe
  TARGET3   = $(EXE_DIR)/$(FILE3).exe
  TARGET4   = $(EXE_DIR)/$(FILE4).exe
  TARGET5   = $(EXE_DIR)/$(FILE5).exe
  TARGET6   = $(EXE_DIR)/$(FILE6).exe
  TARGET7   = $(EXE_DIR)/$(FILE7).exe
  TARGET90  = $(FILE90).obj
else
  TARGET1   = $(EXE_DIR)/$(FILE1)
  TARGET2   = $(EXE_DIR)/$(FILE2)
  TARGET3   = $(EXE_DIR)/$(FILE3)
  TARGET4   = $(EXE_DIR)/$(FILE4)
  TARGET5   = $(EXE_DIR)/$(FILE5)
  TARGET6   = $(EXE_DIR)/$(FILE6)
  TARGET7   = $(EXE_DIR)/$(FILE7)
  TARGET90  = $(FILE90).o
endif


# **********  kernel files must possibly be copied  **********
#
# The environment variable KERNEL_FILES specifies a location
# for kernel files. A program can be executed from any directory
# if it evaluates the environment variable to load a kernel file.
# Otherwise it will not work, if the kernel file isn't available
# in the current directory.
#
KERNEL_FILE_DIR = $(shell echo $(KERNEL_FILES))

ifneq ($(KERNEL_FILE_DIR),)
  TARGET_FILE3_CL = $(KERNEL_FILE_DIR)/$(FILE3_CL)
  TARGET_FILE5_CL = $(KERNEL_FILE_DIR)/$(FILE5_CL)
  TARGET_FILE6_CL = $(KERNEL_FILE_DIR)/$(FILE6_CL)
  TARGET_FILE7_CL = $(KERNEL_FILE_DIR)/$(FILE7_CL)
endif



# **********  rules to make targets  **********
#
ifneq ($(KERNEL_FILE_DIR),)
ALL_TARGETS =	$(TARGET1) $(TARGET2) $(TARGET3) $(TARGET4) $(TARGET5) \
		$(TARGET6) $(TARGET7) \
		$(TARGET_FILE3_CL) $(TARGET_FILE5_CL) $(TARGET_FILE6_CL) \
		$(TARGET_FILE7_CL)
else
ALL_TARGETS =	$(TARGET1) $(TARGET2) $(TARGET3) $(TARGET4) $(TARGET5) \
		$(TARGET6) $(TARGET7)
endif


# optimization report files from Intel compiler if the report is not
# redirected to stderr with "-qopt-report-file=stderr"
#
ALL_OPTRPT =	$(TARGET1).optrpt $(TARGET2).optrpt \
		$(TARGET3).optrpt $(TARGET4).optrpt \
		$(TARGET5).optrpt $(TARGET6).optrpt \
		$(TARGET7).optrpt \
		$(TARGET90).optrpt



all:	$(ALL_TARGETS)


$(TARGET1): $(FILE1).c $(TARGET90)
	$(CC) $(USE_CFLAGS) $(USE_LDFLAGS) -o $(TARGET1) \
	$(FILE1).c $(TARGET90) $(OPENCL_LIBS) $(LIBS)

$(TARGET2): $(FILE2).c $(TARGET90)
	$(CC) $(USE_CFLAGS) $(USE_LDFLAGS) -o $(TARGET2) \
	$(FILE2).c $(TARGET90) $(OPENCL_LIBS) $(LIBS)

$(TARGET3): $(FILE3).c $(TARGET_FILE3_CL) $(TARGET90)
	$(CC) $(USE_CFLAGS) $(USE_LDFLAGS) -o $(TARGET3) \
	$(FILE3).c $(TARGET90) $(OPENCL_LIBS) $(LIBS)

$(TARGET4): $(FILE4).c $(TARGET90)
	$(CC) $(USE_CFLAGS) $(USE_LDFLAGS) -o $(TARGET4) \
	$(FILE4).c $(TARGET90) $(OPENCL_LIBS) $(LIBS)

$(TARGET5): $(FILE5).c $(TARGET_FILE5_CL) $(TARGET90)
	$(CC) $(USE_CFLAGS) $(USE_LDFLAGS) -o $(TARGET5) \
	$(FILE5).c $(TARGET90) $(OPENCL_LIBS) $(LIBS)

$(TARGET6): $(FILE6).c $(TARGET_FILE6_CL) $(TARGET90)
	$(CC) $(USE_CFLAGS) $(USE_LDFLAGS) -o $(TARGET6) \
	$(FILE6).c $(TARGET90) $(OPENCL_LIBS) $(LIBS)

$(TARGET7): $(FILE7).c $(TARGET_FILE7_CL) $(TARGET90)
	$(CC) $(USE_CFLAGS) $(USE_LDFLAGS) -o $(TARGET7) \
	$(FILE7).c $(TARGET90) $(OPENCL_LIBS) $(LIBS)

$(TARGET90): $(FILE90).c
	$(CC) $(USE_CFLAGS) -c -o $(TARGET90) $(FILE90).c

ifneq ($(KERNEL_FILE_DIR),)
  $(TARGET_FILE3_CL): $(FILE3_CL)
	cp -i $(FILE3_CL) $(TARGET_FILE3_CL)

  $(TARGET_FILE5_CL): $(FILE5_CL)
	cp -i $(FILE5_CL) $(TARGET_FILE5_CL)

  $(TARGET_FILE6_CL): $(FILE6_CL)
	cp -i $(FILE6_CL) $(TARGET_FILE6_CL)

  $(TARGET_FILE7_CL): $(FILE7_CL)
	cp -i $(FILE7_CL) $(TARGET_FILE7_CL)
endif


# **********  pseudo rules to clean up  **********
#
clean:
	 rm -f *.o *.obj

clean_all:
  ifneq ($(KERNEL_FILE_DIR),)
	 rm -f	$(ALL_TARGETS) $(ALL_OPTRPT) \
		$(TARGET_FILE3_CL) $(TARGET_FILE5_CL) \
		$(TARGET_FILE6_CL) $(TARGET_FILE7_CL)
	 rm -f	*.o *.obj
  else
	 rm -f	$(ALL_TARGETS) $(ALL_OPTRPT) \
	 rm -f	*.o *.obj
  endif
