# GNUmakefile to build some CUDA programs.
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
# Date: 14.02.2018
#

# common definitions
#
# **********  choose compiler  **********
#
# It's not allowed to add a comment after the next statement or
# similar statements! Otherwise everything between the value and
# the comment character is part of the variable so that comparisons
# with the variable will normally fail in that case!
# Possible values: nvcc (on Cygwin and Linux), clang (only on Linux)
#
CC = nvcc


# necessary files
#
# program names: use the source code file name without file name
#                extension ".c"
# header files:  file name including file name extension
#
FILE1 = hello
FILE2 = add_sequential

# C files using OpenMP pragmas
OPENMP_FILE1 = add_OpenMP_1
OPENMP_FILE2 = add_OpenMP_2

# program names: use the source code file name without file name
#                extension ".cu"
CUDA_FILE1 = hello_kernel_1
CUDA_FILE2 = hello_kernel_2
CUDA_FILE3 = hello_kernel_3
CUDA_FILE4 = deviceProp
CUDA_FILE5 = index_1
CUDA_FILE6 = index_2
CUDA_FILE7 = add_CUDA_1
CUDA_FILE8 = add_CUDA_2
CUDA_FILE9 = divergence_1
CUDA_FILE10 = divergence_2
CUDA_FILE11 = daxpy_1
CUDA_FILE12 = daxpy_2


# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!               Don't change the next statement                   !!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#
# Determine operating system, processor architecture, used compiler for
# "mpicc", and some flags.
#
include GNUmakefile.env
ifneq ($(CUDA_CAPABLE), yes)
  $(warning *** Wrong value for "CC" or wrong operating system.)
  ifeq ($(MYOS), Linux)
    $(error Only "CC=nvcc" and "CC=clang" are allowed)
  else
    ifeq ($(MYOS), Cygwin)
      $(error Only "CC=nvcc" allowed on "Cygwin")
    else
      $(error CUDA isn't available on "$(MYOS)")
    endif
  endif
endif


# **********  choose additional FLAGS and libraries  **********
#
PTHREAD_LIBS = -lpthread
GSL_LIBS     = -lgsl -lgslcblas

ifeq ($(USE_CC), clang)
  AUTOPAR_LIBS = -lomp
else
  AUTOPAR_LIBS = 
endif


ifeq ($(USE_CC), clang)
  ifeq ($(MYOS), Linux)
    # OpenACC not supported.
    # "-std=c11" is not allowed if you want to compile CUDA programs
    # (would result in an error message "invalid argument '-std=c11'
    # not allowed with 'CUDA'").
    #USE_CFLAGS	       += -std=c11 -O2
    USE_CFLAGS	       += -O2 --cuda-gpu-arch=sm_50 -Wno-old-style-cast
    #AUTOPAR_CFLAGS     += -Rpass=".*" -Rpass-missed=".*"
    #OPENACC_CFLAGS     += 
    #OPENMP_CFLAGS      += -Rpass=".*" -Rpass-missed=".*"
    #OPENMP_ACC_CFLAGS  += -Rpass=".*" -Rpass-missed=".*"
    USE_LDFLAGS	       += 
    LIBS	       += -lcudart -lm
  endif
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
  TARGET1 = $(EXE_DIR)/$(FILE1).exe
  TARGET2 = $(EXE_DIR)/$(FILE2).exe

  LIB_TARGET1 = $(EXE_DIR)/$(FILE1).lib
  LIB_TARGET2 = $(EXE_DIR)/$(FILE2).lib

  EXP_TARGET1 = $(EXE_DIR)/$(FILE1).exp
  EXP_TARGET2 = $(EXE_DIR)/$(FILE2).exp

  # Compiler doesn't support OpenMP if OPENMP_CFLAGS is empty
  ifneq ($(OPENMP_CFLAGS),)
    OPENMP_TARGET1 = $(EXE_DIR)/$(OPENMP_FILE1).exe
    OPENMP_TARGET2 = $(EXE_DIR)/$(OPENMP_FILE2).exe

    LIB_OPENMP_TARGET1 = $(EXE_DIR)/$(OPENMP_FILE1).lib
    LIB_OPENMP_TARGET2 = $(EXE_DIR)/$(OPENMP_FILE2).lib

    EXP_OPENMP_TARGET1 = $(EXE_DIR)/$(OPENMP_FILE1).exp
    EXP_OPENMP_TARGET2 = $(EXE_DIR)/$(OPENMP_FILE2).exp
  endif

  # Compiler supports CUDA if CUDA_CAPABLE is "yes"
  ifeq ($(CUDA_CAPABLE), yes)
    CUDA_TARGET1 = $(EXE_DIR)/$(CUDA_FILE1).exe
    CUDA_TARGET2 = $(EXE_DIR)/$(CUDA_FILE2).exe
    CUDA_TARGET3 = $(EXE_DIR)/$(CUDA_FILE3).exe
    CUDA_TARGET4 = $(EXE_DIR)/$(CUDA_FILE4).exe
    CUDA_TARGET5 = $(EXE_DIR)/$(CUDA_FILE5).exe
    CUDA_TARGET6 = $(EXE_DIR)/$(CUDA_FILE6).exe
    CUDA_TARGET7 = $(EXE_DIR)/$(CUDA_FILE7).exe
    CUDA_TARGET8 = $(EXE_DIR)/$(CUDA_FILE8).exe
    CUDA_TARGET9 = $(EXE_DIR)/$(CUDA_FILE9).exe
    CUDA_TARGET10 = $(EXE_DIR)/$(CUDA_FILE10).exe
    CUDA_TARGET11 = $(EXE_DIR)/$(CUDA_FILE11).exe
    CUDA_TARGET12 = $(EXE_DIR)/$(CUDA_FILE12).exe

    LIB_CUDA_TARGET1 = $(EXE_DIR)/$(CUDA_FILE1).lib
    LIB_CUDA_TARGET2 = $(EXE_DIR)/$(CUDA_FILE2).lib
    LIB_CUDA_TARGET3 = $(EXE_DIR)/$(CUDA_FILE3).lib
    LIB_CUDA_TARGET4 = $(EXE_DIR)/$(CUDA_FILE4).lib
    LIB_CUDA_TARGET5 = $(EXE_DIR)/$(CUDA_FILE5).lib
    LIB_CUDA_TARGET6 = $(EXE_DIR)/$(CUDA_FILE6).lib
    LIB_CUDA_TARGET7 = $(EXE_DIR)/$(CUDA_FILE7).lib
    LIB_CUDA_TARGET8 = $(EXE_DIR)/$(CUDA_FILE8).lib
    LIB_CUDA_TARGET9 = $(EXE_DIR)/$(CUDA_FILE9).lib
    LIB_CUDA_TARGET10 = $(EXE_DIR)/$(CUDA_FILE10).lib
    LIB_CUDA_TARGET11 = $(EXE_DIR)/$(CUDA_FILE11).lib
    LIB_CUDA_TARGET12 = $(EXE_DIR)/$(CUDA_FILE12).lib

    EXP_CUDA_TARGET1 = $(EXE_DIR)/$(CUDA_FILE1).exp
    EXP_CUDA_TARGET2 = $(EXE_DIR)/$(CUDA_FILE2).exp
    EXP_CUDA_TARGET3 = $(EXE_DIR)/$(CUDA_FILE3).exp
    EXP_CUDA_TARGET4 = $(EXE_DIR)/$(CUDA_FILE4).exp
    EXP_CUDA_TARGET5 = $(EXE_DIR)/$(CUDA_FILE5).exp
    EXP_CUDA_TARGET6 = $(EXE_DIR)/$(CUDA_FILE6).exp
    EXP_CUDA_TARGET7 = $(EXE_DIR)/$(CUDA_FILE7).exp
    EXP_CUDA_TARGET8 = $(EXE_DIR)/$(CUDA_FILE8).exp
    EXP_CUDA_TARGET9 = $(EXE_DIR)/$(CUDA_FILE9).exp
    EXP_CUDA_TARGET10 = $(EXE_DIR)/$(CUDA_FILE10).exp
    EXP_CUDA_TARGET11 = $(EXE_DIR)/$(CUDA_FILE11).exp
    EXP_CUDA_TARGET12 = $(EXE_DIR)/$(CUDA_FILE12).exp
  endif
else
  TARGET1 = $(EXE_DIR)/$(FILE1)
  TARGET2 = $(EXE_DIR)/$(FILE2)

  # Compiler doesn't support OpenMP if OPENMP_CFLAGS is empty
  ifneq ($(OPENMP_CFLAGS),)
    OPENMP_TARGET1 = $(EXE_DIR)/$(OPENMP_FILE1)
    OPENMP_TARGET2 = $(EXE_DIR)/$(OPENMP_FILE2)
  endif

  # Compiler supports CUDA if CUDA_CAPABLE is "yes"
  ifeq ($(CUDA_CAPABLE), yes)
    CUDA_TARGET1 = $(EXE_DIR)/$(CUDA_FILE1)
    CUDA_TARGET2 = $(EXE_DIR)/$(CUDA_FILE2)
    CUDA_TARGET3 = $(EXE_DIR)/$(CUDA_FILE3)
    CUDA_TARGET4 = $(EXE_DIR)/$(CUDA_FILE4)
    CUDA_TARGET5 = $(EXE_DIR)/$(CUDA_FILE5)
    CUDA_TARGET6 = $(EXE_DIR)/$(CUDA_FILE6)
    CUDA_TARGET7 = $(EXE_DIR)/$(CUDA_FILE7)
    CUDA_TARGET8 = $(EXE_DIR)/$(CUDA_FILE8)
    CUDA_TARGET9 = $(EXE_DIR)/$(CUDA_FILE9)
    CUDA_TARGET10 = $(EXE_DIR)/$(CUDA_FILE10)
    CUDA_TARGET11 = $(EXE_DIR)/$(CUDA_FILE11)
    CUDA_TARGET12 = $(EXE_DIR)/$(CUDA_FILE12)
  endif
endif


# **********  rules to make targets  **********
#
ALL_TARGETS =	$(TARGET1) $(TARGET2)

# Compiler doesn't support OpenMP if OPENMP_CFLAGS is empty
ifneq ($(OPENMP_CFLAGS),)
  ALL_OPENMP_TARGETS = $(OPENMP_TARGET1) $(OPENMP_TARGET2)
else
  ALL_OPENMP_TARGETS = 
endif

# Compiler supports CUDA if CUDA_CAPABLE is "yes"
ifeq ($(CUDA_CAPABLE), yes)
  ALL_CUDA_TARGETS = $(CUDA_TARGET1) $(CUDA_TARGET2) $(CUDA_TARGET3) \
		     $(CUDA_TARGET4) $(CUDA_TARGET5) $(CUDA_TARGET6) \
		     $(CUDA_TARGET7) $(CUDA_TARGET8) $(CUDA_TARGET9) \
		     $(CUDA_TARGET10) $(CUDA_TARGET11) $(CUDA_TARGET12)
else
  ALL_CUDA_TARGETS = 
endif

ifeq ($(MYOS), Cygwin)
  ALL_LIB_TARGETS = \
	$(LIB_TARGET1) $(LIB_TARGET2)

  ALL_EXP_TARGETS = \
	$(EXP_TARGET1) $(EXP_TARGET2)

  # Compiler doesn't support OpenMP if OPENMP_CFLAGS is empty
  ifneq ($(OPENMP_CFLAGS),)
    ALL_LIB_TARGETS += \
	$(LIB_OPENMP_TARGET1) $(LIB_OPENMP_TARGET2)

    ALL_EXP_TARGETS += \
	$(EXP_OPENMP_TARGET1) $(EXP_OPENMP_TARGET2)
  endif

  # Compiler supports CUDA if CUDA_CAPABLE is "yes"
  ifeq ($(CUDA_CAPABLE), yes)
    ALL_LIB_TARGETS += \
	$(LIB_CUDA_TARGET1) $(LIB_CUDA_TARGET2) $(LIB_CUDA_TARGET3) \
	$(LIB_CUDA_TARGET4) $(LIB_CUDA_TARGET5) $(LIB_CUDA_TARGET6) \
	$(LIB_CUDA_TARGET7) $(LIB_CUDA_TARGET8) $(LIB_CUDA_TARGET9) \
	$(LIB_CUDA_TARGET10) $(LIB_CUDA_TARGET11) $(LIB_CUDA_TARGET12)

    ALL_EXP_TARGETS += \
	$(EXP_CUDA_TARGET1) $(EXP_CUDA_TARGET2) $(EXP_CUDA_TARGET3) \
	$(EXP_CUDA_TARGET4) $(EXP_CUDA_TARGET5) $(EXP_CUDA_TARGET6) \
	$(EXP_CUDA_TARGET7) $(EXP_CUDA_TARGET8) $(EXP_CUDA_TARGET9) \
	$(EXP_CUDA_TARGET10) $(EXP_CUDA_TARGET11) $(EXP_CUDA_TARGET12)
  endif
else
  ALL_LIB_TARGETS =	
  ALL_EXP_TARGETS =	
endif


all:	$(ALL_TARGETS) $(ALL_OPENMP_TARGETS) $(ALL_CUDA_TARGETS)

$(TARGET1): $(FILE1).c
	$(CC) $(USE_CFLAGS) $(USE_LDFLAGS) -o $(TARGET1) \
	$(FILE1).c $(LIBS)

$(TARGET2): $(FILE2).c
	$(CC) $(USE_CFLAGS) $(USE_LDFLAGS) -o $(TARGET2) \
	$(FILE2).c $(LIBS)


# Compiler doesn't support OpenMP if OPENMP_CFLAGS is empty
ifneq ($(OPENMP_CFLAGS),)
$(OPENMP_TARGET1): $(OPENMP_FILE1).c
	$(CC) $(USE_CFLAGS) $(OPENMP_CFLAGS) $(USE_LDFLAGS) \
	-o $(OPENMP_TARGET1) $(OPENMP_FILE1).c $(LIBS)

$(OPENMP_TARGET2): $(OPENMP_FILE2).c
	$(CC) $(USE_CFLAGS) $(OPENMP_CFLAGS) $(USE_LDFLAGS) \
	-o $(OPENMP_TARGET2) $(OPENMP_FILE2).c $(LIBS)
endif


# Compiler supports CUDA if CUDA_CAPABLE is "yes"
ifeq ($(CUDA_CAPABLE), yes)
$(CUDA_TARGET1): $(CUDA_FILE1).cu
	$(CC) $(USE_CFLAGS) $(USE_LDFLAGS) -o $(CUDA_TARGET1) \
	$(CUDA_FILE1).cu $(LIBS)

$(CUDA_TARGET2): $(CUDA_FILE2).cu
	$(CC) $(USE_CFLAGS) $(USE_LDFLAGS) -o $(CUDA_TARGET2) \
	$(CUDA_FILE2).cu $(LIBS)

$(CUDA_TARGET3): $(CUDA_FILE3).cu
	$(CC) $(USE_CFLAGS) $(USE_LDFLAGS) -o $(CUDA_TARGET3) \
	$(CUDA_FILE3).cu $(LIBS)

$(CUDA_TARGET4): $(CUDA_FILE4).cu
	$(CC) $(USE_CFLAGS) $(USE_LDFLAGS) -o $(CUDA_TARGET4) \
	$(CUDA_FILE4).cu $(LIBS)

$(CUDA_TARGET5): $(CUDA_FILE5).cu
	$(CC) $(USE_CFLAGS) $(USE_LDFLAGS) -o $(CUDA_TARGET5) \
	$(CUDA_FILE5).cu $(LIBS)

$(CUDA_TARGET6): $(CUDA_FILE6).cu
	$(CC) $(USE_CFLAGS) $(USE_LDFLAGS) -o $(CUDA_TARGET6) \
	$(CUDA_FILE6).cu $(LIBS)

$(CUDA_TARGET7): $(CUDA_FILE7).cu
	$(CC) $(USE_CFLAGS) $(USE_LDFLAGS) -o $(CUDA_TARGET7) \
	$(CUDA_FILE7).cu $(LIBS)

$(CUDA_TARGET8): $(CUDA_FILE8).cu
	$(CC) $(USE_CFLAGS) $(USE_LDFLAGS) -o $(CUDA_TARGET8) \
	$(CUDA_FILE8).cu $(LIBS)

$(CUDA_TARGET9): $(CUDA_FILE9).cu
	$(CC) $(USE_CFLAGS) $(USE_LDFLAGS) -o $(CUDA_TARGET9) \
	$(CUDA_FILE9).cu $(LIBS)

$(CUDA_TARGET10): $(CUDA_FILE10).cu
	$(CC) $(USE_CFLAGS) $(USE_LDFLAGS) -o $(CUDA_TARGET10) \
	$(CUDA_FILE10).cu $(LIBS)

$(CUDA_TARGET11): $(CUDA_FILE11).cu
	$(CC) $(USE_CFLAGS) $(USE_LDFLAGS) -o $(CUDA_TARGET11) \
	$(CUDA_FILE11).cu $(LIBS)

$(CUDA_TARGET12): $(CUDA_FILE12).cu
	$(CC) $(USE_CFLAGS) $(USE_LDFLAGS) -o $(CUDA_TARGET12) \
	$(CUDA_FILE12).cu $(LIBS)
endif


# **********  pseudo rules to clean up  **********
#
clean:
	 rm -f	*.o *.obj

clean_all:
	 rm -f	$(ALL_TARGETS) $(ALL_OPENMP_TARGETS) \
		$(ALL_CUDA_TARGETS) \
		$(ALL_LIB_TARGETS) $(ALL_EXP_TARGETS)
	 rm -f	*.o *.obj
