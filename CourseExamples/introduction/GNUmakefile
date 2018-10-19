# GNUmakefile to build the below mentioned programs using different
# compiler techniques or programming models. Only "gcc-7.x" or newer
# can build all executables an Linux. Other compilers and operating
# systems don't support all models, so that you get only a subset of
# all possible executables.
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
# The following programs will be created if the compiler is available
# and supports the technique.
#
# "normal" C files (optimized sequential version)
# (cc, clang, gcc, icc, mpicc, nvcc, and pgcc from our LinuxLab can
# build these programs)
#   mat_mult_ijk
#   mat_mult_ikj
#   mat_mult_stride_ikj
#   mat_mult_stride_peel_ikj
#   mat_mult_block_ijk
#   mat_mult_block_pqr_ijk
#   mat_mult_block_ikj
#   mat_mult_transpose_ijk
#   mat_mult_trans_unroll_4_ijk
#   mat_mult_trans_unroll_8_ijk
#   mat_mult_int_ijk
#   mat_mult_int_ikj
#   mat_mult_cacheline_ijk
#   mat_mult_cacheline_ikj
#   mat_mult_block_cacheline_ijk
#   mat_mult_block_cacheline_ikj
#   dot_prod_sequential
#
# C files using POSIX threads
# Compiler must support POSIX threads to build these executables
# (GNUmakefile.env must have assigned "yes" to PTHREAD_CAPABLE)
# (cc, clang, gcc, icc, mpicc, nvcc, and pgcc from our LinuxLab can
# build these programs)
#   dot_prod_block_pthread
#   dot_prod_cyclic_pthread
#
# C files using automatic parallelization.
# Compiler must support automatic parallelization to build these
# executables (GNUmakefile.env must have assigned a value to
# AUTOPAR_CFLAGS)
# (cc, clang, gcc, icc, mpicc, and pgcc from our LinuxLab can
# build these programs)
#   mat_mult_ijk
#   mat_mult_ikj
#   mat_mult_stride_ikj
#   mat_mult_stride_peel_ikj
#   mat_mult_block_ijk
#   mat_mult_block_pqr_ijk
#   mat_mult_block_ikj
#   mat_mult_transpose_ijk
#   mat_mult_trans_unroll_4_ijk
#   mat_mult_trans_unroll_8_ijk
#   mat_mult_int_ijk
#   mat_mult_int_ikj
#
# C files using OpenACC pragmas.
# Compiler must support OpenACC to build these executables
# (GNUmakefile.env must have assigned a value to OPENACC_CFLAGS)
# (gcc and pgcc from our LinuxLab can build these programs)
#   dot_prod_OpenACC
#
# C files using OpenCL.
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
# Compiler must support OpenCL to build these executables
# (GNUmakefile.env must have assigned "yes" to OPENCL_CAPABLE)
# (cc, clang, gcc, icc, mpicc, nvcc, and pgcc from our LinuxLab can
# build these programs)
#   dot_prod_OpenCL
#
# C files using OpenMP pragmas.
# Compiler must support OpenMP to build these executables
# (GNUmakefile.env must have assigned a value to OPENMP_CFLAGS)
# (cc, clang, gcc, icc, mpicc, nvcc, and pgcc from our LinuxLab can
# build these programs)
#   mat_mult_OpenMP_ijk
#   mat_mult_int_OpenMP_ijk
#   dot_prod_block_OpenMP
#   dot_prod_cyclic_OpenMP
#
# C files using OpenMP accelerator pragmas.
# (GNUmakefile.env must have assigned a value to OPENMP_ACC_CFLAGS)
# (gcc and clang from our LinuxLab can build these programs)
#   dot_prod_accelerator_OpenMP
#
# C files using "dgemm" from GNU Scientific Library
# (cc, clang, gcc, icc, mpicc, nvcc, and pgcc from our LinuxLab can
# build these programs)
#   mat_mult_dgemm_gsl
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
# Possible values: cc, clang, gcc, icc, mpicc, nvcc, pgcc
#
CC = gcc


# necessary files
#
# program names: use the source code file name without file name
#                extension ".c"
# header files:  file name including file name extension
#

# "normal" C files (optimized sequential version)
FILE1	= mat_mult_ijk
FILE2	= mat_mult_ikj
FILE3	= mat_mult_stride_ikj
FILE4	= mat_mult_stride_peel_ikj
FILE5	= mat_mult_block_ijk
FILE6	= mat_mult_block_pqr_ijk
FILE7	= mat_mult_block_ikj
FILE8	= mat_mult_transpose_ijk
FILE9	= mat_mult_trans_unroll_4_ijk
FILE10	= mat_mult_trans_unroll_8_ijk
FILE11	= mat_mult_int_ijk
FILE12	= mat_mult_int_ikj
FILE13	= mat_mult_cacheline_ijk
FILE14	= mat_mult_cacheline_ikj
FILE15	= mat_mult_block_cacheline_ijk
FILE16	= mat_mult_block_cacheline_ikj
FILE17	= dot_prod_sequential

# C files using POSIX threads
PTHREAD_FILE1	= dot_prod_block_pthread
PTHREAD_FILE2	= dot_prod_cyclic_pthread

# C files using automatic parallelization
AUTO_FILE1	= mat_mult_ijk
AUTO_FILE2	= mat_mult_ikj
AUTO_FILE3	= mat_mult_stride_ikj
AUTO_FILE4	= mat_mult_stride_peel_ikj
AUTO_FILE5	= mat_mult_block_ijk
AUTO_FILE6	= mat_mult_block_pqr_ijk
AUTO_FILE7	= mat_mult_block_ikj
AUTO_FILE8	= mat_mult_transpose_ijk
AUTO_FILE9	= mat_mult_trans_unroll_4_ijk
AUTO_FILE10	= mat_mult_trans_unroll_8_ijk
AUTO_FILE11	= mat_mult_int_ijk
AUTO_FILE12	= mat_mult_int_ikj

# C files using OpenACC pragmas
OPENACC_FILE1	= dot_prod_OpenACC

# C files using OpenCL
OPENCL_FILE1	= dot_prod_OpenCL
OPENCL_FILE1_H	= dot_prod_OpenCL.h
OPENCL_FILE1_CL = dotProdKernel.cl
#
OPENCL_FILE90	= errorCodes

# C files using OpenMP pragmas
OPENMP_FILE1	= mat_mult_OpenMP_ijk
OPENMP_FILE2	= mat_mult_int_OpenMP_ijk
OPENMP_FILE3	= dot_prod_block_OpenMP
OPENMP_FILE4	= dot_prod_cyclic_OpenMP

# C files using OpenMP accelerator pragmas
OPENMP_ACC_FILE1 = dot_prod_accelerator_OpenMP

# C files using "dgemm" from GNU Scientific Library
GSL_FILE1	= mat_mult_dgemm_gsl


# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!               Don't change the next statement                   !!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#
# Determine operating system, processor architecture, used compiler for
# "mpicc", and some flags.
#
include GNUmakefile.env


# **********  choose additional FLAGS and libraries  **********
#
PTHREAD_LIBS = -lpthread
GSL_LIBS     = -lgsl -lgslcblas

ifeq ($(USE_CC), clang)
  AUTOPAR_LIBS	   = -lomp
  OPENMP_ACC_LIBS  = -lomp -lomptarget
else
  AUTOPAR_LIBS	   = 
  OPENMP_ACC_LIBS  = 
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
  TARGET1  = $(EXE_DIR)/$(FILE1).exe
  TARGET2  = $(EXE_DIR)/$(FILE2).exe
  TARGET3  = $(EXE_DIR)/$(FILE3).exe
  TARGET4  = $(EXE_DIR)/$(FILE4).exe
  TARGET5  = $(EXE_DIR)/$(FILE5).exe
  TARGET6  = $(EXE_DIR)/$(FILE6).exe
  TARGET7  = $(EXE_DIR)/$(FILE7).exe
  TARGET8  = $(EXE_DIR)/$(FILE8).exe
  TARGET9  = $(EXE_DIR)/$(FILE9).exe
  TARGET10 = $(EXE_DIR)/$(FILE10).exe
  TARGET11 = $(EXE_DIR)/$(FILE11).exe
  TARGET12 = $(EXE_DIR)/$(FILE12).exe
  TARGET13 = $(EXE_DIR)/$(FILE13).exe
  TARGET14 = $(EXE_DIR)/$(FILE14).exe
  TARGET15 = $(EXE_DIR)/$(FILE15).exe
  TARGET16 = $(EXE_DIR)/$(FILE16).exe
  TARGET17 = $(EXE_DIR)/$(FILE17).exe

  # Compiler supports POSIX threads if PTHREAD_CAPABLE is "yes"
  ifeq ($(PTHREAD_CAPABLE), yes)
    PTHREAD_TARGET1 = $(EXE_DIR)/$(PTHREAD_FILE1).exe
    PTHREAD_TARGET2 = $(EXE_DIR)/$(PTHREAD_FILE2).exe
  endif

  # Compiler doesn't support automatic parallelization if
  # AUTOPAR_CFLAGS is empty
  ifneq ($(AUTOPAR_CFLAGS),)
    AUTO_TARGET1  = $(EXE_DIR)/$(AUTO_FILE1)_auto.exe
    AUTO_TARGET2  = $(EXE_DIR)/$(AUTO_FILE2)_auto.exe
    AUTO_TARGET3  = $(EXE_DIR)/$(AUTO_FILE3)_auto.exe
    AUTO_TARGET4  = $(EXE_DIR)/$(AUTO_FILE4)_auto.exe
    AUTO_TARGET5  = $(EXE_DIR)/$(AUTO_FILE5)_auto.exe
    AUTO_TARGET6  = $(EXE_DIR)/$(AUTO_FILE6)_auto.exe
    AUTO_TARGET7  = $(EXE_DIR)/$(AUTO_FILE7)_auto.exe
    AUTO_TARGET8  = $(EXE_DIR)/$(AUTO_FILE8)_auto.exe
    AUTO_TARGET9  = $(EXE_DIR)/$(AUTO_FILE9)_auto.exe
    AUTO_TARGET10 = $(EXE_DIR)/$(AUTO_FILE10)_auto.exe
    AUTO_TARGET11 = $(EXE_DIR)/$(AUTO_FILE11)_auto.exe
    AUTO_TARGET12 = $(EXE_DIR)/$(AUTO_FILE12)_auto.exe
  endif

  # Compiler doesn't support OpenACC if OPENACC_CFLAGS is empty
  ifneq ($(OPENACC_CFLAGS),)
    OPENACC_TARGET1  = $(EXE_DIR)/$(OPENACC_FILE1).exe
  endif

  # Compiler supports OpenCL if OPENCL_CAPABLE is "yes"
  ifeq ($(OPENCL_CAPABLE), yes)
    OPENCL_TARGET1  = $(EXE_DIR)/$(OPENCL_FILE1).exe
    OPENCL_TARGET90 = $(OPENCL_FILE90).obj
  endif

  # Compiler doesn't support OpenMP if OPENMP_CFLAGS is empty
  ifneq ($(OPENMP_CFLAGS),)
    OPENMP_TARGET1  = $(EXE_DIR)/$(OPENMP_FILE1).exe
    OPENMP_TARGET2  = $(EXE_DIR)/$(OPENMP_FILE2).exe
    OPENMP_TARGET3  = $(EXE_DIR)/$(OPENMP_FILE3).exe
    OPENMP_TARGET4  = $(EXE_DIR)/$(OPENMP_FILE4).exe
  endif

  # Compiler doesn't support OpenMP for accelerators if
  # OPENMP_ACC_CFLAGS is empty
  ifneq ($(OPENMP_ACC_CFLAGS),)
    OPENMP_ACC_TARGET1 = $(EXE_DIR)/$(OPENMP_ACC_FILE1).exe
  endif

  # Compiler supports GNU Scientific Library if GSL_CAPABLE is "yes"
  ifeq ($(GSL_CAPABLE), yes)
    GSL_TARGET1  = $(EXE_DIR)/$(GSL_FILE1).exe
  endif
else
  TARGET1  = $(EXE_DIR)/$(FILE1)
  TARGET2  = $(EXE_DIR)/$(FILE2)
  TARGET3  = $(EXE_DIR)/$(FILE3)
  TARGET4  = $(EXE_DIR)/$(FILE4)
  TARGET5  = $(EXE_DIR)/$(FILE5)
  TARGET6  = $(EXE_DIR)/$(FILE6)
  TARGET7  = $(EXE_DIR)/$(FILE7)
  TARGET8  = $(EXE_DIR)/$(FILE8)
  TARGET9  = $(EXE_DIR)/$(FILE9)
  TARGET10 = $(EXE_DIR)/$(FILE10)
  TARGET11 = $(EXE_DIR)/$(FILE11)
  TARGET12 = $(EXE_DIR)/$(FILE12)
  TARGET13 = $(EXE_DIR)/$(FILE13)
  TARGET14 = $(EXE_DIR)/$(FILE14)
  TARGET15 = $(EXE_DIR)/$(FILE15)
  TARGET16 = $(EXE_DIR)/$(FILE16)
  TARGET17 = $(EXE_DIR)/$(FILE17)

  # Compiler supports POSIX threads if PTHREAD_CAPABLE is "yes"
  ifeq ($(PTHREAD_CAPABLE), yes)
    PTHREAD_TARGET1 = $(EXE_DIR)/$(PTHREAD_FILE1)
    PTHREAD_TARGET2 = $(EXE_DIR)/$(PTHREAD_FILE2)
  endif

  # Compiler doesn't support automatic parallelization if
  # AUTOPAR_CFLAGS is empty
  ifneq ($(AUTOPAR_CFLAGS),)
    AUTO_TARGET1  = $(EXE_DIR)/$(AUTO_FILE1)_auto
    AUTO_TARGET2  = $(EXE_DIR)/$(AUTO_FILE2)_auto
    AUTO_TARGET3  = $(EXE_DIR)/$(AUTO_FILE3)_auto
    AUTO_TARGET4  = $(EXE_DIR)/$(AUTO_FILE4)_auto
    AUTO_TARGET5  = $(EXE_DIR)/$(AUTO_FILE5)_auto
    AUTO_TARGET6  = $(EXE_DIR)/$(AUTO_FILE6)_auto
    AUTO_TARGET7  = $(EXE_DIR)/$(AUTO_FILE7)_auto
    AUTO_TARGET8  = $(EXE_DIR)/$(AUTO_FILE8)_auto
    AUTO_TARGET9  = $(EXE_DIR)/$(AUTO_FILE9)_auto
    AUTO_TARGET10 = $(EXE_DIR)/$(AUTO_FILE10)_auto
    AUTO_TARGET11 = $(EXE_DIR)/$(AUTO_FILE11)_auto
    AUTO_TARGET12 = $(EXE_DIR)/$(AUTO_FILE12)_auto
  endif

  # Compiler doesn't support OpenACC if OPENACC_CFLAGS is empty
  ifneq ($(OPENACC_CFLAGS),)
    OPENACC_TARGET1  = $(EXE_DIR)/$(OPENACC_FILE1)
  endif

  # Compiler supports OpenCL if OPENCL_CAPABLE is "yes"
  ifeq ($(OPENCL_CAPABLE), yes)
    OPENCL_TARGET1  = $(EXE_DIR)/$(OPENCL_FILE1)
    OPENCL_TARGET90 = $(OPENCL_FILE90).o
  endif

  # Compiler doesn't support OpenMP if OPENMP_CFLAGS is empty
  ifneq ($(OPENMP_CFLAGS),)
    OPENMP_TARGET1  = $(EXE_DIR)/$(OPENMP_FILE1)
    OPENMP_TARGET2  = $(EXE_DIR)/$(OPENMP_FILE2)
    OPENMP_TARGET3  = $(EXE_DIR)/$(OPENMP_FILE3)
    OPENMP_TARGET4  = $(EXE_DIR)/$(OPENMP_FILE4)
  endif

  # Compiler doesn't support OpenMP for accelerators if
  # OPENMP_ACC_CFLAGS is empty
  ifneq ($(OPENMP_ACC_CFLAGS),)
    OPENMP_ACC_TARGET1 = $(EXE_DIR)/$(OPENMP_ACC_FILE1)
  endif

  # Compiler supports GNU Scientific Library if GSL_CAPABLE is "yes"
  ifeq ($(GSL_CAPABLE), yes)
    GSL_TARGET1  = $(EXE_DIR)/$(GSL_FILE1)
  endif
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
  OPENCL_TARGET_FILE1_CL = $(KERNEL_FILE_DIR)/$(OPENCL_FILE1_CL)
endif


# **********  rules to make targets  **********
#
ALL_TARGETS =	$(TARGET1) $(TARGET2) $(TARGET3) $(TARGET4) \
		$(TARGET5) $(TARGET6) $(TARGET7) $(TARGET8) \
		$(TARGET9) $(TARGET10) $(TARGET11) $(TARGET12) \
		$(TARGET13) $(TARGET14) $(TARGET15) $(TARGET16) \
		$(TARGET17)

# Compiler supports POSIX threads if PTHREAD_CAPABLE is "yes"
ifeq ($(PTHREAD_CAPABLE), yes)
  ALL_PTHREAD_TARGETS = $(PTHREAD_TARGET1) $(PTHREAD_TARGET2)
endif

# Compiler doesn't support automatic parallelization if
# AUTOPAR_CFLAGS is empty
ifneq ($(AUTOPAR_CFLAGS),)
  ALL_AUTO_TARGETS = $(AUTO_TARGET1) $(AUTO_TARGET2) $(AUTO_TARGET3) \
		     $(AUTO_TARGET4) $(AUTO_TARGET5) $(AUTO_TARGET6) \
		     $(AUTO_TARGET7) $(AUTO_TARGET8) $(AUTO_TARGET9) \
		     $(AUTO_TARGET10) $(AUTO_TARGET11) $(AUTO_TARGET12)
else
  ALL_AUTO_TARGETS = 
endif

# Compiler doesn't support OpenACC if OPENACC_CFLAGS is empty
ifneq ($(OPENACC_CFLAGS),)
  ALL_OPENACC_TARGETS = $(OPENACC_TARGET1)
else
  ALL_OPENACC_TARGETS = 
endif

# Compiler supports OpenCL if OPENCL_CAPABLE is "yes"
ifeq ($(OPENCL_CAPABLE), yes)
  ifneq ($(KERNEL_FILE_DIR),)
    ALL_OPENCL_TARGETS_CL = $(OPENCL_TARGET_FILE1_CL)
    ALL_OPENCL_TARGETS	  = $(OPENCL_TARGET1) $(ALL_OPENCL_TARGETS_CL)
  else
    ALL_OPENCL_TARGETS_CL = 
    ALL_OPENCL_TARGETS = $(OPENCL_TARGET1)
  endif
else
  ALL_OPENCL_TARGETS_CL = 
  ALL_OPENCL_TARGETS = 
endif

# Compiler doesn't support OpenMP if OPENMP_CFLAGS is empty
ifneq ($(OPENMP_CFLAGS),)
  ALL_OPENMP_TARGETS = $(OPENMP_TARGET1) $(OPENMP_TARGET2) \
		       $(OPENMP_TARGET3) $(OPENMP_TARGET4)
else
  ALL_OPENMP_TARGETS = 
endif

# Compiler doesn't support OpenMP for accelerators if OPENMP_ACC_CFLAGS
# is empty
ifneq ($(OPENMP_ACC_CFLAGS),)
  ALL_OPENMP_ACC_TARGETS = $(OPENMP_ACC_TARGET1)
else
  ALL_OPENMP_ACC_TARGETS = 
endif

# Compiler supports GNU Scientific Library if GSL_CAPABLE is "yes"
ifeq ($(GSL_CAPABLE), yes)
  ALL_GSL_TARGETS = $(GSL_TARGET1)
else
  ALL_GSL_TARGETS = 
endif


# optimization report files from Intel compiler if the report is not
# redirected to stderr with "-qopt-report-file=stderr"
#
ALL_OPTRPT =	$(TARGET1).optrpt $(TARGET2).optrpt \
		$(TARGET3).optrpt $(TARGET4).optrpt \
		$(TARGET5).optrpt $(TARGET6).optrpt \
		$(TARGET7).optrpt $(TARGET8).optrpt \
		$(TARGET9).optrpt $(TARGET10).optrpt \
		$(TARGET11).optrpt $(TARGET12).optrpt \
		$(TARGET13).optrpt $(TARGET14).optrpt \
		$(TARGET15).optrpt $(TARGET16).optrpt \
		$(TARGET17).optrpt

# Compiler supports POSIX threads if PTHREAD_CAPABLE is "yes"
ifeq ($(PTHREAD_CAPABLE), yes)
  ALL_PTHREAD_OPTRPT = $(PTHREAD_TARGET1).optrpt $(PTHREAD_TARGET2).optrpt
else
  ALL_PTHREAD_OPTRPT = 
endif

# Compiler doesn't support automatic parallelization if
# AUTOPAR_CFLAGS is empty
ifneq ($(AUTOPAR_CFLAGS),)
  ALL_AUTO_OPTRPT = $(AUTO_TARGET1).optrpt $(AUTO_TARGET2).optrpt \
		    $(AUTO_TARGET3).optrpt $(AUTO_TARGET4).optrpt \
		    $(AUTO_TARGET5).optrpt $(AUTO_TARGET6).optrpt \
		    $(AUTO_TARGET7).optrpt $(AUTO_TARGET8).optrpt \
		    $(AUTO_TARGET9).optrpt $(AUTO_TARGET10).optrpt \
		    $(AUTO_TARGET11).optrpt $(AUTO_TARGET12).optrpt
else
  ALL_AUTO_OPTRPT = 
endif

# Compiler doesn't support OpenACC if OPENACC_CFLAGS is empty
ifneq ($(OPENACC_CFLAGS),)
  ALL_OPENACC_OPTRPT = $(OPENACC_TARGET1).optrpt
else
  ALL_OPENACC_OPTRPT = 
endif

# Compiler supports OpenCL if OPENCL_CAPABLE is "yes"
ifeq ($(OPENCL_CAPABLE), yes)
  ALL_OPENCL_OPTRPT = $(OPENCL_TARGET1).optrpt
else
  ALL_OPENCL_OPTRPT = 
endif

# Compiler doesn't support OpenMP if OPENMP_CFLAGS is empty
ifneq ($(OPENMP_CFLAGS),)
  ALL_OPENMP_OPTRPT = $(OPENMP_TARGET1).optrpt $(OPENMP_TARGET2).optrpt \
		      $(OPENMP_TARGET3).optrpt $(OPENMP_TARGET4).optrpt
else
  ALL_OPENMP_OPTRPT = 
endif

# Compiler doesn't support OpenMP for accelerators if OPENMP_ACC_CFLAGS
# is empty
ifneq ($(OPENMP_ACC_CFLAGS),)
  ALL_OPENMP_ACC_OPTRPT = $(OPENMP_ACC_TARGET1).optrpt
else
  ALL_OPENMP_ACC_OPTRPT = 
endif

# Compiler supports GNU Scientific Library if GSL_CAPABLE is "yes"
ifeq ($(GSL_CAPABLE), yes)
  ALL_GSL_OPTRPT = $(GSL_TARGET1).optrpt
else
  ALL_GSL_OPTRPT = 
endif



all:	$(ALL_TARGETS) $(ALL_PTHREAD_TARGETS) $(ALL_AUTO_TARGETS) \
	$(ALL_OPENACC_TARGETS) $(ALL_OPENCL_TARGETS) \
	$(ALL_OPENCL_TARGETS_CL) \
	$(ALL_OPENMP_TARGETS) $(ALL_OPENMP_ACC_TARGETS) \
	$(ALL_GSL_TARGETS)


$(TARGET1): $(FILE1).c
	$(CC) $(USE_CFLAGS) $(USE_LDFLAGS) -o $(TARGET1) \
	$(FILE1).c $(LIBS)

$(TARGET2): $(FILE2).c
	$(CC) $(USE_CFLAGS) $(USE_LDFLAGS) -o $(TARGET2) \
	$(FILE2).c $(LIBS)

$(TARGET3): $(FILE3).c
	$(CC) $(USE_CFLAGS) $(USE_LDFLAGS) -o $(TARGET3) \
	$(FILE3).c $(LIBS)

$(TARGET4): $(FILE4).c
	$(CC) $(USE_CFLAGS) $(USE_LDFLAGS) -o $(TARGET4) \
	$(FILE4).c $(LIBS)

$(TARGET5): $(FILE5).c
	$(CC) $(USE_CFLAGS) $(USE_LDFLAGS) -o $(TARGET5) \
	$(FILE5).c $(LIBS)

$(TARGET6): $(FILE6).c
	$(CC) $(USE_CFLAGS) $(USE_LDFLAGS) -o $(TARGET6) \
	$(FILE6).c $(LIBS)

$(TARGET7): $(FILE7).c
	$(CC) $(USE_CFLAGS) $(USE_LDFLAGS) -o $(TARGET7) \
	$(FILE7).c $(LIBS)

$(TARGET8): $(FILE8).c
	$(CC) $(USE_CFLAGS) $(USE_LDFLAGS) -o $(TARGET8) \
	$(FILE8).c $(LIBS)

$(TARGET9): $(FILE9).c
	$(CC) $(USE_CFLAGS) $(USE_LDFLAGS) -o $(TARGET9) \
	$(FILE9).c $(LIBS)

$(TARGET10): $(FILE10).c
	$(CC) $(USE_CFLAGS) $(USE_LDFLAGS) -o $(TARGET10) \
	$(FILE10).c $(LIBS)

$(TARGET11): $(FILE11).c
	$(CC) $(USE_CFLAGS) $(USE_LDFLAGS) -o $(TARGET11) \
	$(FILE11).c $(LIBS)

$(TARGET12): $(FILE12).c
	$(CC) $(USE_CFLAGS) $(USE_LDFLAGS) -o $(TARGET12) \
	$(FILE12).c $(LIBS)

$(TARGET13): $(FILE13).c
	$(CC) $(USE_CFLAGS) $(USE_LDFLAGS) -o $(TARGET13) \
	$(FILE13).c $(LIBS)

$(TARGET14): $(FILE14).c
	$(CC) $(USE_CFLAGS) $(USE_LDFLAGS) -o $(TARGET14) \
	$(FILE14).c $(LIBS)

$(TARGET15): $(FILE15).c
	$(CC) $(USE_CFLAGS) $(USE_LDFLAGS) -o $(TARGET15) \
	$(FILE15).c $(LIBS)

$(TARGET16): $(FILE16).c
	$(CC) $(USE_CFLAGS) $(USE_LDFLAGS) -o $(TARGET16) \
	$(FILE16).c $(LIBS)

$(TARGET17): $(FILE17).c
	$(CC) $(USE_CFLAGS) $(USE_LDFLAGS) -o $(TARGET17) \
	$(FILE17).c $(LIBS)


# Compiler supports POSIX threads if PTHREAD_CAPABLE is "yes"
ifeq ($(PTHREAD_CAPABLE), yes)
$(PTHREAD_TARGET1): $(PTHREAD_FILE1).c
	$(CC) $(USE_CFLAGS) $(USE_LDFLAGS) -o $(PTHREAD_TARGET1) \
	$(PTHREAD_FILE1).c $(LIBS) $(PTHREAD_LIBS)

$(PTHREAD_TARGET2): $(PTHREAD_FILE2).c
	$(CC) $(USE_CFLAGS) $(USE_LDFLAGS) -o $(PTHREAD_TARGET2) \
	$(PTHREAD_FILE2).c $(LIBS) $(PTHREAD_LIBS)
endif


# Compiler doesn't support automatic parallelization if
# AUTOPAR_CFLAGS is empty
ifneq ($(AUTOPAR_CFLAGS),)
$(AUTO_TARGET1): $(AUTO_FILE1).c
	$(CC) $(USE_CFLAGS) $(AUTOPAR_CFLAGS) $(USE_LDFLAGS) \
	-o $(AUTO_TARGET1) $(AUTO_FILE1).c $(AUTOPAR_LIBS) $(LIBS)

$(AUTO_TARGET2): $(AUTO_FILE2).c
	$(CC) $(USE_CFLAGS) $(AUTOPAR_CFLAGS) $(USE_LDFLAGS) \
	-o $(AUTO_TARGET2) $(AUTO_FILE2).c $(AUTOPAR_LIBS) $(LIBS)

$(AUTO_TARGET3): $(AUTO_FILE3).c
	$(CC) $(USE_CFLAGS) $(AUTOPAR_CFLAGS) $(USE_LDFLAGS) \
	-o $(AUTO_TARGET3) $(AUTO_FILE3).c $(AUTOPAR_LIBS) $(LIBS)

$(AUTO_TARGET4): $(AUTO_FILE4).c
	$(CC) $(USE_CFLAGS $(AUTOPAR_CFLAGS)) $(USE_LDFLAGS) \
	-o $(AUTO_TARGET4) $(AUTO_FILE4).c $(AUTOPAR_LIBS) $(LIBS)

$(AUTO_TARGET5): $(AUTO_FILE5).c
	$(CC) $(USE_CFLAGS) $(AUTOPAR_CFLAGS) $(USE_LDFLAGS) \
	-o $(AUTO_TARGET5) $(AUTO_FILE5).c $(AUTOPAR_LIBS) $(LIBS)

$(AUTO_TARGET6): $(AUTO_FILE6).c
	$(CC) $(USE_CFLAGS) $(AUTOPAR_CFLAGS) $(USE_LDFLAGS) \
	-o $(AUTO_TARGET6) $(AUTO_FILE6).c $(AUTOPAR_LIBS) $(LIBS)

$(AUTO_TARGET7): $(AUTO_FILE7).c
	$(CC) $(USE_CFLAGS) $(AUTOPAR_CFLAGS) $(USE_LDFLAGS) \
	-o $(AUTO_TARGET7) $(AUTO_FILE7).c $(AUTOPAR_LIBS) $(LIBS)

$(AUTO_TARGET8): $(AUTO_FILE8).c
	$(CC) $(USE_CFLAGS) $(AUTOPAR_CFLAGS) $(USE_LDFLAGS) \
	-o $(AUTO_TARGET8) $(AUTO_FILE8).c $(AUTOPAR_LIBS) $(LIBS)

$(AUTO_TARGET9): $(AUTO_FILE9).c
	$(CC) $(USE_CFLAGS) $(AUTOPAR_CFLAGS) $(USE_LDFLAGS) \
	-o $(AUTO_TARGET9) $(AUTO_FILE9).c $(AUTOPAR_LIBS) $(LIBS)

$(AUTO_TARGET10): $(AUTO_FILE10).c
	$(CC) $(USE_CFLAGS) $(AUTOPAR_CFLAGS) $(USE_LDFLAGS) \
	-o $(AUTO_TARGET10) $(AUTO_FILE10).c $(AUTOPAR_LIBS) $(LIBS)

$(AUTO_TARGET11): $(AUTO_FILE11).c
	$(CC) $(USE_CFLAGS) $(AUTOPAR_CFLAGS) $(USE_LDFLAGS) \
	-o $(AUTO_TARGET11) $(AUTO_FILE11).c $(AUTOPAR_LIBS) $(LIBS)

$(AUTO_TARGET12): $(AUTO_FILE12).c
	$(CC) $(USE_CFLAGS) $(AUTOPAR_CFLAGS) $(USE_LDFLAGS) \
	-o $(AUTO_TARGET12) $(AUTO_FILE12).c $(AUTOPAR_LIBS) $(LIBS)
endif


# Compiler doesn't support OpenACC if OPENACC_CFLAGS is empty
ifneq ($(OPENACC_CFLAGS),)
$(OPENACC_TARGET1): $(OPENACC_FILE1).c
	$(CC) $(USE_CFLAGS) $(OPENACC_CFLAGS) $(USE_LDFLAGS) \
	-o $(OPENACC_TARGET1) $(OPENACC_FILE1).c $(LIBS)
endif


# Compiler supports OpenCL if OPENCL_CAPABLE is "yes"
ifeq ($(OPENCL_CAPABLE), yes)
$(OPENCL_TARGET1): $(OPENCL_FILE1).c $(OPENCL_FILE1_H) \
		   $(OPENCL_FILE1_CL) $(OPENCL_TARGET90)
	$(CC) $(USE_CFLAGS) $(USE_LDFLAGS) \
	-o $(OPENCL_TARGET1) $(OPENCL_FILE1).c \
	$(OPENCL_TARGET90) $(OPENCL_LIBS) $(LIBS)

$(OPENCL_TARGET90): $(OPENCL_FILE90).c
	$(CC) $(USE_CFLAGS) $(OPENCL_CFLAGS) -c \
	-o $(OPENCL_TARGET90) $(OPENCL_FILE90).c

  ifneq ($(KERNEL_FILE_DIR),)
    $(OPENCL_TARGET_FILE1_CL): $(OPENCL_FILE1_CL)
	cp -i $(OPENCL_FILE1_CL) $(OPENCL_TARGET_FILE1_CL)
  endif
endif


# Compiler doesn't support OpenMP if OPENMP_CFLAGS is empty
ifneq ($(OPENMP_CFLAGS),)
$(OPENMP_TARGET1): $(OPENMP_FILE1).c
	$(CC) $(USE_CFLAGS) $(OPENMP_CFLAGS) $(USE_LDFLAGS) \
	-o $(OPENMP_TARGET1) $(OPENMP_FILE1).c $(LIBS)

$(OPENMP_TARGET2): $(OPENMP_FILE2).c
	$(CC) $(USE_CFLAGS) $(OPENMP_CFLAGS) $(USE_LDFLAGS) \
	-o $(OPENMP_TARGET2) $(OPENMP_FILE2).c $(LIBS)

$(OPENMP_TARGET3): $(OPENMP_FILE3).c
	$(CC) $(USE_CFLAGS) $(OPENMP_CFLAGS) $(USE_LDFLAGS) \
	-o $(OPENMP_TARGET3) $(OPENMP_FILE3).c $(LIBS)

$(OPENMP_TARGET4): $(OPENMP_FILE4).c
	$(CC) $(USE_CFLAGS) $(OPENMP_CFLAGS) $(USE_LDFLAGS) \
	-o $(OPENMP_TARGET4) $(OPENMP_FILE4).c $(LIBS)
endif


# Compiler doesn't support OpenMP for accelerators if OPENMP_ACC_CFLAGS
# is empty
ifneq ($(OPENMP_ACC_CFLAGS),)
$(OPENMP_ACC_TARGET1): $(OPENMP_ACC_FILE1).c
	$(CC) $(USE_CFLAGS) $(OPENMP_CFLAGS) $(USE_LDFLAGS) \
	-o $(OPENMP_ACC_TARGET1) $(OPENMP_ACC_FILE1).c \
	$(OPENMP_ACC_LIBS) $(LIBS)
endif


# Compiler supports GNU Scientific Library if GSL_CAPABLE is "yes"
ifeq ($(GSL_CAPABLE), yes)
$(GSL_TARGET1): $(GSL_FILE1).c
	$(CC) $(USE_CFLAGS) $(USE_LDFLAGS) \
	-o $(GSL_TARGET1) $(GSL_FILE1).c $(GSL_LIBS) $(LIBS)
endif



# **********  pseudo rules to clean up  **********
#
clean:
	 rm -f *.o *.obj

clean_all:
  ifneq ($(KERNEL_FILE_DIR),)
	 rm -f  $(ALL_TARGETS) $(ALL_PTHREAD_TARGETS) \
		$(ALL_AUTO_TARGETS) $(ALL_OPENACC_TARGETS) \
		$(ALL_OPENCL_TARGETS) $(ALL_OPENCL_TARGETS_CL) \
		$(ALL_OPENMP_TARGETS) $(ALL_OPENMP_ACC_TARGETS) \
		$(ALL_GSL_TARGETS) \
		$(ALL_OPTRPT) $(ALL_PTHREAD_OPTRPT) $(ALL_AUTO_OPTRPT) \
		$(ALL_OPENACC_OPTRPT) $(ALL_OPENCL_OPTRPT) \
		$(ALL_OPENMP_OPTRPT) $(ALL_OPENMP_ACC_OPTRPT) \
		$(ALL_GSL_OPTRPT)
	 rm -f	*.o *.obj
  else
	 rm -f  $(ALL_TARGETS) $(ALL_PTHREAD_TARGETS) \
		$(ALL_AUTO_TARGETS) $(ALL_OPENACC_TARGETS) \
		$(ALL_OPENCL_TARGETS) \
		$(ALL_OPENMP_TARGETS) $(ALL_OPENMP_ACC_TARGETS) \
		$(ALL_GSL_TARGETS) \
		$(ALL_OPTRPT) $(ALL_PTHREAD_OPTRPT) $(ALL_AUTO_OPTRPT) \
		$(ALL_OPENACC_OPTRPT) $(ALL_OPENCL_OPTRPT) \
		$(ALL_OPENMP_OPTRPT) $(ALL_OPENMP_ACC_OPTRPT) \
		$(ALL_GSL_OPTRPT)
	 rm -f	*.o *.obj
  endif
