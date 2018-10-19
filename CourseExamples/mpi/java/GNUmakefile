# GNUmakefile to build the below mentioned small programs.
#
# The directory of the class-files must be included in CLASSPATH.
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
# Usage:
#   (g)make		make/build class files
#   (g)make clean	remove class files
#
#
# File: GNUmakefile			Author: S. Gross
# Date: 19.07.2018
#


MPIJAVAC = mpijavac

# check if MPIJAVAC is available
#
MYOS = $(shell uname -s | cut -d _ -f 1)
ifeq ($(MYOS), Linux)
  MPIJAVAC_AVAIL = $(shell which $(MPIJAVAC) 2>&1 | cut -d' ' -f1)
  # possible error message using Linux:
  #   which: no mpijavac in ...
  ifeq ($(MPIJAVAC_AVAIL), which:)
    $(warning )
    $(warning *** "$(MPIJAVAC)" not available or not activated in \
	$(HOME)/.cshrc)
    $(error "$(MPIJAVAC)" is only available for Open MPI)
  endif
else
  $(warning )
  $(error "mpijavac" is only available for Open MPI on Linux)
endif


# Choose file name withou extension ".java".
#
FILE1	= InitFinalizeMain
FILE2	= HelloMainWithBarrier
FILE3	= HelloMainWithoutBarrier
FILE4	= HelloMainWithoutMPI
FILE5	= MpiDatatypeSizesMain
FILE6	= MpiDatatypeExtentsMain
FILE7	= SizeExtentMain
FILE8	= MsgSendRecvMain
FILE9	= ColumnSendRecvDouble2DarrayIn1DarrayMain
FILE10	= ColumnSendRecvInt2DarrayIn1DarrayMain
FILE11	= ColumnScatterDouble2DarrayIn1DarrayMain
FILE12	= MatMultWithNproc2DarrayIn1DarrayMain
FILE13	= MatMultWith1toNproc2DarrayIn1DarrayMain
FILE14	= MatMultWithAnyProc2DarrayIn1DarrayMain
FILE15	= EnvironVarMain
FILE16	= BcastIntMain
FILE17	= BcastIntArrayMain
FILE18	= BcastDoubleMain
FILE19	= BcastDoubleArrayMain
FILE20	= BcastStructMain
FILE21	= BcastStructArrayMain
FILE22	= ScatterStructArrayMain
FILE23	= ScatterIntArrayMain
FILE24	= ScatterDoubleArrayMain
FILE25	= ExceptionMpiMain
FILE26	= ExceptionJavaMain

FILE98	= PrintArray
FILE99	= MyStruct


# **********  choose directory for class files  **********
#
# It is a good idea to have one directory for all class files
# which you can add to the environment variable CLASSPATH
# so that you can run your programs from all directories.
#
#CLASS_DIR = .
CLASS_DIR = $(DIRPREFIX_LOCAL)/mpi_classfiles


# **********  choose program name and location  **********
#
TARGET1  = $(CLASS_DIR)/$(FILE1).class
TARGET2  = $(CLASS_DIR)/$(FILE2).class
TARGET3  = $(CLASS_DIR)/$(FILE3).class
TARGET4  = $(CLASS_DIR)/$(FILE4).class
TARGET5  = $(CLASS_DIR)/$(FILE5).class
TARGET6  = $(CLASS_DIR)/$(FILE6).class
TARGET7  = $(CLASS_DIR)/$(FILE7).class
TARGET8  = $(CLASS_DIR)/$(FILE8).class
TARGET9  = $(CLASS_DIR)/$(FILE9).class
TARGET10  = $(CLASS_DIR)/$(FILE10).class
TARGET11  = $(CLASS_DIR)/$(FILE11).class
TARGET12  = $(CLASS_DIR)/$(FILE12).class
TARGET13  = $(CLASS_DIR)/$(FILE13).class
TARGET14  = $(CLASS_DIR)/$(FILE14).class
TARGET15  = $(CLASS_DIR)/$(FILE15).class
TARGET16  = $(CLASS_DIR)/$(FILE16).class
TARGET17  = $(CLASS_DIR)/$(FILE17).class
TARGET18  = $(CLASS_DIR)/$(FILE18).class
TARGET19  = $(CLASS_DIR)/$(FILE19).class
TARGET20  = $(CLASS_DIR)/$(FILE20).class
TARGET21  = $(CLASS_DIR)/$(FILE21).class
TARGET22  = $(CLASS_DIR)/$(FILE22).class
TARGET23  = $(CLASS_DIR)/$(FILE23).class
TARGET24  = $(CLASS_DIR)/$(FILE24).class
TARGET25  = $(CLASS_DIR)/$(FILE25).class
TARGET26  = $(CLASS_DIR)/$(FILE26).class
TARGET98  = $(CLASS_DIR)/$(FILE98).class
TARGET99  = $(CLASS_DIR)/$(FILE99).class
TARGET99a = '$(CLASS_DIR)/$(FILE99)$$Data.class'


# **********  rules to make targets  **********
#
ALL_TARGETS =	$(TARGET1) $(TARGET2) $(TARGET3) $(TARGET4) \
		$(TARGET5) $(TARGET6) $(TARGET7) $(TARGET8) \
		$(TARGET9) $(TARGET10) $(TARGET11) $(TARGET12) \
		$(TARGET13) $(TARGET14) $(TARGET15) $(TARGET16) \
		$(TARGET17) $(TARGET18) $(TARGET19) $(TARGET20) \
		$(TARGET21) $(TARGET22) $(TARGET23) $(TARGET24) \
		$(TARGET25) $(TARGET26) $(TARGET98) $(TARGET99)


all:	$(ALL_TARGETS)

$(TARGET1): $(FILE1).java
	$(MPIJAVAC) -d $(CLASS_DIR) $(FILE1).java

$(TARGET2): $(FILE2).java
	$(MPIJAVAC) -d $(CLASS_DIR) $(FILE2).java

$(TARGET3): $(FILE3).java
	$(MPIJAVAC) -d $(CLASS_DIR) $(FILE3).java

$(TARGET4): $(FILE4).java
	$(MPIJAVAC) -d $(CLASS_DIR) $(FILE4).java

$(TARGET5): $(FILE5).java
	$(MPIJAVAC) -d $(CLASS_DIR) $(FILE5).java

$(TARGET6): $(FILE6).java
	$(MPIJAVAC) -d $(CLASS_DIR) $(FILE6).java

$(TARGET7): $(FILE7).java
	$(MPIJAVAC) -d $(CLASS_DIR) $(FILE7).java

$(TARGET8): $(FILE8).java
	$(MPIJAVAC) -d $(CLASS_DIR) $(FILE8).java

$(TARGET9): $(FILE9).java $(FILE98).java
	$(MPIJAVAC) -d $(CLASS_DIR) $(FILE9).java

$(TARGET10): $(FILE10).java $(FILE98).java
	$(MPIJAVAC) -d $(CLASS_DIR) $(FILE10).java

$(TARGET11): $(FILE11).java $(FILE98).java
	$(MPIJAVAC) -d $(CLASS_DIR) $(FILE11).java

$(TARGET12): $(FILE12).java $(FILE98).java
	$(MPIJAVAC) -d $(CLASS_DIR) $(FILE12).java

$(TARGET13): $(FILE13).java $(FILE98).java
	$(MPIJAVAC) -d $(CLASS_DIR) $(FILE13).java

$(TARGET14): $(FILE14).java $(FILE98).java
	$(MPIJAVAC) -d $(CLASS_DIR) $(FILE14).java

$(TARGET15): $(FILE15).java
	$(MPIJAVAC) -d $(CLASS_DIR) $(FILE15).java

$(TARGET16): $(FILE16).java
	$(MPIJAVAC) -d $(CLASS_DIR) $(FILE16).java

$(TARGET17): $(FILE17).java
	$(MPIJAVAC) -d $(CLASS_DIR) $(FILE17).java

$(TARGET18): $(FILE18).java
	$(MPIJAVAC) -d $(CLASS_DIR) $(FILE18).java

$(TARGET19): $(FILE19).java
	$(MPIJAVAC) -d $(CLASS_DIR) $(FILE19).java

$(TARGET20): $(FILE20).java $(FILE99).java
	$(MPIJAVAC) -d $(CLASS_DIR) $(FILE20).java

$(TARGET21): $(FILE21).java $(FILE99).java
	$(MPIJAVAC) -d $(CLASS_DIR) $(FILE21).java

$(TARGET22): $(FILE22).java $(FILE99).java
	$(MPIJAVAC) -d $(CLASS_DIR) $(FILE22).java

$(TARGET23): $(FILE23).java
	$(MPIJAVAC) -d $(CLASS_DIR) $(FILE23).java

$(TARGET24): $(FILE24).java
	$(MPIJAVAC) -d $(CLASS_DIR) $(FILE24).java

$(TARGET25): $(FILE25).java
	$(MPIJAVAC) -d $(CLASS_DIR) $(FILE25).java

$(TARGET26): $(FILE26).java
	$(MPIJAVAC) -d $(CLASS_DIR) $(FILE26).java

$(TARGET98): $(FILE98).java
	$(MPIJAVAC) -d $(CLASS_DIR) $(FILE98).java

$(TARGET99): $(FILE99).java
	$(MPIJAVAC) -d $(CLASS_DIR) $(FILE99).java


# **********  pseudo rules to clean up  **********
#
clean:
	 rm -f $(ALL_TARGETS) $(TARGET99a)

clean_all: clean
