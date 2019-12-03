# /*
#  * Description:   make file for  Makefile for adaptive real-time dbscan
#  *          example usage:  
#  *
#  * Author:      Tigran Mkhoyan 2019
#  * Email :      t.mkhoyan@tudelft.nl
#  */


.PHONY: all clean

# The program to build
#  comment the desired programm
NAME       := adapativeClustering_dbscan_inverse_noise
NAME       := adapativeClustering_dbscan_inverse_noise

OS=$(shell uname)
CVVERSION=$(shell opencv_version | cut -c 1-1)
# BINDIR ="bin/"

# ----------------------------specify----------------------------------------------#
# ----------------------------Input  ----------------------------------------------#
# ----------------------------path ------------------------------------------------#

OPT_IMAGEPATH_DIR:="d2_c2_f5_g10_30/frames" 			     	# R1.  --> use this for jpeg
# OPT_IMAGEPATH_DIR="d2_c2_f5_g5_30/frames" 			     	# R2
# OPT_IMAGEPATH_DIR:="d3_c2_sweep_g10_30_corr/frames" 	     	# R3
OPT_IMG_TYPE:=tif


OPT_IMAGEPATH_DIR="d2_c2_f5_g10_30/frames_jpg" 			     # R1.  --> use this for jpeg
# # OPT_IMAGEPATH_DIR="d2_c2_f5_g5_30/frames_jpg" 			     # R2
OPT_IMAGEPATH_DIR="d3_c2_sweep_g10_30_corr/frames_jpg" 	     # R3
OPT_IMG_TYPE:=jpg


OPT_COND:=R3

OPT_FILE_OUT_DIR:="out"

# ------------------------------ some variables ----------------------------------------

ifeq ($(OS),Darwin) # Mac OS X
    SUFFIX=osx
    CLOUD_DIR="/Users/tmkhoyan/surfdrive/"
    BASE_DIR="/Users/tmkhoyan/Desktop/data_wt/"
endif

ifeq ($(OS),Linux)  
    SUFFIX=linux
    CLOUD_DIR="/media/tmkhoyan/surfdrive/"
    BASE_DIR="/media/tmkhoyan/Desktop/data_wt/"
endif

#cam 1 or 2
#OPT_CAM="CAM TE"
OPT_CAM="CAM TE" # current
OPT_CAM_TE="CAM TE" # current
OPT_CAM_LE="CAM LE" # current

OPT_IMAGEPATH:=$(BASE_DIR)$(OPT_IMAGEPATH_DIR)

OPT_FILE_OUT_LE=$(OPT_FILE_OUT_DIR)/input_le_$(OPT_IMG_TYPE)_$(OPT_COND)_$(SUFFIX).txt
OPT_FILE_OUT_TE=$(OPT_FILE_OUT_DIR)/input_te_$(OPT_IMG_TYPE)_$(OPT_COND)_$(SUFFIX).txt

# ----------------------------macros  ----------------------------------------------#

# OS problem awk -v path="$(OPT_IMAGEPATH)" quotes
define generate_imagelist_le
	ls $(OPT_IMAGEPATH) | sort -n -t_ -k2 | grep .$(OPT_IMG_TYPE) | grep -i $(OPT_CAM_LE) | gawk '{print}' | awk -v path=$(OPT_IMAGEPATH) '{print path"/"$$0}' >  $(OPT_FILE_OUT_LE)
endef

define generate_imagelist_te
	ls $(OPT_IMAGEPATH) | sort -n -t_ -k2 | grep .$(OPT_IMG_TYPE)| grep -i $(OPT_CAM_TE) | awk -v path=$(OPT_IMAGEPATH) '{print path"/" $$0}' > $(OPT_FILE_OUT_TE)
endef

# Build tools and flags
CXX+=-std=c++11
LD         := $(CXX)
CPPFLAGS   := -I /usr/local/include/opencv4 -I /usr/local/Cellar/tbb/2019_U3_1/include
CXXFLAGS   := -o3#e.g., CXXFLAGS=-g -O0 for debugging
LDFLAGS    := -I /usr/local/include/opencv4 -I /usr/local/Cellar/tbb/2019_U3_1/include
LDLIBS     := -L /usr/local/Cellar/tbb/2019_U3_1/lib -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_videoio -lopencv_imgproc -lopencv_features2d -lopencv_calib3d -pthread 


# Rules for building
all: $(NAME)

$(NAME): $(NAME).o
	$(LD) $(LDFLAGS) -o $@ $^ $(LDLIBS)
# 	$(LD) $(LDFLAGS) -o $@ $^ $(LDLIBS) jetsonGPIO.c

$(NAME).o: $(NAME).cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c -o $@ $<

clean:
	$(RM) $(NAME).o $(NAME)

debug: 
	@echo  "$(OPT_IMAGEPATH)$(OPT_IMAGEPATH)" ;

genimagelist:
	@echo "Imagelist generated see file [imagelist_appended.txt]..." 	;\
	echo "for images at $(OPT_IMAGEPATH) ..."			;\
	$(generate_imagelist_le)	;\
	$(generate_imagelist_te)
