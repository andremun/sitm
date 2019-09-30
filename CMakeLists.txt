cmake_minimum_required(VERSION 2.8.3) 

project(metafeature)

#set( CMAKE_VERBOSE_MAKEFILE on )

message("CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")

# CHECK
if( CMAKE_BINARY_DIR STREQUAL CMAKE_SOURCE_DIR )
    message( FATAL_ERROR "Please select another Build Directory" )
endif()
if( CMAKE_SOURCE_DIR MATCHES " " )
	message( "Your Source Directory contains spaces. If you experience problems when compiling, this can be the cause." )
endif()
if( CMAKE_BINARY_DIR MATCHES " " )
	message( "Your Build Directory contains spaces. If you experience problems when compiling, this can be the cause." )
endif()

#include(FindPkgConfig)

#Build on MASSIVE?
option (MASSIVE "Build on MASSIVE" OFF)

#OpenCV
if(MASSIVE)
  include_directories(/home/toand/dev/opencv-2.4.12/build/include)
  link_directories(/home/toand/dev/opencv-2.4.12/build/lib)
else()
  find_package( OpenCV REQUIRED )
  include_directories( ${OpenCV_INCLUDE_DIRS} )
endif()

#opencv 3
#include_directories(/usr/local/opt/opencv3/include)
#link_directories(/usr/local/opt/opencv3/lib)

#OpenMP
option (USE_OpenMP "Use OpenMP" OFF)
if(USE_OpenMP)
  find_package(OpenMP REQUIRED)
  if(OPENMP_FOUND)
    add_definitions(-DUSE_OpenMP)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  endif()
endif()

#CUDA
option (USE_CUDA "Use CUDA" OFF)
if(USE_CUDA)
  find_package(CUDA REQUIRED)
  if(CUDA_FOUND)
    add_definitions(-DUSE_CUDA)
    include_directories ( ${CUDA_INCLUDE_DIRS} )
  endif()
endif()

# Set output paths
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

#file (COPY ${CMAKE_SOURCE_DIR}/data/ DESTINATION ${CMAKE_BINARY_DIR}/bin )

include_directories(
	src/
  src/cuda/
)

if(MASSIVE)
  set(ALL_LIBS
    -lopencv_core
    -lopencv_ml
    -pthread
    -lz
    -lrt
  )
else()
  set(ALL_LIBS
    #${OpenCV_LIBS}
    -lopencv_core
    -lopencv_ml
  )
endif()

add_subdirectory (src)