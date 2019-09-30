if(USE_CUDA)

	SET( srcs 
					cuda/dmat.cu
	                Data.cpp
	                main.cpp
			)

	cuda_add_executable( sitm ${srcs} )
	target_link_libraries( sitm ${ALL_LIBS} ${CUDA_LIBRARIES} )

else()

	SET( srcs 
	                Data.cpp
	                main.cpp
			)

	add_executable( sitm ${srcs} )
	target_link_libraries( sitm ${ALL_LIBS} )

endif()


