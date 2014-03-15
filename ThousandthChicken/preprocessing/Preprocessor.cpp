/*
 * Preprocessor.cpp
 *
 *  Created on: Mar 8, 2014
 *      Author: aaron
 */

#include "Preprocessor.h"

#include <stdio.h>


#include "../types/image_types.h"
#include "../print_info/print_info.h"

#include "preprocessing_constants.h"


/**
 * @brief Main function of color transformation flow. Should not be called directly though. Use four wrapper functions color_[de]coder_loss[y|less] instead.
 *
 * @param img type_image to will be transformed.
 * @param type Type of color transformation that should be performed. The types are detailed in color_trans_type.
 *
 *
 * @return Returns 0 on sucess.
 */
int color_trans_gpu(type_image *img, color_trans_type type) {
	if(img->num_components != 3) {
		println(INFO, "Error: Color transformation not possible. The number of components != 3.");
		//exit(0);
		return -1;
	}

	//CUDA timing apparatus
	#ifdef COMPUTE_TIME
		cudaEvent_t kernel_start, kernel_stop;

		cudaEventCreate(&kernel_start);
		cudaEventCreate(&kernel_stop);
	#endif

	uint32_t tile_size = 0, i;
	type_tile *tile;
	type_data *comp_a, *comp_b, *comp_c;

	int level_shift = img->num_range_bits - 1;

	int min = img->sign == SIGNED ? -(1 << (img->num_range_bits - 1)) : 0;
	int max = img->sign == SIGNED ? (1 << (img->num_range_bits - 1)) - 1 : (1 << img->num_range_bits) - 1;

	switch(type) {
	case RCT:
//		println_var(INFO, "start: RCT");
		#ifdef COMPUTE_TIME
			cudaEventRecord(kernel_start, 0);
		#endif

		for(i = 0; i < img->num_tiles; i++) {
			tile = &(img->tile[i]);
			comp_a = (&(tile->tile_comp[0]))->img_data_d;
			comp_b = (&(tile->tile_comp[1]))->img_data_d;
			comp_c = (&(tile->tile_comp[2]))->img_data_d;
#ifdef CUDA
			dim3 dimGrid((tile->width + (TILE_SIZEX - 1))/TILE_SIZEX, (tile->height + (TILE_SIZEY - 1))/TILE_SIZEY);
			dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

//			printf("%d\n", level_shift);
/*			int blocks = ( tile_size / (BLOCK_SIZE * BLOCK_SIZE)) + 1;
			dim3 dimGrid(blocks);
			dim3 dimBlock(BLOCK_SIZE* BLOCK_SIZE);*/

			rct_kernel<<<dimGrid, dimBlock>>>( comp_a, comp_b, comp_c, tile->width, tile->height, level_shift );
#endif
		}

		#ifdef COMPUTE_TIME
			cudaEventRecord(kernel_stop, 0);
			cudaEventSynchronize( kernel_stop );
		#endif
		break;

	case TCR:
//		println_var(INFO, "start: TCR");
		#ifdef COMPUTE_TIME
			cudaEventRecord(kernel_start, 0);
		#endif

			for(i = 0; i < img->num_tiles; i++) {
				tile = &(img->tile[i]);
				comp_a = (&(tile->tile_comp[0]))->img_data_d;
				comp_b = (&(tile->tile_comp[1]))->img_data_d;
				comp_c = (&(tile->tile_comp[2]))->img_data_d;

				int blocks = ( tile_size / (BLOCK_SIZE * BLOCK_SIZE)) + 1;
#ifdef CUDA
				dim3 dimGrid(blocks);
				dim3 dimBlock(BLOCK_SIZE* BLOCK_SIZE);

				tcr_kernel<<< dimGrid, dimBlock, 0>>>( comp_a, comp_b, comp_c, tile->width, tile->height, level_shift, min, max);
#endif
			}

		#ifdef COMPUTE_TIME
			cudaEventRecord(kernel_stop, 0);
			cudaEventSynchronize( kernel_stop );
		#endif
		break;

	case ICT:
//		println_var(INFO, "start: ICT");
		#ifdef COMPUTE_TIME
			cudaEventRecord(kernel_start, 0);
		#endif

		for(i = 0; i < img->num_tiles; i++) {
			tile = &(img->tile[i]);
			comp_a = (&(tile->tile_comp[0]))->img_data_d;
			comp_b = (&(tile->tile_comp[1]))->img_data_d;
			comp_c = (&(tile->tile_comp[2]))->img_data_d;
#ifdef CUDA
			dim3 dimGrid((tile->width + (TILE_SIZEX - 1))/TILE_SIZEX, (tile->height + (TILE_SIZEY - 1))/TILE_SIZEY);
			dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
			int level_shift = img->num_range_bits - 1;
			/*int blocks = ( tile_size / (BLOCK_SIZE * BLOCK_SIZE)) + 1;
			dim3 dimGrid(blocks);
			dim3 dimBlock(BLOCK_SIZE* BLOCK_SIZE);*/

			ict_kernel<<< dimGrid, dimBlock>>>( comp_a, comp_b, comp_c, tile->width, tile->height, level_shift );
#endif
		}

		#ifdef COMPUTE_TIME
			cudaEventRecord(kernel_stop, 0);
			cudaEventSynchronize( kernel_stop );
		#endif
		break;

	case TCI:
//		println_var(INFO, "start: TCI");
		#ifdef COMPUTE_TIME
		cudaEventRecord(kernel_start, 0);
		#endif


		for(i = 0; i < img->num_tiles; i++) {
			tile = &(img->tile[i]);
			tile_size = tile->width * tile->height;
			comp_a = (&(tile->tile_comp[0]))->img_data_d;
			comp_b = (&(tile->tile_comp[1]))->img_data_d;
			comp_c = (&(tile->tile_comp[2]))->img_data_d;

			int blocks = ( tile_size / (BLOCK_SIZE * BLOCK_SIZE)) + 1;
#ifdef CUDA
			dim3 dimGrid(blocks);
			dim3 dimBlock(BLOCK_SIZE* BLOCK_SIZE);

			tci_kernel<<< dimGrid, dimBlock, 0>>>( comp_a, comp_b, comp_c, tile->width, tile->height, level_shift, min, max);
#endif
		}

		#ifdef COMPUTE_TIME
			cudaEventRecord(kernel_stop, 0);
			cudaEventSynchronize( kernel_stop );
		#endif
		break;

	}
#ifdef CUDA
	checkCUDAError("color_trans_gpu");
#endif
	float kernel_time;
	#ifdef COMPUTE_TIME
		cudaEventElapsedTime( &kernel_time, kernel_start, kernel_stop );
		cudaEventDestroy( kernel_start );
		cudaEventDestroy( kernel_stop );
		printf("\t\tkernel: %.2f [ms]\n", kernel_time);
	#endif
//	println_end(INFO);
	return 0;
}


/**
 * Lossy color transformation YCbCr -> RGB. Decoder of the color_coder_lossy output.
 *
 * @param img Image to be color-transformated
 * @return Returns the color transformated image. It's just the pointer to the same structure passed in img parameter.
 */
int Preprocessor::color_decoder_lossy(type_image *img) {
	return color_trans_gpu(img, TCI);
}


/**
 * Lossless color transformation YUV -> RGB. Decoder of the color_coder_lossless output.
 *
 * @param img Image to be color-transformated
 * @return Returns the color transformated image. It's just the pointer to the same structure passed in img parameter.
 */
int Preprocessor::color_decoder_lossless(type_image *img) {
	return color_trans_gpu(img, TCR);
}

void Preprocessor::dc_level_shifting(type_image *img, int sign)
{
	uint32_t i, j;
	type_tile *tile;
	type_data *idata;
	int min = 0;
	int max = (1 << img->num_range_bits) - 1;

//	start_measure();

	for(i = 0; i < img->num_tiles; i++)
	{
		tile = &(img->tile[i]);
		for(j = 0; j < img->num_components; j++)
		{
			idata = (&(tile->tile_comp[j]))->img_data_d;
#ifdef CUDA
			dim3 dimGrid((tile->width + (TILE_SIZEX - 1))/TILE_SIZEX, (tile->height + (TILE_SIZEY - 1))/TILE_SIZEY);
			dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
			int level_shift = img->num_range_bits - 1;

			if(sign < 0)
			{
				fdc_level_shift_kernel<<<dimGrid, dimBlock>>>( idata, tile->width, tile->height, level_shift);
			} else
			{
				idc_level_shift_kernel<<<dimGrid, dimBlock>>>( idata, tile->width, tile->height, level_shift, min, max);
			}

			cudaThreadSynchronize();

			checkCUDAError("dc_level_shifting");
#endif
		}
	}

//	stop_measure(INFO);
}

/**
 * @brief Inverse DC level shifting.
 * @param img
 * @param type
 */
void Preprocessor::idc_level_shifting(type_image *img)
{
	dc_level_shifting(img, 1);
}

