/*
 * DiscreteWaveletTansform.cpp
 *
 *  Created on: Mar 8, 2014
 *      Author: aaron
 */

#include "DiscreteWaveletTansform.h"

#include <math.h>
#include <stdlib.h>

#define DWT53	0
#define DWT97	1

/**
 * @brief Perform the inverse wavelet transform on a 2D matrix
 *
 * We assume that top left coordinates u0 and v0 input tile matrix are both even.See Annex F of ISO/EIC IS 15444-1.
 *
 * @param filter Kind of wavelet 53 | 97.
 * @param d_idata Input array.
 * @param d_odata Output array.
 * @param img_size Input image size.
 * @param step Output image size.
 * @param nlevels Number of levels.
 */
type_data *iwt_2d(short filter, type_tile_comp *tile_comp) {
	int i;
	int *sub_x, *sub_y ;
	/* Input data */
	type_data *d_idata = tile_comp->img_data_d;
	/* Result data */
	type_data *d_odata = NULL;

#ifdef CUDA
	/* Image data size */
	const unsigned int smem_size = sizeof(type_data) * tile_comp->width * tile_comp->height;
	cuda_d_allocate_mem((void**) &d_odata, smem_size);
	cudaMemset(d_odata, 0, smem_size);

	int2 img_size = make_int2(tile_comp->width, tile_comp->height);
	int2 step = make_int2(tile_comp->width, tile_comp->height);

	sub_x = (int *)malloc((tile_comp->num_dlvls - 1) * sizeof(int));
	sub_y = (int *)malloc((tile_comp->num_dlvls - 1) * sizeof(int));

	for(i = 0; i < tile_comp->num_dlvls - 1; i++) {
		sub_x[i] = (img_size.x % 2 == 1) ? 1 : 0;
		sub_y[i] = (img_size.y % 2 == 1) ? 1 : 0;
		img_size.y = (int)ceil(img_size.y/2.0);
		img_size.x = (int)ceil(img_size.x/2.0);
	}

	for (i = 0; i < tile_comp->num_dlvls; i++) {
		/* Number of all thread blocks */
		dim3 grid_size = dim3((img_size.x + (PATCHX - 1)) / (PATCHX), (img_size.y
				+ (PATCHY - 1)) / (PATCHY), 1);

//		printf("OK gridx %d, ..gridy %d img_size.x %d img_size.y %d\n", grid_size.x, grid_size.y, img_size.x, img_size.y);

		switch(filter)
		{
			case DWT97:
				iwt97<<<grid_size, dim3(BLOCKSIZEX, BLOCKSIZEY)>>>(d_idata, d_odata, img_size, step);
				break;
			case DWT53:
				iwt53_new<<<grid_size, dim3(BLOCKSIZEX, BLOCKSIZEY)>>>(d_idata, d_odata, img_size, step);
				break;
		}

//		if(tile_comp->tile_comp_no == 0)
//		{
//			char tmp_name[128] = {0};
//			tile_comp->img_data_d = d_idata;
//			sprintf(tmp_name, "dec_idwt_%d.bmp", i);
//			save_img_grayscale(tile_comp->parent_tile->parent_img, tmp_name);
//			tile_comp->img_data_d = d_odata;
//			sprintf(tmp_name, "dec_odwt_%d.bmp", i);
//			save_img_grayscale(tile_comp->parent_tile->parent_img, tmp_name);
//		}

		cudaError_t error;

		// error report
		if(error = cudaGetLastError())
			printf("%s\n", cudaGetErrorString(error));

		/* Copy data between buffers to save previous results */
		if ((tile_comp->num_dlvls - 1) != i) {
//			cudaMemcpy(d_idata, d_odata, smem_size, cudaMemcpyDeviceToDevice);
			cuda_memcpy2d_dtd(d_odata, tile_comp->width * sizeof(type_data), d_idata, tile_comp->width * sizeof(type_data), img_size.x * sizeof(type_data), img_size.y);

			img_size.x = img_size.x * 2 - sub_x[tile_comp->num_dlvls - 2 - i];
			img_size.y = img_size.y * 2 - sub_y[tile_comp->num_dlvls - 2 - i];
		}
	}

	free(sub_x);
	free(sub_y);

	cuda_d_free(d_idata);
#endif

	return d_odata;
}

void DiscreteWaveletTansform::iwt(type_tile *tile)
{
//	println_start(INFO);
	int i;
	type_tile_comp *tile_comp;
	type_image *img = tile->parent_img;

//	save_img(img, "dec_dwt_before.bmp");

	/* Process components from tile */
	for(i = 0; i < tile->parent_img->num_components; i++)
	{
		tile_comp = &(tile->tile_comp[i]);

		/* Do IWT on image data. Lossy. */
		if(img->wavelet_type)
		{
			tile_comp->img_data_d = iwt_2d(DWT97, tile_comp);
		} else /* Lossless */
		{
			tile_comp->img_data_d = iwt_2d(DWT53, tile_comp);
		}
	}
//	save_img(img, "dec_dwt_after.bmp");
//	println_end(INFO);
}



/**
 * @brief Perform the inverse wavelet transform on a 2D matrix
 *
 * We assume that top left coordinates u0 and v0 input tile matrix are both even.See Annex F of ISO/EIC IS 15444-1.
 *
 * @param filter Kind of wavelet 53 | 97.
 * @param d_idata Input array.
 * @param d_odata Output array.
 * @param img_size Input image size.
 * @param step Output image size.
 * @param nlevels Number of levels.
 */
type_data* DiscreteWaveletTansform::iwt_2d(short filter, type_tile_comp *tile_comp) {
	int i;
	int *sub_x, *sub_y ;
	/* Input data */
	type_data *d_idata = tile_comp->img_data_d;
	/* Result data */
	type_data *d_odata = NULL;
	/* Image data size */
	const unsigned int smem_size = sizeof(type_data) * tile_comp->width * tile_comp->height;

#ifdef CUDA
	int2 img_size, step;
	cuda_d_allocate_mem((void**) &d_odata, smem_size);
	cudaMemset(d_odata, 0, smem_size);

	img_size = make_int2(tile_comp->width, tile_comp->height);
	step = make_int2(tile_comp->width, tile_comp->height);


	sub_x = (int *)malloc((tile_comp->num_dlvls - 1) * sizeof(int));
	sub_y = (int *)malloc((tile_comp->num_dlvls - 1) * sizeof(int));

	for(i = 0; i < tile_comp->num_dlvls - 1; i++) {
		sub_x[i] = (img_size.x % 2 == 1) ? 1 : 0;
		sub_y[i] = (img_size.y % 2 == 1) ? 1 : 0;
		img_size.y = (int)ceil(img_size.y/2.0);
		img_size.x = (int)ceil(img_size.x/2.0);
	}

	for (i = 0; i < tile_comp->num_dlvls; i++) {
		/* Number of all thread blocks */
		dim3 grid_size = dim3((img_size.x + (PATCHX - 1)) / (PATCHX), (img_size.y
				+ (PATCHY - 1)) / (PATCHY), 1);

//		printf("OK gridx %d, ..gridy %d img_size.x %d img_size.y %d\n", grid_size.x, grid_size.y, img_size.x, img_size.y);
		switch(filter)
		{
			case DWT97:
				iwt97<<<grid_size, dim3(BLOCKSIZEX, BLOCKSIZEY)>>>(d_idata, d_odata, img_size, step);
				break;
			case DWT53:
				iwt53_new<<<grid_size, dim3(BLOCKSIZEX, BLOCKSIZEY)>>>(d_idata, d_odata, img_size, step);
				break;
		}

//		if(tile_comp->tile_comp_no == 0)
//		{
//			char tmp_name[128] = {0};
//			tile_comp->img_data_d = d_idata;
//			sprintf(tmp_name, "dec_idwt_%d.bmp", i);
//			save_img_grayscale(tile_comp->parent_tile->parent_img, tmp_name);
//			tile_comp->img_data_d = d_odata;
//			sprintf(tmp_name, "dec_odwt_%d.bmp", i);
//			save_img_grayscale(tile_comp->parent_tile->parent_img, tmp_name);
//		}

		cudaError_t error;

		// error report
		if(error = cudaGetLastError())
			printf("%s\n", cudaGetErrorString(error));

		/* Copy data between buffers to save previous results */
		if ((tile_comp->num_dlvls - 1) != i) {
//			cudaMemcpy(d_idata, d_odata, smem_size, cudaMemcpyDeviceToDevice);
			cuda_memcpy2d_dtd(d_odata, tile_comp->width * sizeof(type_data), d_idata, tile_comp->width * sizeof(type_data), img_size.x * sizeof(type_data), img_size.y);

			img_size.x = img_size.x * 2 - sub_x[tile_comp->num_dlvls - 2 - i];
			img_size.y = img_size.y * 2 - sub_y[tile_comp->num_dlvls - 2 - i];
		}
	}

	free(sub_x);
	free(sub_y);
	cuda_d_free(d_idata);
#endif
	return d_odata;
}


