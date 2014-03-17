
#include <stdlib.h>
#include <stdio.h>
#include <string.h>


#include "config/help.h"
#include "config/parameters.h"
#include "preprocessing/Preprocessor.h"
#include <stdio.h>
#include "types/image_types.h"
#include "types/buffered_stream.h"
#include "types/image.h"

#include "print_info/print_info.h"

#include "dwt/DiscreteWaveletTansform.h"
#include "CoefficientCoderGPU.h"

#include "tier1/Dequant.h"
#include <stdio.h>


#include "tier2/codestream.h"
#include "tier2/buffer.h"

#include "file_format/boxes.h"
#include "basic.hpp"


/**
 * @brief Main decoder function. It all starts here.
 * Input parameters:
 * 1) input JPEG 2000 image
 *
 * @return 0 on success
 */
int decode(OpenCLBasic& oclObjects)
{
//	println_start(INFO);
	type_image *img = (type_image *)malloc(sizeof(type_image));
	memset(img, 0, sizeof(type_image));
	img->in_file = "c:\\src\\openjpeg-data\\input\\conformance\\file1.jp2";
	type_parameters *param = (type_parameters*)malloc(sizeof(type_parameters));
	default_config_values(param);
	//init_device(param);

	FILE *fsrc = fopen(img->in_file, "rb");
	if (!fsrc) {
		fprintf(stderr, "Error, failed to open %s for reading\n", img->in_file);
		return 1;
	}

	type_tile *tile;
	unsigned int i;
	DiscreteWaveletTansform dwt;
	Dequant dequant;
	Preprocessor preprocessor;
	CoefficientCoderGPU coefficientCoder(oclObjects);

	if(strstr(img->in_file, ".jp2") != NULL) {
		println(INFO, "It's a JP2 file");

		//parse the JP2 boxes
		jp2_parse_boxes(fsrc, img);
		fclose(fsrc);

		// Do decoding for all tiles
		for(i = 0; i < img->num_tiles; i++) {
			tile = &(img->tile[i]);
			/* Decode data */
			coefficientCoder.decode_tile(tile);
			/* Dequantize data */
			dequant.dequantize_tile(tile);
			/* Do inverse wavelet transform */
			dwt.iwt(tile);
		}

		if(img->use_mct == 1) {
			// lossless decoder
			if(img->wavelet_type == 0) {
				preprocessor.color_decoder_lossless(img);
			}
			else { //lossy decoder
				preprocessor.color_decoder_lossy(img);
			}
		} else if (img->use_part2_mct == 1) {
			//klt.decode_klt(img);
			//part 2 not supported
		} else {
			if(img->sign == UNSIGNED) {
				preprocessor.idc_level_shifting(img);
			}
		}
	}
	else {//It is not a JP2 file.
		type_buffer *src_buff = (type_buffer *) malloc(sizeof(type_buffer));

		init_dec_buffer(fsrc, src_buff);
		fclose(fsrc);

		decode_codestream(src_buff, img);

	//	get_next_box(fsrc);

		// Do decoding for all tiles
		for(i = 0; i < img->num_tiles; i++)	{
			tile = &(img->tile[i]);
			/* Decode data */
			coefficientCoder.decode_tile(tile);
			/* Dequantize data */
			dequant.dequantize_tile(tile);
			/* Do inverse wavelet transform */
			dwt.iwt(tile);
		}

		if(img->use_mct == 1) {
			// lossless decoder
			if(img->wavelet_type == 0) {
				preprocessor.color_decoder_lossless(img);
			}
			else {  //lossy decoder
				preprocessor.color_decoder_lossy(img);
			}
		} else if (img->use_part2_mct == 1) {
			//klt.decode_klt(img);
			//part 2 not supported
		} else {
			if(img->sign == UNSIGNED) {
				preprocessor.idc_level_shifting(img);
			}
		}
	}
	return 0;
}
