/*
 * Preprocessor.h
 *
 *  Created on: Mar 8, 2014
 *      Author: aaron
 */

#ifndef PREPROCESSOR_H_
#define PREPROCESSOR_H_

/* cuda block is a square with a side of BLOCK_SIZE. Actual number of threads in the block is the square of this value*/
#define BLOCK_SIZE 16
#define TILE_SIZEX 32
#define TILE_SIZEY 32

typedef enum {
	RCT, ///Reversible Color Transformation. Encoder part of the lossless flow.
	TCR, ///Decoder part of the lossless flow.
	ICT, ///Irreversible Color Transformation. Encoder part of the lossy flow.
	TCI  ///Decoder part of the lossy flow.
} color_trans_type;

#include "../types/image_types.h"


class Preprocessor {
public:
	void idc_level_shifting(type_image *img);
	int color_decoder_lossy(type_image *img);
	int color_decoder_lossless(type_image *img);

private:
	void dc_level_shifting(type_image *img, int sign);

};

#endif /* PREPROCESSOR_H_ */
