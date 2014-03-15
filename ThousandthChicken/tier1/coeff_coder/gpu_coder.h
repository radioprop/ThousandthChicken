#ifndef __gpu_coder_h__
#define __gpu_coder_h__


#include "../../types/image_types.h"

#ifdef __CDT_PARSER__
#define __global__
#define __device__
#define __shared__
#endif

#define MAX_CODESTREAM_SIZE (4096 * 2) /// TODO: figure out

#define LL_LH_SUBBAND	0
#define HL_SUBBAND		1
#define HH_SUBBAND		2

typedef unsigned char byte;

typedef struct
{
	int significantBits;
	int codingPasses;
	int *coefficients;
	int *h_coefficients;
	int subband;
	int width;
	int height;

	int nominalWidth;
	int nominalHeight;

	int magbits;
	int compType;
	int dwtLevel;
	float stepSize;

	byte *codeStream;
	int length;
	type_codeblock *cblk;
} EntropyCodingTaskInfo;


extern void decode_tile(type_tile *tile);


#endif
