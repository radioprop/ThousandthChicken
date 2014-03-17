#pragma once


#include "../../types/image_types.h"

#define MAX_CODESTREAM_SIZE (4096 * 2) /// TODO: figure out

#define LL_LH_SUBBAND	0
#define HL_SUBBAND		1
#define HH_SUBBAND		2


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

	unsigned char *codeStream;
	int length;
	type_codeblock *cblk;
} EntropyCodingTaskInfo;


