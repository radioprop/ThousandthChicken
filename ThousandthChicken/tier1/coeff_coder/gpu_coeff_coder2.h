#pragma once

typedef struct
{
	int length;
	unsigned char significantBits;
	unsigned char codingPasses;
	unsigned char width;
	unsigned char nominalWidth;
	unsigned char height;
	unsigned char nominalHeight;
	unsigned char stripeNo;
	unsigned char magbits;
	unsigned char subband;
	unsigned char compType;
	unsigned char dwtLevel;
	float stepSize;

	int magconOffset;

	int gpuCoefficientsOffset;
	int gpuCodestreamOffset;
	int gpuSTBufferOffset;
} CodeBlockAdditionalInfo;

