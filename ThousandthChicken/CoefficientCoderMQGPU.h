#pragma once


typedef struct MQEncoder
{
	short L;

	unsigned short A;
	unsigned int C;
	unsigned char CT;
	unsigned char T;

	__global unsigned char *outbuf;

	int CXMPS;
	unsigned char CX;

	unsigned int Ib0;
	unsigned int Ib1;
	unsigned int Ib2;
	unsigned int Ib3;
	unsigned int Ib4;
	unsigned int Ib5;
} MQEncoder;

typedef struct MQDecoder
{
	MQEncoder encoder;
	unsigned char NT;
	int Lmax;
} MQDecoder;