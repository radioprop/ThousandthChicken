#pragma once


#define INIT_BUF_SIZE 128

typedef struct type_buffer type_buffer;

struct type_buffer {
	unsigned char *data;
	unsigned int byte;
	unsigned int bytes_count;
	int bits_count;
	unsigned long int size;
	unsigned char *bp;
	unsigned char *start;
	unsigned char *end;
};

