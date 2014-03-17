#ifndef BUFFERED_STREAM_H_
#define BUFFERED_STREAM_H_


#include <stdio.h>

#include "buffered_stream_type.h"

#define INIT_BUF_SIZE 128


void init_buffer(type_buffer *buffer);
void enlarge_buffer(type_buffer *buffer);
void seek_buffer(type_buffer *buffer, int pos);
void skip_buffer(type_buffer *buffer, int n);
int tell_buffer(type_buffer *buffer);
void write_byte(type_buffer *buffer, unsigned char val);
void write_short(type_buffer *buffer, unsigned short val);
void write_int(type_buffer *buffer, unsigned int val);
void bit_stuffing(type_buffer *buffer);
void write_stuffed_byte(type_buffer *buffer);
void write_zero_bit(type_buffer *buffer);
void write_one_bit(type_buffer *buffer);
void write_bits(type_buffer *buffer, int bits, int n);
void write_array(type_buffer *buffer, unsigned char *in, int length);
void update_buffer_byte(type_buffer *buffer, int pos, unsigned char val);
void write_buffer_to_file(type_buffer *buffer, FILE *fp);
unsigned int read_buffer(type_buffer *buffer, int n);
unsigned int read_bits(type_buffer *buffer, int n);
unsigned int inalign(type_buffer *buffer);
unsigned short peek_marker(type_buffer *buffer);
unsigned char read_byte(type_buffer *buffer);

#endif /* BUFFERED_STREAM_H_ */
