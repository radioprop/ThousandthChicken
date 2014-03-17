#ifndef CODESTREAM_H_
#define CODESTREAM_H_

#include <assert.h>

#include "../types/image_types.h"
#include "../types/buffered_stream.h"

typedef struct type_packet type_packet;

/** Packet parameters */
struct type_packet{
	unsigned short *inclusion;
	unsigned short *zero_bit_plane;
	unsigned short *num_coding_passes;
};

void decode_codestream(type_buffer *buffer, type_image *img);


#endif /* CODESTREAM_H_ */
