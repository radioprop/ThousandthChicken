#include <stdlib.h>
#include <string.h>
#include "codestream.h"
#include "tag_tree_encode.h"
#include "markers.h"
#include "../types/image_types.h"
#include "../config/parameters.h"
#include "../print_info/print_info.h"
#include "codestream_mct.h"
#include "../types/image.h"
#include "../types/buffered_stream.h"


void read_siz_marker(type_buffer *buffer, type_image *img)
{
	int length, i;
	int marker;
	int rsiz;

	/* Read SIZ marker */
	marker = read_buffer(buffer, 2);

	if(marker != SIZ)
	{
		println_var(INFO, "Error: Expected SIZ marker instead of %x", marker);
	}

	/* Allocate coding parameters */
	img->coding_param = (type_coding_param *) malloc(sizeof(type_coding_param));

	/* Lsiz */
	length = read_buffer(buffer, 2);
	/* Rsiz */
	rsiz = read_buffer(buffer, 2);
	/* Chceking Rsiz for MTC as in 15444-2 standard */
	if((rsiz & (1<<15)) && (rsiz & (1<<7)) && !(rsiz & 1)) {
		img->use_part2_mct = 1;
	}
	/* Xsiz - original image width */
	img->width = read_buffer(buffer, 4);
	/* Ysiz - original image height */
	img->height = read_buffer(buffer, 4);
	/* XOsiz - horizontal offset from the origin of the reference grid to the left side of the image area */
	img->coding_param->imgarea_tlx = read_buffer(buffer, 4);
	/* YOsiz - vertical offset from the origin of the reference grid to the top side of the image area */
	img->coding_param->imgarea_tly = read_buffer(buffer, 4);
	/* XTsiz - nominal tile width */
	img->tile_w = read_buffer(buffer, 4);
	/* YTsiz - nominal tile height*/
	img->tile_h = read_buffer(buffer, 4);
	/* XTOsiz - horizontal offset from the origin of the reference grid to the left side of the first tile */
	img->coding_param->tilegrid_tlx = read_buffer(buffer, 4);
	/* YTOsiz - vertical offset from the origin of the reference grid to the top side of the first tile */
	img->coding_param->tilegrid_tly = read_buffer(buffer, 4);
	/* Csiz - number of components */
	img->num_components = read_buffer(buffer, 2);

	/* For every component */
	for (i = 0; i < img->num_components; i++) {
		int tmp;
		/* Ssiz - bit depth and sign of the component samples */
		tmp = read_buffer(buffer, 1);
		img->num_range_bits = (tmp & 0x7f) +1;
		img->sign = (tmp >> 7) == 0 ? UNSIGNED : SIGNED;
		/* XRsiz - component sub-sampling value x-wise */
		img->coding_param->comp_step_x = read_buffer(buffer, 1);
		/* YRsiz - component sub-sampling value x-wise */
		img->coding_param->comp_step_y = read_buffer(buffer, 1);
	}

	img->depth = img->num_components * img->num_range_bits;
	img->num_xtiles = (img->width + (img->tile_w - 1)) / img->tile_w;
	img->num_ytiles = (img->height + (img->tile_h - 1)) / img->tile_h;
	img->num_tiles = img->num_xtiles * img->num_ytiles;

	img->coding_param->imgarea_height = img->height;
	img->coding_param->imgarea_width = img->width;
	img->coding_param->base_step = 1.0 / (float)(1 << (img->num_range_bits - 1));
}

void read_cod_marker(type_buffer *buffer, type_image *img)
{
	int length, pos, i;
	type_parameters *param = (type_parameters *)malloc(sizeof(type_parameters));
	int marker;

	/* Read COD marker */
	marker = read_buffer(buffer, 2);

	if(marker != COD)
	{
		println_var(INFO, "Error: Expected COD marker instead of %x", marker);
	}

	/* Lcod - 12 or 13 + num of decomposition levels bytes. XXX: We assume no precinct partition. */
	length = read_buffer(buffer, 2);
	/* Scod - coding style parameter. XXX: Entropy coder with PPx=15 and PPy=15 and EPH marker shall be used */
	img->coding_style = read_buffer(buffer, 1);
	/* SGcod */
	/* Progression order */
	img->prog_order = read_buffer(buffer, 1);
	/* Number of layers */
	img->num_layers = read_buffer(buffer, 2);
	/* Multiple component transform */
	/* If there are 3 components, multiple component transform is used */
	img->use_mct = read_buffer(buffer, 1);
	/* SPcod */
	/* Number of decomposition levels */
	img->num_dlvls = read_buffer(buffer, 1);
	/* Code-block width and height. TODO: Check */
	param->param_cblk_exp_w = read_buffer(buffer, 1) + 2;
	param->param_cblk_exp_h = read_buffer(buffer, 1) + 2;
	/* Style of the code-block coding passes */
	/* XXX: Only no selective arithmetic coding bypass */
	img->cblk_coding_style = read_buffer(buffer, 1);
	/* Wavelet transform */
	img->wavelet_type = read_buffer(buffer, 1) == 0 ? DWT_97 : DWT_53;

	init_tiles(&img, param);
	/* TODO: In future read precinct partition */
}

/**
 * @brief Currently do nothing.
 *
 * @param buffer
 * @param img
 */
void read_coc_marker(type_buffer *buffer, type_image *img)
{
	/* XXX:Currently we do not use component specific parameters */
}

void read_qcd_marker(type_buffer * buffer, type_image *img)
{
	int length, i, j, k, l, pos, tmp;
	int num_bands;
	int marker;
	type_tile_comp *tile_comp;

	/* Read SOC marker */
	marker = read_buffer(buffer, 2);

	if(marker != QCD)
	{
		println_var(INFO, "Error: Expected QCD marker instead of %x", marker);
	}

	/* Lqcd - 4 + 3 * num_dlvls - no quantization
	 * 		- 5 - derived quantization
	 * 		- 5 + 6 * num_dlvls - expounded quantization */
	/* XXX: Currently we use only derived quantization */
	length = read_buffer(buffer, 2);

	pos = tell_buffer(buffer);

	for(i = 0; i < img->num_tiles; i++)
	{
		for(j = 0; j < img->num_components; j++)
		{
			tile_comp = &(img->tile[i].tile_comp[j]);
			seek_buffer(buffer, pos);
			/* Sqcd - quantization style for all components*/
			tmp = read_buffer(buffer, 1);
			tile_comp->quant_style = tmp & 0x1f;
			tile_comp->num_guard_bits = tmp >> 5;
			if((tile_comp->quant_style != 0) && (tile_comp->quant_style != 1))
			{
				println_var(INFO, "Error: Unsupported quantization style: %d!", tile_comp->quant_style);
			}

			int expn, mant;

			if(tile_comp->quant_style == 0) /* No quantization */
			{
				for (k = 0; k < tile_comp->num_rlvls; k++) {
					type_res_lvl *res_lvl = &(tile_comp->res_lvls[k]);
					for (l = 0; l < res_lvl->num_subbands; l++) {
						type_subband *sb = &(res_lvl->subbands[l]);

						expn = read_buffer(buffer, 1) >> SQCX_EXP_SHIFT;
						mant = 0;

						sb->expn = expn;
						sb->mant = mant;
					}
				}
			} else /* Derived quantization */
			{
				tmp = read_buffer(buffer, 2);
				expn = tmp >> 11;
				mant = tmp & 0x7ff;

				type_subband *sb_ll = &(tile_comp->res_lvls[0].subbands[0]);

				sb_ll->expn = expn;
				sb_ll->mant = mant;

				sb_ll->step_size = (-1.0f - ((float) (sb_ll->mant)) / (1 << 11))
									/ (-1 << sb_ll->expn);

//				printf("%d %d\n", sb_ll->expn, sb_ll->mant);
//				printf("%f\n", sb_ll->step_size);

				for (k = 1; k < tile_comp->num_rlvls; k++) {
					type_res_lvl *res_lvl = &(tile_comp->res_lvls[k]);
					for (l = 0; l < res_lvl->num_subbands; l++) {
						type_subband *sb = &(res_lvl->subbands[l]);

						sb->expn = (sb_ll->expn - ((k * 3 + l) - 1)) > 0 ? (sb_ll->expn - ((k * 3 + l) - 1)) : 0;
						sb->mant = sb_ll->mant;

//						printf("%d %d\n", sb->expn, sb->mant);

						sb->step_size = sb_ll->step_size;

//						printf("%f\n", sb->step_size);
					}
				}
			}
		}
	}
}


void read_main_header(type_buffer *buffer, type_image *img)
{
	unsigned int marker;

	/* Read SOC marker */
	marker = read_buffer(buffer, 2);

	if(marker != SOC)
	{
		println_var(INFO, "Error: Expected SOC(%x) marker instead of %x", SOC, marker);
	}

	/* Read SIZ marker */
	read_siz_marker(buffer, img);
	/* Read COD marker */
	read_cod_marker(buffer, img);
	/* Read COC marker */
	read_coc_marker(buffer, img);
	/* Read QCD marker */
	read_qcd_marker(buffer, img);
	/* Read QCC marker. XXX: Currently we ignore QCC marker */
	/* Read POC and COM markers. XXX:Ignore */
	if(img->use_part2_mct) {
		read_multiple_component_transformations(buffer, img);
	}
}

unsigned int read_tile_header(type_buffer *buffer, type_tile *tile)
{
	unsigned int marker;
	int length, tile_no, total_length;
	int tile_part_no, num_tile_parts;

	/* Read SOT marker */
	marker = read_buffer(buffer, 2);

	if(marker != SOT)
	{
		println_var(INFO, "Error: Expected SOT(%x) marker instead of %x", SOT, marker);
	}

	length = read_buffer(buffer, 2);
	tile_no = read_buffer(buffer, 2);

	if(tile_no != tile->tile_no)
	{
		println_var(INFO, "Error: Expected tile number %d instead of %d", tile->tile_no, tile_no);
	}

	total_length = read_buffer(buffer, 4);

	if(!total_length)
	{
		println_var(INFO, "Error: Length of tile part equals 0");
	}

	tile_part_no = read_buffer(buffer, 1);
	num_tile_parts = read_buffer(buffer, 1);

	if((tile_part_no != 0) || (num_tile_parts != 1))
	{
		println_var(INFO, "Error: Currently unsupported more than one tile parts");
	}

	return total_length;
}

int decode_num_coding_passes(type_buffer *buffer)
{
	int n;

	if (!read_bits(buffer, 1))
		return 1;
	if (!read_bits(buffer, 1))
		return 2;
	if ((n = read_bits(buffer, 2)) != 3)
		return (3 + n);
	if ((n = read_bits(buffer, 5)) != 31)
		return (6 + n);
	return (37 + read_bits(buffer, 7));
}

int get_comma_code(type_buffer *buffer)
{
	int n;

	for(n = 0; read_bits(buffer, 1); n++) ;

	return n;
}

/**
 * @brief Get logarithm of an integer and round downwards.
 *
 * @return Returns log2(a)
 */
int int_floorlog2(int a)
{
	int l;
	for (l = 0; a > 1; l++) {
		a >>= 1;
	}
	return l;
}

/**
 * @brief Returns max value from two integers.
 *
 * @param a
 * @param b
 * @return
 */
int int_max(int a, int b)
{
	return (a > b) ? a : b;
}


void decode_packet_header(type_buffer *buffer, type_res_lvl *res_lvl)
{
	int i, j;
	int layer = 0;
	int packet_present;
	type_tile_comp *tile_comp = res_lvl->parent_tile_comp;
	type_image *img = tile_comp->parent_tile->parent_img;
	type_res_lvl *res_lvl_zero = &(tile_comp->res_lvls[0]);
	type_packet *packet;
	type_subband *sb;
	type_codeblock *cblk;
	unsigned int marker;

	for (i = 0; i < res_lvl->num_subbands; i++) {
		sb = &(res_lvl->subbands[i]);
		sb->inc_tt = tag_tree_create(sb->num_xcblks, sb->num_ycblks);
		sb->zero_bit_plane_tt = tag_tree_create(sb->num_xcblks, sb->num_ycblks);
		tag_tree_reset(sb->inc_tt);
		tag_tree_reset(sb->zero_bit_plane_tt);
		for (j = 0; j < sb->num_cblks; j++) {
			cblk = &(sb->cblks[j]);
			cblk->num_segments = 0;
		}
	}

	if(img->coding_style & USED_SOP)
	{
		/* Read SOP marker */
		marker = read_buffer(buffer, 2);

		if(marker != SOP)
		{
			println_var(INFO, "Error: Expected SOP(%x) marker instead of %x", SOP, marker);
		}
	}

	packet_present = read_bits(buffer, 1);

	if(!packet_present)
	{
		println_var(INFO, "Error: Currently empty packets are unsupported");
	}

	for (i = 0; i < res_lvl->num_subbands; i++) {
		sb = &(res_lvl->subbands[i]);
		for (j = 0; j < sb->num_cblks; j++) {
			int included, increment;
			cblk = &(sb->cblks[j]);

			/* Code block not included yet*/
			if(!cblk->num_segments)
			{
				/* Code-block inclusion */
				included = decode_tag_tree(buffer, sb->inc_tt, cblk->cblk_no, layer + 1);
				//printf("included %d cblkno %d resno %d\n", included, cblk->cblk_no, res_lvl->res_lvl_no);
			} else {
				println_var(INFO, "Error: Currently more than one layer is not supported");
			}

			if(!included)
			{
				println_var(INFO, "Error: Currently more than one layer is not supported");
			}

			if(!cblk->num_segments)
			{
				int k, kmsbs;
				/* Zero bit-plane information */
				for(k = 0; !decode_tag_tree(buffer, sb->zero_bit_plane_tt, cblk->cblk_no, k); k++)
				{
					;
				}
				/* Number of insignificant bits */
				kmsbs = k - 1;
				//printf("kmsbs %d\n", kmsbs);
				/* Number of magnitude bits */
				if(img->wavelet_type == DWT_53)
				{
					sb->mag_bits = sb->expn + tile_comp->num_guard_bits -1;
				} else
				{
					sb->mag_bits = (&res_lvl_zero->subbands[0])->expn - (tile_comp->num_dlvls - res_lvl->dec_lvl_no) + tile_comp->num_guard_bits -1;
				}

//				printf("sb->orient:%d mag_bits: %d expn:%d\n", sb->orient, sb->mag_bits, (&res_lvl_zero->subbands[0])->expn);
				/* Number of significant bits */
				cblk->significant_bits = sb->mag_bits - kmsbs;
				//printf("mag_bits %d significant_bits %d\n", sb->mag_bits, cblk->significant_bits);
				cblk->num_len_bits = 3;
			}
			/* Number of coding passes */
			cblk->num_coding_passes = decode_num_coding_passes(buffer);
			//printf("num_coding_passes %d\n", cblk->num_coding_passes);
			increment = get_comma_code(buffer);
			//printf("increment %d\n", increment);
			cblk->num_len_bits += increment;
			//printf("cblk->numlenbits %d\n", cblk->num_len_bits);
			/* Length of code-block compressed image data */
			cblk->length = read_bits(buffer, cblk->num_len_bits + int_floorlog2(cblk->num_coding_passes));
			//printf("cblk->length %d\n", cblk->length);
		}
	}

	if (inalign(buffer))
	{
		println_var(INFO, "Error: Inaligned packet header");
	}

	if(img->coding_style & USED_EPH)
	{
		/* Read EPH marker */
		marker = read_buffer(buffer, 2);

		if(marker != EPH)
		{
			println_var(INFO, "Error: Expected EPH(%x) marker instead of %x", EPH, marker);
		}
	}
}

//TODO
int sum_size = 0;


void decode_packet_body(type_buffer *buffer, type_res_lvl *res_lvl)
{
	int i, j;
	type_subband *sb;
	type_codeblock *cblk;

	for (i = 0; i < res_lvl->num_subbands; i++) {
		sb = &(res_lvl->subbands[i]);
		for (j = 0; j < sb->num_cblks; j++) {
			cblk = &(sb->cblks[j]);
			/*if(!cblk->num_segments)
			{
				cblk->num_segments++;
			}*/

			cblk->codestream = (unsigned char*)malloc(cblk->length);
			memcpy(cblk->codestream, buffer->bp, cblk->length);

//			int z;
//
//			for(z = 0; z < cblk->length; z++)
//			{
//				printf("%x", cblk->codestream[z]);
//			}
//			printf("\n");

			//printf("memcpy length %d\n", cblk->length);
			skip_buffer(buffer, cblk->length);
		}
	}
}


void decode_tiles(type_buffer *buffer, type_tile *tile)
{
	unsigned int marker, tile_part_length;
	type_image *img = tile->parent_img;
	type_tile_comp *tile_comp;
	type_res_lvl *res_lvl;
	int i;
	int res_no;
	int comp_no;

	tile_part_length = read_tile_header(buffer, tile);
	/* Length of tile part minus SOT, Lsot, Isot, Psot, TPsot, TNsot, SOD */
	tile_part_length -= 14;

	/* Read SOD marker */
	marker = read_buffer(buffer, 2);

	if(marker != SOD)
	{
		println_var(INFO, "Error: Expected SOD(%x) marker instead of %x", SOD, marker);
	}

	/* Currently we only support resolution level - layer - component - position progression */
	/* One precinct for resolution level and one layer for codestream. */
	for (res_no = 0; res_no < img->num_dlvls + 1; res_no++) {
		for (comp_no = 0; comp_no < img->num_components; comp_no++) {
			tile_comp = &(tile->tile_comp[comp_no]);
			res_lvl = &(tile_comp->res_lvls[res_no]);
			/* Decode packet header */
			decode_packet_header(buffer, res_lvl);
			/* Decode packet body */
			decode_packet_body(buffer, res_lvl);
		}
	}

//	println_var(INFO, "tile_part_length %d", tile_part_length);

//	for(i = 0; i < tile_part_length; i++)
//	{
//		read_buffer(buffer, 1);
//	}
}

/**
 * @brief  Main function to decode the codestream.
 *
 * @param buffer
 * @param img
 */
void decode_codestream(type_buffer *buffer, type_image *img)
{
	type_tile *tile;
	unsigned int marker;
	int i;

	read_main_header(buffer, img);

	for (i = 0; i < img->num_tiles; i++) {
		tile = &(img->tile[i]);
		decode_tiles(buffer, tile);
	}

	/* Read EOC marker */
	marker = read_buffer(buffer, 2);

	if(marker != EOC)
	{
		println_var(INFO, "Error: Expected EOC(%x) marker instead of %x", EOC, marker);
	}
}
