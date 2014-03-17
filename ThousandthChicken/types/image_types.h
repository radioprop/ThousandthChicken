#pragma once

#include "../tier2/tag_tree.h"

#include "buffered_stream_type.h"
#include "image_mct.h"

#define UNSIGNED 0U
#define SIGNED 1U

#define DWT_53 0
#define DWT_97 1

#define GUARD_BITS 2

typedef struct type_coding_param type_coding_param;
typedef struct type_image type_image;
typedef struct type_tile type_tile;
typedef struct type_tile_comp type_tile_comp;
typedef struct type_res_lvl type_res_lvl;
typedef struct type_subband type_subband;
typedef struct type_codeblock type_codeblock;
typedef float type_data;

typedef enum
{
	LL, HL, LH, HH
} type_orient;

/** Codeblock coding parameters */
struct type_codeblock
{
	/** Codeblock number in raster order */
	unsigned int cblk_no;

	/** Codeblock number  in the horizontal direction */
	unsigned short no_x;

	/** Codeblock number  in the vertical direction */
	unsigned short no_y;

	/** The x-coordinate of the top-left corner of the codeblock, regarding to subband. */
	unsigned short tlx;

	/** The y-coordinate of the top-left corner of the codeblock, regarding to subband. */
	unsigned short tly;

	/** The x-coordinate of the bottom-right corner of the codeblock, regarding to subband. */
	unsigned short brx;

	/** The y-coordinate of the bottom-right corner of the codeblock, regarding to subband. */
	unsigned short bry;

	/** Codeblock width */
	unsigned short width;

	/** Codeblock height */
	unsigned short height;

	/** Pointer to codeblock data on gpu */
	int *data_d;

	/** Parent subband */
	type_subband *parent_sb;

	/** Code block bytestream */
	unsigned char *codestream;

	/** Codestream length */
	unsigned int length;

	/** Significant bits in codeblock */
	unsigned char significant_bits;

	/** Number of length bits */
	unsigned int num_len_bits;

	/** Number of segments */
	unsigned int num_segments;

	/** Number of coding passes */
	unsigned int num_coding_passes;
};

/** Subband coding parameters */
struct type_subband
{
	/** The orientation of the subband(LL, HL, LH, HH) */
	type_orient orient;

	/** The x-coordinate of the top-left corner of the subband, regarding to tile-component. tbx0 */
	unsigned short tlx;

	/** The y-coordinate of the top-left corner of the subband, regarding to tile-component. tby0 */
	unsigned short tly;

	/** The x-coordinate of the bottom-right corner of the subband, regarding to tile-component. tbx1 */
	unsigned short brx;

	/** The y-coordinate of the bottom-right corner of the subband, regarding to tile-component. tby1 */
	unsigned short bry;

	/** Subband width */
	unsigned short width;

	/** Subband height */
	unsigned short height;

	/** Number of codeblocks in the horizontal direction in subband. */
	unsigned short num_xcblks;

	/** Number of codeblocks in the vertical direction in subband. */
	unsigned short num_ycblks;

	/** Total number of codeblocks in subband */
	unsigned int num_cblks;

	/** Codeblocks in current subband */
	type_codeblock *cblks;

	/** Codeblocks data on gpu */
	int *cblks_data_d;

	/** Codeblocks data on cpu */
	int *cblks_data_h;

	/** Number of magnitude bits in the integer representation of the quantized data */
	unsigned char mag_bits;

	/** Quantization step size */
	type_data step_size;

	/** Convert factor to quantize data */
	type_data convert_factor;

	/** Parent resolution-level */
	type_res_lvl *parent_res_lvl;

	/** Exponent */
	unsigned short expn;

	/** Matissa */
	unsigned short mant;

	/** Inclusion tag tree */
	type_tag_tree *inc_tt;
	/** Zero bit plane tag tree */
	type_tag_tree *zero_bit_plane_tt;
};

/** Resolution-level coding parameters */
struct type_res_lvl
{
	/** Resolution level number. r */
	unsigned char res_lvl_no;

	/** Decomposition level number. nb */
	unsigned char dec_lvl_no;

	/** The x-coordinate of the top-left corner of the tile-component
	 at this resolution. trx0 */
	unsigned short tlx;

	/** The y-coordinate of the top-left corner of the tile-component
	 at this resolution. try0 */
	unsigned short tly;

	/** The x-coordinate of the bottom-right corner of the tile-component
	 at this resolution(plus one). trx1 */
	unsigned short brx;

	/** The y-coordinate of the bottom-right corner of the tile-component
	 at this resolution(plus one). try1 */
	unsigned short bry;

	/** Resolution level width */
	unsigned short width;

	/** Resolution level height */
	unsigned short height;

	/** The exponent value for the precinct width. PPx */
	unsigned char prc_exp_w;

	/** The exponent value for the precinct height PPy */
	unsigned char prc_exp_h;

	/** Number of precincts in the horizontal direction in resolution level. numprecinctswide */
	unsigned short num_hprc;

	/** Number of precincts in the vertical direction in resolution level. numprecinctshigh */
	unsigned short num_vprc;

	/** Total number of precincts. numprecincts */
	unsigned short num_prcs;

	/** Number of subbands */
	unsigned char num_subbands;

	/** Subbands in current resolution level */
	type_subband *subbands;

	/** Parent tile-component */
	type_tile_comp *parent_tile_comp;
};

/** Tile on specific component/channel */
struct type_tile_comp
{
	/** Tile_comp number */
	unsigned int tile_comp_no;

	/** XXX: Tiles on specific components may have different sizes, because components can have various sizes. See ISO B.3 */
	/** Tile-component width. */
	unsigned short width;

	/** Tile-component height */
	unsigned short height;

	/** Number of decomposition levels. NL. COD marker */
	unsigned char num_dlvls;

	/** Number of the resolution levels. */
	unsigned char num_rlvls;

	/** The max exponent value for code-block width */
	/** XXX: Minimum for code-block dimension is 4.
	 * 	Maximum dimension is 64.  */
	unsigned char cblk_exp_w;

	/** The max exponent value for code-block height */
	unsigned char cblk_exp_h;

	/** Nominal codeblock width */
	unsigned char cblk_w;

	/** Nominal codeblock height */
	unsigned char cblk_h;

	/** Quantization style for all components */
	unsigned short quant_style;

	/** Number of guard bits */
	unsigned char num_guard_bits;

	/** Tile component data in the host memory (this is page-locked memory, prepared for copying to device) */
	type_data *img_data;

	/** Tile component data on the GPU */
	type_data *img_data_d;

	/** Resolution levels */
	type_res_lvl *res_lvls;

	/** Parent tile */
	type_tile *parent_tile;
};

struct type_tile
{
	/** Tile number in raster order */
	unsigned int tile_no;

	/** The x-coord of the top left corner of the tile with respect to the original image. tx0 */
	unsigned short tlx;

	/** The y-coord of the top left corner of the tile with respect to the original image. ty0 */
	unsigned short tly;

	/** The x-coord of the bottom right corner of the tile with respect to the original image. tx1 */
	unsigned short brx;

	/** The y-coord of the bottom right corner of the tile with respect to the original image. ty1 */
	unsigned short bry;

	/** Tile width */
	unsigned short width;

	/** Tile height */
	unsigned short height;

	/** Quantization style for each channel (ready for QCD/QCC marker) */
	char QS;

	/** Tile on specific component/channel in host memory */
	type_tile_comp *tile_comp;

	/** Parent image */
	type_image *parent_img;
};

/* Coding parameters */
struct type_coding_param
{
	/* Image area */
	/** The horizontal offset from the origin of the reference grid to the
	 left edge of the image area. XOsiz */
	unsigned short imgarea_tlx;
	/** The vertical offset from the origin of the reference grid to the
	 left edge of the image area. YOsiz */
	unsigned short imgarea_tly;

	/** The horizontal offset from the origin of the reference grid to the
	 right edge of the image area. Xsiz */
	unsigned short imgarea_width;
	/** The vertical offset from the origin of the reference grid to the
	 right edge of the image area. Ysiz */
	unsigned short imgarea_height;

	/* Tile grid */
	/** The horizontal offset from the origin of the tile grid to the
	 origin of the reference grid. XTOsiz */
	unsigned short tilegrid_tlx;
	/** The vertical offset from the origin of the tile grid to the
	 origin of the reference grid. YTOsiz */
	unsigned short tilegrid_tly;

	/** The component horizontal sampling factor. XRsiz */
	unsigned short comp_step_x;

	/** The component vertical sampling factor. YRsiz */
	unsigned short comp_step_y;

	/** Base step size */
	type_data base_step;

	/** Target size when using PCRD */
	unsigned int target_size;
};

struct type_image
{
	/** Input file name */
	const char *in_file;

	/** Input header file name */
	const char *in_hfile;

	/** Output file name */
	const char *out_file;

	/** Configuration file */
	const char *conf_file;

	/** Mct_compression_method: 0 klt, 2 wavelet */
	unsigned char mct_compression_method;

	/** BSQ file type */
	unsigned char bsq_file;

	/** BIP file type */
	unsigned char bip_file;

	/** BIL file type */
	unsigned char bil_file;

	/** Image width */
	unsigned short width;

	/** Image height */
	unsigned short height;

	/** Number of channels/components. Csiz */
	unsigned short num_components;

	/** Original bit depth. XXX: Should be separate for every component */
	unsigned short depth;

	/** Data sign. XXX: Should be separate for every component.
	 * Really separate? i think we can safely assume all components are either signed or unsigned (q)*/
	unsigned char sign;

	/** Nominal number of decomposition levels */
	unsigned char num_dlvls;

	/** Type of wavelet transform: lossless DWT_53, lossy DWT_97. COD marker */
	unsigned char wavelet_type;

	/** Area allocated in the device memory */
	int area_alloc;

	/** The nominal tile width. XTsiz. SIZ marker */
	unsigned short tile_w;
	/** The nominal tile height. YTsiz. SIZ marker */
	unsigned short tile_h;

	/** Number of tiles in horizontal direction. nunXtiles */
	unsigned short num_xtiles;
	/** Number of tiles in vertical direction. numYtiles */
	unsigned short num_ytiles;
	/** Number of all tiles */
	unsigned int num_tiles;

	/** Was the MCT used? */
	unsigned char use_mct;

	/** Was MCT as in 15444-2 standard */
	unsigned char use_part2_mct;

	/** Data for MCT as in 15444-2 */
	type_multiple_component_transformations* mct_data;

	/** Nominal range of bits */
	unsigned char num_range_bits;

	/** Coding style for all components */
	int coding_style;

	/** Codeblock coding style */
	unsigned char cblk_coding_style;

	/** Progression order */
	unsigned char prog_order;

	/** Number of layers */
	unsigned short num_layers;

	/** Initial real-image data on GPU, used only in read_image and color transformation,
	 * after tiling use pointers in tile->tile_comp.*/
	type_data *img_data_d;
	/** Real image data is in this array of tiles. */
	type_tile *tile;

	/** Coding parameters */
	type_coding_param *coding_param;
};

