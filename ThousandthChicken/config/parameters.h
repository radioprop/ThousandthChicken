#ifndef PARAMETERS_H_
#define PARAMETERS_H_

typedef struct type_parameters type_parameters;

/** Image parameters */
struct type_parameters {
	unsigned short param_tile_w; /// Tile width. According to this and param_tile_height all the parameters of the tiles are set. -1 is no tiling (only one tile which covers entire image).
	unsigned short param_tile_h; /// Tile height. According to this and param_tile_width all the parameters of the tiles are set. -1 is no tiling (only one tile which covers entire image).
	unsigned char param_tile_comp_dlvls;
	unsigned char param_cblk_exp_w; ///Maximum codeblock size is 2^6 x 2^6 ( 64 x 64 ).
	unsigned char param_cblk_exp_h; ///Maximum codeblock size is 2^6 x 2^6 ( 64 x 64 ).
	unsigned char param_wavelet_type; ///Lossy encoding
	unsigned char param_use_mct;//Multi-component transform
	unsigned char param_device;//Which device use
	unsigned int param_target_size;//Target size of output file
	float param_bp;//Bits per pixel per component
	unsigned char param_use_part2_mct; // Multiple component transform as in 15444-2
	unsigned char param_mct_compression_method; // 0 klt 2 wavelet
	unsigned int param_mct_klt_iterations; // max number of iterations of Gram-Schmidt algorithm
	float param_mct_klt_border_eigenvalue; // cut-off for dumping components 
	float param_mct_klt_err; // error sufficient for Gram-Schmit algorithm to end iteration
};


int parse_config(const char *filename, type_parameters *param);
void default_config_values(type_parameters *param);

#endif /* PARAMETERS_H_ */
