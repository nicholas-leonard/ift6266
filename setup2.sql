/*
A PostgreSQL database schema for Hyper Parameter Search (HPS) using pylearn2.
Each pylearn2 class has its own Table.

TODO:
    add Foreign Key constraints
*/

--DROP SCHEMA hps2 CASCADE;
CREATE SCHEMA hps2; --Hyper parameter search

CREATE TABLE hps2.config (
	config_id	SERIAL,
	config_class	VARCHAR(255)[],
	dataset_name	VARCHAR(255),
	random_seed	INT4 DEFAULT 7777,
	ext_array	INT8[],
	start_time	TIMESTAMP,
	end_time	TIMESTAMP,
	PRIMARY KEY (config_id)
);


--DROP TABLE hps2.model_mlp CASCADE;
CREATE TABLE hps2.model_mlp (
	layer_array			INT8,
	batch_size			INT4,
	input_space_id			INT8,
	dropout_include_probs 		FLOAT4[],
	dropout_scales 			FLOAT4[],
	dropout_input_include_prob 	FLOAT4,
	dropout_input_scale 		FLOAT4,
	weight_decay			FLOAT4[],
	nvis				INT4,
	PRIMARY KEY (config_id)
) INHERITS (hps2.config);

/* Space */

CREATE TABLE hps2.space (
	space_id	SERIAL,
	space_class	VARCHAR(255),
	PRIMARY KEY (space_id)
);
CREATE TABLE hps2.space_conv2DSpace(
	num_row		INT4 NOT NULL,
	num_column	INT4 NOT NULL,
	num_channel	INT4 NOT NULL,
	PRIMARY KEY (space_id)
) INHERITS (hps2.space);

  
  
/* Extensions */

CREATE TABLE hps2.extension (
	ext_id		SERIAL,
	ext_class	VARCHAR(255) NOT NULL,
	PRIMARY KEY (ext_id)
);

CREATE TABLE hps2.ext_ExponentialDecayOverEpoch(
	decay_factor	FLOAT4, 
	min_lr		FLOAT4,
	PRIMARY KEY (ext_id)	
) INHERITS (hps2.extension);

CREATE TABLE hps2.ext_MomentumAdjustor(
	final_momentum	FLOAT4, 
	start_epoch	INT4, 
	saturate_epoch	INT4,
	PRIMARY KEY (ext_id)
) INHERITS (hps2.extension);


/* Termination Criteria */


CREATE TABLE hps2.term_epochcounter (
	ec_max_epoch	INT4
) INHERITS (hps2.config);

CREATE TABLE hps2.term_monitorbased (
	proportional_decrease	FLOAT4 DEFAULT 0.01,
	mb_max_epoch		INT4 DEFAULT 30,
	channel_name		VARCHAR(255) DEFAULT 'Validation Missclassification'
) INHERITS (hps2.config);

/* Training Algorithms */

--DROP TABLE hps2.train_sgd CASCADE;
CREATE TABLE hps2.train_sgd (
	learning_rate		FLOAT4 NOT NULL,
	batch_size		INT4,
	init_momentum   	FLOAT4,
	train_iteration_mode	VARCHAR(255) DEFAULT 'random_uniform'
) INHERITS (hps2.config);


--DROP TABLE hps2.config CASCADE;


--DROP TABLE hps2.config_mlp_sgd_mb_ec;
CREATE TABLE hps2.config_mlp_sgd_mb_ec (
	PRIMARY KEY (config_id)
) INHERITS (hps2.model_mlp, hps2.train_sgd, hps2.term_monitorbased, hps2.term_epochcounter);

--DROP TABLE hps2.training_log;
CREATE TABLE hps2.training_log (
	config_id	INT8,
	epoch_count	INT4,
	channel_name	VARCHAR(255),
	channel_value	FLOAT4,
	PRIMARY KEY (config_id, epoch_count, channel_name)
);

/* MLP layers */
	
--DROP TABLE hps2.layer CASCADE;
CREATE TABLE hps2.layer (
	layer_id	SERIAL,
	layer_class	VARCHAR(255),
	layer_name	VARCHAR(255),
	PRIMARY KEY (layer_id)
);

CREATE TABLE hps2.layer_rectifiedlinear(
	dim		INT4,
	irange 		FLOAT4,
	istdev 		FLOAT4,
	sparse_init 	FLOAT4,
	sparse_stdev 	FLOAT4 DEFAULT 1.0,
	include_prob 	FLOAT4 DEFAULT 1.0,
	init_bias 	FLOAT4,
	W_lr_scale 	FLOAT4,
	b_lr_scale 	FLOAT4,
	left_slope 	FLOAT4 DEFAULT 0.0,
	max_row_norm 	FLOAT4,
	max_col_norm 	FLOAT4,
	use_bias 	BOOLEAN DEFAULT True,
	PRIMARY KEY (layer_id)
) INHERITS (hps2.layer);


CREATE TABLE hps2.layer_softmax (
	n_classes	INT4, 
	irange 		FLOAT4,
        istdev 		FLOAT4,
        sparse_init 	FLOAT4, 
        W_lr_scale 	FLOAT4,
        b_lr_scale 	FLOAT4, 
        max_row_norm 	FLOAT4,
        no_affine 	BOOLEAN DEFAULT False,
        max_col_norm 	FLOAT4,
	PRIMARY KEY (layer_id)
) INHERITS (hps2.layer);

CREATE TABLE hps2.layer_maxout (
	 num_units		INT4,
	 num_pieces		INT4,
	 pool_stride 		INT4,
	 randomize_pools 	BOOLEAN DEFAULT False,
	 irange 		FLOAT4,
	 sparse_init 		INT4,
	 sparse_stdev 		FLOAT4 DEFAULT 1.0,
	 include_prob 		FLOAT4 DEFAULT 1.0,
	 init_bias		FLOAT4 DEFAULT 0.,
	 W_lr_scale 		FLOAT4,
	 b_lr_scale 		FLOAT4,
	 max_col_norm 		FLOAT4,
	 max_row_norm 		FLOAT4,
	 min_zero		BOOLEAN DEFAULT False
) INHERITS (hps2.layer);

CREATE TABLE hps2.layer_ConvRectifiedLinear (
	output_channels	INT4,
	kernel_width	INT4,
	pool_width	INT4,
	pool_stride	INT4,
	irange 		FLOAT4,
        border_mode 	VARCHAR(255) DEFAULT 'valid',
	sparse_init 	FLOAT4,
	include_prob 	FLOAT4 DEFAULT 1.0,
	init_bias 	FLOAT4 DEFAULT 0,
	W_lr_scale 	FLOAT4,
	b_lr_scale 	FLOAT4,
	left_slope 	FLOAT4 DEFAULT 0.0,
	max_kernel_norm FLOAT4,
	PRIMARY KEY (layer_id)
) INHERITS (hps2.layer);

CREATE TABLE hps2.layer_MaxoutConvC01B (
	num_channels		INT4,
	num_pieces		INT4,
	kernel_width		INT4,
	pool_width		INT4,
	pool_stride		INT4,
	irange			FLOAT4,
	init_bias 		FLOAT4 DEFAULT 0,
	W_lr_scale 		FLOAT4,
	b_lr_scale 		FLOAT4,
	pad 			FLOAT4,
	fix_pool_shape 		BOOLEAN DEFAULT False,
	fix_pool_stride 	BOOLEAN DEFAULT False,
	fix_kernel_shape 	BOOLEAN DEFAULT False,
	partial_sum 		FLOAT4 DEFAULT 1,
	tied_b  		BOOLEAN DEFAULT False,
	max_kernel_norm		FLOAT4,
	input_normalization 	FLOAT4,
	output_normalization 	FLOAT4,
	PRIMARY KEY (layer_id)
) INHERITS (hps2.layer);   

CREATE TABLE hps2.layer_sigmoid (
	dim		INT4,
	irange 		FLOAT4,
	sparse_init 	INT4,
	sparse_stdev 	FLOAT4 DEFAULT 1.0,
	include_prob 	FLOAT4 DEFAULT 1.0,
	init_bias	FLOAT4 DEFAULT 0.,
	W_lr_scale 	FLOAT4,
	b_lr_scale 	FLOAT4,
	max_col_norm 	FLOAT4,
	max_row_norm 	FLOAT4,
	PRIMARY KEY (layer_id)
) INHERITS (hps2.layer); 