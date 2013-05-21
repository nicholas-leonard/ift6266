/*
A PostgreSQL database schema for Hyper Parameter Search (HPS) using pylearn2.
Each pylearn2 class has its own Table.

TODO:
    add Foreign Key constraints
*/

--DROP SCHEMA hps3 CASCADE;
CREATE SCHEMA hps3; --Hyper parameter search

--DROP TABLE hps3.config;
CREATE TABLE hps3.config (
	config_id		SERIAL,
	task_id			INT8,
	random_seed		INT4 DEFAULT 7777,
	ext_array		INT8[],
	start_time		TIMESTAMP,
	end_time		TIMESTAMP,
	dataset_id		INT8,
	worker_name		VARCHAR(255),
	channel_array		INT8[],
	PRIMARY KEY (config_id)
);

/* Dataset */

CREATE TABLE hps3.dataset (
	dataset_id		SERIAL,
	preprocess_array	INT8[],
	train_ddm_id		INT8,
	valid_ddm_id		INT8,
	test_ddm_id		INT8,
	PRIMARY KEY (dataset_id)
);

CREATE TABLE hps3.ddm (
	ddm_id		SERIAL,
	ddm_class	VARCHAR(255),
	PRIMARY KEY (ddm_id)
);

CREATE TABLE hps3.ddm_cifar100 (
	which_set	VARCHAR(255), 
	center 		BOOLEAN DEFAULT False,
	gcn 		FLOAT4, 
	toronto_prepro 	BOOLEAN DEFAULT False,
	axes 		VARCHAR(255) DEFAULT 'b01c',
	start		INT4, 
	stop 		INT4, 
	one_hot 	BOOLEAN DEFAULT True,
	PRIMARY KEY (ddm_id)
) INHERITS (hps3.ddm);

CREATE TABLE hps3.model (
	model_class	VARCHAR(255),
	PRIMARY KEY (config_id)
) INHERITS (hps3.config);

--DROP TABLE hps3.model_mlp CASCADE;
CREATE TABLE hps3.model_mlp (
	layer_array			INT8[],
	batch_size			INT4 DEFAULT 32,
	input_space_id			INT8,
	nvis				INT4,
	PRIMARY KEY (config_id)
) INHERITS (hps3.model);

/* Cost */

CREATE TABLE hps3.cost (
	cost_id		SERIAL,
	cost_class	VARCHAR(255),
	PRIMARY KEY (cost_id)
);

CREATE TABLE hps3.cost_mlp (
	cost_type		VARCHAR(255) DEFAULT 'default', 
	missing_target_value	FLOAT4,
	cost_name		VARCHAR(255),
	default_dropout_prob	FLOAT4 DEFAULT 0.5,
	default_dropout_scale	FLOAT4 DEFAULT 2.0,
	PRIMARY KEY (cost_id)
) INHERITS (hps3.cost);

--DROP TABLE hps3.channel;
CREATE TABLE hps3.channel (
	channel_id		SERIAL,
	channel_class		VARCHAR(255),
	monitoring_datasets	VARCHAR(255)[] DEFAULT '{valid,test}',
	PRIMARY KEY (channel_id)
);
        
/* Space */

--DROP TABLE hps3.space CASCADE;
CREATE TABLE hps3.space (
	space_id	SERIAL,
	space_class	VARCHAR(255),
	PRIMARY KEY (space_id)
);
--DROP TABLE hps3.space_conv2dspace;
CREATE TABLE hps3.space_conv2DSpace(
	num_row		INT4 NOT NULL,
	num_column	INT4 NOT NULL,
	num_channel	INT4 NOT NULL,
	axes 		VARCHAR(255) DEFAULT 'b01c',
	PRIMARY KEY (space_id)
) INHERITS (hps3.space);

--DROP TABLE hps3.space_vector CASCADE;
CREATE TABLE hps3.space_vector (
	dim		INT4 NOT NULL,
	sparse		BOOLEAN DEFAULT FALSE,
	PRIMARY KEY (space_id)
) INHERITS (hps3.space);
  
  
/* Extensions */

CREATE TABLE hps3.extension (
	ext_id		SERIAL,
	ext_class	VARCHAR(255) NOT NULL,
	PRIMARY KEY (ext_id)
);

--DROP TABLE hps3.ext_ExponentialDecayOverEpoch;
CREATE TABLE hps3.ext_ExponentialDecayOverEpoch(
	decay_factor	FLOAT4, 
	min_lr_scale	FLOAT4,
	PRIMARY KEY (ext_id)	
) INHERITS (hps3.extension);

CREATE TABLE hps3.ext_MomentumAdjustor(
	final_momentum	FLOAT4, 
	start_epoch	INT4, 
	saturate_epoch	INT4,
	PRIMARY KEY (ext_id)
) INHERITS (hps3.extension);


/* Termination Criteria */

CREATE TABLE hps3.termination (
	term_id		SERIAL,
	term_class	VARCHAR(255),
	PRIMARY KEY (term_id)
);

CREATE TABLE hps3.term_epochcounter (
	max_epoch	INT4,
	PRIMARY KEY (term_id)
) INHERITS (hps3.termination);

--DROP TABLE hps3.term_monitorbased;
CREATE TABLE hps3.term_monitorbased (
	proportional_decrease	FLOAT4 DEFAULT 0.01,
	max_epoch		INT4 DEFAULT 30,
	channel_name		VARCHAR(255) DEFAULT 'valid_hps_cost',
	PRIMARY KEY (term_id)
) INHERITS (hps3.termination);

/* Training Algorithms */

CREATE TABLE hps3.train (
	train_class	VARCHAR(255),
	PRIMARY KEY (config_id)
) INHERITS (hps3.config);

--DROP TABLE hps3.train_sgd CASCADE;
CREATE TABLE hps3.train_sgd (
	term_array		INT8[],
	learning_rate		FLOAT4 NOT NULL,
	batch_size		INT4,
	init_momentum   	FLOAT4,
	train_iteration_mode	VARCHAR(255) DEFAULT 'random_uniform',
	cost_array		INT8[],
	PRIMARY KEY (config_id)
) INHERITS (hps3.train);


--DROP TABLE hps3.config CASCADE;

--DROP TABLE hps3.config_mlp_sgd;
CREATE TABLE hps3.config_mlp_sgd (
	PRIMARY KEY (config_id)
) INHERITS (hps3.model_mlp, hps3.train_sgd);

--DROP TABLE hps3.training_log;
CREATE TABLE hps3.training_log (
	config_id	INT8,
	epoch_count	INT4,
	channel_name	VARCHAR(255),
	channel_value	FLOAT4,
	PRIMARY KEY (config_id, epoch_count, channel_name)
);

/* MLP layers */
	
--DROP TABLE hps3.layer CASCADE;
CREATE TABLE hps3.layer (
	layer_id		SERIAL,
	layer_class		VARCHAR(255),
	layer_name		VARCHAR(255),
	dropout_scale		FLOAT4,
        dropout_probability	FLOAT4,
        weight_decay		FLOAT4, 
        l1_weight_decay		FLOAT4,
	PRIMARY KEY (layer_id)
);

CREATE TABLE hps3.layer_rectifiedlinear(
	dim		INT4,
	irange 		FLOAT4,
	istdev 		FLOAT4,
	sparse_init 	INT4,
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
) INHERITS (hps3.layer);

--DROP TABLE hps3.layer_linear;
CREATE TABLE hps3.layer_linear(
	dim		INT4,
	irange 		FLOAT4,
	istdev 		FLOAT4,
	sparse_init 	FLOAT4,
	sparse_stdev 	FLOAT4 DEFAULT 1.0,
	include_prob 	FLOAT4 DEFAULT 1.0,
	init_bias 	FLOAT4,
	W_lr_scale 	FLOAT4,
	b_lr_scale 	FLOAT4,
	max_row_norm 	FLOAT4,
	max_col_norm 	FLOAT4,
	use_bias 	BOOLEAN DEFAULT True,
	softmax_columns BOOLEAN DEFAULT False,
	PRIMARY KEY (layer_id)
) INHERITS (hps3.layer);

CREATE TABLE hps3.layer_softmax (
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
) INHERITS (hps3.layer);

--DROP TABLE hps3.layer_maxout;
CREATE TABLE hps3.layer_maxout (
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
	 min_zero		BOOLEAN DEFAULT False,
	 PRIMARY KEY (layer_id)
) INHERITS (hps3.layer);

CREATE TABLE hps3.layer_ConvRectifiedLinear (
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
) INHERITS (hps3.layer);

CREATE TABLE hps3.layer_Convolution (
	output_channels	INT4,
	kernel_width	INT4,
	pool_width	INT4,
	pool_stride	INT4,
	irange 		FLOAT4,
        border_mode 	VARCHAR(255) DEFAULT 'valid',
        activation_function VARCHAR(255) DEFAULT 'tanh',
	sparse_init 	FLOAT4,
	include_prob 	FLOAT4 DEFAULT 1.0,
	init_bias 	FLOAT4 DEFAULT 0,
	W_lr_scale 	FLOAT4,
	b_lr_scale 	FLOAT4,
	max_kernel_norm FLOAT4,
	PRIMARY KEY (layer_id)
) INHERITS (hps3.layer);

CREATE TABLE hps3.layer_MaxoutConvC01B (
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
) INHERITS (hps3.layer);   

CREATE TABLE hps3.layer_sigmoid (
	dim		INT4,
	irange 		FLOAT4,
	istdev 		FLOAT4,
	sparse_init 	INT4,
	sparse_stdev 	FLOAT4 DEFAULT 1.0,
	include_prob 	FLOAT4 DEFAULT 1.0,
	init_bias	FLOAT4 DEFAULT 0.,
	W_lr_scale 	FLOAT4,
	b_lr_scale 	FLOAT4,
	max_col_norm 	FLOAT4,
	max_row_norm 	FLOAT4,
	PRIMARY KEY (layer_id)
) INHERITS (hps3.layer); 

CREATE TABLE hps3.layer_tanh (
	dim		INT4,
	irange 		FLOAT4,
	istdev 		FLOAT4,
	sparse_init 	INT4,
	sparse_stdev 	FLOAT4 DEFAULT 1.0,
	include_prob 	FLOAT4 DEFAULT 1.0,
	init_bias	FLOAT4 DEFAULT 0.,
	W_lr_scale 	FLOAT4,
	b_lr_scale 	FLOAT4,
	max_col_norm 	FLOAT4,
	max_row_norm 	FLOAT4,
	PRIMARY KEY (layer_id)
) INHERITS (hps3.layer); 

/* Preprocessing */

CREATE TABLE hps3.preprocess (
	preprocess_id 		SERIAL,
	preprocess_class	VARCHAR(255),
	PRIMARY KEY (preprocess_id)
);

CREATE TABLE hps3.preprocess_standardize (
	global_mean		BOOLEAN DEFAULT False, 
	global_std		BOOLEAN DEFAULT False, 
	std_eps			FLOAT4 DEFAULT 1e-4,
	PRIMARY KEY (preprocess_id)
) INHERITS (hps3.preprocess);

CREATE TABLE hps3.preprocess_zca (
	n_components		INT4, 
	n_drop_components	INT4,
        filter_bias		FLOAT4 DEFAULT 0.1,
	PRIMARY KEY (preprocess_id)
) INHERITS (hps3.preprocess);

CREATE TABLE hps3.preprocess_gcn (
	subtract_mean		BOOLEAN DEFAULT True, 
	std_bias		FLOAT4 DEFAULT 10.0, 
	use_norm		BOOLEAN DEFAULT False,
	PRIMARY KEY (preprocess_id)
) INHERITS (hps3.preprocess);

/* Functions */


CREATE OR REPLACE FUNCTION hps3.get_best(config_id INT4, channel_name VARCHAR(1000)) 
RETURNS TABLE (epoch INT4, value FLOAT4, rank INT4) AS $$
	SELECT epoch_count, channel_value, rank::INT4
	FROM	(
		SELECT config_id, epoch_count, channel_value, rank()
			OVER (PARTITION BY config_id ORDER BY channel_value DESC, epoch_count ASC)
		FROM hps3.training_log
		WHERE channel_name = $2 AND config_id = $1
		) AS a
$$ LANGUAGE SQL;

CREATE OR REPLACE FUNCTION hps3.get_channel(config_id INT4, channel_name VARCHAR(1000), epoch INT4) 
RETURNS FLOAT4 AS $$
	SELECT channel_value
	FROM hps3.training_log
	WHERE channel_name = $2 AND config_id = $1 AND epoch_count = $3
$$ LANGUAGE SQL;


CREATE OR REPLACE FUNCTION hps3.get_last(config_id INT4, channel_name VARCHAR(1000)) 
RETURNS FLOAT4 AS $$
	SELECT channel_value
	FROM hps3.training_log
	WHERE channel_name = $2 AND config_id = $1
	ORDER BY epoch_count DESC LIMIT 1
$$ LANGUAGE SQL;


CREATE OR REPLACE FUNCTION hps3.get_end(config_id INT4) 
RETURNS INT4 AS $$
	SELECT MAX(epoch_count)
	FROM hps3.training_log
	WHERE config_id = $1
$$ LANGUAGE SQL;
