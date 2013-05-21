/*
A PostgreSQL database schema for Hyper Parameter Search (HPS) using pylearn2.
Each pylearn2 class has its own Table.

TODO:
    add Foreign Key constraints
*/

--DROP SCHEMA hps2 CASCADE;
CREATE SCHEMA hps2; --Hyper parameter search

CREATE TABLE hps2.config (
	config_id		SERIAL,
	config_class		VARCHAR(255)[],
	dataset_name		VARCHAR(255),
	task_id			INT8,
	random_seed		INT4 DEFAULT 7777,
	ext_array		INT8[],
	start_time		TIMESTAMP,
	end_time		TIMESTAMP,
	dataset_id		INT8,
	cost_id			INT8,
	PRIMARY KEY (config_id)
);

CREATE TABLE hps2.cost (
	cost_id		SERIAL,
	cost_class	VARCHAR(255),
	PRIMARY KEY (cost_id)
);

CREATE TABLE hps2.cost_mlp (
	cost

);

CREATE TABLE hps2.dataset (
	dataset_id		SERIAL,
	preprocess_array	INT8[],
	train_ddm_id		INT8,
	valid_ddm_id		INT8,
	test_ddm_id		INT8,
	PRIMARY KEY (dataset_id)
);

CREATE TABLE hps2.ddm (
	ddm_id		SERIAL,
	ddm_class	VARCHAR(255),
	PRIMARY KEY (ddm_id)
);

CREATE TABLE hps2.ddm_cifar100 (
	which_set	VARCHAR(255), 
	center 		BOOLEAN DEFAULT False,
	gcn 		FLOAT4, 
	toronto_prepro 	BOOLEAN DEFAULT False,
	axes 		VARCHAR(255) DEFAULT 'b01c',
	start		INT4, 
	stop 		INT4, 
	one_hot 	BOOLEAN DEFAULT False,
	PRIMARY KEY (ddm_id)
) INHERITS (hps2.ddm);


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

--DROP TABLE hps2.layer_linear;
CREATE TABLE hps2.layer_linear(
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

CREATE TABLE hps2.layer_Convolution (
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

-- Mixtures:

CREATE TABLE hps2.gater (
	 gater_id	SERIAL,
	 gater_class	VARCHAR(255),
	 PRIMARY KEY (gater_id)
);

CREATE TABLE hps2.gater_kmeans (
	 irange 		FLOAT4 DEFAULT 0.05,
	 gating_protocol 	VARCHAR(255) DEFAULT 'nearest',
	 stochastic_masking 	BOOLEAN DEFAULT False,
	 distribution_function 	VARCHAR(255) DEFAULT 'divide by sum',
	 W_lr_scale 		FLOAT4,
	 PRIMARY KEY (gater_id)		  
) INHERITS (hps2.gater);

CREATE TABLE hps2.layer_starmixture (
	 dim			INT4,
	 num_experts		INT4,
	 expert_dim		INT4,
	 gater_id		INT8,
	 expert_activation 	VARCHAR(255),
	 output_activation 	VARCHAR(255),
	 cost_function 		VARCHAR(255),
	 irange 		FLOAT4 DEFAULT 0.05,
	 istdev 		FLOAT4,
	 sparse_init 		INT4,
	 sparse_stdev 		FLOAT4 DEFAULT 1.0,
	 init_bias 		FLOAT4 DEFAULT 0.,
	 W_lr_scale 		FLOAT4,
	 b_lr_scale 		FLOAT4,
	 max_col_norm0 		FLOAT4,
	 max_col_norm1		FLOAT4,
	 use_bias 		BOOLEAN DEFAULT True,
	 PRIMARY KEY (layer_id)
) INHERITS (hps2.layer);

--DROP TABLE hps2.cost_kmeans CASCADE;
CREATE TABLE hps2.cost_kmeans (
	kmeans_coeff		FLOAT4[],
	PRIMARY KEY (config_id)
) INHERITS (hps2.config);

CREATE TABLE hps2.cost_mixture (
	gater_autonomy	FLOAT4[],
	expert_autonomy FLOAT4[],
	PRIMARY KEY (config_id)
) INHERITS (hps2.config);

--DROP TABLE hps2.config_mlp_sgd_mb_ec_km;
CREATE TABLE hps2.config_mlp_sgd_mb_ec_km (
	PRIMARY KEY (config_id)
) INHERITS (hps2.model_mlp, hps2.train_sgd, hps2.term_monitorbased, hps2.term_epochcounter, hps2.cost_kmeans);

CREATE TABLE hps2.config_mlp_sgd_mb_ec_mix (
	PRIMARY KEY (config_id)
) INHERITS (hps2.model_mlp, hps2.train_sgd, hps2.term_monitorbased, hps2.term_epochcounter, hps2.cost_mixture);

CREATE TABLE hps2.config_mlp_sgd_mb_ec (
	PRIMARY KEY (config_id)
) INHERITS (hps2.model_mlp, hps2.train_sgd, hps2.term_monitorbased, hps2.term_epochcounter);


CREATE TABLE hps2.preprocess (
	preprocess_id 		SERIAL,
	preprocess_class	VARCHAR(255),
	PRIMARY KEY (preprocess_id)
);

CREATE TABLE hps2.preprocess_standardize (
	global_mean		BOOLEAN DEFAULT False, 
	global_std		BOOLEAN DEFAULT False, 
	std_eps			FLOAT4 DEFAULT 1e-4,
	PRIMARY KEY (preprocess_id)
) INHERITS (hps2.preprocess);

CREATE TABLE hps2.preprocess_zca (
	n_components		INT4, 
	n_drop_components	INT4,
        filter_bias		FLOAT4 DEFAULT 0.1,
	PRIMARY KEY (preprocess_id)
) INHERITS (hps2.preprocess);

CREATE TABLE hps2.preprocess_gcn (
	subtract_mean		BOOLEAN DEFAULT True, 
	std_bias		FLOAT4 DEFAULT 10.0, 
	use_norm		BOOLEAN DEFAULT False,
	PRIMARY KEY (preprocess_id)
) INHERITS (hps2.preprocess);


