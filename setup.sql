/*
A PostgreSQL database schema for Hyper Parameter Search (HPS) using pylearn2.
Each pylearn2 class has its own Table.

TODO:
    add Foreign Key constraints
*/

--DROP SCHEMA hps CASCADE;
CREATE SCHEMA hps; --Hyper parameter search


CREATE TABLE hps.user (
	user_id		SERIAL,
	user_name	VARCHAR(255),
	user_password	BYTEA,
	PRIMARY KEY (user_id)
);

--DROP TABLE hps.dataset;
CREATE TABLE hps.dataset (
	dataset_id	SERIAL,
	dataset_desc	TEXT,
	dataset_nvis	INT4,
	dataset_nout	INT4,
	PRIMARY KEY (dataset_id)
);

CREATE TABLE hps.user_dataset (
	user_id		INT8,
	dataset_id	INT8,
	PRIMARY KEY (user_id, dataset_id)
);


CREATE TABLE hps.user_model (
	user_id		INT8,
	experiment_id	INT8,
	PRIMARY KEY (user_id, experiment_id)
);

/* 
An experiment is conducted on a set of models
that have something that needs to be optimized.
*/
--DROP TABLE hps.experiment;
CREATE TABLE hps.experiment (
	experiment_id		SERIAL,
	experiment_desc		TEXT,
	PRIMARY KEY (experiment_id)
);

/* Model : MLP */
CREATE TABLE hps.model (
	model_id	SERIAL,
	model_class	VARCHAR(255) NOT NULL,
	PRIMARY KEY (model_id)
);
CREATE TABLE hps.model_mlp (
	input_layer_id		INT8,
	output_layer_id		INT8,
	PRIMARY KEY (model_id)
) INHERITS (hps.model);

CREATE TABLE hps.mlp_graph (
	model_id		INT8,
	input_layer_id		INT8,
	output_layer_id		INT8,
	PRIMARY KEY (model_id, input_layer_id, output_layer_id)
);

/* Weight Initializer */
CREATE TABLE hps.init (
	init_id		SERIAL,
	init_class	VARCHAR(255),
	PRIMARY KEY(init_id)
);

--DROP TABLE hps.init_uniform;
CREATE TABLE hps.init_uniform (
	init_range	FLOAT4,
	PRIMARY KEY(init_id)
) INHERITS (hps.init);


CREATE TABLE hps.init_normal (
	init_stdev	FLOAT4,
	PRIMARY KEY(init_id)
) INHERITS (hps.init);

--DROP TABLE hps.init_sparse;
CREATE TABLE hps.init_sparse (
	init_sparseness	FLOAT4,
	init_stdev	FLOAT4 DEFAULT 1.0,
	--mask_weights 	FLOAT4,
	PRIMARY KEY(init_id)
) INHERITS (hps.init);



CREATE TABLE hps.init_conv

/* MLP layers */
	
--DROP TABLE hps.layer CASCADE;
CREATE TABLE hps.layer (
	layer_id	SERIAL,
	layer_class	VARCHAR(255),
	layer_name	VARCHAR(255),
	dim		INT4,
	dropout_prob	FLOAT4,
	dropout_scale	FLOAT4,
	PRIMARY KEY (layer_id)
);

CREATE TABLE hps.layer_linear (
	init_id		INT8,
	init_bias 	FLOAT4 DEFAULT 0,
	W_lr_scale 	FLOAT4,
	b_lr_scale 	FLOAT4,
	max_row_norm 	FLOAT4,
	max_col_norm 	FLOAT4,
	--copy_input 	BOOLEAN DEFAULT FALSE,
	PRIMARY KEY (layer_id)
) INHERITS (hps.layer);

CREATE TABLE hps.layer_tanh (
	PRIMARY KEY (layer_id)
) INHERITS (hps.layer_linear);

CREATE TABLE hps.layer_sigmoid (
	PRIMARY KEY (layer_id)
) INHERITS (hps.layer_linear);

CREATE TABLE hps.layer_rectifiedlinear(
	left_slope 	FLOAT4 DEFAULT 0.0,
	PRIMARY KEY (layer_id)
) INHERITS (hps.layer_linear);


CREATE TABLE hps.layer_softmax (
	PRIMARY KEY (layer_id)
) INHERITS (hps.layer_linear);



CREATE TABLE hps.layer_softmaxpool (
	detector_layer_dim	INT4,
	pool_size		INT4,
	init_id			INT8,
	init_bias 		FLOAT4,
	W_lr_scale 		FLOAT4,
	b_lr_scale 		FLOAT4,
	--mask_weights
	PRIMARY KEY (layer_id)
) INHERITS (hps.layer);

CREATE TABLE hps.layer_ConvRectifiedLinear (
	output_channels	INT4,
	kernel_shape1	INT4,
	kernel_shape2	INT4,
	pool_shape1	INT4,
	pool_shape2	INT4,
	pool_stride	INT4,
	border_mode 	VARCHAR(255) DEFAULT 'valid',
	init_bias 	FLOAT4 DEFAULT 0,
	init_conv_id	INT8,
	W_lr_scale 	FLOAT4,
	b_lr_scale 	FLOAT4,
	left_slope 	FLOAT4 DEFAULT 0.0,
	max_kernel_norm FLOAT4,
	PRIMARY KEY (layer_id)
) INHERITS (hps.layer);

/* Extensions */

CREATE TABLE hps.extension (
	ext_id		SERIAL,
	ext_class	VARCHAR(255) NOT NULL,
	PRIMARY KEY (ext_id)
);

CREATE TABLE hps.ext_ExponentialDecayOverEpoch(
	decay_factor	FLOAT4, 
	min_lr		FLOAT4,
	PRIMARY KEY (ext_id)	
) INHERITS (hps.extension);

CREATE TABLE hps.ext_MomentumAdjustor(
	final_momentum	FLOAT4, 
	start_epoch	INT4, 
	saturate_epoch	INT4,
	PRIMARY KEY (ext_id)
) INHERITS (hps.extension);


--DROP TABLE hps.ext_multi;
CREATE TABLE hps.ext_multi (
	ext_array	INT8[],
	PRIMARY KEY (ext_id)
) INHERITS (hps.extension);


/* Termination Criteria */

CREATE TABLE hps.termination (
	term_id		SERIAL,
	term_class	VARCHAR(255) NOT NULL,
	PRIMARY KEY (term_id)
);
CREATE TABLE hps.term_epochcounter (
	max_epochs	INT4,
	PRIMARY KEY (term_id)
) INHERITS (hps.termination);

--DROP TABLE hps.term_multi;
CREATE TABLE hps.term_multi (
	term_array	INT8[],
	PRIMARY KEY (term_id)
) INHERITS (hps.termination);

/* Costs */

CREATE TABLE hps.cost (
	cost_id		SERIAL,
	cost_class	VARCHAR(255),
	cost_desc	TEXT,
	PRIMARY KEY (cost_id)
);

CREATE TABLE hps.cost_weightDecay (
	decay_coeff	FLOAT4,
	PRIMARY KEY (cost_id)
) INHERITS (hps.cost);

CREATE TABLE hps.cost_methodCost (
	method_name	VARCHAR(255),
	supervised	BOOLEAN,
	PRIMARY KEY (cost_id)
) INHERITS (hps.cost);

CREATE TABLE hps.cost_multi (
	cost_array	INT8[],
	PRIMARY KEY (cost_id)
) INHERITS (hps.cost);

/* Training Algorithms */

--DROP TABLE hps.trainingAlgorithm CASCADE;
CREATE TABLE hps.trainingAlgorithm (
	train_id	SERIAL,
	train_class	VARCHAR(255) NOT NULL,
	train_desc	TEXT,
	PRIMARY KEY (train_id)
);

CREATE TABLE hps.train_stochasticGradientDescent (
	learning_rate		FLOAT4 NOT NULL,
	cost_id 		INT8 NOT NULL,
	--batch_size		INT4, in config since used by both TrainingAlgorithm and Model.
	term_id			INT8 NOT NULL,
	init_momentum   	FLOAT4,
	train_iteration_mode	VARCHAR(255) DEFAULT 'random_uniform',
	PRIMARY KEY (train_id)
) INHERITS (hps.trainingAlgorithm);

/*
BEGIN;

SELECT config_id 
FROM hps.stack 
WHERE stack_id = 1 AND start_time IS NULL 
LIMIT 1 FOR UPDATE;


UPDATE hps.stack 
SET start_time = now() 
WHERE stack_id = 12 AND config_id = 32;

COMMIT;
*/

CREATE TABLE hps.validation_accuracy (
	config_id	INT8,
	accuracy	FLOAT4,
	PRIMARY KEY (config_id)
);

--DROP TABLE hps.config CASCADE;
CREATE TABLE hps.config (
	config_id	SERIAL,
	model_id	INT8 NOT NULL,
	ext_id		INT8,
	train_id	INT8 NOT NULL,
	dataset_id	INT8 NOT NULL,
	random_seed	INT4 DEFAULT 7777,
	batch_size	INT4 NOT NULL,
	experiment_id	INT8 NOT NULL, 
	start_time	TIMESTAMP,
	end_time	TIMESTAMP,
	PRIMARY KEY (config_id)
);

