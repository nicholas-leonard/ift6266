/* dataset */

INSERT INTO hps.dataset (dataset_desc, dataset_nvis, dataset_nout)
VALUES ('Emotional Recognition:
	input: 48x48 pixel gray values (between 0 and 255) 
	target: emotion category (beween 0 and 6: anger=0, disgust=1, 
	fear=2, happy=3, sad=4, surprise=5, neutral=6)
	auxiliary information which can be used but not as 
	input for predicting emotions at test time: identity 
	(-1: unknown, positive integers = ID, not all contiguous integer values). 
	The auxiliary file contains the identity associated with all the training examples.',
	2304, 7)
RETURNING dataset_id;--1

/* experiments */

INSERT INTO hps.experiment (experiment_desc)
VALUES ('Trying out rectified linear layers in MLP')
RETURNING experiment_id;--1

/* initializers */

INSERT INTO hps.init_uniform (init_class, init_range)
VALUES ('uniform', 0) 
RETURNING init_id;--1

INSERT INTO hps.init_sparse (init_class, init_sparseness)
VALUES ('sparse', 0.03) 
RETURNING init_id;--2

/* layers */

INSERT INTO hps.layer_rectifiedlinear (
	layer_class, layer_name,
	dim, init_id)
VALUES ('rectifiedlinear', 'rectified', 400, 2)
RETURNING layer_id;--9

INSERT INTO hps.layer_rectifiedlinear (
	layer_class, layer_name,
	dim, init_id)
VALUES ('rectifiedlinear', 'rectified', 500, 2)
RETURNING layer_id;--1

INSERT INTO hps.layer_softmax (
	layer_class, layer_name,
	dim, init_id)
VALUES ('softmax', 'output', 7, 1)
RETURNING layer_id;--2


INSERT INTO hps.layer_rectifiedlinear (
	dropout_prob,
	layer_class, layer_name,
	dim, init_id)
VALUES (0.2, 'rectifiedlinear', 'rectified', 1000, 2)
RETURNING layer_id;--4

INSERT INTO hps.layer_rectifiedlinear (
	dropout_prob,
	layer_class, layer_name,
	dim, init_id)
VALUES (0.5, 'rectifiedlinear', 'rectified2', 1000, 2)
RETURNING layer_id;--6


INSERT INTO hps.layer_rectifiedlinear (
	dropout_prob,
	layer_class, layer_name,
	dim, init_id)
VALUES (0.2, 'rectifiedlinear', 'rectified1', 1500, 2)
RETURNING layer_id;--7

INSERT INTO hps.layer_rectifiedlinear (
	dropout_prob,
	layer_class, layer_name,
	dim, init_id)
VALUES (0.5, 'rectifiedlinear', 'rectified2', 1500, 2)
RETURNING layer_id;--8

INSERT INTO hps.layer_softmax (
	dropout_prob,
	layer_class, layer_name,
	dim, init_id)
VALUES (0.5, 'softmax', 'output', 7, 1)
RETURNING layer_id;--5

/* extensions */
INSERT INTO hps.ext_ExponentialDecayOverEpoch(ext_class, decay_factor, min_lr) 
VALUES ('exponentialdecayoverepoch', 0.998, 0.000001)
RETURNING ext_id;--1

INSERT INTO hps.ext_MomentumAdjustor(ext_class, final_momentum, start_epoch, saturate_epoch) 
VALUES ('momentumadjustor', 0.99, 0, 300)
RETURNING ext_id;--2

INSERT INTO hps.ext_multi (ext_array)
VALUES ('{1,2}'::INT8[])
RETURNING ext_id;--3

/* models (MLP) */

INSERT INTO hps.model_mlp (model_class, input_layer_id, output_layer_id)
VALUES ('mlp', 1, 2)
RETURNING model_id;--1

INSERT INTO hps.mlp_graph (model_id, input_layer_id, output_layer_id)
VALUES (1, 1, 2)

INSERT INTO hps.model_mlp (model_class, input_layer_id, output_layer_id)
VALUES ('mlp', 4, 5)
RETURNING model_id;--2

INSERT INTO hps.mlp_graph (model_id, input_layer_id, output_layer_id)
VALUES (2, 4, 5)

INSERT INTO hps.model_mlp (model_class, input_layer_id, output_layer_id)
VALUES ('mlp', 4, 5)
RETURNING model_id;--3

INSERT INTO hps.mlp_graph (model_id, input_layer_id, output_layer_id)
VALUES (3, 4, 6), (3, 6, 5);

INSERT INTO hps.model_mlp (model_class, input_layer_id, output_layer_id)
VALUES ('mlp', 7, 5)
RETURNING model_id;--4

INSERT INTO hps.mlp_graph (model_id, input_layer_id, output_layer_id)
VALUES (4, 7, 5)

INSERT INTO hps.model_mlp (model_class, input_layer_id, output_layer_id)
VALUES ('mlp', 7, 5)
RETURNING model_id;--5

INSERT INTO hps.mlp_graph (model_id, input_layer_id, output_layer_id)
VALUES (5, 7, 8), (5, 8, 5);

INSERT INTO hps.model_mlp (model_class, input_layer_id, output_layer_id)
VALUES ('mlp', 9, 2)
RETURNING model_id;--6

INSERT INTO hps.mlp_graph (model_id, input_layer_id, output_layer_id)
VALUES (6, 9, 2)

/* termination criteria*/

INSERT INTO hps.term_epochcounter (term_class, max_epochs)
VALUES ('epochcounter', 1000)
RETURNING term_id;--1

INSERT INTO hps.term_epochcounter (term_class, max_epochs)
VALUES ('epochcounter', 1500)
RETURNING term_id;--2

/* training algorithms */

INSERT INTO hps.train_stochasticGradientDescent (
	train_class, learning_rate, term_id, init_momentum)
VALUES ('stochasticgradientdescent', 0.00001, 1, 0.5)
RETURNING train_id;

INSERT INTO hps.train_stochasticGradientDescent (
	train_class, learning_rate, term_id, init_momentum)
VALUES ('stochasticgradientdescent', 0.000001, 1, 0.5)
RETURNING train_id;--2

INSERT INTO hps.train_stochasticGradientDescent (
	train_class, learning_rate, term_id, init_momentum)
VALUES ('stochasticgradientdescent', 0.0001, 1, 0.5)
RETURNING train_id;--3

INSERT INTO hps.train_stochasticGradientDescent (
	train_class, learning_rate, term_id, init_momentum)
VALUES ('stochasticgradientdescent', 0.001, 1, 0.5)
RETURNING train_id;--4

/* configurations */

INSERT INTO hps.config (model_id, ext_id, train_id, dataset_id,
			batch_size, experiment_id)
VALUES (1, 3, 1, 1, 32, 1);

INSERT INTO hps.config (model_id, ext_id, train_id, dataset_id,
			batch_size, experiment_id)
VALUES (1, 3, 1, 1, 64, 1);

INSERT INTO hps.config (model_id, ext_id, train_id, dataset_id,
			batch_size, experiment_id)
VALUES (1, 3, 1, 1, 128, 1);

INSERT INTO hps.config (model_id, ext_id, train_id, dataset_id,
			batch_size, experiment_id)
VALUES 	(1, 3, 2, 1, 16, 1),
	(1, 3, 2, 1, 32, 1),
	(1, 3, 2, 1, 64, 1);

INSERT INTO hps.config (model_id, ext_id, train_id, dataset_id,
			batch_size, experiment_id)
VALUES 	(1, 3, 3, 1, 16, 1),
	(1, 3, 3, 1, 32, 1),
	(1, 3, 3, 1, 64, 1);


INSERT INTO hps.config (model_id, ext_id, train_id, dataset_id,
			batch_size, experiment_id)
VALUES 	(2, 3, 1, 1, 16, 1),
	(2, 3, 1, 1, 32, 1),
	(2, 3, 1, 1, 64, 1),
	(2, 3, 2, 1, 16, 1),
	(2, 3, 2, 1, 32, 1),
	(2, 3, 2, 1, 64, 1),
	(2, 3, 3, 1, 16, 1),
	(2, 3, 3, 1, 32, 1),
	(2, 3, 3, 1, 64, 1);

INSERT INTO hps.config (model_id, ext_id, train_id, dataset_id,
			batch_size, experiment_id)
VALUES 	(3, 3, 1, 1, 16, 1),
	(3, 3, 1, 1, 32, 1),
	(3, 3, 1, 1, 64, 1),
	(3, 3, 2, 1, 16, 1),
	(3, 3, 2, 1, 32, 1),
	(3, 3, 2, 1, 64, 1),
	(3, 3, 3, 1, 16, 1),
	(3, 3, 3, 1, 32, 1),
	(3, 3, 3, 1, 64, 1);

INSERT INTO hps.config (model_id, ext_id, train_id, dataset_id,
			batch_size, experiment_id)
VALUES 	(4, 3, 1, 1, 16, 1),
	(4, 3, 1, 1, 32, 1),
	(4, 3, 1, 1, 64, 1),
	(4, 3, 2, 1, 16, 1),
	(4, 3, 2, 1, 32, 1),
	(4, 3, 2, 1, 64, 1),
	(4, 3, 3, 1, 16, 1),
	(4, 3, 3, 1, 32, 1),
	(4, 3, 3, 1, 64, 1);

INSERT INTO hps.config (model_id, ext_id, train_id, dataset_id,
			batch_size, experiment_id)
VALUES 	(5, 3, 1, 1, 16, 1),
	(5, 3, 1, 1, 32, 1),
	(5, 3, 1, 1, 64, 1),
	(5, 3, 2, 1, 16, 1),
	(5, 3, 2, 1, 32, 1),
	(5, 3, 2, 1, 64, 1),
	(5, 3, 3, 1, 16, 1),
	(5, 3, 3, 1, 32, 1),
	(5, 3, 3, 1, 64, 1);

INSERT INTO hps.config (model_id, ext_id, train_id, dataset_id,
			batch_size, experiment_id)
VALUES 	(6, 3, 2, 1, 16, 1),
	(6, 3, 2, 1, 32, 1),
	(6, 3, 2, 1, 64, 1),
	(6, 3, 2, 1, 128, 1);


--from best model:
---0.734375;45;4;3;3;64;"rectifiedlinear";"rectified1";1500;0.2;"softmax";"output";7;0.5
INSERT INTO hps.model_mlp (model_class, input_layer_id, output_layer_id)
VALUES ('mlp', 7, 5)
RETURNING model_id;--4

INSERT INTO hps.layer_rectifiedlinear (
	dropout_prob,
	layer_class, layer_name,
	dim, init_id)
VALUES (0.2, 'rectifiedlinear', 'rectified1', 1500, 2)
RETURNING layer_id;--7

INSERT INTO hps.mlp_graph (model_id, input_layer_id, output_layer_id)
VALUES (4, 7, 5)

INSERT INTO hps.config (model_id, ext_id, train_id, dataset_id, batch_size, experiment_id)
VALUES 	(4, 3, 3, 1, 8, 1);

INSERT INTO hps.train_stochasticGradientDescent (
	train_class, learning_rate, term_id, init_momentum)
VALUES ('stochasticgradientdescent', 0.0001, 1, 0.5)
RETURNING train_id;--3

INSERT INTO hps.train_stochasticGradientDescent (
	train_class, learning_rate, term_id, init_momentum)
VALUES ('stochasticgradientdescent', 0.0001, 1, 0.1)
RETURNING train_id;--5

INSERT INTO hps.train_stochasticGradientDescent (
	train_class, learning_rate, term_id, init_momentum)
VALUES ('stochasticgradientdescent', 0.0001, 1, 0.0)
RETURNING train_id;--6

INSERT INTO hps.train_stochasticGradientDescent (
	train_class, learning_rate, term_id, init_momentum)
VALUES ('stochasticgradientdescent', 0.0001, 2, 0.5)
RETURNING train_id;--7


INSERT INTO hps.train_stochasticGradientDescent (
	train_class, learning_rate, term_id, init_momentum)
VALUES ('stochasticgradientdescent', 0.001, 2, 0.5)
RETURNING train_id;--8

/* extensions */
INSERT INTO hps.ext_ExponentialDecayOverEpoch(ext_class, decay_factor, min_lr) 
VALUES ('exponentialdecayoverepoch', 0.998, 0.000001)
RETURNING ext_id;--1

INSERT INTO hps.ext_MomentumAdjustor(ext_class, final_momentum, start_epoch, saturate_epoch) 
VALUES ('momentumadjustor', 0.99, 0, 300)
RETURNING ext_id;--2

INSERT INTO hps.ext_multi (ext_array)
VALUES ('{1,2}'::INT8[])
RETURNING ext_id;--3

INSERT INTO hps.ext_ExponentialDecayOverEpoch(ext_class, decay_factor, min_lr) 
VALUES ('exponentialdecayoverepoch', 0.998, 0.00001)
RETURNING ext_id;--5

INSERT INTO hps.ext_MomentumAdjustor(ext_class, final_momentum, start_epoch, saturate_epoch) 
VALUES ('momentumadjustor', 0.99, 100, 500)
RETURNING ext_id;--4

INSERT INTO hps.ext_multi (ext_array)
VALUES ('{1,4}'::INT8[]), ('{5,2}'::INT8[]), ('{5,4}'::INT8[])
RETURNING ext_id;--6, 7, 8

-- variants:

INSERT INTO hps.layer_rectifiedlinear (
	dropout_prob,
	layer_class, layer_name,
	dim, init_id)
VALUES (0.2, 'rectifiedlinear', 'rectified1', 2000, 2)
RETURNING layer_id;--10

INSERT INTO hps.layer_rectifiedlinear (
	dropout_prob,
	layer_class, layer_name,
	dim, init_id)
VALUES (0.2, 'rectifiedlinear', 'rectified1', 2000, 2)
RETURNING layer_id;

INSERT INTO hps.layer_rectifiedlinear (
	dropout_prob,
	layer_class, layer_name,
	dim, init_id)
VALUES (0.2, 'rectifiedlinear', 'rectified1', 2500, 2)
RETURNING layer_id;--11

INSERT INTO hps.model_mlp (model_class, input_layer_id, output_layer_id)
VALUES ('mlp', 10, 5)
RETURNING model_id;--7

INSERT INTO hps.model_mlp (model_class, input_layer_id, output_layer_id)
VALUES ('mlp', 11, 5)
RETURNING model_id;--8

INSERT INTO hps.mlp_graph (model_id, input_layer_id, output_layer_id)
VALUES (7, 10, 5);

INSERT INTO hps.mlp_graph (model_id, input_layer_id, output_layer_id)
VALUES (8, 11, 5);


INSERT INTO hps.config (model_id, ext_id, train_id, dataset_id, batch_size, experiment_id)
VALUES 	(4, 3, 3, 1, 64, 1)

INSERT INTO hps.config (model_id, ext_id, train_id, dataset_id, batch_size, experiment_id)
VALUES 	(4, 3, 3, 1, 8, 1),
	(4, 3, 4, 1, 16, 1),
	(4, 3, 4, 1, 8, 1);

INSERT INTO hps.config (model_id, ext_id, train_id, dataset_id, batch_size, experiment_id)
VALUES 	(4, 3, 4, 1, 64, 1),
	(7, 3, 3, 1, 64, 1),
	(8, 3, 3, 1, 64, 1),
	(4, 3, 5, 1, 64, 1),
	(4, 2, 4, 1, 64, 1),
	(4, 1, 4, 1, 64, 1),
	(4, 1, 6, 1, 64, 1),
	(4, 6, 4, 1, 64, 1),
	(4, 7, 4, 1, 64, 1),
	(4, 8, 4, 1, 64, 1),
	(4, 6, 5, 1, 64, 1),
	(4, 7, 5, 1, 64, 1),
	(4, 8, 5, 1, 64, 1);

--best one hidden layer:
--7;3;3;1;64

INSERT INTO hps.config (model_id, ext_id, train_id, dataset_id, batch_size, experiment_id)
VALUES 	(7, 3, 7, 1, 64, 1),
	(7, 3, 7, 1, 2, 1),
	(7, 3, 7, 1, 8, 1),
	(7, 3, 7, 1, 16, 1),
	(7, 3, 7, 1, 32, 1),
	(7, 3, 7, 1, 128, 1),
	(7, 3, 8, 1, 64, 1),
	(7, 3, 8, 1, 2, 1),
	(7, 3, 8, 1, 8, 1),
	(7, 3, 8, 1, 16, 1),
	(7, 3, 8, 1, 32, 1),
	(7, 3, 8, 1, 128, 1);
	--add ext_id differences


--best two hidden layer:
--0.701172;35;5;3;3;32;"rectifiedlinear";"rectified1";1500;0.2;"rectifiedlinear";"rectified2";1500;0.5
INSERT INTO hps.config (model_id, ext_id, train_id, dataset_id, batch_size, experiment_id)
VALUES 	(5, 3, 4, 1, 64, 1),

--old
INSERT INTO hps.experiment (experiment_desc)
VALUES ('tried rectified linear with 4000 and 5000 weights, and explored different measures of sparseness')
RETURNING experiment_id--4

INSERT INTO hps.experiment (experiment_desc)
VALUES ('fixed an error: generating best model')
RETURNING experiment_id--5

--best rectified linear 5000 dim: trying new init patterns:

INSERT INTO hps.experiment (experiment_desc)
VALUES ('grid search on batch_sizes, training algorithms (momentums) and models (normal initialization with different init biases)')
RETURNING experiment_id--6

INSERT INTO hps.config (model_id, train_id, dataset_id, batch_size, experiment_id) (
	SELECT model_id,train_id,1,batch_size,6
	FROM ( SELECT generate_series(20,31) AS model_id ) AS a,
		( SELECT unnest('{16,19,18}'::INT4[]) AS train_id) AS b,
	     ( SELECT unnest('{16,64,256}'::INT4[]) AS batch_size ) AS c
);

--uses prior knowledge (conv net):

INSERT INTO hps.experiment (experiment_desc)
VALUES ('grid search on conv net with 1 convrectlinear layer (kernel shape, output channels, max_norm) and a softmax output, training algorithms (momentums and learning rates)')
RETURNING experiment_id--7

INSERT INTO hps.config (model_id, train_id, dataset_id, batch_size, experiment_id) (
	SELECT model_id,train_id,1,64,7
	FROM ( SELECT generate_series(32,39) AS model_id ) AS a,
		( SELECT unnest('{22,23,24,25,26,27}'::INT4[]) AS train_id) AS b
);

--trial 3 (conv net):

INSERT INTO hps.experiment (experiment_desc)
VALUES ('')
RETURNING experiment_id--

INSERT INTO hps.config (model_id, train_id, dataset_id, batch_size, experiment_id) (
	SELECT model_id,train_id,1,batch_size,8
	FROM ( SELECT unnest('{45,46,47,48,49,50}'::INT4[]) AS model_id ) AS a,
		( SELECT unnest('{30,31,32}'::INT4[]) AS train_id) AS b,
		( SELECT unnest('{16,64,256}'::INT4[]) AS batch_size) AS c
);


--experiment template (use 2 screens):



INSERT INTO hps.experiment (experiment_desc)
VALUES ('')
RETURNING experiment_id

/* layers */

INSERT INTO hps.layer_rectifiedlinear (
	dropout_prob,
	layer_class, layer_name,
	dim, init_id)
VALUES (0.2, 'rectifiedlinear', 'rectified1', 2000, 2)
RETURNING layer_id;--10

INSERT INTO hps.layer_rectifiedlinear (
	dropout_prob,
	layer_class, layer_name,
	dim, init_id)
VALUES (0.2, 'rectifiedlinear', 'rectified1', 2000, 2)
RETURNING layer_id;--10

INSERT INTO hps.config (model_id, train_id, dataset_id, batch_size, experiment_id) (
	SELECT model_id,train_id,1,batch_size,8
	FROM ( SELECT unnest('{45,46,47,48,49,50}'::INT4[]) AS model_id ) AS a,
		( SELECT unnest('{30,31,32}'::INT4[]) AS train_id) AS b,
		( SELECT unnest('{16,64,256}'::INT4[]) AS batch_size) AS c
);

