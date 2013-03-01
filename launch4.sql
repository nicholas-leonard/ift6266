
INSERT INTO hps2.space_conv2dspace (space_class, num_row, num_column, num_channel)
VALUES ('conv2dspace', 42, 42, 1)
RETURNING space_id;--2

INSERT INTO hps2.config_mlp_sgd_mb_ec (config_class,
					dataset_name,
					task_id,
					layer_array,
					batch_size, learning_rate, init_momentum,
					input_space_id, 
					weight_decay,
					ec_max_epoch, 
					ext_array)				
VALUES ('{mlp,sgd,monitorbased,epochcounter}'::VARCHAR(255)[],'Emotion Recognition Augmented', 2, '{18,19,3}'::INT8[],
	64, .001, .5, 2, '{.00005, .00005, .00005 }'::FLOAT4[], 1000, '{1,2}'::INT8[])
RETURNING config_id;

INSERT INTO hps2.config_mlp_sgd_mb_ec (config_class,
					dataset_name,
					task_id,
					layer_array,
					batch_size, learning_rate, init_momentum,
					input_space_id, 
					weight_decay,
					ec_max_epoch, 
					ext_array)				
VALUES ('{mlp,sgd,monitorbased,epochcounter}'::VARCHAR(255)[],'Emotion Recognition Augmented', 4, '{4,5,3}'::INT8[],
	32, .001, .5, 2, '{.00005, .00005, .00005 }'::FLOAT4[], 1000, '{1,2}'::INT8[]),
	('{mlp,sgd,monitorbased,epochcounter}'::VARCHAR(255)[],'Emotion Recognition Augmented', 5, '{4,5,6,3}'::INT8[],
	32, .001, .5, 2, '{.00001, .00005, .00005 , .00005}'::FLOAT4[], 1000, '{1,2}'::INT8[]),
	('{mlp,sgd,monitorbased,epochcounter}'::VARCHAR(255)[],'Emotion Recognition Augmented', 4, '{12,13,3}'::INT8[],
	32, .001, .5, 2, '{.00005, .00005, .00005 }'::FLOAT4[], 1000, '{1,2}'::INT8[]),
	('{mlp,sgd,monitorbased,epochcounter}'::VARCHAR(255)[],'Emotion Recognition Augmented', 5, '{12,13,14,3}'::INT8[],
	32, .001, .5, 2, '{.00001, .00005, .00005, .00005}'::FLOAT4[], 1000, '{1,2}'::INT8[]),
	('{mlp,sgd,monitorbased,epochcounter}'::VARCHAR(255)[],'Emotion Recognition Augmented', 5, '{25,28,27,3}'::INT8[],
	32, .001, .5, 2, '{.00001, .00005, .00005, .00005}'::FLOAT4[], 1000, '{1,2}'::INT8[]),
	('{mlp,sgd,monitorbased,epochcounter}'::VARCHAR(255)[],'Emotion Recognition Augmented', 4, '{38,39,3}'::INT8[],
	32, .001, .5, 2, '{.00005, .00005, .00005 }'::FLOAT4[], 1000, '{1,2}'::INT8[])
RETURNING config_id;41-46