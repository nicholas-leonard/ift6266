
INSERT INTO hps2.layer_convrectifiedlinear (layer_name, layer_class, output_channels, irange, kernel_width, pool_width, pool_stride, max_kernel_norm)
VALUES 	('h1', 'convrectifiedlinear', 64, 0.05, 5, 4, 2, 1.9365),
	('h2', 'convrectifiedlinear', 96, 0.05, 5, 4, 2, 1.9365),
	('h3', 'convrectifiedlinear', 128, 0.05, 5, 4, 2, 1.9365)
RETURNING layer_id;--4,5,6

INSERT INTO hps2.layer_softmax(layer_name, layer_class, max_col_norm, n_classes, istdev)
VALUES 	('output', 'softmax', 1.9365, 7, .05)
RETURNING layer_id;--3

INSERT INTO hps2.ext_momentumAdjustor(ext_class, start_epoch, saturate_epoch, final_momentum)
VALUES ('momentumadjustor', 1,10,.99)
RETURNING ext_id;--1

INSERT INTO hps2.space_conv2dspace(space_class, num_row, num_column, num_channel)
VALUES ('conv2dspace', 48, 48, 1) 
RETURNING space_id;--1


INSERT INTO hps2.config_mlp_sgd_mb_ec (config_class,
					layer_array,
					batch_size, learning_rate, init_momentum,
					input_space_id, 
					dropout_include_probs,
					dropout_input_include_prob,
					weight_decay,
					ec_max_epoch, 
					ext_array)				
VALUES ('{mlp,sgd,monitorbased,epochcounter}'::VARCHAR(255)[],'{4,5,6,3}'::INT8[], 16, .01, .5, 1,
	'{0.5,0.5,0.5,1.0}'::FLOAT4[], 0.8, '{.000005, .000005, .000005, .000005 }'::FLOAT4[], 1000, '{1,2}'::INT8[])
RETURNING config_id;--3

INSERT INTO hps2.config_mlp_sgd_mb_ec (config_class,
					layer_array,
					batch_size, learning_rate, init_momentum,
					input_space_id, 
					dropout_include_probs,
					dropout_input_include_prob,
					weight_decay,
					ec_max_epoch, 
					ext_array)				
VALUES ('{mlp,sgd,monitorbased,epochcounter}'::VARCHAR(255)[],'{4,5,6,3}'::INT8[], 32, .01, .5, 1,
	'{0.5,0.5,0.5,1.0}'::FLOAT4[], 0.8, '{.000005, .000005, .000005, .000005 }'::FLOAT4[], 1000, '{1,2}'::INT8[])
RETURNING config_id;


INSERT INTO hps2.layer_convrectifiedlinear (layer_name, layer_class, output_channels, irange, kernel_width, pool_width, pool_stride, max_kernel_norm)
VALUES 	('h1', 'convrectifiedlinear', 96, 0.05, 5, 4, 2, 1.9365),
	('h2', 'convrectifiedlinear', 96, 0.05, 5, 4, 2, 1.9365),
	('h3', 'convrectifiedlinear', 128, 0.05, 5, 4, 2, 1.9365)
RETURNING layer_id;

INSERT INTO hps2.config_mlp_sgd_mb_ec (config_class,
					layer_array,
					batch_size, learning_rate, init_momentum,
					input_space_id, 
					dropout_include_probs,
					dropout_input_include_prob,
					weight_decay,
					ec_max_epoch, 
					ext_array)				
VALUES ('{mlp,sgd,monitorbased,epochcounter}'::VARCHAR(255)[],'{9,10,11,3}'::INT8[], 32, .01, .5, 1,
	'{0.5,0.5,0.5,1.0}'::FLOAT4[], 0.8, '{.000005, .000005, .000005, .000005 }'::FLOAT4[], 1000, '{1,2}'::INT8[])
RETURNING config_id;

INSERT INTO hps2.layer_convrectifiedlinear (layer_name, layer_class, output_channels, irange, kernel_width, pool_width, pool_stride, max_kernel_norm)
VALUES 	('h1', 'convrectifiedlinear', 96, 0.05, 5, 4, 2, 1.9365),
	('h2', 'convrectifiedlinear', 128, 0.05, 5, 4, 2, 1.9365),
	('h3', 'convrectifiedlinear', 128, 0.05, 5, 4, 2, 1.9365)
RETURNING layer_id;


INSERT INTO hps2.config_mlp_sgd_mb_ec (config_class,
					layer_array,
					batch_size, learning_rate, init_momentum,
					input_space_id, 
					dropout_include_probs,
					dropout_input_include_prob,
					weight_decay,
					ec_max_epoch, 
					ext_array)				
VALUES ('{mlp,sgd,monitorbased,epochcounter}'::VARCHAR(255)[],'{12,13,14,3}'::INT8[], 32, .01, .5, 1,
	'{0.5,0.5,0.5,1.0}'::FLOAT4[], 0.8, '{.000005, .000005, .000005, .000005 }'::FLOAT4[], 1000, '{1,2}'::INT8[])
RETURNING config_id;

INSERT INTO hps2.layer_convrectifiedlinear (layer_name, layer_class, output_channels, irange, kernel_width, pool_width, pool_stride, max_kernel_norm)
VALUES 	('h1', 'convrectifiedlinear', 96, 0.05, 5, 4, 2, 1.9365),
	('h2', 'convrectifiedlinear', 128, 0.05, 5, 4, 2, 1.9365),
	('h3', 'convrectifiedlinear', 170, 0.05, 5, 4, 2, 1.9365)
RETURNING layer_id;


INSERT INTO hps2.config_mlp_sgd_mb_ec (config_class,
					layer_array,
					batch_size, learning_rate, init_momentum,
					input_space_id, 
					dropout_include_probs,
					dropout_input_include_prob,
					weight_decay,
					ec_max_epoch, 
					ext_array)				
VALUES ('{mlp,sgd,monitorbased,epochcounter}'::VARCHAR(255)[],'{15,16,17,3}'::INT8[], 32, .01, .5, 1,
	'{0.5,0.5,0.5,1.0}'::FLOAT4[], 0.8, '{.000005, .000005, .000005, .000005 }'::FLOAT4[], 1000, '{1,2}'::INT8[])
RETURNING config_id;