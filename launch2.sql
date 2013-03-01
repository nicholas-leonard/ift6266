
INSERT INTO hps2.layer_convrectifiedlinear (layer_name, layer_class, output_channels, irange, kernel_width, pool_width, pool_stride, max_kernel_norm)
VALUES 	('h1', 'convrectifiedlinear', 64, 0.05, 5, 4, 2, 1.9365),
	('h2', 'convrectifiedlinear', 64, 0.05, 5, 4, 2, 1.9365)
RETURNING layer_id;--7,8

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
					weight_decay,
					ec_max_epoch, 
					ext_array)				
VALUES ('{mlp,sgd,monitorbased,epochcounter}'::VARCHAR(255)[],'{7,8,3}'::INT8[], 64, .001, .5, 1,
	'{.00005, .00005, .00005 }'::FLOAT4[], 1000, '{1}'::INT8[])
RETURNING config_id;--6




INSERT INTO hps2.config_mlp_sgd_mb_ec (config_class,
					layer_array,
					batch_size, learning_rate, init_momentum,
					input_space_id, 
					weight_decay,
					ec_max_epoch, 
					ext_array)				
VALUES ('{mlp,sgd,monitorbased,epochcounter}'::VARCHAR(255)[],'{18,19,3}'::INT8[], 32, .001, .5, 1,
	'{.00005, .00005, .00005 }'::FLOAT4[], 1000, '{1,2}'::INT8[])
RETURNING config_id;--15



INSERT INTO hps2.layer_convrectifiedlinear (layer_name, layer_class, output_channels, irange, kernel_width, pool_width, pool_stride, max_kernel_norm)
VALUES 	('h1', 'convrectifiedlinear', 96, 0.05, 7, 4, 2, 1.9365),
	('h2', 'convrectifiedlinear', 96, 0.05, 7, 4, 2, 1.9365)
RETURNING layer_id;--20, 21


INSERT INTO hps2.config_mlp_sgd_mb_ec (config_class,
					layer_array,
					batch_size, learning_rate, init_momentum,
					input_space_id, 
					weight_decay,
					ec_max_epoch, 
					ext_array)				
VALUES ('{mlp,sgd,monitorbased,epochcounter}'::VARCHAR(255)[],'{20,21,3}'::INT8[], 64, .001, .5, 1,
	'{.00005, .00005, .00005 }'::FLOAT4[], 1000, '{1,2}'::INT8[])
RETURNING config_id;


INSERT INTO hps2.layer_convrectifiedlinear (layer_name, layer_class, output_channels, irange, kernel_width, pool_width, pool_stride, max_kernel_norm)
VALUES 	('h1', 'convrectifiedlinear', 32, 0.05, 5, 4, 2, 1.9365),
	('h2', 'convrectifiedlinear', 64, 0.05, 5, 4, 2, 1.9365),
	('h3', 'convrectifiedlinear', 96, 0.05, 5, 4, 2, 1.9365)
RETURNING layer_id;--22,23,24

INSERT INTO hps2.config_mlp_sgd_mb_ec (config_class,
					layer_array,
					batch_size, learning_rate, init_momentum,
					input_space_id, 
					weight_decay,
					ec_max_epoch, 
					ext_array)				
VALUES ('{mlp,sgd,monitorbased,epochcounter}'::VARCHAR(255)[],'{22,23,24,3}'::INT8[], 32, .001, .5, 1,
	'{.00005, .00005, .00005, .00005 }'::FLOAT4[], 1000, '{1,2}'::INT8[])
RETURNING config_id;

INSERT INTO hps2.layer_convrectifiedlinear (layer_name, layer_class, output_channels, irange, kernel_width, pool_width, pool_stride, max_kernel_norm)
VALUES 	('h1', 'convrectifiedlinear', 64, 0.05, 5, 4, 2, 1.9365),
	('h2', 'convrectifiedlinear', 64, 0.05, 5, 4, 2, 1.9365),
	('h3', 'convrectifiedlinear', 64, 0.05, 5, 4, 2, 1.9365)
RETURNING layer_id;--25,26,27

INSERT INTO hps2.config_mlp_sgd_mb_ec (config_class,
					layer_array,
					batch_size, learning_rate, init_momentum,
					input_space_id, 
					weight_decay,
					ec_max_epoch, 
					ext_array)				
VALUES ('{mlp,sgd,monitorbased,epochcounter}'::VARCHAR(255)[],'{25,26,27,3}'::INT8[], 32, .001, .5, 1,
	'{.00005, .00005, .00005, .00005 }'::FLOAT4[], 1000, '{1,2}'::INT8[])
RETURNING config_id;

INSERT INTO hps2.config_mlp_sgd_mb_ec (config_class,
					layer_array,
					batch_size, learning_rate, init_momentum,
					input_space_id, 
					ec_max_epoch, 
					ext_array)				
VALUES ('{mlp,sgd,monitorbased,epochcounter}'::VARCHAR(255)[],'{18,19,3}'::INT8[], 32, .001, .5, 1,
	1000, '{1,2}'::INT8[])
RETURNING config_id;

INSERT INTO hps2.layer_convrectifiedlinear (layer_name, layer_class, output_channels, irange, kernel_width, pool_width, pool_stride, max_kernel_norm)
VALUES 	('h1', 'convrectifiedlinear', 32, 0.05, 5, 4, 2, 1.9365),
	('h2', 'convrectifiedlinear', 48, 0.05, 5, 4, 2, 1.9365),
	('h3', 'convrectifiedlinear', 64, 0.05, 5, 4, 2, 1.9365)
RETURNING layer_id;--31,32,33

INSERT INTO hps2.config_mlp_sgd_mb_ec (config_class,
					layer_array,
					batch_size, learning_rate, init_momentum,
					input_space_id, 
					weight_decay,
					ec_max_epoch, 
					ext_array)				
VALUES ('{mlp,sgd,monitorbased,epochcounter}'::VARCHAR(255)[],'{31,32,33,3}'::INT8[], 64, .001, .5, 1,
	'{.00005, .00005, .00005, .00005 }'::FLOAT4[], 1000, '{1,2}'::INT8[])
RETURNING config_id;

INSERT INTO hps2.layer_convrectifiedlinear (layer_name, layer_class, output_channels, irange, kernel_width, pool_width, pool_stride, max_kernel_norm)
VALUES 	('h1', 'convrectifiedlinear', 64, 0.05, 5, 4, 2, 1.9365),
	('h2', 'convrectifiedlinear', 96, 0.05, 5, 4, 2, 1.9365)
RETURNING layer_id;--34,35

INSERT INTO hps2.config_mlp_sgd_mb_ec (config_class,
					layer_array,
					batch_size, learning_rate, init_momentum,
					input_space_id, 
					weight_decay,
					ec_max_epoch, 
					ext_array)				
VALUES ('{mlp,sgd,monitorbased,epochcounter}'::VARCHAR(255)[],'{34,35,3}'::INT8[], 64, .001, .5, 1,
	'{.00005, .00005, .00005 }'::FLOAT4[], 1000, '{1,2}'::INT8[])
RETURNING config_id;

INSERT INTO hps2.layer_convrectifiedlinear (layer_name, layer_class, output_channels, irange, kernel_width, pool_width, pool_stride, max_kernel_norm)
VALUES 	('h1', 'convrectifiedlinear', 32, 0.05, 5, 4, 2, 1.9365),
	('h2', 'convrectifiedlinear', 32, 0.05, 5, 4, 2, 1.9365)
RETURNING layer_id;--36,37

INSERT INTO hps2.config_mlp_sgd_mb_ec (config_class,
					layer_array,
					batch_size, learning_rate, init_momentum,
					input_space_id, 
					weight_decay,
					ec_max_epoch, 
					ext_array)				
VALUES ('{mlp,sgd,monitorbased,epochcounter}'::VARCHAR(255)[],'{36,37,3}'::INT8[], 128, .001, .5, 1,
	'{.00005, .00005, .00005 }'::FLOAT4[], 1000, '{1,2}'::INT8[])
RETURNING config_id;

INSERT INTO hps2.layer_convrectifiedlinear (layer_name, layer_class, output_channels, irange, kernel_width, pool_width, pool_stride, max_kernel_norm)
VALUES 	('h1', 'convrectifiedlinear', 48, 0.05, 5, 4, 2, 1.9365),
	('h2', 'convrectifiedlinear', 96, 0.05, 5, 4, 2, 1.9365)
RETURNING layer_id;--38,39

INSERT INTO hps2.config_mlp_sgd_mb_ec (config_class,
					layer_array,
					batch_size, learning_rate, init_momentum,
					input_space_id, 
					weight_decay,
					ec_max_epoch, 
					ext_array)				
VALUES ('{mlp,sgd,monitorbased,epochcounter}'::VARCHAR(255)[],'{38,39,3}'::INT8[], 64, .001, .5, 1,
	'{.00005, .00005, .00005 }'::FLOAT4[], 1000, '{1,2}'::INT8[])
RETURNING config_id;

INSERT INTO hps2.config_mlp_sgd_mb_ec (config_class,
					layer_array,
					batch_size, learning_rate, init_momentum,
					input_space_id, 
					weight_decay,
					ec_max_epoch, 
					ext_array)				
VALUES ('{mlp,sgd,monitorbased,epochcounter}'::VARCHAR(255)[],'{18,19,3}'::INT8[], 64, .005, .5, 1,
	'{.00005, .00005, .00005 }'::FLOAT4[], 1000, '{1,2}'::INT8[])
RETURNING config_id;

INSERT INTO hps2.config_mlp_sgd_mb_ec (config_class,
					layer_array,
					batch_size, learning_rate, init_momentum,
					input_space_id, 
					weight_decay,
					ec_max_epoch, 
					ext_array)				
VALUES ('{mlp,sgd,monitorbased,epochcounter}'::VARCHAR(255)[],'{38,39,3}'::INT8[], 64, .01, .5, 1,
	'{.00005, .00005, .00005 }'::FLOAT4[], 1000, '{1,2}'::INT8[])
RETURNING config_id;

INSERT INTO hps2.config_mlp_sgd_mb_ec (config_class,
					layer_array,
					batch_size, learning_rate, init_momentum,
					input_space_id, 
					weight_decay,
					ec_max_epoch, 
					ext_array)				
VALUES ('{mlp,sgd,monitorbased,epochcounter}'::VARCHAR(255)[],'{38,39,3}'::INT8[], 32, .01, .5, 1,
	'{.00005, .00005, .00005 }'::FLOAT4[], 1000, '{1,2}'::INT8[])
RETURNING config_id;

INSERT INTO hps2.config_mlp_sgd_mb_ec (config_class,
					layer_array,
					batch_size, learning_rate, init_momentum,
					input_space_id, 
					weight_decay,
					ec_max_epoch, 
					ext_array)				
VALUES ('{mlp,sgd,monitorbased,epochcounter}'::VARCHAR(255)[],'{38,39,3}'::INT8[], 64, .0001, .5, 1,
	'{.00005, .00005, .00005 }'::FLOAT4[], 1000, '{1,2}'::INT8[])
RETURNING config_id;


INSERT INTO hps2.config_mlp_sgd_mb_ec (config_class,
					layer_array,
					batch_size, learning_rate, init_momentum,
					input_space_id, 
					weight_decay,
					ec_max_epoch, 
					ext_array)				
VALUES ('{mlp,sgd,monitorbased,epochcounter}'::VARCHAR(255)[],'{18,19,3}'::INT8[], 64, .01, .5, 1,
	'{.00005, .00005, .00005 }'::FLOAT4[], 1000, '{1,2}'::INT8[])
RETURNING config_id;

INSERT INTO hps2.config_mlp_sgd_mb_ec (config_class,
					layer_array,
					batch_size, learning_rate, init_momentum,
					input_space_id, 
					weight_decay,
					ec_max_epoch, 
					ext_array)				
VALUES ('{mlp,sgd,monitorbased,epochcounter}'::VARCHAR(255)[],'{18,19,3}'::INT8[], 32, .01, .5, 1,
	'{.00005, .00005, .00005 }'::FLOAT4[], 1000, '{1,2}'::INT8[])
RETURNING config_id;

INSERT INTO hps2.config_mlp_sgd_mb_ec (config_class,
					layer_array,
					batch_size, learning_rate, init_momentum,
					input_space_id, 
					weight_decay,
					ec_max_epoch, 
					ext_array)				
VALUES ('{mlp,sgd,monitorbased,epochcounter}'::VARCHAR(255)[],'{18,19,3}'::INT8[], 64, .0001, .5, 1,
	'{.00005, .00005, .00005 }'::FLOAT4[], 1000, '{1,2}'::INT8[])
RETURNING config_id;