
INSERT INTO hps2.space_conv2dspace (space_class, num_row, num_column, num_channel)
VALUES ('conv2dspace', 42, 42, 1)
RETURNING space_id;--2

num_channels integer,
  num_pieces integer,
  kernel_width integer,
  pool_width integer,
  pool_stride integer,
  irange real,
  init_bias real DEFAULT 0,
  w_lr_scale real,
  b_lr_scale real,
  pad real,
  fix_pool_shape boolean DEFAULT false,
  fix_pool_stride boolean DEFAULT false,
  fix_kernel_shape boolean DEFAULT false,
  partial_sum real DEFAULT 1,
  tied_b boolean DEFAULT false,
  max_kernel_norm real,
  input_normalization real,
  output_normalization real,

INSERT INTO hps2.layer_maxoutconvc01b (
	layer_name, layer_class, num_channels, num_pieces, kernel_width, pool_width, pool_stride, irange, max_kernel_norm) (
	SELECT 'm1', 'maxoutconvc01b', num_channels, num_pieces, kernel_width, 4, 2, 0.05,  max_kernel_norm
	FROM (SELECT unnest('{64,96}'::INT4[]) AS num_channels) AS a,
		(SELECT unnest('{4,5}'::INT4[]) AS num_pieces) AS b,
		(SELECT unnest('{5,7}'::INT4[]) AS kernel_width) AS c,
		(SELECT unnest('{2,4}'::INT4[]) AS max_kernel_norm) AS d
)RETURNING layer_id;--183-198

INSERT INTO hps2.layer_maxoutconvc01b (
	layer_name, layer_class, num_channels, num_pieces, kernel_width, pool_width, pool_stride, irange, max_kernel_norm) (
	SELECT 'm2', 'maxoutconvc01b', num_channels, num_pieces, kernel_width, 4, 2, 0.05,  max_kernel_norm
	FROM (SELECT unnest('{64,96}'::INT4[]) AS num_channels) AS a,
		(SELECT unnest('{4,5}'::INT4[]) AS num_pieces) AS b,
		(SELECT unnest('{5,7}'::INT4[]) AS kernel_width) AS c,
		(SELECT unnest('{2,4}'::INT4[]) AS max_kernel_norm) AS d
)RETURNING layer_id;--199-214


INSERT INTO hps2.config_mlp_sgd_mb_ec (config_class,dataset_name,task_id,
					layer_array,batch_size, learning_rate, init_momentum, input_space_id, 
					ec_max_epoch, ext_array, mb_max_epoch,
					dropout_include_probs, dropout_input_include_prob) (				
	SELECT '{mlp,sgd,monitorbased,epochcounter}'::VARCHAR(255)[],'Emotion Recognition Full', 10, 
		ARRAY[layer1, layer2, layer3]::INT8[], 32, lr, .5, 3, 
		1000, '{1,2}'::INT8[], 40,
		'{0.5,0.5,1.0}'::FLOAT4[], 1.0
	FROM 	(SELECT generate_series(183,198,1) AS layer1, generate_series(199,214,1) AS layer2, 3 AS layer3) AS a,
			(SELECT unnest('{0.001,0.0001}'::FLOAT4[]) AS lr) AS b
) RETURNING config_id;--1216-1247

--BEGIN; UPDATE hps2.config_mlp_sgd_mb_ec SET batch_size=8 WHERE config_id >= 1218 AND config_id <= 1247; COMMIT;
--BEGIN; UPDATE hps2.config_mlp_sgd_mb_ec SET input_space_id=3 WHERE config_id >= 1184 AND config_id <= 1215; COMMIT;
--BEGIN; DELETE FROM hps2.config_mlp_sgd_mb_ec WHERE config_id >= 1184 AND config_id <= 1215; COMMIT;

INSERT INTO hps2.config_mlp_sgd_mb_ec (config_class,dataset_name,task_id,
					layer_array,batch_size, learning_rate, init_momentum, input_space_id, 
					ec_max_epoch, ext_array, mb_max_epoch,
					dropout_include_probs, dropout_input_include_prob) (				
	SELECT '{mlp,sgd,monitorbased,epochcounter}'::VARCHAR(255)[],'Emotion Recognition', 11, 
		ARRAY[layer1, layer2, layer3]::INT8[], 8, lr, .5, 3, 
		1000, '{1,2}'::INT8[], 40,
		'{0.5,0.5,1.0}'::FLOAT4[], 1.0
	FROM 	(SELECT generate_series(183,198,1) AS layer1, generate_series(199,214,1) AS layer2, 3 AS layer3) AS a,
			(SELECT unnest('{0.001,0.0001}'::FLOAT4[]) AS lr) AS b
) RETURNING config_id;--1248-1279


--phase 2:


INSERT INTO hps2.layer_maxoutconvc01b (
	layer_name, layer_class, num_channels, num_pieces, kernel_width, pool_width, pool_stride, irange, max_kernel_norm, pad) (
	SELECT 'm1', 'maxoutconvc01b', num_channels, num_pieces, 5, 4, 2, 0.05,  max_kernel_norm, 2
	FROM (SELECT unnest('{48,64}'::INT4[]) AS num_channels) AS a,
		(SELECT unnest('{5,6}'::INT4[]) AS num_pieces) AS b,
		(SELECT unnest('{4,6}'::INT4[]) AS max_kernel_norm) AS d
)RETURNING layer_id;--215-222


INSERT INTO hps2.layer_maxoutconvc01b (
	layer_name, layer_class, num_channels, num_pieces, kernel_width, pool_width, pool_stride, irange, max_kernel_norm, pad) (
	SELECT 'm2', 'maxoutconvc01b', num_channels, num_pieces, 5, 4, 2, 0.05,  max_kernel_norm, 2
	FROM (SELECT unnest('{48,64}'::INT4[]) AS num_channels) AS a,
		(SELECT unnest('{5,6}'::INT4[]) AS num_pieces) AS b,
		(SELECT unnest('{4,6}'::INT4[]) AS max_kernel_norm) AS d
)RETURNING layer_id;--223-230

--phase 3
INSERT INTO hps2.config_mlp_sgd_mb_ec (config_class,dataset_name,task_id,
					layer_array,batch_size, learning_rate, init_momentum, input_space_id, 
					ec_max_epoch, ext_array, mb_max_epoch,
					dropout_include_probs, dropout_input_include_prob) (				
	SELECT '{mlp,sgd,monitorbased,epochcounter}'::VARCHAR(255)[],'Emotion Recognition', 13, 
		ARRAY[layer1, layer2, layer3, layer4]::INT8[], 8, 0.0001, .5, 3, 
		1000, '{3,4}'::INT8[], 10,
		'{0.5,0.5,1.0}'::FLOAT4[], diip
	FROM 	(SELECT generate_series(215,222,1) AS layer1, generate_series(223,230,1) AS layer2, 
			generate_series(76,126, 25) AS layer3, 3 AS layer4) AS a,
			(SELECT unnest('{1.0,0.8}'::FLOAT4[]) AS diip) AS b
) RETURNING config_id;--1280-1295