
INSERT INTO fkd.ddm_facialkeypoint(ddm_class,which_set, start, stop)
VALUES 	('facialkeypoint','train',0,6000),
	('facialkeypoint','train',6000,7049)
RETURNING ddm_id;--7-8

INSERT INTO hps3.dataset(preprocess_array,train_ddm_id,valid_ddm_id)
VALUES ('{1,2}'::INT8[], 7,8), ('{1}'::INT8[],7,8), ('{1,2,3}'::INT8[], 7,8)
RETURNING dataset_id;--5-7

INSERT INTO hps3.layer_convrectifiedlinear (layer_class, output_channels, 
	irange, kernel_width, pool_width, pool_stride, max_kernel_norm)
VALUES 	('convrectifiedlinear', 64, 0.05, 5, 3, 2, 1.9)
RETURNING layer_id;--81

INSERT INTO hps3.layer_convolution (layer_class, activation_function, border_mode, output_channels, 
	irange, kernel_width, pool_width, pool_stride, max_kernel_norm)
VALUES 	('convolution', 'tanh', 'full', 64, 0.05, 5, 3, 2, 1.9)
RETURNING layer_id;--82

INSERT INTO hps3.layer_tanh (layer_class, dim, irange)(
	SELECT 'tanh', dim, 0.05 FROM (SELECT unnest('{100,500,1000}'::FLOAT4[]) AS dim) AS b
) RETURNING layer_id;--85-87

INSERT INTO hps3.layer_linear(layer_class, dim, irange)
VALUES ('linear', 30, 0.05)
RETURNING layer_id;--84

--DELETE FROM hps3.config_mlp_sgd WHERE learning_rate >= 0.09 AND config_id BETWEEN 290 AND 325 
INSERT INTO hps3.config_mlp_sgd (
	model_class,train_class,task_id,dataset_id,input_space_id,
	ext_array,term_array,layer_array,cost_array,
	batch_size,learning_rate, init_momentum, random_seed) (
	SELECT 'mlp','sgd',4,dataset_id,2,
		'{1,2}'::INT8[],'{1,2}'::INT8[],ARRAY[82,82,layer,84]::INT8[],'{4}'::INT8[],
		8,lr,0.5, random()*1000000 AS random_seed
	FROM (SELECT unnest('{.01,.001,.0001}'::FLOAT4[]) AS lr) AS a,
	     (SELECT unnest('{5,6,7}'::FLOAT4[]) AS dataset_id) AS b,
	     (SELECT generate_series(85,87,1) AS layer) AS c
	ORDER BY random_seed
) RETURNING config_id;--290-325


--BEGIN; UPDATE hps3.config_mlp_sgd_mb_ec SET task_id = 25 WHERE config_id BETWEEN 5880 AND 5891; COMMIT;
/*BEGIN; UPDATE hps3.config_mlp_sgd_mb_ec_mix 
SET ext_array = '{3}'::FLOAT4[], init_momentum=0
WHERE task_id=21; COMMIT;*/

--BEGIN; UPDATE hps3.layer_starmixture SET learning_rate=learning_rate*10 WHERE layer_id BETWEEN 231 AND 326; COMMIT;
--BEGIN; DELETE FROM hps3.training_log WHERE config_id = 5520;