
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

INSERT INTO hps3.layer_rectifiedlinear (layer_class, dim, sparse_init, init_bias, max_col_norm) (
	SELECT 'rectifiedlinear', dim, 15, 0, Null
	FROM (SELECT unnest('{1000,2000}'::INT4[]) AS dim) AS a
)RETURNING layer_id;--110,111

INSERT INTO hps3.layer_softmax (layer_class,n_classes,irange)
VALUES ('softmax',98,0.05) RETURNING layer_id;--109

--DELETE FROM hps3.config_mlp_sgd WHERE config_id BETWEEN 351 AND 355 
INSERT INTO hps3.config_mlp_sgd (
	model_class,train_class,task_id,dataset_id,input_space_id,
	ext_array,term_array,layer_array,cost_array,
	batch_size,learning_rate, init_momentum, random_seed) (
	SELECT 'mlp','sgd',4,7,2,
		'{1,2}'::INT8[],'{1,2}'::INT8[],ARRAY[82,82,layer,109]::INT8[],'{1}'::INT8[],
		8,lr,0.5, random()*1000000 AS random_seed
	FROM (SELECT unnest('{.001,.0001}'::FLOAT4[]) AS lr) AS a,
	     (SELECT generate_series(110,111,1) AS layer) AS c
	ORDER BY random_seed
) RETURNING config_id;--351-354

INSERT INTO fkd.layer_multisoftmax(layer_class, n_groups, n_classes, irange)
VALUES ('multisoftmax', 30, 98, 0.05) RETURNING layer_id;--113

INSERT INTO hps3.config_mlp_sgd (
	model_class,train_class,task_id,dataset_id,input_space_id,
	ext_array,term_array,layer_array,cost_array,
	batch_size,learning_rate, init_momentum, random_seed) (
	SELECT 'mlp','sgd',5,5,2,
		'{1,2}'::INT8[],'{1,2}'::INT8[],ARRAY[81,81,layer,113]::INT8[],'{7}'::INT8[],
		8,lr,0.5, random()*1000000 AS random_seed
	FROM (SELECT unnest('{.001,.0001}'::FLOAT4[]) AS lr) AS a,
	     (SELECT generate_series(110,111,1) AS layer) AS c
) RETURNING config_id;--357-360

INSERT INTO hps3.layer_rectifiedlinear (layer_class, dim, sparse_init, init_bias, max_col_norm) (
	SELECT 'rectifiedlinear', dim, 15, 0, Null
	FROM (SELECT unnest('{3000,4000,5000}'::INT4[]) AS dim) AS a
)RETURNING layer_id;--147-149

INSERT INTO hps3.config_mlp_sgd (
	model_class,train_class,task_id,dataset_id,input_space_id,
	ext_array,term_array,layer_array,cost_array,
	batch_size,learning_rate, init_momentum, random_seed) (
	SELECT 'mlp','sgd',5,5,2,
		'{1,2}'::INT8[],'{1,2}'::INT8[],ARRAY[81,81,layer,113]::INT8[],'{7}'::INT8[],
		8,0.001,0.5, random()*1000000 AS random_seed
	FROM (SELECT generate_series(147,149,1) AS layer) AS c
) RETURNING config_id;--420-422

INSERT INTO hps3.config_mlp_sgd (
	model_class,train_class,task_id,dataset_id,input_space_id,
	ext_array,term_array,layer_array,cost_array,
	batch_size,learning_rate, init_momentum, random_seed) (
	SELECT 'mlp','sgd',5,5,2,
		'{1,2}'::INT8[],'{1,2}'::INT8[],ARRAY[81,81,81,layer,113]::INT8[],'{7}'::INT8[],
		8,0.001,0.5, random()*1000000 AS random_seed
	FROM (SELECT unnest('{110,111,147,148,149}'::INT4[]) AS layer) AS a
) RETURNING config_id;--423-427

INSERT INTO hps3.config_mlp_sgd (
	model_class,train_class,task_id,dataset_id,input_space_id,
	ext_array,term_array,layer_array,cost_array,
	batch_size,learning_rate, init_momentum, random_seed) (
	SELECT 'mlp','sgd',5,5,2,
		'{1,2}'::INT8[],'{1,2}'::INT8[],ARRAY[81,81,81,layer,113]::INT8[],'{7}'::INT8[],
		8,0.001,0.5, random()*1000000 AS random_seed
	FROM (SELECT unnest('{110,111,147,148,149}'::INT4[]) AS layer) AS a
) RETURNING config_id;--428-432

/* More hiddens */

0.001;"{81,81,147,113}"

INSERT INTO hps3.layer_rectifiedlinear (layer_class, dim, sparse_init, init_bias, max_col_norm) (
	SELECT 'rectifiedlinear', dim, 15, 0, Null
	FROM (SELECT unnest('{6000,7000}'::INT4[]) AS dim) AS a
)RETURNING layer_id;--251-252

INSERT INTO hps3.config_mlp_sgd (
	model_class,train_class,task_id,dataset_id,input_space_id,
	ext_array,term_array,layer_array,cost_array,
	batch_size,learning_rate, init_momentum, random_seed,description) (
	SELECT 'mlp','sgd',14,5,2,
		'{1,2}'::INT8[],'{1,2}'::INT8[],ARRAY[81,81,layer,113]::INT8[],'{7}'::INT8[],
		8,0.001,0.5, random()*1000000 AS random_seed, 'from 420, more hiddens'
	FROM (SELECT generate_series(148,149,1) AS layer) AS c
	UNION ALL
	SELECT 'mlp','sgd',14,5,2,
		'{1,2}'::INT8[],'{1,2}'::INT8[],ARRAY[layer,113]::INT8[],'{7}'::INT8[],
		8,0.001,0.5, random()*1000000 AS random_seed, 'mlp with multisoftmax'
	FROM (SELECT generate_series(251,252,1) AS layer) AS c
	ORDER BY random_seed
) RETURNING config_id;--1043-1046

--BEGIN; UPDATE hps3.config_mlp_sgd SET start_time=NULL,end_time=NULL,task_id=5 WHERE config_id BETWEEN 357 AND 360; COMMIT;
/*BEGIN; UPDATE hps3.config_mlp_sgd_mb_ec_mix 
SET ext_array = '{3}'::FLOAT4[], init_momentum=0
WHERE task_id=21; COMMIT;*/

--BEGIN; UPDATE hps3.layer_starmixture SET learning_rate=learning_rate*10 WHERE layer_id BETWEEN 231 AND 326; COMMIT;
--BEGIN; DELETE FROM hps3.training_log WHERE config_id = 5520;