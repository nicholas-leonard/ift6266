
INSERT INTO fkd.ddm_facialkeypoint(ddm_class,which_set, start, stop)
VALUES 	('facialkeypoint','train',0,6000),
	('facialkeypoint','train',6000,7049)
RETURNING ddm_id;--4-5

INSERT INTO hps2.dataset(preprocess_array,train_ddm_id,valid_ddm_id)
VALUES ('{1,2}'::INT8[], 4,5)
RETURNING dataset_id;--5

INSERT INTO hps2.dataset(preprocess_array,train_ddm_id,valid_ddm_id)
VALUES ('{1}'::INT8[], 4,5)
RETURNING dataset_id;--6

INSERT INTO hps2.dataset(preprocess_array,train_ddm_id,valid_ddm_id)
VALUES ('{1,2,3}'::INT8[], 4,5)
RETURNING dataset_id;--7


INSERT INTO hps2.layer_convrectifiedlinear (layer_name, layer_class, output_channels, 
	irange, kernel_width, pool_width, pool_stride, max_kernel_norm)
VALUES 	('h1', 'convrectifiedlinear', 64, 0.05, 5, 4, 2, 1.9),
	('h2', 'convrectifiedlinear', 64, 0.05, 5, 4, 2, 1.9)
RETURNING layer_id;--894-895

INSERT INTO hps2.layer_linear(layer_name, layer_class, dim, irange)
VALUES ('output', 'linear', 30, 0.05)
RETURNING layer_id;--896

INSERT INTO hps2.config_mlp_sgd_mb_ec (config_class,
					dataset_name,
					task_id,
					layer_array,
					batch_size, learning_rate, init_momentum,
					input_space_id,
					ec_max_epoch, 
					ext_array,
					random_seed, dataset_id, proportional_decrease) (
SELECT '{mlp,sgd,monitorbased,epochcounter,mtc}'::VARCHAR(255)[],'FacialKeypoints', 21, ARRAY[894,895,896]::INT8[],
	32, lr, .5, 5, 100, '{3,4}'::INT8[], random()*10000000 AS random_seed, dataset_id, 0.0001
FROM (SELECT unnest('{0.1,.01,.001,.0001}'::FLOAT4[]) AS lr) AS b,
     (SELECT unnest('{5,6,7}'::FLOAT4[]) AS dataset_id) AS e
ORDER BY random_seed DESC
LIMIT 50
) RETURNING config_id;--5880-5891


--BEGIN; UPDATE hps2.config_mlp_sgd_mb_ec SET task_id = 25 WHERE config_id BETWEEN 5880 AND 5891; COMMIT;
/*BEGIN; UPDATE hps2.config_mlp_sgd_mb_ec_mix 
SET ext_array = '{3}'::FLOAT4[], init_momentum=0
WHERE task_id=21; COMMIT;*/

--BEGIN; UPDATE hps2.layer_starmixture SET learning_rate=learning_rate*10 WHERE layer_id BETWEEN 231 AND 326; COMMIT;
--BEGIN; DELETE FROM hps2.training_log WHERE config_id = 5520;