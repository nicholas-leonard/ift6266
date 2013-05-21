

INSERT INTO hps2.config_mlp_sgd_mb_ec_km (config_class,
					dataset_name,
					task_id,
					layer_array,
					batch_size, learning_rate, init_momentum,
					nvis, 
					weight_decay,
					ec_max_epoch, 
					ext_array,
					kmeans_coeff)				
VALUES ('{mlp,sgd,monitorbased,epochcounter,kmeans}'::VARCHAR(255)[],'CIFAR-100', 20, '{18,19,3}'::INT8[],
	64, .001, .5, 2, '{.00005, .00005, .00005 }'::FLOAT4[], 1000, '{1,2}'::INT8[])
RETURNING config_id;
