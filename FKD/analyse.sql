SELECT 	a.config_id, b.epoch_count AS epoch, j.epoch_count AS end, a.dataset_id, p.preprocess_array,
	b.channel_value AS optimum_mse, j.channel_value AS final_mse, 
	k.channel_value AS train_error, l.channel_value AS final_train_error, 
	a.init_momentum, a.learning_rate, a.layer_array,  a.batch_size, 
	--a.dropout_include_probs, a.dropout_input_include_prob, a.dropout_input_scale, a.weight_decay, 
	a.ext_array, a.worker_name, a.end_time
FROM 	hps2.config_mlp_sgd_mb_ec AS a,
	(
		SELECT config_id, epoch_count, channel_value, rank()
			OVER (PARTITION BY config_id ORDER BY channel_value ASC, epoch_count ASC)
		FROM hps2.training_log
		WHERE channel_name = 'Validation MSE'
	) AS b, 
	(
		SELECT config_id, epoch_count, channel_value, rank()
			OVER (PARTITION BY config_id ORDER BY epoch_count DESC)
		FROM hps2.training_log
		WHERE channel_name = 'Validation MSE'
	) AS j, 
	(
		SELECT config_id, epoch_count, channel_value, rank()
			OVER (PARTITION BY config_id ORDER BY epoch_count DESC)
		FROM hps2.training_log
		WHERE channel_name = 'train_objective'
	) AS l, hps2.training_log AS k, hps2.dataset AS p
WHERE  a.config_id = b.config_id AND b.rank = 1  
	AND a.config_id = k.config_id AND k.channel_name = 'train_objective' AND k.epoch_count = b.epoch_count 
	AND a.config_id = j.config_id AND j.rank = 1 AND a.config_id = l.config_id AND l.rank = 1
	AND task_id = 25 AND a.dataset_id = p.dataset_id 
ORDER BY optimum_mse ASC


