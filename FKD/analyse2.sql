SELECT 	a.task_id, a.config_id, b.epoch_count AS epoch, j.epoch_count AS end, a.dataset_id, p.preprocess_array,
	b.channel_value AS optimum_valid_nll, j.channel_value AS final_valid_nll, 
	k.channel_value AS optimum_train_nll, l.channel_value AS final_train_nll, 
	a.init_momentum, a.learning_rate, a.layer_array,  a.batch_size, 
	--a.dropout_include_probs, a.dropout_input_include_prob, a.dropout_input_scale, a.weight_decay, 
	a.ext_array, a.worker_name, a.start_time, a.end_time
FROM 	hps3.config_mlp_sgd AS a,
	(
		SELECT config_id, epoch_count, channel_value, rank()
			OVER (PARTITION BY config_id ORDER BY channel_value ASC, epoch_count ASC)
		FROM hps3.training_log
		WHERE channel_name = 'valid_hps_cost'
	) AS b, 
	(
		SELECT config_id, epoch_count, channel_value, rank()
			OVER (PARTITION BY config_id ORDER BY epoch_count DESC)
		FROM hps3.training_log
		WHERE channel_name = 'valid_hps_cost'
	) AS j, 
	(
		SELECT config_id, epoch_count, channel_value, rank()
			OVER (PARTITION BY config_id ORDER BY epoch_count DESC)
		FROM hps3.training_log
		WHERE channel_name = 'train_objective'
	) AS l, hps3.training_log AS k, hps3.dataset AS p
WHERE  a.config_id = b.config_id AND b.rank = 1  
	AND a.config_id = k.config_id AND k.channel_name = 'train_objective' AND k.epoch_count = b.epoch_count 
	AND a.config_id = j.config_id AND j.rank = 1 AND a.config_id = l.config_id AND l.rank = 1
	AND (task_id = 5 OR task_id = 14) AND a.dataset_id = p.dataset_id 
ORDER BY optimum_valid_nll ASC

/*
SELECT a.epoch_count, a.channel_value, b.channel_value
FROM hps3.training_log AS a, hps3.training_log AS b
WHERE a.config_id = 358 AND (b.config_id, b.epoch_count) = (a.config_id, a.epoch_count)
	AND a.channel_name = 'train_objective' AND b.channel_name = 'valid_hps_cost'
ORDER BY epoch_count ASC

SELECT * FROM hps3.training_log WHERE config_id = 332 AND channel_name = 'train_objective'
ORDER BY epoch_count DESC

SELECT 	a.config_id, b.epoch_count AS epoch, j.epoch_count AS end, s.dim, a.dataset_id, p.preprocess_array,
	b.channel_value AS optimum_valid_nll, j.channel_value AS final_valid_nll, 
	k.channel_value AS optimum_train_nll, l.channel_value AS final_train_nll, 
	a.init_momentum, a.learning_rate, a.layer_array,  a.batch_size, 
	--a.dropout_include_probs, a.dropout_input_include_prob, a.dropout_input_scale, a.weight_decay, 
	a.ext_array, a.worker_name, a.start_time, a.end_time
FROM 	hps3.config_mlp_sgd AS a,
	(
		SELECT config_id, epoch_count, channel_value, rank()
			OVER (PARTITION BY config_id ORDER BY channel_value ASC, epoch_count ASC)
		FROM hps3.training_log
		WHERE channel_name = 'valid_hps_cost'
	) AS b, 
	(
		SELECT config_id, epoch_count, channel_value, rank()
			OVER (PARTITION BY config_id ORDER BY epoch_count DESC)
		FROM hps3.training_log
		WHERE channel_name = 'valid_hps_cost'
	) AS j, 
	(
		SELECT config_id, epoch_count, channel_value, rank()
			OVER (PARTITION BY config_id ORDER BY epoch_count DESC)
		FROM hps3.training_log
		WHERE channel_name = 'train_objective'
	) AS l, hps3.training_log AS k, hps3.dataset AS p, hps3.layer_rectifiedlinear AS s
WHERE  a.config_id = b.config_id AND b.rank = 1 AND a.layer_array[3] = s.layer_id
	AND a.config_id = k.config_id AND k.channel_name = 'train_objective' AND k.epoch_count = b.epoch_count 
	AND a.config_id = j.config_id AND j.rank = 1 AND a.config_id = l.config_id AND l.rank = 1
	AND task_id = 5 AND a.dataset_id = p.dataset_id 
ORDER BY optimum_valid_nll ASC
*/