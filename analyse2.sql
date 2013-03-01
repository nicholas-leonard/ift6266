/* Emotion Recognition Augmented */
SELECT 	a.config_id, b.epoch_count AS epoch, j.epoch_count AS end, b.channel_value AS accuracy, j.channel_value AS final_accuracy, 
	k.channel_value AS train_error, a.batch_size, a.init_momentum, a.learning_rate, a.layer_array, 
	a.dropout_include_probs, a.dropout_input_include_prob, a.dropout_input_scale, a.weight_decay, a.ext_array
FROM 	hps2.config_mlp_sgd_mb_ec AS a,
	(
		SELECT config_id, epoch_count, channel_value, rank()
			OVER (PARTITION BY config_id ORDER BY channel_value DESC, epoch_count ASC)
		FROM hps2.training_log
		WHERE channel_name = 'Validation Classification Accuracy'
	) AS b, 
	(
		SELECT config_id, epoch_count, channel_value, rank()
			OVER (PARTITION BY config_id ORDER BY epoch_count DESC)
		FROM hps2.training_log
		WHERE channel_name = 'Validation Classification Accuracy'
	) AS j, hps2.training_log AS k
WHERE  a.config_id = b.config_id AND b.rank = 1  
	AND a.config_id = k.config_id AND k.channel_name = 'train_output_misclass' AND k.epoch_count = b.epoch_count 
	AND a.config_id = j.config_id AND j.rank = 1
	AND dataset_name = 'Emotion Recognition Augmented'
ORDER BY accuracy DESC

SELECT 	a.config_id, b.epoch_count AS epoch, j.epoch_count AS end, b.channel_value AS accuracy, j.channel_value AS final_accuracy, 
	k.channel_value AS train_error, a.batch_size, a.init_momentum, a.learning_rate, a.layer_array, 
	a.dropout_include_probs, a.dropout_input_include_prob, a.weight_decay, a.ext_array
FROM 	hps2.config_mlp_sgd_mb_ec AS a,
	(
		SELECT config_id, epoch_count, channel_value, rank()
			OVER (PARTITION BY config_id ORDER BY channel_value DESC, epoch_count ASC)
		FROM hps2.training_log
		WHERE channel_name = 'Validation Classification Accuracy'
	) AS b, 
	(
		SELECT config_id, epoch_count, channel_value, rank()
			OVER (PARTITION BY config_id ORDER BY epoch_count DESC)
		FROM hps2.training_log
		WHERE channel_name = 'Validation Classification Accuracy'
	) AS j, hps2.training_log AS k
WHERE  a.config_id = b.config_id AND b.rank = 1  
	AND a.config_id = k.config_id AND k.channel_name = 'train_output_misclass' AND k.epoch_count = b.epoch_count 
	AND a.config_id = j.config_id AND j.rank = 1
	AND dataset_name = 'Mighty Quest'
ORDER BY accuracy DESC
--DELETE FROM hps2.training_log WHERE config_id = 33