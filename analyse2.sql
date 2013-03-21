/* Emotion Recognition Maxout */
SELECT 	a.config_id, b.epoch_count AS epoch, j.epoch_count AS end, b.channel_value AS accuracy, j.channel_value AS final_accuracy, 
	k.channel_value AS train_error, l.channel_value AS final_train_error, a.batch_size, a.init_momentum, a.learning_rate, a.layer_array, 
	a.dropout_include_probs, a.dropout_input_include_prob, a.dropout_input_scale, a.weight_decay, a.ext_array,
	m.*
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
	) AS j, 
	(
		SELECT config_id, epoch_count, channel_value, rank()
			OVER (PARTITION BY config_id ORDER BY epoch_count DESC)
		FROM hps2.training_log
		WHERE channel_name = 'train_output_misclass'
	) AS l, hps2.training_log AS k, hps2.layer_maxoutconvc01b AS m
WHERE  a.config_id = b.config_id AND b.rank = 1  AND a.layer_array[1] = m.layer_id
	AND a.config_id = k.config_id AND k.channel_name = 'train_output_misclass' AND k.epoch_count = b.epoch_count 
	AND a.config_id = j.config_id AND j.rank = 1 AND a.config_id = l.config_id AND l.rank = 1
	AND dataset_name = 'Emotion Recognition'
ORDER BY accuracy DESC

/* Emotion Recognition */
SELECT 	a.config_id, b.epoch_count AS epoch, j.epoch_count AS end, b.channel_value AS accuracy, j.channel_value AS final_accuracy, 
	k.channel_value AS train_error, l.channel_value AS final_train_error, a.batch_size, a.init_momentum, a.learning_rate, a.layer_array, 
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
	) AS j, 
	(
		SELECT config_id, epoch_count, channel_value, rank()
			OVER (PARTITION BY config_id ORDER BY epoch_count DESC)
		FROM hps2.training_log
		WHERE channel_name = 'train_output_misclass'
	) AS l, hps2.training_log AS k
WHERE  a.config_id = b.config_id AND b.rank = 1  
	AND a.config_id = k.config_id AND k.channel_name = 'train_output_misclass' AND k.epoch_count = b.epoch_count 
	AND a.config_id = j.config_id AND j.rank = 1 AND a.config_id = l.config_id AND l.rank = 1
	AND dataset_name = 'Emotion Recognition'
ORDER BY accuracy DESC

/* Emotion Recognition Full */
SELECT 	a.config_id, b.epoch_count AS epoch, j.epoch_count AS end, b.channel_value AS accuracy, j.channel_value AS final_accuracy, 
	k.channel_value AS train_error, l.channel_value AS final_train_error, a.batch_size, a.init_momentum, a.learning_rate, a.layer_array, 
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
	) AS j, 
	(
		SELECT config_id, epoch_count, channel_value, rank()
			OVER (PARTITION BY config_id ORDER BY epoch_count DESC)
		FROM hps2.training_log
		WHERE channel_name = 'train_output_misclass'
	) AS l, hps2.training_log AS k
WHERE  a.config_id = b.config_id AND b.rank = 1  
	AND a.config_id = k.config_id AND k.channel_name = 'train_output_misclass' AND k.epoch_count = b.epoch_count 
	AND a.config_id = j.config_id AND j.rank = 1 AND a.config_id = l.config_id AND l.rank = 1
	AND dataset_name = 'Emotion Recognition Full'
ORDER BY accuracy DESC

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
	AND dataset_name = 'Emotion recognition'
ORDER BY accuracy DESC
--BEGIN; DELETE FROM hps2.training_log WHERE config_id >= 33 AND config_id <= 37; COMMIT;
--BEGIN; DELETE FROM hps2.training_log WHERE config_id >= 56;
--UPDATE train

SELECT * FROM hps2.layer_softmax;