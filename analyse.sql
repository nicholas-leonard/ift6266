/* rectified linear */
SELECT 	b.epoch_count AS epoch, j.epoch_count AS end, b.channel_value AS accuracy, j.channel_value AS final_accuracy, 
	k.channel_value AS train_error, a.batch_size, g.init_momentum, g.learning_rate, d.layer_name, d.dim, d.dropout_prob, d.init_id, d.init_bias, d.max_col_norm,
	/*e.layer_class, e.dim, e.dropout_prob,*/ h.cost_desc, a.config_id, a.model_id, a.ext_id, a.train_id, c.input_layer_id
FROM 	hps.config AS a, hps.model_mlp AS c, hps.layer_rectifiedlinear AS d, /*hps.layer AS e,*/ hps.mlp_graph AS f, 
	hps.train_stochasticGradientDescent AS g, hps.cost AS h, 
	(
		SELECT config_id, epoch_count, channel_value, rank()
			OVER (PARTITION BY config_id ORDER BY channel_value DESC, epoch_count ASC)
		FROM hps.training_log
		WHERE channel_name = 'Validation Classification Accuracy'
	) AS b, 
	(
		SELECT config_id, epoch_count, channel_value, rank()
			OVER (PARTITION BY config_id ORDER BY epoch_count DESC)
		FROM hps.training_log
		WHERE channel_name = 'Validation Classification Accuracy'
	) AS j, hps.training_log AS k
WHERE c.model_id = a.model_id	
	AND c.input_layer_id = d.layer_id AND (c.model_id, c.input_layer_id) = (f.model_id, f.input_layer_id)
	/*AND f.output_layer_id = e.layer_id */AND a.train_id = g.train_id AND g.cost_id = h.cost_id
	AND a.config_id = b.config_id AND b.rank = 1  
	AND a.config_id = k.config_id AND k.channel_name = 'train_output_misclass' AND k.epoch_count = b.epoch_count 
	AND a.config_id = j.config_id AND j.rank = 1
	--AND experiment_id = 9 --AND (experiment_id = 6 OR experiment_id = 5 OR experiment_id = 4) --AND a.config_id = 159
ORDER BY accuracy DESC

--SELECT * FROM hps.training_log WHERE config_id = 378

/* rectified linear: maginalize */
SELECT 	h.cost_desc,
	AVG(b.epoch_count), AVG(b.channel_value) AS accuracy, AVG(j.channel_value) AS final_accuracy, 
	AVG(k.channel_value) AS train_error, COUNT(*)
FROM 	hps.config AS a, hps.model_mlp AS c, hps.layer_rectifiedlinear AS d, /*hps.layer AS e,*/ hps.mlp_graph AS f, 
	hps.train_stochasticGradientDescent AS g, hps.cost AS h, hps.term_epochCounter AS i,
	(
		SELECT config_id, epoch_count, channel_value, rank()
			OVER (PARTITION BY config_id ORDER BY channel_value DESC, epoch_count ASC)
		FROM hps.training_log
		WHERE channel_name = 'Validation Classification Accuracy'
	) AS b, hps.training_log AS j, hps.training_log AS k
WHERE c.model_id = a.model_id	
	AND c.input_layer_id = d.layer_id AND (c.model_id, c.input_layer_id) = (f.model_id, f.input_layer_id)
	/*AND f.output_layer_id = e.layer_id */AND a.train_id = g.train_id AND g.cost_id = h.cost_id
	AND g.term_id = i.term_id AND a.config_id = b.config_id AND b.rank = 1  
	AND a.config_id = k.config_id AND k.channel_name = 'train_output_misclass' AND k.epoch_count = b.epoch_count 
	AND a.config_id = j.config_id AND j.channel_name = 'Validation Classification Accuracy' AND j.epoch_count = i.max_epochs
	--AND experiment_id = 6
GROUP BY h.cost_desc
ORDER BY accuracy DESC


/* convrectifiedlinear */
SELECT 	b.epoch_count AS epoch, j.epoch_count AS end, b.channel_value AS accuracy, j.channel_value AS final_accuracy, 
	k.channel_value AS train_error, a.batch_size, g.init_momentum, g.learning_rate, d.layer_name, d.output_channels,
	d.kernel_shape, d.max_kernel_norm, d.dropout_prob, d.init_id,
	/*e.layer_class, e.dim, e.dropout_prob,*/ h.cost_desc, a.config_id, a.model_id, a.ext_id, a.train_id, c.input_layer_id
FROM 	hps.config AS a, hps.model_mlp AS c, hps.layer_convrectifiedlinear AS d, /*hps.layer AS e,*/ hps.mlp_graph AS f, 
	hps.train_stochasticGradientDescent AS g, hps.cost AS h,
	(
		SELECT config_id, epoch_count, channel_value, rank()
			OVER (PARTITION BY config_id ORDER BY channel_value DESC, epoch_count ASC)
		FROM hps.training_log
		WHERE channel_name = 'Validation Classification Accuracy'
	) AS b, 
	(
		SELECT config_id, epoch_count, channel_value, rank()
			OVER (PARTITION BY config_id ORDER BY epoch_count DESC)
		FROM hps.training_log
		WHERE channel_name = 'Validation Classification Accuracy'
	) AS j, hps.training_log AS k
WHERE c.model_id = a.model_id	
	AND c.input_layer_id = d.layer_id AND (c.model_id, c.input_layer_id) = (f.model_id, f.input_layer_id)
	/*AND f.output_layer_id = e.layer_id */AND a.train_id = g.train_id AND g.cost_id = h.cost_id
	AND a.config_id = b.config_id AND b.rank = 1  
	AND a.config_id = k.config_id AND k.channel_name = 'train_output_misclass' AND k.epoch_count = b.epoch_count 
	AND a.config_id = j.config_id AND j.rank = 1
	--AND experiment_id = 8
ORDER BY accuracy DESC


SELECT 	b.cost_desc,
	AVG(b.epoch_count) AS best_epoch_avg, AVG(b.channel_value) AS avg_accuracy, AVG(i.channel_value) AS avg_final_accuracy, 
	AVG(k.channel_value) AS train_error, COUNT(*)
FROM 	(
		SELECT d.cost_desc, a.config_id, epoch_count, channel_value, rank()
			OVER (PARTITION BY d.cost_desc ORDER BY channel_value DESC, epoch_count ASC)
		FROM 	(
			SELECT a.config_id, epoch_count, channel_value, rank()
			OVER (PARTITION BY a.config_id ORDER BY channel_value DESC, epoch_count ASC)
			FROM hps.training_log AS a
			WHERE channel_name = 'Validation Classification Accuracy'
			) AS a, hps.config AS b, hps.train_stochasticGradientDescent AS c, hps.cost AS d, hps.model_mlp AS e,
			hps.layer_rectifiedlinear AS f
		WHERE a.config_id = b.config_id AND b.train_id = c.train_id AND c.cost_id = d.cost_id AND rank = 1
			AND b.model_id = e.model_id AND e.input_layer_id = f.layer_id
	) AS b, 
	(
		SELECT config_id, epoch_count, channel_value, rank()
			OVER (PARTITION BY config_id ORDER BY epoch_count DESC)
		FROM hps.training_log
		WHERE channel_name = 'Validation Classification Accuracy'
	) AS i, hps.training_log AS k
WHERE b.config_id = k.config_id AND k.channel_name = 'train_output_misclass' AND k.epoch_count = b.epoch_count 
	AND b.config_id = i.config_id AND i.rank = 1 AND b.rank <= 5
GROUP BY b.cost_desc
ORDER BY avg_accuracy DESC

--DELETE FROM hps.training_log WHERE config_id = 384