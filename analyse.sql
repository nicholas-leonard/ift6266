SELECT 	b.accuracy, a.batch_size,
	g.init_momentum, g.learning_rate,
	d.layer_class, d.dim, d.dropout_prob,
	e.layer_class, e.dim, e.dropout_prob,
	h.cost_desc, i.max_epochs,
	a.config_id, a.model_id, a.ext_id, a.train_id 
FROM hps.config AS a, hps.validation_accuracy AS b, hps.model_mlp AS c, 
     hps.layer AS d, hps.layer AS e, hps.mlp_graph AS f, hps.train_stochasticGradientDescent AS g,
     hps.cost AS h, hps.term_epochCounter AS i
WHERE a.config_id = b.config_id
	AND c.model_id = a.model_id	
	AND c.input_layer_id = d.layer_id
	AND (c.model_id, c.input_layer_id) = (f.model_id, f.input_layer_id)
	AND f.output_layer_id = e.layer_id
	AND a.train_id = g.train_id
	AND g.cost_id = h.cost_id
	AND g.term_id = i.term_id
ORDER BY accuracy DESC

SELECT *
FROM hps.model_mlp AS a, hps.layer AS b
WHERE (a.model_id = 4 OR a.model_id = 1)
	AND a.input_layer_id = b.layer_id
