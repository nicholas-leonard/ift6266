/*
A PostgreSQL database schema for Hyper Parameter Search (HPS) using pylearn2.
Each pylearn2 class has its own Table.

TODO:
    add Foreign Key constraints
*/

--DROP SCHEMA fkd CASCADE;
CREATE SCHEMA fkd; --Hyper parameter search

--DROP TABLE fkd.ddm_FacialKeypoint;
CREATE TABLE fkd.ddm_FacialKeypoint (
	which_set	VARCHAR(255), 
	axes 		VARCHAR(255) DEFAULT 'b01c',
	start		INT4, 
	stop 		INT4, 
	PRIMARY KEY (ddm_id)
) INHERITS (hps3.ddm);

CREATE TABLE fkd.layer_multisoftmax (
	n_groups	INT4,
	n_classes	INT4, 
	irange 		FLOAT4,
        istdev 		FLOAT4,
        sparse_init 	FLOAT4, 
        W_lr_scale 	FLOAT4,
        b_lr_scale 	FLOAT4, 
        max_row_norm 	FLOAT4,
        no_affine 	BOOLEAN DEFAULT False,
        max_col_norm 	FLOAT4,
	PRIMARY KEY (layer_id)
) INHERITS (hps3.layer);