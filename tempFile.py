def build_two_output_GMF_from_hps(best_hps):
    """
    Rebuilds a GMF model with two outputs (vector + score)
    using the tuned hyperparameters.
    """

    # 1) Extract tuned hyperparameters
    gmf_dim = best_hps['gmf_dim']
    gmf_user_emb_regularization = best_hps['gmf_user_emb_regularization']
    gmf_item_emb_regularization = best_hps['gmf_item_emb_regularization']
    gmf_user_vec_dropoutRate = best_hps['gmf_user_vec_dropoutRate']
    gmf_item_vec_dropoutRate = best_hps['gmf_item_vec_dropoutRate']
    learning_rate = best_hps['lr']

    # 2) Inputs
    user_input = Input(shape=(1,), name='userId')
    item_input = Input(shape=(1,), name='movieId')

    # 3) Embedding layers
    user_emb = Embedding(
        input_dim=num_users,
        output_dim=gmf_dim,
        embeddings_regularizer=regularizers.l2(gmf_user_emb_regularization),
        name='gmf_user_emb'
    )(user_input)

    item_emb = Embedding(
        input_dim=num_items,
        output_dim=gmf_dim,
        embeddings_regularizer=regularizers.l2(gmf_item_emb_regularization),
        name='gmf_item_emb'
    )(item_input)

    # 4) Flatten + Dropout
    user_vec = Flatten(name='gmf_user_vec')(user_emb)
    user_vec = Dropout(gmf_user_vec_dropoutRate, name='gmf_user_vec_dropout')(user_vec)

    item_vec = Flatten(name='gmf_item_vec')(item_emb)
    item_vec = Dropout(gmf_item_vec_dropoutRate, name='gmf_item_vec_dropout')(item_vec)

    # 5) Interaction vector (latent features)
    gmf_vector = Multiply(name='gmf_vector')([user_vec, item_vec])

    # 6) Score head
    gmf_score = Dense(1, activation='linear', name='prediction')(gmf_vector)

    # 7) Build two‐output model
    model = Model(
        inputs=[user_input, item_input],
        outputs=[gmf_vector, gmf_score],
        name='GMF_two_output'
    )

    # 8) Compile so that only the score head contributes to loss
    model.compile(
        optimizer=Adam(learning_rate),
        loss={
            'gmf_vector': None,  # no direct loss on the vector
            'prediction': 'mse'  # supervision only on the score
        },
        loss_weights={
            'gmf_vector': 0.0,
            'prediction': 1.0
        }
    )

    return model
