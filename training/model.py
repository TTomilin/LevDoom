from tensorflow import math
from tensorflow.keras import backend as K, initializers
from tensorflow.keras.initializers import he_uniform
from tensorflow.keras.layers import BatchNormalization, Activation, Masking, Embedding, RepeatVector
from tensorflow.keras.layers import Flatten, Input, Dense, Conv2D, Lambda, TimeDistributed, LSTM, add, concatenate
from tensorflow.keras.losses import Huber, categorical_crossentropy
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.engine.base_layer import InputSpec
from typing import Tuple

BIAS_SIGMA = 0.017  # https://arxiv.org/pdf/1706.10295.pdf


def build_base_cnn(input_shape: Tuple[int], noisy: bool) -> Tuple:
    conv_layer = NoisyConv2D if noisy else Conv2D
    input_layer = Input(shape=input_shape)
    x = conv_layer(32, (8, 8), strides=(4, 4), activation='relu', kernel_initializer=he_uniform())(input_layer)
    x = conv_layer(64, (4, 4), strides=(2, 2), activation='relu', kernel_initializer=he_uniform())(x)
    x = conv_layer(64, (3, 3), strides=(1, 1), activation='relu', kernel_initializer=he_uniform())(x)
    x = Flatten()(x)
    return input_layer, x


def dqn(input_shape: Tuple[int], action_size: int, learning_rate: float, noisy: bool) -> Model:
    # Build the convolutional network section and flatten the output
    state_input, state_hidden = build_base_cnn(input_shape, noisy)

    dense_layer = NoisyDense if noisy else Dense
    output = dense_layer(action_size, activation='linear')(state_hidden)

    model = Model(inputs=state_input, outputs=output)
    model.compile(loss=Huber(), optimizer=Adam(lr=learning_rate))
    return model


def drqn(input_shape: Tuple[int], action_size: int, learning_rate: float, noisy: bool) -> Model:
    conv_layer = NoisyConv2D if noisy else Conv2D
    dense_layer = NoisyDense if noisy else Dense
    model = Sequential()
    model.add(TimeDistributed(conv_layer(32, (8, 8), strides=(4, 4), activation='relu', input_shape=input_shape)))
    model.add(TimeDistributed(conv_layer(64, (4, 4), strides=(2, 2), activation='relu')))
    model.add(TimeDistributed(conv_layer(64, (3, 3), activation='relu')))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(512, activation='tanh'))  # Use last trace for training
    model.add(dense_layer(action_size, activation='linear'))
    model.compile(loss=Huber(), optimizer=Adam(lr=learning_rate))
    return model


def dueling_dqn(input_shape: Tuple[int], action_size: int, learning_rate: float, noisy: bool) -> Model:
    # Build the convolutional network section and flatten the output
    state_input, x = build_base_cnn(input_shape, noisy)

    # Determine the type of the fully collected layer
    dense_layer = NoisyDense if noisy else Dense

    # State value tower - V
    state_value = dense_layer(256, activation='relu', kernel_initializer=he_uniform())(x)
    state_value = dense_layer(1, kernel_initializer=he_uniform())(state_value)
    state_value = Lambda(lambda s: K.expand_dims(s[:, 0], axis=-1), output_shape=(action_size,))(state_value)

    # Action advantage tower - A
    action_advantage = dense_layer(256, activation='relu', kernel_initializer=he_uniform())(x)
    action_advantage = dense_layer(action_size, kernel_initializer=he_uniform())(action_advantage)
    action_advantage = Lambda(lambda a: a[:, :] - K.mean(a[:, :], keepdims=True), output_shape=(action_size,))(
        action_advantage)

    # Merge to state-action value function Q
    state_action_value = add([state_value, action_advantage])

    model = Model(inputs=state_input, outputs=state_action_value)
    model.compile(loss=Huber(), optimizer=Adam(lr=learning_rate))
    return model


def dueling_drqn(input_shape: Tuple[int], action_size: int, learning_rate: float) -> Model:
    max_action_sequence_length = 5
    input_action_space_size = action_size + 2
    end_token = action_size + 1

    state_model_input = Input(shape=input_shape)
    state_model = Conv2D(16, (3, 3), strides=(2, 2), activation='elu', input_shape=input_shape,
                         kernel_initializer=he_uniform(), trainable=True)(state_model_input)
    state_model = Conv2D(32, (3, 3), strides=(2, 2), activation='elu', kernel_initializer=he_uniform(),
                         trainable=True)(state_model)
    state_model = Conv2D(64, (3, 3), strides=(2, 2), activation='elu', kernel_initializer=he_uniform(),
                         trainable=True)(state_model)
    state_model = Conv2D(128, (3, 3), strides=(1, 1), activation='elu', kernel_initializer=he_uniform())(
        state_model)
    state_model = Conv2D(256, (3, 3), strides=(1, 1), activation='elu', kernel_initializer=he_uniform())(
        state_model)
    state_model = Flatten()(state_model)
    state_model = Dense(512, activation='elu', kernel_initializer=he_uniform())(state_model)
    state_model = RepeatVector(max_action_sequence_length)(state_model)

    action_model_input = Input(shape=(max_action_sequence_length,))
    action_model = Masking(mask_value=end_token, input_shape=(max_action_sequence_length,))(action_model_input)
    action_model = Embedding(input_dim=input_action_space_size, output_dim=100,
                             embeddings_initializer=he_uniform(), input_length=max_action_sequence_length)(
        action_model)
    action_model = TimeDistributed(Dense(100, kernel_initializer=he_uniform(), activation='elu'))(action_model)

    # x = concatenate([state_model, action_model], concat_axis = -1)
    x = concatenate([state_model, action_model], axis=-1)
    x = LSTM(512, return_sequences=True, activation='elu', kernel_initializer=he_uniform())(x)

    # state value tower - V
    state_value = TimeDistributed(Dense(256, activation='elu', kernel_initializer=he_uniform()))(x)
    state_value = TimeDistributed(Dense(1, kernel_initializer=he_uniform()))(state_value)
    state_value = Lambda(lambda s: K.repeat_elements(s, rep=action_size, axis=2))(state_value)

    # Action advantage tower - A
    action_advantage = TimeDistributed(Dense(256, activation='elu', kernel_initializer=he_uniform()))(x)
    action_advantage = TimeDistributed(Dense(action_size, kernel_initializer=he_uniform()))(action_advantage)
    action_advantage = TimeDistributed(Lambda(lambda a: a - K.mean(a, keepdims=True, axis=-1)))(action_advantage)

    # Merge to state-action value function Q
    state_action_value = add([state_value, action_advantage])

    model = Model(inputs=[state_model_input, action_model_input], outputs=state_action_value)
    model.compile(Adam(lr=learning_rate), Huber())
    model.summary()
    return model


def value_distribution_network(input_shape: Tuple[int], action_size: int, learning_rate: float,
                               num_atoms=51) -> Model:
    """ Value Distribution Model with States as inputs and output Probability Distributions for all Actions """
    state_input, x = build_base_cnn(input_shape)
    x = Dense(512, activation='elu')(x)

    distribution_list = []
    for i in range(action_size):
        distribution_list.append(Dense(num_atoms, activation='softmax')(x))

    model = Model(inputs=state_input, outputs=distribution_list)
    model.compile(loss=categorical_crossentropy, optimizer=Adam(lr=learning_rate))
    return model


def reinforce(input_shape: Tuple[int], action_size: int, learning_rate: float) -> Model:
    model = Sequential()
    model.add(Conv2D(32, 8, 8, strides=(4, 4), input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(64, 4, 4, strides=(2, 2)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(64, 3, 3))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(64))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(32))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(output_dim=action_size, activation='softmax'))
    model.compile(loss=categorical_crossentropy, optimizer=Adam(lr=learning_rate))
    return model


def dfp_network(input_shape: Tuple[int], action_size: int, learning_rate: float, measurement_size=3,
                n_timesteps=6) -> Model:
    """ Neural Network for Direct Future Prediction (DFP) """

    # Perception Feature
    state_input, perception_feat = build_base_cnn(input_shape)
    perception_feat = Dense(512, activation='elu')(perception_feat)

    # Measurement Feature
    measurement_input = Input(shape=(measurement_size,))
    measurement_feat = Dense(128, activation='elu')(measurement_input)
    measurement_feat = Dense(128, activation='elu')(measurement_feat)
    measurement_feat = Dense(128, activation='elu')(measurement_feat)

    # Goal Feature
    goal_size = measurement_size * n_timesteps
    goal_input = Input(shape=(goal_size,))
    goal_feat = Dense(128, activation='elu')(goal_input)
    goal_feat = Dense(128, activation='elu')(goal_feat)
    goal_feat = Dense(128, activation='elu')(goal_feat)

    concat_feat = concatenate([perception_feat, measurement_feat, goal_feat])

    measurement_pred_size = measurement_size * n_timesteps  # 3 measurements, 6 timesteps

    expectation_stream = Dense(measurement_pred_size, activation='elu')(concat_feat)

    prediction_list = []
    for i in range(action_size):
        action_stream = Dense(measurement_pred_size, activation='elu')(concat_feat)
        prediction_list.append(add([action_stream, expectation_stream]))

    model = Model(inputs=[state_input, measurement_input, goal_input], outputs=prediction_list)

    adam = Adam(lr=learning_rate)
    model.compile(loss='mse', optimizer=adam)

    return model


class NoisyDense(Dense):
    def __init__(self, units, **kwargs):
        self.output_dim = units
        super(NoisyDense, self).__init__(units, **kwargs)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        self.input_dim = input_shape[-1]

        self.kernel = self.add_weight(shape=(self.input_dim, self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=None,
                                      constraint=None)

        # Second kernel (trainable weights) for random control
        self.kernel_sigma = self.add_weight(shape=(self.input_dim, self.units),
                                            initializer=initializers.Constant(BIAS_SIGMA),
                                            name='sigma_kernel',
                                            regularizer=None,
                                            constraint=None)

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=None,
                                        constraint=None)

            # Trainable, control the randomness of the bias
            self.bias_sigma = self.add_weight(shape=(self.units,),
                                              initializer=initializers.Constant(BIAS_SIGMA),
                                              name='bias_sigma',
                                              regularizer=None,
                                              constraint=None)
        else:
            self.bias = None

        self.input_spec = InputSpec(min_ndim=2, axes={-1: self.input_dim})
        self.built = True

    def call(self, inputs):
        # Generate random value matrix (newly generated with each call) - vector version
        self.kernel_epsilon = K.random_normal(shape=(self.input_dim, self.units))

        w = self.kernel + math.multiply(self.kernel_sigma, self.kernel_epsilon)
        output = K.dot(inputs, w)

        if self.use_bias:
            # Generate random bias vector
            self.bias_epsilon = K.random_normal(shape=(self.units,))

            bias = self.bias + math.multiply(self.bias_sigma, self.bias_epsilon)
            output = output + bias
        if self.activation is not None:
            output = self.activation(output)
        return output


class NoisyConv2D(Conv2D):
    """ In principle identical to the dense layer, only the (filter) kernel and the output have one more dimension """

    def build(self, input_shape):
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        self.input_dim = input_shape[channel_axis]
        self.kernel_shape = self.kernel_size + (self.input_dim, self.filters)

        self.kernel = self.add_weight(shape=self.kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)

        self.kernel_sigma = self.add_weight(shape=self.kernel_shape,
                                            initializer=initializers.Constant(BIAS_SIGMA),
                                            name='kernel_sigma',
                                            regularizer=self.kernel_regularizer,
                                            constraint=self.kernel_constraint)

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.filters,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)

            self.bias_sigma = self.add_weight(shape=(self.filters,),
                                              initializer=initializers.Constant(BIAS_SIGMA),
                                              name='bias_sigma',
                                              regularizer=self.bias_regularizer,
                                              constraint=self.bias_constraint)
        else:
            self.bias = None

        self.input_spec = InputSpec(ndim=self.rank + 2,
                                    axes={channel_axis: self.input_dim})
        self.built = True

    def call(self, inputs):
        # add noise to kernel
        self.kernel_epsilon = K.random_normal(shape=self.kernel_shape)

        w = self.kernel + math.multiply(self.kernel_sigma, self.kernel_epsilon)

        outputs = K.conv2d(
            inputs,
            w,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate)

        if self.use_bias:
            self.bias_epsilon = K.random_normal(shape=(self.filters,))

            b = self.bias + math.multiply(self.bias_sigma, self.bias_epsilon)
            outputs = K.bias_add(
                outputs,
                b,
                data_format=self.data_format)

        if self.activation is not None:
            return self.activation(outputs)
        return outputs
