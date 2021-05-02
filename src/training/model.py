from enum import Enum, auto

from tensorflow.keras import backend as K
from tensorflow.keras.initializers import he_uniform
from tensorflow.keras.layers import BatchNormalization, Activation, Masking, Embedding, RepeatVector
from tensorflow.keras.layers import Flatten, Input, Dense, Conv2D, Lambda, TimeDistributed, LSTM, add, concatenate
from tensorflow.keras.losses import Huber, categorical_crossentropy
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from typing import Tuple


class ModelVersion:
    def __init__(self, version = 0):
        self.version = version


class Algorithm(Enum):
    DFP = auto()
    DQN = auto()
    DRQN = auto()
    DDRQN = auto()
    C51_DDQN = auto()
    REINFORCE = auto()
    DUELING_DDQN = auto()
    DUELING_DDRQN = auto()


def build_base_cnn(input_shape: Tuple[int]) -> Tuple:
    input_layer = Input(shape = input_shape)
    x = Conv2D(32, (8, 8), strides = (4, 4), activation = 'relu', kernel_initializer = he_uniform())(input_layer)
    x = Conv2D(64, (4, 4), strides = (2, 2), activation = 'relu', kernel_initializer = he_uniform())(x)
    x = Conv2D(64, (3, 3), strides = (1, 1), activation = 'relu', kernel_initializer = he_uniform())(x)
    x = Flatten()(x)
    return input_layer, x


# Model from https://flyyufelix.github.io/2017/10/12/dqn-vs-pg.html
def dueling_dqn(input_shape: Tuple[int], action_size: int, learning_rate: float) -> Model:
    state_input, x = build_base_cnn(input_shape)

    # State value tower - V
    state_value = Dense(256, activation = 'relu', kernel_initializer = he_uniform())(x)
    state_value = Dense(1, kernel_initializer = he_uniform())(state_value)
    state_value = Lambda(lambda s: K.expand_dims(s[:, 0], axis = -1), output_shape = (action_size,))(state_value)

    # Action advantage tower - A
    action_advantage = Dense(256, activation = 'relu', kernel_initializer = he_uniform())(x)
    action_advantage = Dense(action_size, kernel_initializer = he_uniform())(action_advantage)
    action_advantage = Lambda(lambda a: a[:, :] - K.mean(a[:, :], keepdims = True), output_shape = (action_size,))(
        action_advantage)

    # Merge to state-action value function Q
    state_action_value = add([state_value, action_advantage])

    model = Model(inputs = state_input, outputs = state_action_value)
    model.compile(loss = Huber(), optimizer = Adam(lr = learning_rate))
    return model


# Model from https://github.com/itaicaspi/keras-dqn-doom/blob/master/main.py
def dueling_drqn(input_shape: Tuple[int], action_size: int, learning_rate: float) -> Model:
    max_action_sequence_length = 5
    input_action_space_size = action_size + 2
    end_token = action_size + 1

    state_model_input = Input(shape = input_shape)
    state_model = Conv2D(16, (3, 3), strides = (2, 2), activation = 'elu', input_shape = input_shape, kernel_initializer = he_uniform(), trainable = True)(state_model_input)
    state_model = Conv2D(32, (3, 3), strides = (2, 2), activation = 'elu', kernel_initializer = he_uniform(), trainable = True)(state_model)
    state_model = Conv2D(64, (3, 3), strides = (2, 2), activation = 'elu', kernel_initializer = he_uniform(), trainable = True)(state_model)
    state_model = Conv2D(128, (3, 3), strides = (1, 1), activation = 'elu', kernel_initializer = he_uniform())(state_model)
    state_model = Conv2D(256, (3, 3), strides = (1, 1), activation = 'elu', kernel_initializer = he_uniform())(state_model)
    state_model = Flatten()(state_model)
    state_model = Dense(512, activation = 'elu', kernel_initializer = he_uniform())(state_model)
    state_model = RepeatVector(max_action_sequence_length)(state_model)

    action_model_input = Input(shape = (max_action_sequence_length,))
    action_model = Masking(mask_value = end_token, input_shape = (max_action_sequence_length,))(action_model_input)
    action_model = Embedding(input_dim = input_action_space_size, output_dim = 100, embeddings_initializer = he_uniform(), input_length = max_action_sequence_length)(action_model)
    action_model = TimeDistributed(Dense(100, kernel_initializer = he_uniform(), activation = 'elu'))(action_model)

    # x = concatenate([state_model, action_model], concat_axis = -1)
    x = concatenate([state_model, action_model], axis = -1)
    x = LSTM(512, return_sequences = True, activation = 'elu', kernel_initializer = he_uniform())(x)

    # state value tower - V
    state_value = TimeDistributed(Dense(256, activation = 'elu', kernel_initializer = he_uniform()))(x)
    state_value = TimeDistributed(Dense(1, kernel_initializer = he_uniform()))(state_value)
    state_value = Lambda(lambda s: K.repeat_elements(s, rep = action_size, axis = 2))(state_value)

    # Action advantage tower - A
    action_advantage = TimeDistributed(Dense(256, activation = 'elu', kernel_initializer = he_uniform()))(x)
    action_advantage = TimeDistributed(Dense(action_size, kernel_initializer = he_uniform()))(action_advantage)
    action_advantage = TimeDistributed(Lambda(lambda a: a - K.mean(a, keepdims = True, axis = -1)))(action_advantage)

    # Merge to state-action value function Q
    state_action_value = add([state_value, action_advantage])

    model = Model(inputs = [state_model_input, action_model_input], outputs = state_action_value)
    model.compile(Adam(lr = learning_rate), Huber())
    model.summary()
    return model


# Model from https://flyyufelix.github.io/2017/10/12/dqn-vs-pg.html
def value_distribution_network(input_shape: Tuple[int], action_size: int, learning_rate: float, num_atoms = 51) -> Model:
    """Model Value Distribution
    With States as inputs and output Probability Distributions for all Actions
    """
    state_input, x = build_base_cnn(input_shape)
    x = Dense(512, activation = 'elu')(x)

    distribution_list = []
    for i in range(action_size):
        distribution_list.append(Dense(num_atoms, activation = 'softmax')(x))

    model = Model(inputs = state_input, outputs = distribution_list)
    model.compile(loss = categorical_crossentropy, optimizer = Adam(lr = learning_rate))
    return model


# Model from https://flyyufelix.github.io/2017/10/12/dqn-vs-pg.html
def drqn(input_shape: Tuple[int], action_size: int, learning_rate: float) -> Model:
    model = Sequential()
    model.add(TimeDistributed(Conv2D(32, (8, 8), strides = (4, 4), activation = 'elu'), input_shape = input_shape))
    model.add(TimeDistributed(Conv2D(64, (4, 4), strides = (2, 2), activation = 'elu')))
    model.add(TimeDistributed(Conv2D(64, (3, 3), activation = 'elu')))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(512, activation = 'tanh'))  # Use last trace for training
    model.add(Dense(action_size, activation = 'linear'))
    model.compile(loss = Huber(), optimizer = Adam(lr = learning_rate))
    return model


# Model from https://flyyufelix.github.io/2017/10/12/dqn-vs-pg.html
def reinforce(input_shape: Tuple[int], action_size: int, learning_rate: float) -> Model:
    model = Sequential()
    model.add(Conv2D(32, 8, 8, strides = (4, 4), input_shape = input_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(64, 4, 4, strides = (2, 2)))
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
    model.add(Dense(output_dim = action_size, activation = 'softmax'))
    model.compile(loss = categorical_crossentropy, optimizer = Adam(lr = learning_rate))
    return model


# Model from https://github.com/flyyufelix/Direct-Future-Prediction-Keras
def dfp_network(input_shape: Tuple[int], action_size: int, learning_rate: float, measurement_size = 3, n_timesteps = 6) -> Model:
    """
    Neural Network for Direct Future Prediction (DFP)
    """

    # Perception Feature
    state_input, perception_feat = build_base_cnn(input_shape)
    perception_feat = Dense(512, activation = 'elu')(perception_feat)

    # Measurement Feature
    measurement_input = Input(shape = (measurement_size,))
    measurement_feat = Dense(128, activation = 'elu')(measurement_input)
    measurement_feat = Dense(128, activation = 'elu')(measurement_feat)
    measurement_feat = Dense(128, activation = 'elu')(measurement_feat)

    # Goal Feature
    goal_size = measurement_size * n_timesteps
    goal_input = Input(shape = (goal_size,))
    goal_feat = Dense(128, activation = 'elu')(goal_input)
    goal_feat = Dense(128, activation = 'elu')(goal_feat)
    goal_feat = Dense(128, activation = 'elu')(goal_feat)

    concat_feat = concatenate([perception_feat, measurement_feat, goal_feat])

    measurement_pred_size = measurement_size * n_timesteps  # 3 measurements, 6 timesteps

    expectation_stream = Dense(measurement_pred_size, activation = 'elu')(concat_feat)

    prediction_list = []
    for i in range(action_size):
        action_stream = Dense(measurement_pred_size, activation = 'elu')(concat_feat)
        prediction_list.append(add([action_stream, expectation_stream]))

    model = Model(inputs = [state_input, measurement_input, goal_input], outputs = prediction_list)

    adam = Adam(lr = learning_rate)
    model.compile(loss = 'mse', optimizer = adam)

    return model
