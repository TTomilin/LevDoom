from enum import Enum, auto

from keras.layers import BatchNormalization, Activation
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Flatten, Input, Dense, Conv2D, Lambda, TimeDistributed, LSTM, add, concatenate
from tensorflow.keras.losses import Huber, categorical_crossentropy
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam


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


def build_base_cnn(input_shape):
    state_input = Input(shape = input_shape)
    x = Conv2D(32, (8, 8), strides = (4, 4), activation = 'relu')(state_input)
    x = Conv2D(64, (4, 4), strides = (2, 2), activation = 'relu')(x)
    x = Conv2D(64, (3, 3), activation = 'relu')(x)
    x = Flatten()(x)
    return state_input, x


# Model from https://flyyufelix.github.io/2017/10/12/dqn-vs-pg.html
def dueling_dqn(input_shape: [], action_size: int, learning_rate = 0.0001) -> Model:
    state_input, x = build_base_cnn(input_shape)

    # State value tower - V
    state_value = Dense(256, activation = 'relu')(x)
    state_value = Dense(1, kernel_initializer = 'uniform')(state_value)
    state_value = Lambda(lambda s: K.expand_dims(s[:, 0], axis = -1), output_shape = (action_size,))(state_value)

    # Action advantage tower - A
    action_advantage = Dense(256, activation = 'relu')(x)
    action_advantage = Dense(action_size)(action_advantage)
    action_advantage = Lambda(lambda a: a[:, :] - K.mean(a[:, :], keepdims = True), output_shape = (action_size,))(
        action_advantage)

    # Merge to state-action value function Q
    state_action_value = add([state_value, action_advantage])

    model = Model(inputs = state_input, outputs = state_action_value)
    model.compile(loss = Huber(), optimizer = Adam(lr = learning_rate))
    return model


# Model from https://flyyufelix.github.io/2017/10/12/dqn-vs-pg.html
def value_distribution_network(input_shape: [], action_size: int, learning_rate = 0.0001, num_atoms = 51):
    """Model Value Distribution
    With States as inputs and output Probability Distributions for all Actions
    """
    state_input, x = build_base_cnn(input_shape)
    x = Dense(512, activation = 'relu')(x)

    distribution_list = []
    for i in range(action_size):
        distribution_list.append(Dense(num_atoms, activation = 'softmax')(x))

    model = Model(inputs = state_input, outputs = distribution_list)
    model.compile(loss = categorical_crossentropy, optimizer = Adam(lr = learning_rate))
    return model


# Model from https://flyyufelix.github.io/2017/10/12/dqn-vs-pg.html
def drqn(input_shape: [], action_size: int, learning_rate = 0.0001) -> Model:
    model = Sequential()
    model.add(TimeDistributed(Conv2D(32, (8, 8), strides = (4, 4), activation = 'relu'), input_shape = input_shape))
    model.add(TimeDistributed(Conv2D(64, (4, 4), strides = (2, 2), activation = 'relu')))
    model.add(TimeDistributed(Conv2D(64, (3, 3), activation = 'relu')))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(512, activation = 'tanh'))  # Use last trace for training
    model.add(Dense(action_size, activation = 'linear'))
    model.compile(loss = Huber(), optimizer = Adam(lr = learning_rate))
    return model


# Model from https://flyyufelix.github.io/2017/10/12/dqn-vs-pg.html
def reinforce(input_shape: [], action_size: int, learning_rate = 0.0001) -> Model:
    model = Sequential()
    model.add(Conv2D(32, 8, 8, strides = (4, 4), input_shape = (input_shape)))
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
def dfp_network(input_shape, action_size, learning_rate, measurement_size = 3, num_timesteps = 6):
    """
    Neural Network for Direct Future Prediction (DFP)
    """

    # Perception Feature
    state_input, perception_feat = build_base_cnn(input_shape)
    perception_feat = Dense(512, activation = 'relu')(perception_feat)

    # Measurement Feature
    measurement_input = Input(shape = (measurement_size,))
    measurement_feat = Dense(128, activation = 'relu')(measurement_input)
    measurement_feat = Dense(128, activation = 'relu')(measurement_feat)
    measurement_feat = Dense(128, activation = 'relu')(measurement_feat)

    # Goal Feature
    goal_size = measurement_size * num_timesteps
    goal_input = Input(shape = (goal_size,))
    goal_feat = Dense(128, activation = 'relu')(goal_input)
    goal_feat = Dense(128, activation = 'relu')(goal_feat)
    goal_feat = Dense(128, activation = 'relu')(goal_feat)

    concat_feat = concatenate([perception_feat, measurement_feat, goal_feat])

    measurement_pred_size = measurement_size * num_timesteps  # 3 measurements, 6 timesteps

    expectation_stream = Dense(measurement_pred_size, activation = 'relu')(concat_feat)

    prediction_list = []
    for i in range(action_size):
        action_stream = Dense(measurement_pred_size, activation = 'relu')(concat_feat)
        prediction_list.append(add([action_stream, expectation_stream]))

    model = Model(inputs = [state_input, measurement_input, goal_input], outputs = prediction_list)

    adam = Adam(lr = learning_rate)
    model.compile(loss = 'mse', optimizer = adam)

    return model
