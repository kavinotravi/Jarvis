import glob
import logging
import os
import numpy as np
import re
import soundfile
from keras.models import model_from_json
from numpy.lib.stride_tricks import as_strided

from char_map import char_map, index_map

logger = logging.getLogger(__name__)

def conv_output_length(input_length, filter_size, border_mode, stride,
                       dilation=1):
    """ Compute the length of the output sequence after 1D convolution along
        time. Note that this function is in line with the function used in
        Convolution1D class from Keras.
    Params:
        input_length (int): Length of the input sequence.
        filter_size (int): Width of the convolution kernel.
        border_mode (str): Only support `same` or `valid`.
        stride (int): Stride size used in 1D convolution.
        dilation (int)
    """
    if input_length is None:
        return None
    assert border_mode in {'same', 'valid'}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if border_mode == 'same':
        output_length = input_length
    elif border_mode == 'valid':
        output_length = input_length - dilated_filter_size + 1
    return (output_length + stride - 1) // stride

def save_model(save_dir, model, train_costs, validation_costs, index=None):
    """ Save the model and costs into a directory
    Params:
        save_dir (str): Directory used to store the model
        model (keras.models.Model)
        train_costs (list(float))
        validation_costs (list(float))
        index (int): If this is provided, add this index as a suffix to
            the weights (useful for checkpointing during training)
    """
    logger.info("Checkpointing model to: {}".format(save_dir))
    model_config_path = os.path.join(save_dir, 'model_config.json')
    with open(model_config_path, 'w') as model_config_file:
        model_json = model.to_json()
        model_config_file.write(model_json)
    if index is None:
        weights_format = 'model_weights.h5'
    else:
        weights_format = 'model_{}_weights.h5'.format(index)
    model_weights_file = os.path.join(save_dir, weights_format)
    model.save_weights(model_weights_file, overwrite=True)
    np.savez(os.path.join(save_dir, 'costs.npz'), train=train_costs,
             validation=validation_costs)


def load_model(load_dir, weights_file=None):
    """ Load a model and its weights from a directory
    Params:
        load_dir (str): Path the model directory
        weights_file (str): If this is not passed in, try to load the latest
            model_*weights.h5 file in the directory
    Returns:
        model (keras.models.Model)
    """
    def atoi(text):
        return int(text) if text.isdigit() else text

    def natural_keys(text):
        # From http://stackoverflow.com/questions/5967500
        return [atoi(c) for c in re.split('(\d+)', text)]

    model_config_file = os.path.join(load_dir, 'model_config.json')
    model_config = open(model_config_file).read()
    model = model_from_json(model_config)

    if weights_file is None:
        # This will find all files of name model_*weights.h5
        # We try to use the latest one saved
        weights_files = glob.glob(os.path.join(load_dir, 'model_*weights.h5'))
        weights_files.sort(key=natural_keys)
        model_weights_file = weights_files[-1]  # Use the latest model
    else:
        model_weights_file = weights_file
    model.load_weights(model_weights_file)
    return model


def argmax_decode(prediction):
    """ Decode a prediction using the highest probable character at each
        timestep. Then, simply convert the integer sequence to text
    Params:
        prediction (np.array): timestep * num_characters
    """
    int_sequence = []
    for timestep in prediction:
        int_sequence.append(np.argmax(timestep))
    tokens = []
    c_prev = -1
    for c in int_sequence:
        if c == c_prev:
            continue
        if c != 0:  # Blank
            tokens.append(c)
        c_prev = c

    text = ''.join([index_map[i] for i in tokens])
    return text


def text_to_int_sequence(text):
    """ Use a character map and convert text to an integer sequence """
    int_sequence = []
    for c in text:
        if c == ' ':
            ch = char_map['<SPACE>']
        else:
            ch = char_map[c]
        int_sequence.append(ch)
    return int_sequence


def configure_logging(console_log_level=logging.INFO,
                      console_log_format=None,
                      file_log_path=None,
                      file_log_level=logging.INFO,
                      file_log_format=None,
                      clear_handlers=False):
    """Setup logging.

    This configures either a console handler, a file handler, or both and
    adds them to the root logger.

    Args:
        console_log_level (logging level): logging level for console logger
        console_log_format (str): log format string for console logger
        file_log_path (str): full filepath for file logger output
        file_log_level (logging level): logging level for file logger
        file_log_format (str): log format string for file logger
        clear_handlers (bool): clear existing handlers from the root logger

    Note:
        A logging level of `None` will disable the handler.
    """
    if file_log_format is None:
        file_log_format = \
            '%(asctime)s %(levelname)-7s (%(name)s) %(message)s'

    if console_log_format is None:
        console_log_format = \
            '%(asctime)s %(levelname)-7s (%(name)s) %(message)s'

    # configure root logger level
    root_logger = logging.getLogger()
    root_level = root_logger.level
    if console_log_level is not None:
        root_level = min(console_log_level, root_level)
    if file_log_level is not None:
        root_level = min(file_log_level, root_level)
    root_logger.setLevel(root_level)

    # clear existing handlers
    if clear_handlers and len(root_logger.handlers) > 0:
        print("Clearing {} handlers from root logger."
              .format(len(root_logger.handlers)))
        for handler in root_logger.handlers:
            root_logger.removeHandler(handler)

    # file logger
    if file_log_path is not None and file_log_level is not None:
        log_dir = os.path.dirname(os.path.abspath(file_log_path))
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
        file_handler = logging.FileHandler(file_log_path)
        file_handler.setLevel(file_log_level)
        file_handler.setFormatter(logging.Formatter(file_log_format))
        root_logger.addHandler(file_handler)

    # console logger
    if console_log_level is not None:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(console_log_level)
        console_handler.setFormatter(logging.Formatter(console_log_format))
        root_logger.addHandler(console_handler)
