import time
from lib import path_lib

MODEL = 'default'
VARIANT = 'default'

LEVEL_PATH = 'PATH'
LEVEL_PARAM = 'PARAM'
LEVEL_DATA = 'DATA'
LEVEL_DETAIL = 'DETAIL'
LEVEL_RET = 'RESULT'
LEVEL_MSG = 'MSG'
LEVEL_NOTICE = 'NOTICE'
LEVEL_WARNING = 'WARNING'
LEVEL_ERROR = 'ERROR'


def add(_id, function, message, _level=LEVEL_MSG, output=False):
    # construct log message
    _time = str(time.strftime('%Y-%m-%d %H:%M:%S'))
    string = f'{_id} : {_level} : {_time} : {function} : {message}\n'

    # write log
    with open(path_lib.get_relative_file_path('log', MODEL, f'{VARIANT}.log'), 'ab') as f:
        f.write(string.encode('utf-8'))

    # show to console
    if output:
        print(string.strip())


def new_line(output=False):
    with open(path_lib.get_relative_file_path('log', MODEL, f'{VARIANT}.log'), 'ab') as f:
        f.write('\n'.encode('utf-8'))

    if output:
        print('')


def new_paragraph(output=False):
    string = '\n\n----------------------------------------------\n'
    with open(path_lib.get_relative_file_path('log', MODEL, f'{VARIANT}.log'), 'ab') as f:
        f.write(string.encode('utf-8'))

    if output:
        print(string)
