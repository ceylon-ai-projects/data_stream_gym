from baseconv import BaseConverter

x = 0.01
encoder_base = BaseConverter('USD')

filter_ranges = [
    {'range': (-100, -3 * x), 'label': 'D'},
    {'range': (-3 * x, -2 * x), 'label': 'D'},
    {'range': (-2 * x, -x), 'label': 'D'},
    {'range': (-x, x), 'label': 'S'},
    {'range': (x, 2 * x), 'label': 'U'},
    {'range': (2 * x, 3 * x), 'label': 'U'},
    {'range': (3 * x, 100), 'label': 'U'},
]


def pattern_to_action(pattern):
    pt_i = pattern[:1][0]
    pt_t = pattern[-1:][0]
    pt = pt_i + pt_t
    # print(pt)
    if pt == 'UU' or pt == 'SU':
        return 2
    elif pt == 'DD' or pt == 'SD':
        return 0
    else:
        return 1


def encode_column_to_range_index(x, i=0, alpha=0.001):
    # print(x,i)
    for f in filter_ranges:
        if f['range'][0] <= x + (alpha * i) <= f['range'][1]:
            return f['label']
    # print("None",x)
    return "U"


def decode_column_to_int(x):
    return int(encoder_base.decode(x)) / float(encoder_base.decode("".rjust(len(x), 'D')))



encode_column_to_range_index(-0.02)

decode_column_to_int("UUUDSDSSS")
encoder_base.decode("DDDDDDDDD")