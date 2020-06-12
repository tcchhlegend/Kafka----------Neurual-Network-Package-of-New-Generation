def create_none_dict(li: list):
    vals = [None] * len(li)
    return dict(zip(li, vals))


def to_tuple(shape):
    if hasattr(shape, '__iter__'):
        return tuple(shape)
    else:
        return shape,
