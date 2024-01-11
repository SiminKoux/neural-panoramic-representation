
def check_dims_dsets(dsets):
    test = next(iter(dsets))
    return test.height, test.width

def check_names_dsets(dsets):
    test = next(iter(dsets))
    return test.names