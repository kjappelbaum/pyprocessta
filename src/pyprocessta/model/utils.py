def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for mod in model.model.children():
        for m2 in mod.children():
            for m3 in m2.children():
                if m3.__class__.__name__.startswith('Dropout'):
                    m3.train()