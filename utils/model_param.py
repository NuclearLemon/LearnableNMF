class ModelParam:

    def __init__(self, batch, layer, lr_set, seed, rho, epoch):
        self.batch, self.layer = batch, layer
        self.lr_set = lr_set
        self.seed = seed
        self.rho, self.epoch = rho, epoch

