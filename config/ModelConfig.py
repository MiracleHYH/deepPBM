class VAEConfig:
    def __init__(self, device, img_size, latent_dim, feature_row, feature_col):
        self.Device = device
        self.L1 = 32
        self.L2 = 64
        self.L3 = 128
        self.L4 = 128
        self.L5 = 2400
        self.LatentDim = latent_dim
        self.KernelSize = (4, 4)
        self.PoolSize = 2
        self.Stride = 2
        self.EPS = 1e-5
        self.ImageSize = img_size
        self.FeatureRow = feature_row
        self.FeatureCol = feature_col
