class TrajectoryState:
    def __init__(self):
        self.x_predict = None
        self.P_predict = None
        self.x_update = None
        self.P_update = None
        self.score = None
        self.dets = None

