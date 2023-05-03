class Status:
    stat = None

    def __init__(self, epoch):
        if Status.stat is None:
            self.max = epoch
            self.count = 0
            self.tic_mark = 100
            self.env = ""
            self.param = ""
            self.value = ""
            Status.stat = self

    @staticmethod
    def get_status():
        return Status.stat

    def tic_epoch(self, msg=None):
        #if msg is not None:
            #print("\r"+msg)

        self.count += 1
        ratio = self.count / self.max
        done = int(ratio * self.tic_mark)
        notdone = self.tic_mark - done
        line = "\r[{hash}{dash}] {percent:.4f}% {env} -> {param}= {value}".format(hash="#" * done,
                                                                                dash="_" * notdone,
                                                                                percent=ratio * 100,
                                                                                env=self.env,
                                                                                param=self.param,
                                                                                value=self.value)
        print(line, end="")

    def print(self, msg):
        #print("\r"+msg)
        ratio = self.count / self.max
        done = int(ratio * self.tic_mark)
        notdone = self.tic_mark - done
        line = "\r[{hash}{dash}] {percent:.4f}% {env} -> {param}= {value}".format(hash="#" * done,
                                                                                dash="_" * notdone,
                                                                                percent=ratio * 100,
                                                                                env=self.env,
                                                                                param=self.param,
                                                                                value=self.value)
        print(line, end="")

    def set_env(self, env):
        self.env = env

    def set_param(self, param):
        self.param = param

    def set_value(self, value):
        self.value = str(value)
