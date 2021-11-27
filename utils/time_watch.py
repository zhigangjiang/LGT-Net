""" 
@Date: 2021/07/18
@description:
"""
import time


class TimeWatch:
    def __init__(self, name="", logger=None):
        self.name = name
        self.start = time.time()
        self.logger = logger

    def __del__(self):
        end = time.time()
        output = f"{self.name} | time use {(end - self.start):.2f}s."
        if self.logger:
            self.logger.info(output)
        else:
            print(output)


if __name__ == '__main__':
    w = TimeWatch("__main__")
    time.sleep(2)