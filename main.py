#!/usr/bin/env python3

from activity_recognition_engine import ActivityRecognitionEngine
from gabriel_server.local_engine import runner


INPUT_QUEUE_MAXSIZE = 60
PORT = 9099
TOKENS = 2


def main():
    def engine_setup():
        return ActivityRecognitionEngine()

    runner.run(engine_setup, ActivityRecognitionEngine.ENGINE_NAME, INPUT_QUEUE_MAXSIZE, PORT, TOKENS)


if __name__ == '__main__':
    main()
