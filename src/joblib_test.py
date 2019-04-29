import logging
from joblib import Parallel, delayed, Logger

logging.basicConfig(format='%(asctime)-15s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.DEBUG, filename='joblib.log', filemode='a')

logger = logging.getLogger("test")


def do_something(num):
    logging.basicConfig(format='%(name)s - %(levelname)s - %(message)s',
                        level=logging.DEBUG, filename='joblib.log', filemode='a')
    # slog = logging.getLogger('joblib')
    # logging.info('handling {}'.format(num))
    vlog = Logger()
    vlog.warn('handling {}'.format(num))
    # vlog.format(num)
    return num * num


if __name__ == '__main__':
    logger.info('start')
    items = Parallel(n_jobs=4)(delayed(do_something)(x) for x in range(100))
    logger.info('finished')

