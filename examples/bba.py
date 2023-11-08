from matplotlib import pyplot as plt

from bact_analysis_bessyii.bba import app
import logging
# logging.basicConfig(level=logging.INFO)
logger= logging.getLogger('bact-analysis-bessyii')

if __name__ == "__main__":
    import sys
    try:
        name, uid = sys.argv
    except ValueError:
        print("need one argument! a uid")
        if True:
            sys.exit()

    # uid = '9ba454c7-f709-4c42-84b3-410b5ac05d9d'
    # uid = "e60215ff-62ea-4d3b-a968-f6b0d9d9ee9d"
    print(f"Using uid for testing {uid}")

    try:
        app.main(uid)
    except Exception as exc:
        raise exc
    finally:
        plt.show()