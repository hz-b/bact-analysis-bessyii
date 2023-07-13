from bact_analysis_bessyii.bba import app


if __name__ == "__main__":
    import sys
    try:
        _, uid = sys.argv
    except ValueError:
        print("need one argument! a uid")
        sys.exit()
    app.main(uid)
