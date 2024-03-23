import at
from dt4acc.model.twiss import Twiss
from lat2db.tools.factories.pyat import factory
from pymongo import MongoClient


def get_bessyii_machine_calculator(
    mongo_url: str="mongodb://127.0.0.1:27017/",
    database: str="bessyii",
    collection: str="machines"
):
    """
    Todo:
        need to work on return, currently only returing twiss
    Args:
        mongo_url:
        database:
        collection:

    Returns:

    """
    # get database
    client = MongoClient(mongo_url)
    db = client[database]
    collection = db[collection]
    lattice_in_json_format = collection.find_one()

    seq = factory(lattice_in_json_format)
    assert(seq)
    ring = at.Lattice(seq, name='bessy2', periodicity=1, energy=1.7e9)

    # set up of calculation choice
    ring.enable_6d()  # Should 6D be default?
    # Set main cavity phases
    ring.set_cavity_phase(cavpts='CAV*')
    return ring

def twiss_from_at() -> Twiss:
    from dt4acc.resources.bessy2_sr_reflat import bessy2Lattice
    from dt4acc.calculator.pyat_calculator import PyAtTwissCalculator

    twiss_calculator = PyAtTwissCalculator(get_bessyii_machine_calculator())
    return twiss_calculator.calculate()
